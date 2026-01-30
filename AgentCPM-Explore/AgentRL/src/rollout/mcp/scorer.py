"""
Scorer for MCP tool-using tasks.
"""
import re
import string
import asyncio
from typing import Dict, Any, Callable
from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError
from log import logger


def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def bool_mapping(s):
    """Map boolean values to yes/no."""
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def em_check(prediction, golden_answers):
    """Exact match check."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(bool_mapping(golden_answer))
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluation assistant.
    Your task is to judge whether the predicted answer correctly answers the question, using the labeled answer as the ground truth.


    Evaluation Criteria:
    - **Primary focus**: Does the predicted answer correctly capture the **key information** needed to answer the question?
    - Use the labeled answer as the ground truth to identify what the key information is.
    - Focus on **factual correctness and alignment of core information** rather than completeness or surface similarity.
    - **Key information alignment** (correct model information retrieval) is the most important factor.
    - Minor rewordings, synonyms, or paraphrases that preserve the key information should be considered correct.
    - **Missing non-critical/supplementary information** that can be covered is acceptable - still considered **correct**.
    - **Additional relevant details or more specific information** beyond the labeled answer is acceptable - still considered **correct**.
    - Only mark as **incorrect** if:
      * The predicted answer misses **critical/key information** required to answer the question
      * The predicted answer provides a **factually incorrect answer**
      * The predicted answer is **unrelated to or contradicts** the question's requirements

    ---
    Question:
    {question}

    Labeled Answer (Ground Truth):
    {labeled_answer}

    Predicted Answer:
    {pred_answer}
    ---

    Final Decision:
    Please respond with only one word:
    - "Correct" → if the predicted answer correctly captures the key information to answer the question (even if missing minor details or including additional details).
    - "Incorrect" → if it misses critical information or is unrelated to the question.

    Answer:"""


async def llm_judge_score(solution_str: str, ground_truth: Dict[str, Any], scorer_agent_config: Dict[str, Any]) -> bool:
    """
    Use LLM to judge whether predicted answer and ground truth are equivalent.

    Args:
        solution_str: Predicted answer
        ground_truth: Ground truth dict, contains "query" and "answer" fields
        scorer_agent_config: scorer_agent config dict, contains models list and other config

    Returns:
        - True: If LLM judges as "Correct"
        - False: If LLM judges as "Incorrect" or API call fails
    """
    # Validate inputs
    if not solution_str or not solution_str.strip():
        return False
    
    labeled_answer = ground_truth.get("answer", "")
    if not labeled_answer or not str(labeled_answer).strip():
        return False

    # Get prompt template from config, fallback to default if not provided
    judge_prompt_template = scorer_agent_config.get("judge_prompt_template")
    if not judge_prompt_template:
        # Fallback to default template
        judge_prompt_template = DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE
        logger.debug("Using default judge prompt template (not found in config)")
    else:
        logger.debug("Using judge prompt template from config")
    
    prompt = judge_prompt_template.format(
        question=ground_truth.get("query", ""),
        labeled_answer=str(labeled_answer),
        pred_answer=solution_str
    )

    models = scorer_agent_config.get("models", [])
    max_retries = scorer_agent_config.get("max_retries", 5)
    retry_delay = scorer_agent_config.get("retry_delay", 1)
    backoff_factor = scorer_agent_config.get("backoff_factor", 2)
    
    if not models:
        logger.error("No models configured in scorer_agent config")
        return False
    
    # Try each model in config, in order
    for i, model_config in enumerate(models):
        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model")
        
        if not api_key or not base_url or not model_name:
            logger.warning(f"Model {i+1} in scorer_agent config is missing required fields (api_key, base_url, or model), skipping")
            continue
        
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Trying scorer_agent model {i+1}: {model_name}")
            
            delay = retry_delay
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                    content = response.choices[0].message.content.strip().lower()

                    if "correct" in content and "incorrect" not in content:
                        return True
                    elif "incorrect" in content:
                        return False

                except OpenAIAPIError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor**attempt
                    else:
                        logger.warning(f"Scorer agent model {i+1} ({model_name}) API error after {max_retries} attempts: {e}")
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor**attempt
                    else:
                        logger.warning(f"Scorer agent model {i+1} ({model_name}) error after {max_retries} attempts: {e}")
                        break
        except Exception as e:
            logger.warning(f"Scorer agent model {i+1} ({model_name}) call exception: {e}")
            continue

    logger.error(f"All {len(models)} models in scorer_agent config failed")
    return False


async def agentcpm_scorer(model_answer: str, ground_truth: Dict[str, Any], scorer_agent_config: Dict[str, Any]) -> float:
    """
    Scorer for agentcpm tasks using LLM judge.
    
    Args:
        model_answer: Model predicted answer
        ground_truth: Ground truth answer, can be dict {"query": ..., "answer": ...} or string
        scorer_agent_config: scorer_agent config dict, contains models list and other config
    """
    # Validate model_answer
    if not model_answer or not str(model_answer).strip():
        return 0.0
    
    # Ensure model_answer is a string
    model_answer = str(model_answer).strip()
    
    if isinstance(ground_truth, dict):
        # Use LLM judge for scoring
        is_correct = await llm_judge_score(model_answer, ground_truth, scorer_agent_config)
        return float(is_correct)
    elif isinstance(ground_truth, str):
        # If ground_truth is string, convert to dict format
        ground_truth_str = str(ground_truth).strip()
        if not ground_truth_str:
            return 0.0
        ground_truth_dict = {"query": "", "answer": ground_truth_str}
        is_correct = await llm_judge_score(model_answer, ground_truth_dict, scorer_agent_config)
        return float(is_correct)
    else:
        return 0.0


class MCPScorerFactory:
    """Factory for creating scorers."""
    
    _func_map = {
        "agentcpm": agentcpm_scorer,
        "asearcher": agentcpm_scorer,  # Keep asearcher as alias for backward compatibility
    }
    
    @classmethod
    def get_scorer(cls, name: str) -> Callable:
        """
        Get a scorer by name.
        
        Args:
            name: Name of the scorer
            
        Returns:
            Scorer function (async)
        """
        if name not in cls._func_map:
            logger.warning(f"Scorer '{name}' not found, using 'agentcpm' as default")
            return cls._func_map.get("agentcpm", agentcpm_scorer)
        return cls._func_map[name]
    
    @classmethod
    def register_scorer(cls, name: str, scorer_func: Callable):
        """
        Register a new scorer.
        
        Args:
            name: Name of the scorer
            scorer_func: Scorer function (should be async)
        
        Example:
            ```python
            from rollout.mcp.scorer import MCPScorerFactory
            from typing import Dict, Any

            async def my_custom_scorer(
                model_answer: str, 
                ground_truth: Dict[str, Any], 
                scorer_agent_config: Dict[str, Any]
            ) -> float:
                \"\"\"
                Custom scorer function

                Args:
                    model_answer: Model predicted answer (string)
                    ground_truth: Ground truth dict, contains "query" and "answer" fields
                    scorer_agent_config: scorer_agent config dict

                Returns:
                    Score (float between 0.0 and 1.0)
                \"\"\"
                # Implement custom scoring logic
                # Can use model config from scorer_agent_config for LLM judge
                # Or implement other scoring strategies (e.g., exact match, fuzzy match, etc.)
                return score

            # Register custom scorer
            MCPScorerFactory.register_scorer("custom_scorer", my_custom_scorer)
            ```
        """
        cls._func_map[name] = scorer_func

