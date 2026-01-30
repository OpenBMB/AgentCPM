"""
MCP-specific data conversion functions.
"""
import statistics
from typing import AsyncGenerator, Any, Optional
from log import logger
from configs import AgentTrainingConfig
from databases import Record, DispatchedSamplingTask, RecordData
from training.datasets import is_contained_in_prefix, preprocess_mm_messages_for_sample
from .models import MCPTask


def should_skip_record(record: Record, args: Optional[AgentTrainingConfig] = None) -> bool:
    """
    Check if a record should be skipped based on status and final_answer.
    
    Args:
        record: The Record to check
        
    Returns:
        True if the record should be skipped, False otherwise
    """
    # Skip ABANDONED records
    if record.status == Record.Status.ABANDONED:
        logger.debug(f"Record {record.id} skipped: ABANDONED status")
        return True
        
    return False


def should_skip_sample(
    sample: DispatchedSamplingTask,
    args: Optional[AgentTrainingConfig] = None
) -> tuple[bool, Optional[str]]:
    """
    Check if a sample should be skipped based on various criteria.
    
    Args:
        sample: The DispatchedSamplingTask to check
        args: Optional AgentTrainingConfig for filtering parameters
        
    Returns:
        Tuple of (should_skip, reason). reason is None if should not skip.
    """
    # Skip samples with invalid status
    if sample.status not in [sample.Status.COMPLETED, sample.Status.TOOLFAILED, sample.Status.FORMATERROR, sample.Status.REPEATERROR]:
        return True, f"status {sample.status}"
    
    # Skip samples with context issues to avoid oom
    max_train_tokens = getattr(args, "max_train_tokens", (getattr(args, "max_tokens", None))) if args else None
    min_train_tokens = getattr(args, "min_train_tokens", None) if args else None
    
    if max_train_tokens is not None or min_train_tokens is not None:
        total_tokens = sample.response.get("usage", {}).get("total_tokens", 0)
        if max_train_tokens is not None and total_tokens > max_train_tokens:
            return True, f"total_tokens {total_tokens} > max_train_tokens {max_train_tokens}"
        if min_train_tokens is not None and total_tokens < min_train_tokens:
            return True, f"total_tokens {total_tokens} < min_train_tokens {min_train_tokens}"
    
    # You could skip samples with finish_reason == "length", but it is not necessary.
    # finish_reason = sample.response.get("choices", [{}])[0].get("finish_reason", None)
    # if finish_reason == "length":
    #     return True, "finish_reason == 'length'"
    
    return False, None


async def convert_record_to_data_mcp(
    record: Record,
    args: AgentTrainingConfig = None
) -> AsyncGenerator[RecordData, Any]:
    """Convert a Record object to a data dictionary with MCP-specific logic.
    
    This function includes:
    - record-level status filtering
    - sample-level status filtering
    - Leave-one-out advantage calculation
    - Advantage scaling
    - Auto-prefix merging
    """
    logger.info(f"Converting record {record.id} to data with MCP-specific logic.")
    task = await record.task.fetch(True)
    
    # Skip records based on status and final_answer
    if should_skip_record(record, args):
        return
    
    # Collect valid samples from trajectory
    samples: list[DispatchedSamplingTask] = []
    for item in record.traj:
        try:
            sample = await item.fetch(True)
            if sample is None:
                logger.warning(f"Sample {item.id} not found in database, skipping.")
                continue
        except Exception as e:
            logger.warning(f"Failed to fetch sample {item.id}: {type(e).__name__}: {str(e)}, skipping.")
            continue
        
        # Check if sample should be skipped
        should_skip, reason = should_skip_sample(sample, args)
        if should_skip:
            logger.info(f"Skipping sample {sample.id} due to: {reason}")
            continue
        
        samples.append(sample)

    # try to merge samples according to the prefix
    logger.info(f"[convert_mcp] Record {record.id}: Starting merge logic with {len(samples)} samples")
    unique_samples = []
    data_queue: list[RecordData] = []
    step = len(samples)
    for sample in reversed(samples):
        step -= 1
        contained = False
        matched_idx = None
        for idx, usample in enumerate(unique_samples):
            is_contained = is_contained_in_prefix(sample, usample)
            if is_contained:
                contained = True
                matched_idx = idx
                break
        
        # Calculate advantage with leave-one-out logic (similar to tool_datasets.py)
        score = sample.score if sample.score is not None else record.score
        
        # Calculate leave-one-out advantage (similar to tool_datasets.py)
        try:
            if task.scores and len(task.scores) > 1:
                # Find first matching index and exclude it
                try:
                    idx_to_remove = task.scores.index(score)
                    other_scores = task.scores[:idx_to_remove] + task.scores[idx_to_remove+1:]
                except ValueError:
                    # If score not found in task.scores, use all scores
                    other_scores = task.scores
                
                if other_scores:
                    mean_others = statistics.mean(other_scores)
                    # Calculate advantage without dividing by std
                    advantage = score - mean_others
                else:
                    advantage = 0
            else:
                advantage = 0
        except Exception as e:
            logger.error(f"Failed to calculate advantage for sample {sample.id}: {type(e).__name__}: {str(e)}")
            advantage = 0

        # use the advantage of the sample if it is not None and smaller than the calculated advantage
        if sample.advantage is not None:
            advantage = min(sample.advantage, advantage)
        # advantage clipping
        if sample.score <= 0.5:
            advantage = min(advantage, 0)
        else:
            advantage = max(advantage, 0)

        
        # Apply advantage scaling (similar to tool_datasets.py)
        adv_scaling = getattr(args, "adv_scaling", 1.0)
        advantage = advantage * adv_scaling

        response_index = len(sample.request["messages"])
        if not contained:
            unique_samples.append(sample)
            data_queue.append(
                RecordData(
                    messages=sample.request["messages"] + [sample.response["choices"][0]["message"]],
                    tools=sample.request.get("tools",None),
                    scores={response_index: score},
                    advantages={response_index: advantage},
                    logprobs={response_index: sample.response["choices"][0].get("logprobs", None)},
                    reward=record.score,
                    step=step,
                    created_at_step={response_index: sample.created_at_step}
                )
            )
        else:
            # data should be replaced
            # matched_idx is the matched index since we break when found
            target_data = data_queue[matched_idx]  # ✅ Use matched_idx instead of idx
            # Ensure messages list is long enough
            while len(target_data.messages) <= response_index:
                target_data.messages.append(None)
            target_data.messages[response_index] = sample.response["choices"][0]["message"]
            target_data.scores[response_index] = score
            target_data.advantages[response_index] = advantage
            target_data.logprobs[response_index] = sample.response["choices"][0].get("logprobs", None)
            target_data.created_at_step[response_index] = sample.created_at_step

    valid_data = []
    for data in data_queue:
        data.messages = await preprocess_mm_messages_for_sample(data.messages)
        # remove indexed with advantage < minimal advantage
        removed_index = list(filter(lambda idx: data.advantages[idx] < args.minimal_advantage if args.minimal_advantage else (args.drop_zero_advantage and data.advantages[idx] == 0), data.advantages.keys()))
        if removed_index:
            for idx in removed_index:
                data.scores.pop(idx)
                data.advantages.pop(idx)
        
        if data.scores:
            valid_data.append(data)
    
    if valid_data:
        logger.info("Find {} samples in record {}.".format(len(valid_data),record.id))
        for data in reversed(valid_data):
            yield data

