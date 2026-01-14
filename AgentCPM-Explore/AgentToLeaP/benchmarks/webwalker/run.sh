#!/bin/bash

# =====================================================================================
# --- User Configuration Section ---
# Please modify all parameters you want to evaluate here

# --- Switch: Whether to return thought in content ("true" or "false") ---
RETURN_THOUGHT_TO_LLM="true"

# --- Switch: Whether to enable browser processor ("true" or "false") ---
USE_BROWSER_PROCESSOR="true"

# --- Switch: Whether to enable Context Manager ("true" or "false") ---
USE_CONTEXT_MANAGER="false"

# --- Pass@k and Temperature, Top_p, Presence_penalty, Max_tokens configuration ---
PASS_K=8
TEMPERATURE=1
TOP_P=1
PRESENCE_PENALTY=0
MAX_TOKENS=16384

# --- 1. Main model configuration (e.g.: DeepSeek API) ---
PROVIDER="openai"
MODEL_NAME=""             # Corresponds to "model" parameter in API request
RESULT_DIR_NAME=""        # Short identifier for generating result folder name
BASE_URL=""
API_KEY=""


# --- 2. Processor model configuration (e.g.: Qwen on Tianyi Cloud) ---
PROCESSOR_PROVIDER="openai"
PROCESSOR_MODEL_NAME="" # Corresponds to "model" parameter in API request
PROCESSOR_BASE_URL=""
PROCESSOR_API_KEY="dada"
# ===========================================================

# --- 3. General evaluation configuration ---
MANAGER_URL=""
EVALUATION_ROOT_DIR="" # Root directory for storing evaluation results
FILES_DIR="" # Directory for dataset attached files

MAX_SAMPLES=-1
NUM_PROCESSES=120
MAX_INTERACTIONS=200

# --- 4. Custom tool call text tags---
TOOL_START_TAG="<tool_call>"
TOOL_END_TAG="</tool_call>"

# --- 5. Tokenizer path ---
TOKENIZER_PATH="" # Optional: HuggingFace tokenizer path or model name

# --- Switch: Whether to use llm-as-judge for dataset results, will replace the original code scoring method ---
USE_LLM_JUDGE="true"
LLM_JUDGE_API_KEY=""
LLM_JUDGE_API_BASE=""

# =====================================================================================
# --- Execution Section (usually no modification needed) ---
# =====================================================================================

# --- Auto-build paths ---
# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# Get the name of this directory as benchmark name
BENCHMARK_NAME=$(basename "$SCRIPT_DIR")
# Build input and output paths
INPUT_FILE="$SCRIPT_DIR/$BENCHMARK_NAME.jsonl"
RAW_OUTPUT_DIR="$EVALUATION_ROOT_DIR/_temp_raw_outputs/${BENCHMARK_NAME}_${RESULT_DIR_NAME}"

echo "========================================="
echo "Starting Benchmark Evaluation and Analysis (Dual Model + Context Manager)"
echo "========================================="
echo "  - Benchmark: $BENCHMARK_NAME"
echo "  --- Main Model ---"
echo "  - Model Name (API): $MODEL_NAME"
echo "  - Result Identifier: $RESULT_DIR_NAME"
echo "  - API Address: $BASE_URL"
echo "  --- Processor Model ---"
echo "  - Model Name (API): $PROCESSOR_MODEL_NAME"
echo "  - API Address: $PROCESSOR_BASE_URL"
echo "  -----------------------------------------"
echo "  - Tool Server: $MANAGER_URL"
echo "  - Evaluation Results Root Directory: $EVALUATION_ROOT_DIR"
echo "-----------------------------------------"


USE_BROWSER_PROCESSOR_FLAG=""
if [[ "$USE_BROWSER_PROCESSOR" == "true" ]]; then
    USE_BROWSER_PROCESSOR_FLAG="--use-browser-processor"
    echo "  - Browser Processor: Enabled"
else
    echo "  - Browser Processor: Disabled"
fi

RETURN_THOUGHT_FLAG=""
if [[ "$RETURN_THOUGHT_TO_LLM" == "true" ]]; then
    RETURN_THOUGHT_FLAG="--return-thought"
    echo "  - Return Thought Process: Enabled"
else
    echo "  - Return Thought Process: Disabled"
fi

USE_CONTEXT_MANAGER_FLAG=""
if [[ "$USE_CONTEXT_MANAGER" == "true" ]]; then
    USE_CONTEXT_MANAGER_FLAG="--use-context-manager"
    echo "  - Context Manager: Enabled"
else
    echo "  - Context Manager: Disabled"
fi

LLM_JUDGE_ARGS=()
if [[ "$USE_LLM_JUDGE" == "true" ]]; then
    LLM_JUDGE_ARGS+=( "--llm-judge" )
    LLM_JUDGE_ARGS+=( "--llm-judge-api-key" "$LLM_JUDGE_API_KEY" )
    LLM_JUDGE_ARGS+=( "--llm-judge-api-base" "$LLM_JUDGE_API_BASE" )
    echo "  - LLM as Judge:   [!!! ENABLED !!!]"
    echo "  - LLM Judge API:  $LLM_JUDGE_API_BASE"
else
    echo "  - LLM as Judge:   [Disabled]"
fi
echo "-----------------------------------------"


export MONGO_URI=""
export MONGO_DB=""
SAFE_MODEL_NAME=$(echo "$RESULT_DIR_NAME" | tr '/: ' '___')
export MONGO_COL="${BENCHMARK_NAME}_${SAFE_MODEL_NAME}"
export MONGO_CREATE_INDEX="0"

echo "[preflight] env mongo vars:"
env | grep -E '^MONGO_' || echo "[preflight] (none)"
python - <<'PY'
import os
keys = ["MONGO_URI","MONGO_DB","MONGO_COL","MONGO_CREATE_INDEX"]
print("[preflight python]", {k: (os.getenv(k)[:40]+"..." if os.getenv(k) and k=="MONGO_URI" else os.getenv(k)) for k in keys})
PY

python "$SCRIPT_DIR/../../run_evaluation.py" \
    --provider "$PROVIDER" \
    --model "$MODEL_NAME" \
    --model-name "$RESULT_DIR_NAME" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --processor-provider "$PROCESSOR_PROVIDER" \
    --processor-model "$PROCESSOR_MODEL_NAME" \
    --processor-base-url "$PROCESSOR_BASE_URL" \
    --processor-api-key "$PROCESSOR_API_KEY" \
    --benchmark-name "$BENCHMARK_NAME" \
    --manager-url "$MANAGER_URL" \
    --input-file "$INPUT_FILE" \
    --output-dir "$RAW_OUTPUT_DIR" \
    --output-base-dir "$EVALUATION_ROOT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --num-processes "$NUM_PROCESSES" \
    --max-interactions "$MAX_INTERACTIONS" \
    --tool-start-tag "$TOOL_START_TAG" \
    --tool-end-tag "$TOOL_END_TAG" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --files-dir "$FILES_DIR" \
    $USE_BROWSER_PROCESSOR_FLAG \
    $RETURN_THOUGHT_FLAG \
    $USE_CONTEXT_MANAGER_FLAG \
    "${LLM_JUDGE_ARGS[@]}" \
    --k "$PASS_K" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --presence-penalty "$PRESENCE_PENALTY" \
    --max-tokens "$MAX_TOKENS"

echo "========================================="
echo "All tasks completed."
echo "========================================="
read -p "Press any key to continue..."