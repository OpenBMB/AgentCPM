#!/bin/bash
set -euo pipefail

# --- 用户配置区 ---
NUM_PROCESSES=<YOUR_NUM_PROCESSES>
if [ $# -lt 1 ]; then
  echo "用法: $0 <RUN_NAME>"
  exit 1
fi
RUN_NAME="$1"

FILE_DIR=$(cd "$(dirname "$0")" && pwd)
# ---------------------
# 创建日志目录
mkdir -p logs/as

cleanup() {
  echo -e "\n>>> Cleanup: killing training processes..."
  pkill -f 'accelerate launch' || true
}
trap cleanup EXIT SIGINT SIGTERM

echo "Training directory: $FILE_DIR"
echo "Launching training locally..."

export TOKENIZERS_PARALLELISM=false

#多机多卡配置
export MACHINE_RANK=0             # 节点编号
export NUM_MACHINES=1             # 总机器数
export MASTER_ADDR="<YOUR_MASTER_ADDR>" 
export MASTER_PORT="<YOUR_MASTER_PORT>"

# 通信配置
export NCCL_SOCKET_IFNAME="<YOUR_NCCL_SOCKET_IFNAME>"
export GLOO_SOCKET_IFNAME="<YOUR_GLOO_SOCKET_IFNAME>"
export TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_TIMEOUT=14400000
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL="<YOUR_NCCL_NET_GDR_LEVEL>"
export NCCL_IB_HCA="<YOUR_NCCL_IB_HCA>"
unset NCCL_IB_GID_INDEX
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export MONITOR_INTERVAL=5
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export LOG_LEVEL=debug

# swanlab配置
export SWANLAB_API_KEY="<YOUR_SWANLAB_API_KEY>"
export SWANLAB_PROJECT="<YOUR_SWANLAB_PROJECT>"
export SWANLAB_WORKSPACE="<YOUR_SWANLAB_WORKSPACE>"

accelerate launch \
    --config_file assets/fsdp2_dst.yml \
    --num_processes=$NUM_PROCESSES \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$NUM_MACHINES \
    \
    src/main.py \
    --model_name_or_path "qwen3-4b-thinking-2507" \
    --trainer "QwenTrainer" \
    --db_connection_string "mongodb://11.11.22.3:27016/$RUN_NAME?replicaSet=rs0" \
    --cpu_ram_efficient_loading true \
    --weight_decay 0.05 \
    --optim_args 'beta1=0.9,beta2=0.98'\
    --token_level_loss true \
    --max_steps 1000 \
    --num_train_epochs 5 \
    --adv_scaling 2.0 \
    --max_tokens 131072 \
    --loss_calculater "GRPO" \
    --max_trained_count 1 \
    --train_file "assets/fetch/gaia_text.jsonl" \
    --dataloader_drop_last true \
    --dataloader_num_workers 1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --learning_rate 2e-6 \
    --warmup_steps 5 \
    --lr_scheduler_type "cosine" \
    --bf16 \
    --tp_size 4 \
    --pp_size 1 \
    --activation_offloading false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 48 \
    --seed 42 \
    --eval_strategy "no" \
    --eval_steps 5 \
    --run_name $RUN_NAME \
    --report_to swanlab \
    --save_only_model true \
    --save_steps 3 \
    --dataloader_prefetch_factor 4 \
    --balance_sample true \
    --accelerator_config '{"split_batches":true}' \
    --pad_to_multiple_of 4096 \
    \
    --enable_sampling true \
    --agent_config_path "assets/agent_config.yml" \
    --mcp_manager_url "http://11.11.23.2:9700/mcpapi" \
    --max_new_tokens 32768 \
    --max_prompt_tokens 128768 \
    --num_generations 8 \
    --tool_call_parser "qwen" \
    --preferred_sampling_params '{"temperature": 1, "top_p": 1, "min_p": 0, "top_k": -1}' \
    --presence_penalty 0.0 \
    --frequency_penalty 0.0 \
    --inf_tp_size 1 \
    --inf_mem_ratio 0.8 \
    --max_concurrent_samples_per_process 4 \
    --target_concurrency 3 \
    --max_turns 200 \
    --repetition_early_stop_times 4 \
    --enable_repetition_compress false \
    \
    --output_dir output/$RUN_NAME \
    > logs/as/${RUN_NAME}_train_${MACHINE_RANK}.log 2>&1

