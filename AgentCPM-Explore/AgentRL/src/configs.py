# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import TrainingArguments

@dataclass
class AgentTrainingConfig(TrainingArguments):
    r"""
    Configuration class for the [`AgentTrainingConfig`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
    """
    trainer: Literal["QwenTrainer","MiniCPMVTrainer", "QwenVLTrainer", "QwenMoETrainer"] = field(
        default="QwenTrainer",
        metadata={
            "help": "Trainer class to use for training. See `training/trainer/` for available options."
        }
    )

    # Parameters that control the dataset
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the training dataset file. This should be a JSON or JSONL file containing the training data. "
            "If not provided, the dataset will be read from the database."
        },
    )

    max_pixels: Optional[int] = field(
        default=1920*1080,
        metadata={
            "help": "Maximum number of pixels for the vision input. Images larger than this will be resized. "
        }
    )


    remove_experience: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove experience in samples."
        }
    )

    drop_zero_advantage: bool = field(
        default=True,
        metadata={
            "help": "Whether to drop samples with zero advantage."
        }
    )
    
    shift_labels: bool = field(
        default=True,
        metadata={
            "help": "Whether to shift the labels for causal language modeling."
        }
    )
    

    db_connection_string: str = field(
        default="",
        metadata={"help": "Database connection string for storing trajectories."}
    )

    oss_connection_string: str = field(
        default="",
        metadata={
            "help": "MinIO connection string in format: minio://access_key:secret_key@endpoint/bucket?secure=true|false.",
        }
    )

    enable_oss: bool = field(
        default=False,
        metadata={
            "help": "Global switch to enable object storage (OSS/MinIO) integration."
        }
    )

    skip_non_positive_advantage: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip samples with non-positive advantage. If set to `True`, samples with non-positive advantage will be skipped during training."
        }
    )

    # Parameters that control the model
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model checkpoint or model identifier from the Hugging Face Hub "
            "If the `model` argument of the `GRPOTrainer` is provided as a string, this will be used to initialize the model."
        },
    )

    attention_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": "Attention implementation to use for the model. This can be one of 'flash_attention_2' or 'sdpa'. "
        }
    )

    output_router_logits: bool = field(
        default=False,
        metadata={
            "help": "Whether to output the router logits from the MoE layers. "
        }
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to trust the remote code when loading the model. "
            "This is useful when using models that require custom code to be loaded. "
        }
    )

    cpu_ram_efficient_loading: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the model in a CPU RAM efficient way. "
            "This is useful when the model is too large to fit in GPU memory. "
        }
    )

    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "Data type for the model. This can be one of 'float16', 'bfloat16'. "
            "The default is 'bfloat16', which is suitable for most models. "
        }
    )

    tp_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism size for the model. This is used to split the model across multiple devices. "
            "For example, if set to 8, the model will be split across 8 devices."
        },
    )

    pp_size: int = field(
        default=1,
        metadata={
            "help": "Pipeline parallelism size for the model. This is used to split the model across multiple devices in a pipeline fashion. "
            "For example, if set to 4, the model will be split across 4 devices in a pipeline fashion."
        },
    )

    ep_size: int = field(
        default=1,
        metadata={
            "help": "Expert parallelism size for the model. This is used to split the experts across multiple devices. "
            "For example, if set to 2, the experts will be split across 2 devices."
        },
    )

    cp_size: int = field(
        default=1,
        metadata={
            "help": "Context parallelism size for the model. This is used to split the context across multiple devices. "
        }
    )

    pad_to_multiple_of: int = field(
        default=1,
        metadata={
            "help": "Padding to multiple of for the model. This is used to pad the inputs to the multiple of the specified value. "
        }
    )

    bypass_causal_mask_check: bool = field(
        default=False,
        metadata={
            "help": "Whether to bypass the causal mask check for SDPA attention. "
            "This is useful when the attention mask is known to be causal but does not strictly follow the expected format."
        }
    )

    enable_loss_parallel: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable loss parallelism. This is useful when the model's vocabulary size is too large to fit in GPU memory."
        }
    )
    
    loss_seq_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Sequence chunk size for computing loss. This is useful when the model's sequence length is too long to fit in GPU memory."
            "Warning: setting this to a very small value may slow down the training."
        }
    )
    
    activation_offloading: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable activation offloading. This is useful when the model is too large to fit in GPU memory. "
            "When enabled, activations will be offloaded to CPU memory during the forward pass. "
        },
    )

    decoder_sparse_step: Optional[int] = field(
        default=None,
        metadata={
            "help": "Sparse step for the decoder. If set, the decoder will use sparse moe experts with the specified sparse step. "
        }
    )

    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )


    pad_to_maximum: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad the inputs to the maximum length. "
            "This is useful when using pipeline parallelism to ensure that all batches have the same length. "
        }
    )

    dynamic_batching: bool = field(
        default=False,
        metadata={
            "help": "Whether to use dynamic batching during batching. "
            "If set to `True`, batches will be padded to the max_prompt_tokens by merging samples dynamically. "
        }
    )

    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )

    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    epsilon_higher: float = field(
        default=0.1,
        metadata={
            "help": "Delta of epsilon value for higher bound clipping. See DAPO(https://arxiv.org/abs/2503.14476) paper for more details.",
        }
    )

    beta1: Optional[float] = field(
        default=None,
        metadata={
            "help": "Beta1 parameter for the CE-GRPO algorithm. If not set, the low stop-gradient samples will not be reweighted."
        }
    )

    beta2: Optional[float] = field(
        default=None,
        metadata={
            "help": "Beta2 parameter for the CE-GRPO algorithm. If not set, the high stop-gradient samples will not be reweighted."
        }
    )

    tune_vision: bool = field(
        default=True,
        metadata={
            "help": "Whether to tune the vision part of the VL model."
        }
    )


    adv_scaling: float = field(
        default=1.0,
        metadata={
            "help": "Scaling factor for the advantage. This is used to scale the advantage before computing the loss. "
        }
    )

    max_tokens: int = field(
        default=65536,
        metadata={
            "help": "Maximum number of tokens to generate. This is used to limit the number of tokens generated by the model. "
        }
    )

    balance_sample: bool = field(
        default=True,
        metadata={
            "help": "Whether to balance the sample. If set to `True`, the sample will be balanced. "
        }
    )

    token_level_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to compute token-level loss in GRPO. If set to `False`, only the loss on current rank will be reduced."
        }
    )

    out_of_date_decay: Optional[float] = field(
        default=None,
        metadata={
            "help": "Decay factor for out-of-date samples in the database. Useful with off-policy training."
        }
    )
        
    importance_weight_cap_ratio: float = field(
        default=4.0,
        metadata={
            "help": "The cap for behavior importance weight ratio to mitigate high variance. "
        }
    )
    
    strict_in_bound: bool = field(
        default=False,
        metadata={
            "help": "Whether to strictly enforce the importance weight bounds in [1-epsilon,1+epsilon+higher] by dropping out-of-bound samples."
        }
    )
    
    log_entropy: bool = field(
        default=False,
        metadata={
            "help": "Whether to log the token-level entropy during training. If set to `True`, the entropy will be computed and logged in the training metrics."
        }
    )
    
    skip_length_normalization: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip length normalization when computing the GRPO loss."
        }
    )

    loss_calculater: Literal["GRPO","CrossEntropy","GSPO","MINIRL"] = field(
        default="GRPO",
        metadata={
            "help": "Loss function to use for training. This can be one of 'GRPO' or 'CrossEntropy'. "
        }
    )

    max_trained_count: int = field(
        default=1,
        metadata={
            "help": "Maximum number of trained count for each record. "
        }
    )

    retrained_interval: int = field(
        default=1,
        metadata={
            "help": "Interval between the record repeat training steps."
        }
    )

    minimal_advantage: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum advantage value for sampling. If set, only samples with advantage greater than this value will be considered."
        }
    )

    # ============================================================================
    # Parameters that control the sampler
    # ============================================================================
    
    enable_sampling: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable sampling. If set to `False`, the training will not sample any data and will only train on the provided dataset."
        },
    )

    sync_sampling: bool = field(
        default=False,
        metadata={
            "help": "Whether to use synchronous sampling. If set to `True`, the training will wait for all sampling tasks to complete before proceeding to the next step."
        },
    )

    max_new_tokens: Optional[int] = field(
        default=8196,
        metadata={
            "help": "Maximum number of tokens to sample in each generation. "
        }
    )

    max_prompt_tokens: Optional[int] = field(
        default=5120000,
        metadata={
            "help": "Maximum number of prompt tokens in each generation. "
        }
    )

    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )

    tool_call_parser: Literal[
        "qwen",
        "qwen25",
        "mistral",
        "llama3",
        "deepseekv3",
        "pythonic",
        "kimi_k2",
    ] = field(
        default="qwen25",
        metadata={
            "help":"Specify the parser for handling tool-call interactions. Options include: 'qwen25', 'mistral', 'llama3', 'deepseekv3', 'pythonic', and 'kimi_k2'.",
        }
    )

    model_name: str = field(
        default="train-model",
        metadata={"help": "Name of the model to use for sampling."}
    )

    inf_tp_size: int = field(
        default=1,
        metadata={"help": "Tensor parallelism size for the inference model."}
    )

    inf_ep_size: int = field(
        default=1,
        metadata={"help": "Expert parallelism size for the inference model."}
    )

    max_concurrent_samples_per_process: int = field(
        default=1,
        metadata={
            "help": "Maximum number of concurrent samples that can be processed by a single process."
        }
    )

    target_concurrency: int = field(
        default=1,
        metadata={
            "help": "Target concurrency for each inference services. Increasing this if you find that the inference service is not fully utilized."
        }
    )

    inf_mem_ratio: float = field(
        default=0.5,
        metadata={
            "help": "Memory fraction for the inference service. This is used to limit the memory usage of the inference service. "
            "It is a fraction of the total memory available on the device. "
            "For example, if set to 0.7, the inference service will use at most 70% of the total memory available on the device."
        },
    )

    top_k: int = field(
        default=1,
        metadata={"help": "Number of top-k tokens that should be stored for calculating the importance sampling."}
    )

    presence_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Presence penalty for sampling. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."
        }
    )

    frequency_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Frequency penalty for sampling. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."
        }
    )

    preferred_sampling_params: str = field(
        default='{"temperature": 1, "top_p": 1, "min_p": 0, "top_k": -1}',
        metadata={
            "help": "Preferred sampling parameters for inference as JSON string. This is a JSON string containing parameters like 'temperature', 'top_p', 'min_p', 'top_k' that will be used during sampling. Example: '{\"temperature\": 1, \"top_p\": 1, \"min_p\": 0, \"top_k\": -1}'"
        }
    )

    
    resample_error_records: bool = field(
        default=False,
        metadata={
            "help": "Whether to resample records that encountered errors during previous sampling attempts."
        }
    )


    # Parameters that control MCP sampling
    mcp_server_name: str = field(
        default="all",
        metadata={
            "help": (
                "Which MCP server to call. "
                "In MCPAPI mode (manager URL ends with /mcpapi), multiple servers can be aggregated. "
                "Set a specific server name or use 'all' to aggregate every server. "
                "Examples: 'web_search' targets only that server; 'all' aggregates all."
            )
        }
    )

    mcp_manager_url: str = field(
        default="http://localhost:9876/mcpapi",
        metadata={
            "help": (
                "Endpoint for the MCP manager; it also selects the mode:\n"
                "- MCPAPI (custom aggregator): URL ends with /mcpapi → uses MCPAPIHandler, supports multi-server aggregation and tool filtering.\n"
                "  Example: 'http://localhost:9700/mcpapi' (custom manager with multiple servers).\n"
                "- MCPSDK (official protocol): other URLs → uses official SDK (MCPHandler/MCPManager), connects to a single official MCP server (SSE/stdio).\n"
                "  Example: 'http://localhost:8088/mcp' (official streamable HTTP service).\n"
                "Must include scheme/host/port/path. The sampler routes to MCPAPIHandler or MCPHandler based on this."
            )
        }
    )

    mcp_auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional Authorization token (Bearer) for MCP streamable HTTP (official MCPSDK/SSE). "
                "If set, sent as 'Authorization: Bearer <token>' when connecting to the MCP server. "
                "Ignored for the custom MCPAPI aggregator."
            )
        }
    )

    max_turns: int = field(
        default=25,
        metadata={
            "help": "Maximum number of turns for MCP tool-using tasks."
        }
    )

    min_turns: int = field(
        default=1,
        metadata={
            "help": "Minimum number of turns for records to be trained. If the number of turns is less than this value, the record will be skipped."
        }
    )

    repetition_early_stop_times: int = field(
        default=-1,
        metadata={
            "help": "Number of repetitions before early stopping. When repetition is detected, if the same content appears this many times, the sampler will stop early. Set to -1 or 0 to disable early stopping. Default is -1 (disabled)."
        }
    )

    enable_repetition_compress: bool = field(
        default=False,
        metadata={
            "help": "Whether to trigger history compression when repetition is detected. If set to `True`, the sampler will compress conversation history when the response content matches a previous message exactly."
        }
    )

    new_context: Optional[Literal["disabled", "discard_all_tools", "discard_all"]] = field(
        default="disabled",
        metadata={
            "help": "Reset mode when repetition or context limit is reached. Options: 'disabled' (exit normally), 'discard_all_tools' (clear all tool messages and reset messages), 'discard_all' (clear all tool messages, generate summary, and replace messages after the 2nd one with summary)."
        }
    )

    agent_config_path: Optional[str] = field(
        default="assets/agent_config.yml",
        metadata={
            "help": "Path to the agent configuration YAML file. This file contains configuration for browse_agent and scorer_agent with multiple models for fallback. "
            "The scorer_agent configuration replaces the old judge_api_key, judge_base_url, and judge_model parameters."
        }
    )

    reserve_context_error_count: int = field(
        default=0,
        metadata={
            "help": "Number of context error records (context_limit_error and turns_limit_error) to randomly reserve for training. "
            "This helps preserve important information about context-related failures while avoiding training on too many error records."
        }
    )

    efficient_rewarding: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable efficient rewarding. If set to `True`, the sampler will reward the records with efficient way additionally."
        }
    )

    # ============================================================================
    # End of sampler parameters
    # ============================================================================