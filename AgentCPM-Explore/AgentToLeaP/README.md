# AgentToLeaP

<p align="center">
  【<a href="README_zh.md">中文</a> | English】
</p>

<p align="center">
  <strong>Evaluation Framework for LLM Agents</strong>
</p>

<p align="center">
  A comprehensive and extensible evaluation framework designed to measure the performance<br>
  of LLM agents across a wide range of complex, long-horizon benchmarks.
</p>

---

## ✨ Features

- 📊 **Multi-Benchmark Support** - Out-of-the-box support for GAIA, HLE, BrowseComp, Frames, WebWalkerQA, and more.
- 🚀 **Parallel Execution** - High-performance multi-process engine for concurrent task evaluation.
- 📈 **Automated Reporting** - Detailed success/fail analysis with reasoning trajectory and automated scoring.
- 🛠️ **MCP Integration** - Seamlessly connects with **AgentDock** for secure tool use and environment interaction.
- 🧩 **Extensible Design** - Easily add new datasets and custom evaluation logic with minimal configuration.

## 🏗️ Architecture

```text
AgentToLeaP/
├── benchmarks/                 # Benchmark-specific configurations and scripts
│   ├── gaia/                  
│   ├── hle/                    
│   └── ...                     # Other 8+ integrated benchmarks
├── context/                    # Agent context management
├── run_evaluation.py           # Main entry point for parallel execution
├── evaluate_and_report.py      # Core scoring and report generation logic
├── gaia_report_generator.py    # Specialized report generator for GAIA
├── data_test_copy.py           # Evaluation task implementation
└── browser_processor.py        # Web content purification and processing
```

## 📚 Supported Benchmarks

The framework provides pre-configured evaluation logic and scripts for the following benchmarks. You can find their configurations in the `benchmarks/` directory (Note: Dataset files must be prepared separately as described in the [Dataset Preparation](#3-dataset-preparation) section).

| Benchmark | Description | Source |
| :--- | :--- | :--- |
| **HLE-text-2158** | A subset of HLE for text-only tasks. | [[Paper]](https://arxiv.org/abs/2501.14249) |
| **GAIA Validation** | A benchmark for General AI Assistants. | [[Paper]](https://arxiv.org/abs/2311.12983) |
| **GAIA-text-103** | A subset of GAIA Validation for text-only tasks. | [[Paper]](https://arxiv.org/abs/2505.22648) |
| **WebWalkerQA** | A dataset focusing on web-based navigation and complex QA tasks. | [[Paper]](https://arxiv.org/abs/2501.07572) |
| **BrowseComp-EN** | Web browsing and comprehension tasks. | [[Paper]](https://arxiv.org/abs/2504.12516) |
| **BrowseComp-ZH** | A Chinese version of BrowseComp. | [[Paper]](https://arxiv.org/abs/2504.19314) |
| **SEAL-0** | Benchmark designed to test model reasoning when faced with contradictory web information. | [[Paper]](https://arxiv.org/abs/2506.01062) |
| **Frames** | A comprehensive set for measuring factuality, information retrieval, and multi-hop reasoning. | [[Paper]](https://arxiv.org/abs/2409.12941) |
| **XBench-DeepSearch** | A benchmark for deep research agents. | [[Website]](https://xbench.org/agi/aisearch) |

## 🚀 Quick Start

### 1. Environment Setup (Recommended)

The easiest way to run evaluations is using our pre-built Docker image (supports amd64/arm64 architectures) which contains all necessary dependencies:

```bash
docker pull yuyangfu/agenttoleap-eval:v2.0
docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval:v2.0
docker exec -it agenttoleap /bin/bash
cd /workspace
```

### 2. Quick Test (Optional)

To quickly verify the environment with a single task before running full benchmarks, you can use the `quickstart.py` script located in the project root:

1. **Configure**: Open `quickstart.py` in the project root and fill in your `API_KEY`, `MODEL_NAME`, and `MANAGER_URL` in the `[USER CONFIGURATION]` section.
2. **Run**:
   ```bash
   # From the project root directory
   python quickstart.py
   ```
3. **Check Results**: Interaction logs and reasoning chains will be saved in `outputs/quickstart_results/dialog.json`.

### 3. Dataset Preparation

Due to copyright restrictions, we do not provide the original dataset files. You need to download the datasets from their official sources and convert them into the required `.jsonl` format.

For each benchmark integrated in the `benchmarks/` directory, you should place a `.jsonl` file with the **same name as the directory** inside that directory (e.g., `benchmarks/gaia/gaia.jsonl`).

**Required JSONL Format:**
Each line must be a JSON object containing:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `task_id` | String / Int | Unique identifier for the task |
| `Question` | String | The complete question or instruction sent to the model |
| `Final answer` | String / Num | The reference answer used for automated evaluation |
| `file_name` | String / List | (Optional) Path(s) to attached files for the task |

**Example:**
```json
{"task_id": "validation_0", "Question": "What is the capital of France?", "Final answer": "Paris"}
```

### 4. Run a Benchmark

Navigate to a benchmark directory and execute the provided `run.sh` script:

```bash
cd AgentToLeaP/benchmarks/gaia
# Edit run.sh to configure your API_KEY and MODEL_NAME
bash run.sh
```

## ⚙️ Configuration

Evaluations are primarily configured via environment variables in the `run.sh` scripts.

### 1. Primary Model Configuration
| Variable | Example | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `"Qwen3-4B"` | Name of the model under evaluation (API `model` field) |
| `BASE_URL` | `"https://api.openai.com/v1"` | Primary model API base URL |
| `API_KEY` | `"sk-..."` | Primary model API key |
| `RESULT_DIR_NAME` | `"Qwen3-4B-test-0109"` | Result identifier used to generate output directory name |

### 2. Auxiliary Model Configuration
| Variable | Example | Description |
|----------|---------|-------------|
| `PROCESSOR_MODEL_NAME` | `"Qwen3-14B"` | Auxiliary model for summarization and long-context processing |
| `PROCESSOR_BASE_URL` | `"..."` | Auxiliary model API base URL |

### 3. Evaluation Environment
| Variable | Example | Description |
|----------|---------|-------------|
| `MANAGER_URL` | `"http://localhost:8000/mcpapi"` | Address of the AgentDock service |
| `EVALUATION_ROOT_DIR` | `"/path/to/outputs"` | Root directory for evaluation outputs |
| `FILES_DIR` | `"/path/to/files"` | Directory for benchmark attachments/files |

### 4. Control & Sampling Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_PROCESSES` | `10` | Number of concurrent evaluation workers |
| `MAX_INTERACTIONS` | `50` | Maximum interaction turns per task |
| `USE_LLM_JUDGE` | `"true"` | Whether to use an LLM as the judge (recommended) |
| `PASS_K` | `8` | Pass@k sampling runs |
| `TEMPERATURE` | `1.0` | Sampling temperature |
| `TOP_P` | `1.0` | Top-p sampling |
| `MAX_TOKENS` | `16384` | Maximum tokens per generation |

## 📊 Results & Reports

After evaluation, results are saved in the directory specified by `EVALUATION_ROOT_DIR`.

### Directory Structure
```text
evaluation_outputs/ (EVALUATION_ROOT_DIR)
├── _temp_raw_outputs/                      # [All tasks] Raw evaluation logs
│   └── gaia_Qwen3-4B-test-0109/            # Named as ${BENCHMARK}_${RESULT_DIR_NAME}
│       ├── task_id_1/
│       │   ├── dialog.json                 # Model dialogue trajectory
│       │   ├── result.json                 # Complete task result
│       │   └── trace.json                  # Execution trace
│
├── [Benchmark Specific Reports]            # Success/fail analysis and MD reports
│   └── Qwen3-4B-test-0109/
│       ├── success/                        # Details of correctly answered tasks
│       ├── fail/                           # Details of incorrectly answered tasks
│       └── *_main_report.md                # Human-readable summary report
```

- **`dialog.json`**: Full interaction trace including thoughts and tool calls.
- **`result.json`**: Final output and scoring result for each task.
- **`*_report.md`**: Detailed success/fail analysis with reasoning trajectory.

## ➕ Adding a Custom Benchmark

This framework is designed to be easily extensible. To add a new evaluation dataset:

1. **Create a directory**: Create a new folder under `benchmarks/`, for example `my_custom_bench`.
2. **Prepare the data**: Inside this folder, create a `.jsonl` file with the **same name** (e.g., `my_custom_bench.jsonl`).
3. **Data format**: Each line must be a JSON object containing the following fields:

   | Field Name | Type | Description |
   | :--- | :--- | :--- |
   | `task_id` | String / Int | Unique identifier for the task |
   | `Question` | String | The complete question or instruction sent to the model |
   | `Final answer` | String / Num | The reference answer used for automated evaluation |

   **Example (`my_custom_bench.jsonl`):**
   ```json
   {"task_id": 1, "Question": "What is 1 + 1?", "Final answer": "2"}
   ```
4. **Configure the script**: Copy the `run.sh` file from any existing benchmark (like `hle_text`) into the new directory. Adjust the environment variables to point to your new data, and you're ready to run.

## 📄 License

This module is part of the AgentCPM-Explore project and is released under the [Apache-2.0](../../LICENSE) license.
