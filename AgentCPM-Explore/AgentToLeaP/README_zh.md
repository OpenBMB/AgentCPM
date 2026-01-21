# AgentToLeaP

<p align="center">
  【中文 | <a href="README.md">English</a>】
</p>

<p align="center">
  <strong>大模型智能体评测框架</strong>
</p>

<p align="center">
  一个全面且可扩展的评测框架，旨在衡量 LLM 智能体在各种复杂、长程任务评测基准上的性能。
</p>

---

## ✨ 特性

- 📊 **多榜单支持** - 开箱即用支持 GAIA, HLE, BrowseComp, Frames, WebWalkerQA 等。
- 🚀 **并行执行** - 高性能多进程引擎，支持并发任务评测。
- 📈 **自动报告** - 详尽的成功/失败分析，包含推理轨迹和自动评分。
- 🛠️ **MCP 集成** - 与 **AgentDock** 无缝连接，确保安全的工具使用和环境交互。
- 🧩 **可扩展设计** - 只需少量配置即可轻松添加新数据集和自定义评测逻辑。

## 🏗️ 架构说明

```text
AgentToLeaP/
├── benchmarks/                 # 特定评测基准的配置和脚本
│   ├── gaia/                  
│   ├── hle/                    
│   └── ...                     # 其他 8+ 已集成的评测基准
├── context/                    # 智能体上下文管理
├── run_evaluation.py           # 并行执行的主入口
├── evaluate_and_report.py      # 核心评分和报告生成逻辑
├── gaia_report_generator.py    # GAIA 专用报告生成器
├── data_test_copy.py           # 评测任务具体实现
└── browser_processor.py        # 网页内容清洗与处理
```

## 📚 支持的评测基准

本框架为以下评测基准提供了预配置的评测逻辑和运行脚本。相关配置可以在 `benchmarks/` 目录中找到（注意：数据集文件需按照[数据集准备](#3-数据集准备)章节自行准备）。

| 评测基准 | 描述 | 来源 |
| :--- | :--- | :--- |
| **HLE-text-2158** | HLE 的纯文本任务子集。 | [[Paper]](https://arxiv.org/abs/2501.14249) |
| **GAIA Validation** | 通用 AI 助手评测基准。 | [[Paper]](https://arxiv.org/abs/2311.12983) |
| **GAIA-text-103** | GAIA Validation 的纯文本任务子集。 | [[Paper]](https://arxiv.org/abs/2505.22648) |
| **WebWalkerQA** | 专注于网页导航及复杂问答任务的数据集。 | [[Paper]](https://arxiv.org/abs/2501.07572) |
| **BrowseComp-EN** | 网页浏览与理解任务评测。 | [[Paper]](https://arxiv.org/abs/2504.12516) |
| **BrowseComp-ZH** | BrowseComp 的中文版本。 | [[Paper]](https://arxiv.org/abs/2504.19314) |
| **SEAL-0** | 旨在测试模型在面对冲突网页信息时推理能力的基准。 | [[Paper]](https://arxiv.org/abs/2506.01062) |
| **Frames** | 用于评估事实准确性、信息检索及多步推理能力的综合测试集。 | [[Paper]](https://arxiv.org/abs/2409.12941) |
| **XBench-DeepSearch** | 深度研究型智能体评测基准。 | [[Website]](https://xbench.org/agi/aisearch) |

## 🚀 快速上手

### 1. 环境准备 (推荐)

运行评测最简单的方法是使用我们预构建的 Docker 镜像（支持 amd64/arm64 架构），其中包含所有必要的依赖项：

```bash
docker pull yuyangfu/agenttoleap-eval:v2.0
docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval:v2.0
docker exec -it agenttoleap /bin/bash
cd /workspace
```

### 2. 快速测试 (可选)

在运行完整的评测基准之前，如果您想通过单个任务快速验证环境，可以使用根目录下的 `quickstart.py` 脚本：

1. **配置**: 打开项目根目录下的 `quickstart.py`，在 `[USER CONFIGURATION]` 区域填入您的 `API_KEY`、`MODEL_NAME` 和 `MANAGER_URL`。
2. **运行**:
   ```bash
   # 在项目根目录下执行
   python quickstart.py
   ```
3. **查看结果**: 交互日志和推理链将保存在 `outputs/quickstart_results/dialog.json` 中。

### 3. 数据集准备

由于版权限制，我们不直接提供原始数据集文件。您需要从官方渠道下载数据集，并将其转换为所需的 `.jsonl` 格式。

对于 `benchmarks/` 目录中已集成的每个评测基准，您需要在对应文件夹下放置一个与**目录同名**的 `.jsonl` 文件（例如 `benchmarks/gaia/gaia.jsonl`）。

**数据格式要求：**
每一行必须是一个包含以下字段的 JSON 对象：

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `task_id` | String / Int | 任务的唯一标识符 |
| `Question` | String | 发送给模型的完整问题或指令 |
| `Final answer` | String / Num | 用于自动评测的标准答案 |
| `file_name` | String / List | (可选) 与任务相关的附件文件路径 |

**示例：**
```json
{"task_id": "validation_0", "Question": "法国的首都是哪里？", "Final answer": "巴黎"}
```

### 4. 运行评测基准

进入评测基准目录并执行提供的 `run.sh` 脚本：

```bash
cd AgentToLeaP/benchmarks/gaia
# 编辑 run.sh 以配置你的 API_KEY 和 MODEL_NAME
bash run.sh
```

## ⚙️ 配置说明

评测主要通过 `run.sh` 脚本中的环境变量进行配置。

### 1. 主模型配置 (Primary Model)
| 变量名 | 示例值 | 描述 |
|----------|---------|-------------|
| `MODEL_NAME` | `"Qwen3-4B"` | 待评测模型名称 (API 中的 `model` 字段) |
| `BASE_URL` | `"https://api.openai.com/v1"` | 主模型 API 基础地址 |
| `API_KEY` | `"sk-..."` | 主模型 API 密钥 |
| `RESULT_DIR_NAME` | `"Qwen3-4B-test-0109"` | 结果标识符，用于生成输出目录名 |

### 2. 辅助模型配置 (Auxiliary Model)
| 变量名 | 示例值 | 描述 |
|----------|---------|-------------|
| `PROCESSOR_MODEL_NAME` | `"Qwen3-14B"` | 用于摘要和长上下文处理的辅助模型 |
| `PROCESSOR_BASE_URL` | `"..."` | 辅助模型 API 基础地址 |

### 3. 评测环境配置
| 变量名 | 示例值 | 描述 |
|----------|---------|-------------|
| `MANAGER_URL` | `"http://localhost:8000/mcpapi"` | AgentDock 服务地址 |
| `EVALUATION_ROOT_DIR` | `"/path/to/outputs"` | 评测输出的根目录 |
| `FILES_DIR` | `"/path/to/files"` | 评测基准附件/文件所在目录 |

### 4. 控制与采样参数
| 变量名 | 默认值 | 描述 |
|----------|---------|-------------|
| `NUM_PROCESSES` | `10` | 并行评测的工作进程数 |
| `MAX_INTERACTIONS` | `50` | 每个任务的最大交互轮数 |
| `USE_LLM_JUDGE` | `"true"` | 是否使用 LLM 作为裁判 (推荐) |
| `PASS_K` | `8` | Pass@k 采样次数 |
| `TEMPERATURE` | `1.0` | 采样温度 |
| `TOP_P` | `1.0` | Top-p 采样 |
| `MAX_TOKENS` | `16384` | 单次生成的最大 Token 数 |

## 📊 结果与报告

评测完成后，结果保存在 `EVALUATION_ROOT_DIR` 指定的目录中。

### 目录结构
```text
evaluation_outputs/ (EVALUATION_ROOT_DIR)
├── _temp_raw_outputs/                      # [所有任务] 原始评测日志
│   └── gaia_Qwen3-4B-test-0109/            # 命名格式为 ${BENCHMARK}_${RESULT_DIR_NAME}
│       ├── task_id_1/
│       │   ├── dialog.json                 # 模型对话轨迹
│       │   ├── result.json                 # 完整的任务结果
│       │   └── trace.json                  # 执行追踪
│
├── [特定评测报告]                           # 成功/失败分析及 MD 报告
│   └── Qwen3-4B-test-0109/
│       ├── success/                        # 正确回答任务的详情
│       ├── fail/                           # 错误回答任务的详情
│       └── *_main_report.md                # 可读性强的汇总报告
```

- **`dialog.json`**: 包含思考过程和工具调用的完整交互轨迹。
- **`result.json`**: 每个任务的最终输出和评分结果。
- **`*_report.md`**: 带有推理轨迹的详细成功/失败分析。

## ➕ 添加自定义评测基准

本框架设计为易于扩展。要添加新的评测数据集：

1. **创建目录**: 在 `benchmarks/` 下创建一个新文件夹，例如 `my_custom_bench`。
2. **准备数据**: 在该文件夹内创建一个**同名**的 `.jsonl` 文件（例如 `my_custom_bench.jsonl`）。
3. **数据格式**: 每一行必须是一个包含以下字段的 JSON 对象：

   | 字段名 | 类型 | 描述 |
   | :--- | :--- | :--- |
   | `task_id` | String / Int | 任务的唯一标识符 |
   | `Question` | String | 发送给模型的完整问题或指令 |
   | `Final answer` | String / Num | 用于自动评测的标准答案 |

   **示例 (`my_custom_bench.jsonl`):**
   ```json
   {"task_id": 1, "Question": "1 + 1 等于几?", "Final answer": "2"}
   ```
4. **配置脚本**: 从任何现有基准（如 `gaia`）复制 `run.sh` 文件到新目录。调整环境变量指向你的新数据，即可开始评测。

## 📄 开源协议

本模块是 AgentCPM-Explore 项目的一部分，遵循 [Apache-2.0](../../LICENSE) 协议开源。
