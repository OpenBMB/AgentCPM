

<div align="center">
  <img src="../assets/AgentCPM-Explore-logo.png" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【中文 | <a href="README.md">English</a>】
</p>

<p align="center">
  <a href="#概述">概述</a> •
  <a href="#安装">安装</a> •
  <a href="#模型训练">模型训练</a> •
  <a href="https://huggingface.co/openbmb/AgentCPM-Explore">模型下载</a> •
  <a href="#一键测评">一键测评</a> •
  <a href="#开发共建">开发共建</a> 
</p>

# 最新消息

* [2026-01-12] 🚀🚀🚀我们开源了基于仅4B参数的训练的智能体大模型AgentCPM-Explore，成功闯入GAIA、HLE、BrowseComp等8个经典长难智能体任务榜单，同级别SOTA的表现带来更长行为链路、更准确的深度调研能力，由此突破端侧智能体的性能壁垒。

# 概述

**AgentCPM-Explore** 是由[清华大学自然语言处理实验室（THUNLP）](https://nlp.csai.tsinghua.edu.cn)、[中国人民大学](http://ai.ruc.edu.cn/)与[面壁智能](https://modelbest.cn/en)联合开发的开源智能体大模型，基于 [Qwen3-4B-thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) 构建，拥有 40 亿参数，让大模型的长程任务处理能力落地端侧。AgentCPM-Explore的亮点包括：

- 首个以 4B 全量参数登入 GAIA、HLE、BrowseComp 等 8 个长程复杂智能体任务榜单的端侧智能体模型。

- 可实现超过 100 轮的连续环境交互，支持多源信息交叉验证、搜索策略动态调整、实时核验最新信息，持续深度探索直至任务完成。

- 全流程开源，包括智能体全异步强化学习训练框架与工具沙盒统一管理平台，支持社区共建与自定义扩展。


演示案例（倍速）：

https://github.com/user-attachments/assets/f8487889-d17a-447e-9aef-2608f4c84a83



实验结果：
<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>GAIA文本子集</th>
      <th>BrowseComp</th>
      <th>BrowseComp (ZH)</th>
      <th>HLE</th>
      <th>Frames</th>
      <th>WebWalkerQA</th>
      <th>Seal-0</th>
      <th>xbench-DeepSearch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9"><strong>Closed-Source Models</strong></td>
    </tr>
    <tr>
      <td>Claude-4.5-sonnet</td>
      <td>71.2%</td>
      <td>19.6%</td>
      <td>40.8%</td>
      <td>24.5%</td>
      <td>85.0%</td>
      <td>/</td>
      <td>53.4%</td>
      <td>66.0%</td>
    </tr>
    <tr>
      <td>Gemini Deep Research</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>26.9%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>Deepseek-V3.2</td>
      <td>63.5%</td>
      <td>67.6%</td>
      <td>65.0%</td>
      <td>40.8%</td>
      <td>80.2%</td>
      <td>/</td>
      <td>38.5%</td>
      <td>71.0%</td>
    </tr>
    <tr>
      <td>Minimax-M2</td>
      <td>75.7%</td>
      <td>44.0%</td>
      <td>48.5%</td>
      <td>31.8%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>72.0%</td>
    </tr>
    <tr>
      <td>OpenAI-GPT-5-high</td>
      <td>76.4%</td>
      <td>54.9%</td>
      <td>65.0%</td>
      <td>35.2%</td>
      <td>/</td>
      <td>/</td>
      <td>51.4%</td>
      <td>77.8%</td>
    </tr>
    <tr>
      <td>GLM-4.6</td>
      <td>71.9%</td>
      <td>45.1%</td>
      <td>49.5%</td>
      <td>30.4%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>70.0%</td>
    </tr>
    <tr>
      <td>Kimi-Researcher</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>26.9%</td>
      <td>78.8%</td>
      <td>/</td>
      <td>36.0%</td>
      <td>69.0%</td>
    </tr>
    <tr>
      <td>Seed-1.8</td>
      <td>87.4%</td>
      <td>67.6%</td>
      <td>81.3%</td>
      <td>40.9%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td colspan="9"><strong>Open-Source Models</strong></td>
    </tr>
    <tr>
      <td>MiroThinker 8B</td>
      <td>66.4%</td>
      <td>31.1%</td>
      <td>40.2%</td>
      <td>21.5%</td>
      <td>80.6%</td>
      <td>60.6%</td>
      <td>40.4%</td>
      <td>60.6%</td>
    </tr>
    <tr>
      <td>Tongyi DeepResearch 30B</td>
      <td>70.9%</td>
      <td>43.4%</td>
      <td>46.7%</td>
      <td>32.9%</td>
      <td>90.6%</td>
      <td>72.2%</td>
      <td>/</td>
      <td>75.0%</td>
    </tr>
    <tr>
      <td>ASearcher QWQ 32B v2</td>
      <td>58.7%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>74.5%</td>
      <td>/</td>
      <td>/</td>
      <td>51.1%</td>
    </tr>
    <tr>
      <td>iterresearch-30B-A3B</td>
      <td>72.8%</td>
      <td>37.3%</td>
      <td>45.2%</td>
      <td>28.8%</td>
      <td>71.0%</td>
      <td>/</td>
      <td>39.6%</td>
      <td>/</td>
    </tr>
    <tr>
      <td>WebSailor-V2-30B-A3B (RL)</td>
      <td>74.1%</td>
      <td>35.3%</td>
      <td>44.1%</td>
      <td>30.6%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>73.7%</td>
    </tr>
    <tr>
      <td>WebLeaper-30B-A3B-RUC</td>
      <td>73.2%</td>
      <td>38.8%</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>48.6%</td>
      <td>72.0%</td>
    </tr>
    <tr>
      <td>WebDancer (QwQ-32B)</td>
      <td>51.5%</td>
      <td>3.8%</td>
      <td>18.0%</td>
      <td>/</td>
      <td>/</td>
      <td>47.9%</td>
      <td>/</td>
      <td>38.3%</td>
    </tr>
    <tr>
      <td>⭐ <strong>AgentCPM-Explore 4B</strong></td>
      <td>63.9%</td>
      <td>24.1%</td>
      <td>29.1%</td>
      <td>19.1%</td>
      <td>82.7%</td>
      <td>68.1%</td>
      <td>40.5%</td>
      <td>70.0%</td>
    </tr>
  </tbody>
</table>




# 安装

## 环境需求

- Docker 和 Docker Compose
- Python 3.10+
- 至少 8GB 内存（推荐 16GB+）

## AgentDock 工具沙盒平台

AgentDock 是 AgentCPM-Explore 的统一工具沙盒管理平台，提供 MCP (Model Context Protocol) 服务的容器化部署与管理能力。

**核心架构：**

| 组件 | 端口 | 说明 |
| :--- | :--- | :--- |
| `agentdock-manager` | 8080 | 管理面板，提供容器生命周期管理、健康监控、API 路由 |
| `agentdock-mongodb` | 27017 | 状态持久化存储 |
| `agentdock-node-full` | 8004/8092 | 全功能 MCP 节点，支持 GitHub、Slack、文档处理等工具 |
| `agentdock-node-explore` | 8014/8102 | 搜索探索节点，支持网页搜索、内容抓取、代码执行等工具 |

**快速部署：**

```bash
# 1. 进入 AgentDock 目录
cd AgentDock

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置 MongoDB 密码 and 可选的 API Keys

# 3. 一键启动
docker compose up -d

# 4. 访问管理面板
open http://localhost:8080
```

**环境变量配置 (.env)：**

```bash
# 必填：MongoDB 认证
MONGODB_USERNAME=admin
MONGODB_PASSWORD=your_password

# 可选：搜索功能 API Keys
JINA_API_KEY=your_jina_key        # Jina Reader API
GOOGLE_SERP_API_KEY=your_serp_key # Google 搜索 API
```

## QuickStart

- **多模型多工具协作环境部署**：首先启动 AgentDock 工具沙盒平台，提供统一的 MCP 工具服务。和 API 模型协作时，配置模型的 `BASE_URL` 和 `API_KEY`；和本地 host 的模型协作时，确保模型服务可访问。在 `config.toml` 文件中配置工具所需的使用参数。

- **启动环境**：开箱即用，一键启动。AgentDock 统一工具沙盒管理平台支持 `docker compose up -d` 一键启动所有服务，包括管理面板、数据库和工具节点。

- **启动执行**：通过 QuickStart 脚本快速体验框架的核心能力，无需繁琐配置即可运行一个完整的 Agent 任务。

0. **准备评测环境 (推荐)**：
   我们提供了一个预装好所有评测依赖的 Docker 镜像，建议直接拉取镜像并在容器内运行：
   
   ```bash
   # 拉取镜像
   docker pull yuyangfu/agenttoleap-eval
   
   # 启动容器 (请根据实际路径修改 -v 参数)
   docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval
   
   # 进入容器
   docker exec -it agenttoleap /bin/bash
   cd /workspace
   ```

1. **配置与运行**：
   打开根目录下的 `quickstart.py`，在 `[USER CONFIGURATION]` 区域进行简单配置：
   
   - **自定义任务**：修改 `QUERY` 变量为您想要测试的指令（例如："查一下昨晚的欧冠比赛结果"）。
   - **模型信息**：填入您的 LLM `API_KEY`、`MODEL_NAME` 和 `BASE_URL`。
   - **工具服务**：设置 `MANAGER_URL` 为您的 MCP 工具服务器地址（例如 `http://localhost:8000`，请确保该服务已先行启动）。

   配置完成后，直接运行：

   ```bash
   python quickstart.py
   ```

   *脚本会自动创建一个演示任务（默认查询今日的ArXiv计算机科学论文），生成执行脚本并启动评测流程。*

2. **查看结果**：
   运行完成后，结果将保存在 `outputs/quickstart_results/` 目录下。您可以查看 `dialog.json` 获取完整的交互轨迹（包含工具调用、思维链等）。

   *注：QuickStart模式默认跳过了自动评分步骤，仅用于展示Agent执行能力。*

若完全复现相关结果，则需要对齐网页信息摘要模型的启动设置。以本地host模型为例，模型以sglang的形式启动，进行如下配置：

```bash
export SUMMARY_MODEL="Qwen3-14b"
export SUMMARY_BASE_URL="YOUR-BASE-URL"
export SUMMARY_API_KEY="YOUR-API-KEY"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python sglang_init.py \
--model-path YOUR-MODEL-PATH \
--port YOUR-BASE-URL \
--tp-size 1 \
--dp-size 8 \
--api-key YOUR-API-KEY \
--served-model-name YOUR-MODEL-NAME \
--mcp_manager_url YOUR-SERVER-IP-ADDRESS
```

# 模型训练

我们的训练基于自研AgentRL框架进行。

> **详细训练文档**: 请参阅 [AgentRL 训练文档](AgentRL/README_zh.md) 获取完整的训练指南，包括环境配置、数据准备、训练脚本配置等详细说明。

# 一键测评

我们提供了一套完整的自动化评测框架，支持对GAIA、HLE等8个经典智能体任务评测集进行一键测评。每个评测集支持独立管理，并将结果统一输出，便于开发者基于本框架加入新测试集。

> **注意**：为了确保评测环境的一致性，强烈建议在上述 **QuickStart** 中提到的 Docker 容器内执行评测。

关于详细参数配置、报告说明及自定义评测集的更多细节，请参阅 [AgentToLeaP 测评文档](AgentToLeaP/README_zh.md)。

## 1. 核心参数配置

在运行评测前，请修改对应 `AgentToLeaP/benchmarks` 目录下的启动脚本（如 `AgentToLeaP/benchmarks/gaia/run.sh`）。

| 参数变量 | 示例值 | 说明 |
| :--- | :--- | :--- |
| `MODEL_NAME` | "Qwen3-4B" | 被测模型名称 (API `model`字段) |
| `BASE_URL` | "..." | 主模型 API Base URL |
| `API_KEY` | "sk-..." | 主模型 API Key |
| `MANAGER_URL` | "..." | 工具服务器 (AgentDock) 地址 |

## 2. 运行评测

以 **GAIA** Benchmark 为例：

```bash
# 1. 进入对应 benchmark 的目录
cd AgentToLeaP/benchmarks/gaia

# 2. 修改 run.sh 中的参数配置 

# 3. 启动评测脚本
bash run.sh
```

## 3. 查看报告

评测结果将保存在 `EVALUATION_ROOT_DIR` 指定的目录下。包含交互轨迹 `dialog.json`、原始结果 `result.json` 以及各任务的详细报告。

## 4. 添加自定义评测集

本框架支持轻松扩展新的评测数据集。只需遵循以下步骤：

1.  **创建目录**：在 `AgentToLeaP/benchmarks/` 下新建一个文件夹。
2.  **准备数据**：在该文件夹内创建一个同名的 `.jsonl` 文件。
3.  **配置脚本**：复制现有 `run.sh` 脚本并调整环境变量。

更详细的步骤请参考 [AgentToLeaP 测评文档](AgentToLeaP/README_zh.md)。






# 开发共建

## 自定义工具接入

如果开发者想使用自定义的工具接入环境进行训练 and 评测，可以通过以下步骤配置：

**1. 创建 MCP 工具服务**

在 `AgentDock/agentdock-node-explore/mcp_servers/` 目录下创建新的工具服务：

```bash
mkdir mcp_servers/my_custom_tool
```

**2. 实现工具逻辑**

创建符合 MCP 协议的工具服务（Python 示例）：

```python
# mcp_servers/my_custom_tool/server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-custom-tool")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="my_tool",
            description="自定义工具描述",
            inputSchema={"type": "object", "properties": {...}}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_tool":
        # 实现工具逻辑
        result = process(arguments)
        return [TextContent(type="text", text=result)]
```

**3. 注册工具到配置文件**

编辑 `config.toml`，添加新工具：

```toml
[mcpServers.my_custom_tool]
command = "python"
args = ["mcp_servers/my_custom_tool/server.py"]
env = { MY_API_KEY = "your_key" }  # 可选：环境变量
```

**4. 重启服务生效**

```bash
docker compose restart agentdock-node-explore
```

## 自定义模型接入

一个或多个工具批量纳入统一管理平台后，以 Qwen3 系列模型为例，即可执行如下指令进行推理：

```bash
python quickstart.py \
    --model_name "Qwen3-4B" \
    --base_url "http://localhost:8000/v1" \
    --api_key "your_api_key" \
    --manager_url "http://localhost:8080"
```

如需切换模型，需要查阅对应模型相关的文档以获取其工具调用的 special token，在 `src/tool_parser/` 目录增加工具调用的 parser 来解析工具调用，由此访问工具服务获取执行结果。

# 致谢

本项目的实现离不开开源社区的支持与贡献。我们在开发过程中参考并使用了多个优秀的开源框架、模型和数据资源，包括
[verl](https://github.com/volcengine/verl)、
[trl](https://github.com/huggingface/trl)、
[TongYi Deep Research](https://github.com/Alibaba-NLP/DeepResearch)、
[DeepSeek](https://www.deepseek.com/)；
同时也受益于以下项目与数据集：
[ASearcher](https://github.com/inclusionAI/ASearcher)、
[WebExplorer](https://github.com/hkust-nlp/WebExplorer)、
[NVIDIA Nemotron](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)、
[DeepDive](https://github.com/THUDM/DeepDive)、
[WebWalker](https://aclanthology.org/2025.acl-long.508/)、
[MiroVerse](https://hf-mirror.com/datasets/miromind-ai/MiroVerse-v0.1)、
[HybridQA](https://hf-mirror.com/datasets/wenhu/hybrid_qa)，
以及 [MegaScience](https://hf-mirror.com/datasets/MegaScience/MegaScience)。

感谢上述项目的作者和维护者为开源生态所做出的贡献。



# 贡献

项目负责人：陈颢天

项目贡献者：陈颢天, 从鑫, 樊昇达, 符煜洋, 龚子沁, 卢雅西, 李奕杉, 牛博也, 潘成骏, 宋子骏, 汪华东, 吴叶赛, 吴玥莹, 谢子昊, 闫宇坤, 张众

项目指导人：林衍凯, 刘知远, 孙茂松

# 引用

如果 **AgentCPM-Explore** 对您的研究有所帮助，您可以按照如下方式进行引用：

```bibtex
@software{AgentCPMExplore2026,
  title  = {AgentCPM-Explore: An End-to-End Infrastructure for Training and Evaluating LLM Agents},
  author = {Haotian Chen, Xin Cong, Shengda Fan, Yuyang Fu, Ziqin Gong, Yaxi Lu, Yishan Li, Boye Niu, Chengjun Pan, Zijun Song, Huadong Wang, Yesai Wu, Yueying Wu, Zihao Xie, Yukun Yan, Zhong Zhang, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM-Explore}
}
```
