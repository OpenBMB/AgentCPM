<div align="center">
  <img src="../assets/AgentCPM-Explore-logo.png" alt="AgentCPM-Explore Logo" width="400em"></img>
</div>

<p align="center">
    【<a href="README_zh.md">中文</a> | English】
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#model-training">Model Training</a> •
  <a href="https://huggingface.co/openbmb/AgentCPM-Explore">Model Download</a> •
  <a href="#one-click-evaluation">One-Click Evaluation</a> •
  <a href="#community-development">Community Development</a>
</p>

# Latest News

* [2026-01-12] 🚀🚀🚀 We open-sourced **AgentCPM-Explore**, an agent foundation model trained with only **4B parameters**, which successfully entered **8 classic long-horizon and hard agent benchmarks** including **GAIA, HLE, and BrowseComp**. It achieves **state-of-the-art performance within the same parameter scale**, enabling longer action chains and more accurate deep research capabilities, thereby breaking the performance ceiling of **on-device agents**.

# Overview

**AgentCPM-Explore** is an open-source agent foundation model jointly developed by the [THUNLP](https://nlp.csai.tsinghua.edu.cn), [Renmin University of China](http://ai.ruc.edu.cn/), and [ModelBest](https://modelbest.cn/en). It is built upon [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) with **4 billion parameters**, bringing long-horizon task-solving capabilities of large models to **on-device deployment**.

Key highlights of AgentCPM-Explore include:

- The **first on-device agent model with only 4B full parameters** to enter **8 long-horizon and complex agent benchmarks**, including GAIA, HLE, and BrowseComp.
- Supports **over 100 turns of continuous environment interaction**, enabling multi-source information cross-validation, dynamic search strategy adjustment, real-time verification of up-to-date information, and sustained deep exploration until task completion.
- **Fully open-sourced pipeline**, including an asynchronous agent reinforcement learning framework and a unified tool sandbox management platform, supporting community-driven development and custom extensions.

Demo (accelerated playback):

https://github.com/user-attachments/assets/f2b3bb20-ccd5-4b61-8022-9f6e90992baa




Experimental Results:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>GAIA (text only)</th>
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

# Installation

## Requirements

- Docker & Docker Compose
- Python 3.10+
- At least 8GB RAM (16GB+ recommended)

## AgentDock Tool Sandbox Platform

**AgentDock** is the unified tool sandbox management platform for AgentCPM-Explore. It provides containerized deployment and management for MCP (Model Context Protocol) services.

**Core Architecture:**

| Component | Port | Description |
| :--- | :--- | :--- |
| `agentdock-manager` | 8080 | Management UI, container lifecycle management, health monitoring, API routing |
| `agentdock-mongodb` | 27017 | Persistent state storage |
| `agentdock-node-full` | 8004/8092 | Full-featured MCP node (GitHub, Slack, document processing, etc.) |
| `agentdock-node-explore` | 8014/8102 | Exploration node (web search, crawling, code execution, etc.) |

**Quick Deployment:**

```bash
# 1. Enter the AgentDock folder
cd AgentDock

# 2. Set the environment variables
cp .env.example .env
# Editing .env file，setting the password of MongoDB and optional API Keys

# 3. One-click startup
docker compose up -d

# 4. Access the management dashboard
open http://localhost:8080
```

**Set the environment variables (.env):**

```bash
# Required: MongoDB authentication
MONGODB_USERNAME=admin
MONGODB_PASSWORD=your_password

# Optional: API Keys of search tools
JINA_API_KEY=your_jina_key        # Jina Reader API
GOOGLE_SERP_API_KEY=your_serp_key # Google Search API
```


## QuickStart

- **QuickStart tutorial video (setup & run)**: https://www.youtube.com/watch?v=j3dtYY9KCd0  
  *Recommended: follow along in the provided evaluation Docker container to avoid environment discrepancies.*

- **Multi-model, multi-tool collaborative environment setup**: First, start the AgentDock tool sandbox platform to provide unified MCP (Model Context Protocol) tool services. When working with API-based models, configure the model’s `BASE_URL` and `API_KEY`. When working with locally hosted models, ensure the model service is accessible. Configure the required tool parameters in the `config.toml` file.

- **Launch the environment**: Out of the box, one-click startup. The AgentDock unified tool sandbox platform supports launching all services with a single `docker compose up -d` command, including the management dashboard, database, and tool nodes.

- **Run execution**: Quickly experience the core capabilities of the framework via the QuickStart script, allowing you to run a complete Agent task without complex configuration.

0. **Prepare Evaluation Environment (Recommended)**:  
   We provide a Docker image with all evaluation dependencies pre-installed. It is recommended to pull the image and run it directly:

   ```bash
   # Pull the image (Supports amd64/arm64 architectures)
   docker pull yuyangfu/agenttoleap-eval:v2.0
   
   # Start the container (Adjust the -v path as needed)
   docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval:v2.0
   
   # Enter the container
   docker exec -it agenttoleap /bin/bash
   cd /workspace
   ```

1. **Configure and run**:  
  Open `quickstart.py` in the project root directory and make simple configurations in the `[USER CONFIGURATION]` section:

  - **Custom task**: Modify the `QUERY` variable to the instruction you want to test (e.g., “Check the results of last night’s UEFA Champions League matches”).
  - **Model information**: Provide your LLM `API_KEY`, `MODEL_NAME`, and `BASE_URL`.
  - **Tool service**: Set `MANAGER_URL` to the address of your MCP tool server (e.g., `http://localhost:8000`; make sure the service is already running).

  After configuration, run:

  ```bash
  python quickstart.py
  ```

  The script will automatically create a demo task (by default, querying today’s arXiv computer science papers), generate the execution workflow, and start the evaluation process.

2. **View Results**

  After execution completes, results will be saved under the `outputs/quickstart_results/` directory. You can inspect `dialog.json` to obtain the full interaction trace, including tool calls and reasoning chains.

  *Note: In QuickStart mode, automatic scoring is skipped by default and is intended only to demonstrate the Agent’s execution capabilities.*

  To fully reproduce the reported results, the startup configuration of the web information summarization model must be aligned. Taking a locally hosted model as an example, the model is launched via sglang with the following configuration:


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

# Model Training

Our training is based on our in-house **AgentRL** framework.

> **Detailed training documentation**: Please refer to [AgentRL Training Documentation](AgentRL/README.md) for a complete training guide, including environment setup, data preparation, training script configuration, and other details.

# One-Click Evaluation

We provide a complete automated evaluation framework that supports **one-click evaluation** on **8 classic agent benchmarks**, including GAIA and HLE. Each benchmark can be managed independently, while results are exported in a unified format—making it easy for developers to add new benchmarks on top of this framework.

> **Note**: To ensure consistency in the evaluation environment, it is strongly recommended to run the evaluation within the Docker container mentioned in the **QuickStart** section. 

For detailed parameter configuration, report explanation, and instructions on adding custom benchmarks, please refer to the [AgentToLeaP Documentation](AgentToLeaP/README.md).

## 1. Core Parameter Configuration

Before running evaluation, please edit the corresponding launch script under `AgentToLeaP/benchmarks/` (e.g., `AgentToLeaP/benchmarks/gaia/run.sh`).

| Variable | Example | Description |
| :--- | :--- | :--- |
| `MODEL_NAME` | "Qwen3-4B" | Name of the model under evaluation (API `model` field) |
| `BASE_URL` | "..." | Primary model API base URL |
| `API_KEY` | "sk-..." | Primary model API key |
| `MANAGER_URL` | "..." | Tool server (AgentDock) endpoint |

## 2. Run Evaluation

Take the **GAIA** benchmark as an example:

```bash
# 1. Enter the benchmark folder
cd AgentToLeaP/benchmarks/gaia

# 2. Adjust the configs in run.sh

# 3. Launch the evaluation
bash run.sh
```

## 3. Viewing Reports

Evaluation results will be saved under the directory specified by `EVALUATION_ROOT_DIR`. It includes the interaction trajectory `dialog.json`, raw results `result.json`, and detailed reports for each task.

## 4. Adding a Custom Benchmark

This framework is designed to be easily extensible. To add a new evaluation dataset:

1. **Create a directory**: Create a new folder under `AgentToLeaP/benchmarks/`.
2. **Prepare the data**: Inside this folder, create a `.jsonl` file with the same name.
3. **Configure the script**: Copy any existing `run.sh` and adjust environment variables.

For more detailed instructions, please refer to the [AgentToLeaP Documentation](AgentToLeaP/README.md).


# Community Development

## Integrating Custom Tools

If developers want to integrate custom tools into the environment for training and evaluation, they can configure them by following the steps below:

**1. Create an MCP tool service**

Create a new tool service under the `AgentDock/agentdock-node-explore/mcp_servers/` directory:

```bash
mkdir mcp_servers/my_custom_tool
```

**2. Implement the tool logic**

Create a tool service that conforms to the MCP protocol (Python example):


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
            description="tool description",
            inputSchema={"type": "object", "properties": {...}}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_tool":
        result = process(arguments)
        return [TextContent(type="text", text=result)]
```

**3. Register the tool in the configuration file**

Edit `config.toml` and add the new tool:

```toml
[mcpServers.my_custom_tool]
command = "python"
args = ["mcp_servers/my_custom_tool/server.py"]
env = { MY_API_KEY = "your_key" } 
```

**4. Restart the service to apply changes**

```bash
docker compose restart agentdock-node-explore
```

## Integrating Custom Models

Once one or more tools have been batch-registered into the unified management platform, you can run inference commands using models such as the Qwen3 series as an example:

```bash
python quickstart.py \
    --model_name "Qwen3-4B" \
    --base_url "http://localhost:8000/v1" \
    --api_key "your_api_key" \
    --manager_url "http://localhost:8080"
```

If you need to switch to a different model, please refer to the corresponding model documentation to obtain the required *special tokens* for tool calling. Then, add a corresponding tool-call parser under the `src/tool_parser/` directory to parse the model’s tool invocation format, enabling access to the tool services and retrieval of execution results.

# Acknowledge

This project builds upon and integrates ideas, tools, and resources from several open-source frameworks and models, including
[verl](https://github.com/volcengine/verl),
[trl](https://github.com/huggingface/trl),
[TongYi Deep Research](https://github.com/Alibaba-NLP/DeepResearch),
[DeepSeek](https://www.deepseek.com/),
as well as datasets such as
[ASearcher](https://github.com/inclusionAI/ASearcher),
[WebExplorer](https://github.com/hkust-nlp/WebExplorer),
[NVIDIA Nemotron](https://huggingface.co/collections/nvidia/nemotron-post-training-v3),
[DeepDive](https://github.com/THUDM/DeepDive),
[WebWalker](https://aclanthology.org/2025.acl-long.508/),
[MiroVerse-Voyager1.0](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1),
[HybridQA](https://huggingface.co/datasets/wenhu/hybrid_qa),
and [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience).

# Contributions

Project Lead: Haotian Chen

Contributors (in alphabetical order): Haotian Chen, Xin Cong, Shengda Fan, Yuyang Fu, Ziqin Gong, Yaxi Lu, Yishan Li, Boye Niu, Chengjun Pan, Zijun Song, Huadong Wang, Yesai Wu, Yueying Wu, Zihao Xie, Yukun Yan, Zhong Zhang

Project Supervisor: Yankai Lin, Zhiyuan Liu, Maosong Sun

---

# Citation

If **AgentCPM-Explore** is useful for your research, please cite the codebase:

```bibtex
@software{AgentCPMExplore2026,
  title  = {AgentCPM-Explore: An End-to-End Infrastructure for Training and Evaluating LLM Agents},
  author = {Haotian Chen, Xin Cong, Shengda Fan, Yuyang Fu, Ziqin Gong, Yaxi Lu, Yishan Li, Boye Niu, Chengjun Pan, Zijun Song, Huadong Wang, Yesai Wu, Yueying Wu, Zihao Xie, Yukun Yan, Zhong Zhang, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM-Explore}
}
```
