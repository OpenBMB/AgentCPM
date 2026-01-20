

<div align="center">
  <img src="./assets/light.svg" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【<a href="README_zh.md">中文</a> | English】
</p>



# Latest News

* [2026-01-20] 🚀🚀🚀 We have open-sourced **AgentCPM-Report**, built on MiniCPM4.1-8B, which can rival top closed-source commercial systems for report generation such as Gemini-2.5-pro-DeepResearch.

* [2026-01-12] 🚀🚀🚀 We have open-sourced **AgentCPM-Explore**—an agent LLM with only **4B parameters**—along with all code for training, inference, and the tool sandbox environment. It successfully made it onto eight classic long-horizon and challenging agent leaderboards, including GAIA, HLE, and BrowseComp. Its SOTA performance at this scale enables longer action chains and more accurate Deep Research, breaking the performance barrier for on-device agents.


## Table of Contents

- [Overview](#overview)
- [Model List](#model-list)
- [AgentCPM-Explore](#agentcpm-explore)
  - [Demo](#demo)
  - [QuickStart](#quickstart)
- [AgentCPM-Report](#agentcpm-report)
  - [Demo](#demo-1)
  - [QuickStart](#quickstart-1)
- [License](#license)
- [Citation](#citation)
- [Explore More](#explore-more)


# Overview
AgentCPM is a series of open-source LLM agents jointly developed by [THUNLP (Tsinghua NLP Lab)](https://nlp.csai.tsinghua.edu.cn), [Renmin University of China](http://ai.ruc.edu.cn/), [ModelBest](https://modelbest.cn/en), and the [OpenBMB community](https://www.openbmb.cn/home). To address challenges faced by agents in real-world applications—such as limited long-horizon capability, autonomy, and generalization—we propose a series of model building approaches. Recently, the team has focused on comprehensively building deep research capabilities for agents, releasing [AgentCPM-Explore](./AgentCPM-Explore), a deep-search LLM agent, and [AgentCPM-Report](./AgentCPM-Report), a deep-research LLM agent.


# Model List

| Model            | Download Links                                                                                                                                | Open-Sourced Content | Technical Report | How to Use |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-----------|
| [AgentCPM-Explore](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Explore)          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Explore)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Explore/)                  |  [AgentDock](./AgentCPM-Explore/AgentDock): unified tool sandbox management & scheduling platform  <br> [AgentRL](./AgentCPM-Explore/AgentRL): fully asynchronous agent reinforcement learning framework  <br> [AgentToLeaP](./AgentCPM-Explore/AgentToLeaP): one-click evaluation framework for agent tool-learning capability | Coming Soon | [README.md](./AgentCPM-Explore)
| [AgentCPM-Report](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Report)          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Report)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Report/)                  |  [UltraRAG](https://github.com/OpenBMB/UltraRAG): low-code RAG framework   | Coming Soon | [README.md](./AgentCPM-Report)


## AgentCPM-Explore

The AgentCPM team has focused on systematically building agents’ deep research capabilities and released **AgentCPM-Explore**, a deep-search LLM agent. **AgentCPM-Explore** is the first open-source agent model with 4B parameters to appear on eight widely used long-horizon agent benchmarks such as GAIA, XBench, etc.

Key highlights:

- **SOTA at 4B Scale**: Best-in-class among same-size models, matches or surpasses 8B models, rivals some 30B+ and closed-source LLMs.

- **Deep Exploration**: 100+ turns of continuous interaction with multi-source cross-validation and dynamic strategy adjustment.

- **End-to-End Open Source**: Complete training and evaluation infrastructure for community development and custom extensions.


### Demo

Demo examples (speed up):

https://github.com/user-attachments/assets/f2b3bb20-ccd5-4b61-8022-9f6e90992baa


### QuickStart

- **Multi-model, multi-tool collaborative environment setup**: First, start the AgentDock tool sandbox platform to provide unified MCP (Model Context Protocol) tool services. When working with API-based models, configure the model’s `BASE_URL` and `API_KEY`. When working with locally hosted models, ensure the model service is accessible. Configure the required tool parameters in the `config.toml` file.

- **Launch the environment**: Out of the box, one-click startup. The AgentDock unified tool sandbox platform supports launching all services with a single `docker compose up -d` command, including the management dashboard, database, and tool nodes.

- **Run execution**: Quickly experience the core capabilities of the framework via the QuickStart script, allowing you to run a complete Agent task without complex configuration.

0. **Prepare Evaluation Environment (Recommended)**:  
   We provide a Docker image with all evaluation dependencies pre-installed. It is recommended to pull the image and run it directly:

   ```bash
   # 1. Enter the project folder
   cd AgentCPM-Explore
   
   # 2. Pull the image
   docker pull yuyangfu/agenttoleap-eval:v1.0
   
   # 3. Start the container (Adjust the -v path as needed)
   docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval:v1.0
   
   # 4. Enter the container
   docker exec -it agenttoleap /bin/bash
   cd /workspace
   ```

1. **Configure and run**:  
  Open `quickstart.py` and make simple configurations in the `[USER CONFIGURATION]` section:

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


## AgentCPM-Report

### Introduction
**AgentCPM-Report** is built on the 8B-parameter base model [MiniCPM4.1](https://github.com/OpenBMB/MiniCPM). It takes user instructions as input and autonomously generates long-form reports. Highlights include:

- **Extreme Performance, Minimal Footprint**: Through an average of 40 rounds of deep retrieval and nearly 100 rounds of chain-of-thought reasoning, it achieves comprehensive information mining and restructuring, enabling edge-side models to produce logically rigorous, deeply insightful long-form articles with tens of thousands of words. With just 8 billion parameters, it delivers performance on par with top-tier closed-source systems in deep research tasks.  
- **Physical Isolation, Local Security**: Specifically designed for high-privacy scenarios, it supports fully offline and agile local deployment, completely eliminating the risk of cloud data leaks. Leveraging our UltraRAG framework, it efficiently mounts and understands your local private knowledge base, securely transforming core confidential data into highly valuable professional decision-making reports without ever leaving its domain.


### Demo
<div align="center">
  <a href="https://www.youtube.com/watch?v=d5XWONt0PWo"><img src="https://img.youtube.com/vi/d5XWONt0PWo/0.jpg", width=70%></a>
</div>


### QuickStart
#### Docker Deployment
<div align="center">
  <a href="https://www.youtube.com/watch?v=ze8qJRrass4"><img src="https://img.youtube.com/vi/ze8qJRrass4/0.jpg", width=70%></a>
</div>

We provide a minimal one-click docker-compose deployment integrated into UltraRAG, which includes the RAG framework UltraRAG2.0, the model inference framework vllm, and the Milvus vector database. If you want CPU inference, we also provide a llama.cpp-based version for GGUF-format models—simply replace `docker-compose.yml` with `docker-compose.cpu.yml`.

``` bash
git clone git@github.com:OpenBMB/UltraRAG.git
cd UltraRAG
git checkout agentcpm-report-demo
cd agentcpm-report-demo
cp env.example .env
docker-compose -f docker-compose.yml up -d --build
docker-compose -f docker-compose.yml logs -f ultrarag-ui
``` 
The first startup needs to pull images, download models, and set up the environment, which may take about 30 minutes.
Then open `http://localhost:5050`. If you can see the GUI, the deployment is successful.
Follow the UI instructions to upload local files, chunk them, and build the index. Then, in the Chat panel, select AgentCPM-Report in the pipeline to start your workflow!

(Optional) You can import Wiki2024 from [Wiki2024](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark/tree/master/corpus/wiki24) as a writing database.

You can read more tutorials about AgentCPM-Report from the [tutorial](https://ultrarag.openbmb.cn/pages/en/demo/deepresearch).


# License

* The code in this repository is released under the [Apache-2.0](./LICENSE) license.

# Citation

If **AgentCPM-Explore** is useful for your research, please cite the codebase:

```bibtex
@software{AgentCPMExplore2026,
  title  = {AgentCPM-Explore: An End-to-End Infrastructure for Training and Evaluating LLM Agents},
  author = {Haotian Chen and Xin Cong and Shengda Fan and Yuyang Fu and Ziqin Gong and Yaxi Lu and Yishan Li and Boye Niu and Chengjun Pan and Zijun Song and Huadong Wang and Yesai Wu and Yueying Wu and Zihao Xie and Yukun Yan and Zhong Zhang and Yankai Lin and Zhiyuan Liu and Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```

If **AgentCPM-Report** is helpful for your research, you can cite it as follows:

```bibtex
@software{AgentCPMReport2026,
  title  = {AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch},
  author = {Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```


# Explore More

- [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- [UltraRAG](https://github.com/OpenBMB/UltraRAG)
