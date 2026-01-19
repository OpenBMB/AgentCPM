# AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch

<a href='https://huggingface.co/openbmb/AgentCPM-Report'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report-blue'>
<a href='https://huggingface.co/openbmb/AgentCPM-Report-gguf'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report--gguf-blue'>
<a href='https://github.cpm/OpenBMB/UltraRAG'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report--gguf-blue'>

<p align="center">

| <a href="./README_zh.md"><b>简体中文</b></a> |
| :---: |
| <b>English</b> |

</p>

## News
- [2026-01-20] 🚀🚀🚀 We open-sourced AgentCPM-Report based on MiniCPM4.1-8B, capable of matching top closed-source commercial systems like Gemini-2.5-pro-DeepResearch in report generation.

## Overview
AgentCPM-Report is an open-source large language model agent jointly developed by [THUNLP](https://nlp.csai.tsinghua.edu.cn), Renmin University of China [RUCBM](https://github.com/RUCBM), and [ModelBest](https://modelbest.cn/en). Based on the [MiniCPM4.1](https://github.com/OpenBMB/MiniCPM4.1) 8B parameter foundation model, it accepts user instructions as input and autonomously generates long reports. Its highlights include:

- **Significant Advantage in Insight and Comprehensiveness**: The first 8B edge-side model to surpass closed-source DeepResearch systems in deep research report generation tasks, redefining the performance ceiling for small-scale agent systems, especially achieving SOTA results in the Insight metric.
- **Lightweight and Local Deployment**: Supports agile local deployment, enabling scalable knowledge base construction based on frameworks like UltraRAG, completing report generation that is even more professional and in-depth than large models. Lightweight models and local knowledge base support make it possible to deploy deep research report writing systems on personal computers, providing a foundation for report writing based on personal privacy data or private domain data.

## Demo Cases
`YouTube link or Bilibili link for the video`

## Quick Start
### Docker Deployment
We have implemented a simple one-click docker-compose deployment integrated into UltraRAG, including the RAG framework UltraRAG2.0, model inference framework vllm, and vector database milvus. If you want to use CPU inference, we also have a version using llama.cpp for gguf format models; simply convert `docker-compose.yml` to `docker-compose.cpu.yml`.

```bash
git clone git@github.com:OpenBMB/UltraRAG.git
cd UltraRAG
git checkout agentcpm-report-demo
cd agentcpm-report-demo
cp env.example .env
docker-compose -f docker-compose.yml up -d --build
docker-compose -f docker-compose.yml logs -f ultrarag-ui
``` 
The first startup requires pulling images, downloading models, and configuring the environment, which takes about 30 minutes.
After that, open http://localhost:5050. If you see the graphical interface, your deployment is successful.
You can follow the interface instructions to upload local files, slice them, and build indexes; then select AgentCPM-Report in the Chat section pipeline to start your process!
(Optional) You can import Wiki2024 from https://huggingface.co/datasets/UltraRAG/UltraRAG_Benchmark as a writing database.
You can read more tutorials about AgentCPM-Report at https://ultrarag.openbmb.cn/pages/cn/pipeline/agentcpm-report.

### Code Structure
```
AgentCPM-Report/
├── agentcpm-report-demo/  # Contains docker-compose configuration for one-click deployment
├── examples/              # Contains configuration examples for AgentCPM-Report
├── prompts/               # Contains Prompt templates required for report generation
├── servers/               # Custom service implementation (mainly AgentCPM-Report pipeline)
└── UltraRAG/              # Deployment framework UltraRAG, already integrated with deployment content
```

## Methods
Key features of AgentCPM-Report include:
- **Writing Mode More Aligned with Human Cognition**: Proposes the "Writing as Reasoning" execution mode, where the agent autonomously decides whether to adjust the writing plan based on the writing content, truly achieving "planning while writing" like a human, and constantly gaining new insights during the writing process.
- **Autonomous Decision-making and Deepening**: Endows the agent with more autonomy, allowing it to autonomously decide whether to formally submit or continue deepening based on the current writing results.
- **Multi-stage Reinforcement Learning**: Decomposes the report generation goal into four atomic capabilities: planning, retrieval, writing, and decision-making. **We adopt a three-stage training strategy**: First, cold start via SFT; second, design specific reward functions (such as using "trajectory pruning" to optimize decision-making and "recall rate" to optimize retrieval) to independently reinforce each atomic capability (Atomic Skill RL), ensuring training efficiency and stability; finally, conduct full-process reinforcement learning (Pipeline RL) with overall report quality as the goal to ensure optimal collaboration among modules.

## Evaluation
| DeepResearch Bench            | Overall | Comprehensiveness | Insight | Instruction Following | Readability |
|-------------------------------|---------|-------------------|---------|-----------------------|-------------|
| Doubao-research               | 44.34   | 44.84             | 40.56   | 47.95                 | 44.69       |
| Claude-research               | 45      | 45.34             | 42.79   | 47.58                 | 44.66       |
| OpenAI-deepresearch           | 46.45   | 46.46             | 43.73   | 49.39                 | 47.22       |
| Gemini-2.5-Pro-deepresearch   | 49.71   | 49.51             | 49.45   | 50.12                 | 50          |
| WebWeaver(Qwen3-30B-A3B)      | 46.77   | 45.15             | 45.78   | 49.21                 | 47.34       |
| WebWeaver(Claude-Sonnet-4)    | 50.58   | 51.45             | 50.02   | 50.81                 | 49.79       |
| Enterprise-DR(Gemini-2.5-Pro) | 49.86   | 49.01             | 50.28   | 50.03                 | 49.98       |
| RhinoInsigh(Gemini-2.5-Pro)   | 50.92   | 50.51             | 51.45   | 51.72                 | 50          |
| AgentCPM-Report               | 50.11   | 50.54             | 52.64   | 48.87                 | 44.17       |

| DeepConsult                   | Avg. | Win   | Tie   | Lose  |
|-------------------------------|------|-------|-------|-------|
| Doubao-research               | 5.42 | 29.95 | 40.35 | 29.7  |
| Claude-research               | 4.6  | 25    | 38.89 | 36.11 |
| OpenAI-deepresearch           | 5    | 0     | 100   | 0     |
| Gemini-2.5-Pro-deepresearch   | 6.7  | 61.27 | 31.13 | 7.6   |
| WebWeaver(Qwen3-30B-A3B)      | 4.57 | 28.65 | 34.9  | 36.46 |
| WebWeaver(Claude-Sonnet-4)    | 6.96 | 66.86 | 10.47 | 22.67 |
| Enterprise-DR(Gemini-2.5-Pro) | 6.82 | 71.57 | 19.12 | 9.31  |
| RhinoInsigh(Gemini-2.5-Pro)   | 6.82 | 68.51 | 11.02 | 20.47 |
| AgentCPM-Report               | 6.6  | 57.6  | 13.73 | 28.68 |

| DeepResearch Gym            | Avg.  | Clarity  | Depth  | Balance  | Breadth  | Support  | Insightfulness |
|-----------------------------|-------|----------|--------|----------|----------|----------|----------------|
| Doubao-research             | 84.46 | 68.85    | 93.12  | 83.96    | 93.33    | 84.38    | 83.12          |
| Claude-research             | 80.25 | 86.67    | 96.88  | 84.41    | 96.56    | 26.77    | 90.22          |
| OpenAI-deepresearch         | 91.27 | 84.90    | 98.10  | 89.80    | 97.40    | 88.40    | 89.00          |
| Gemini-2.5-pro-deepresearch | 96.02 | 90.71    | 99.90  | 93.37    | 99.69    | 95.00    | 97.45          |
| WebWeaver (Qwen3-30b-a3b)   | 77.27 | 71.88    | 85.51  | 75.80    | 84.78    | 63.77    | 81.88          |
| WebWeaver (Claude-sonnet-4) | 96.77 | 90.50    | 99.87  | 94.30    | 100.00   | 98.73    | 97.22          |
| AgentCPM-Report             | 98.48 | 95.1     | 100.0  | 98.5     | 100.0    | 97.3     | 100.0          |


Our evaluation datasets include DeepResearch Bench, DeepConsult, and DeepResearch Gym. The knowledge base used during writing includes about 2.7 million Arxiv papers (https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv) and about 200,000 internal webpage summaries.

## Citation

If **AgentCPM-Report** is helpful for your research, please cite it as follows:

```bibtex
@software{AgentCPMReport2026,
  title  = {AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch},
  author = {Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Cong Xin, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```
