# AgentCPM-Report：Gemini-2.5-pro-DeepResearch水平的本地DeepResearch

<a href='https://huggingface.co/openbmb/AgentCPM-Report'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report-blue'>
<a href='https://huggingface.co/openbmb/AgentCPM-Report-gguf'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report--gguf-blue'>
<a href='https://github.cpm/OpenBMB/UltraRAG'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20GitHub-UltraRAG-blue'>

<p align="center">

| 
<b>简体中文</b>
|
<a href="./README.md"><b>English</b></a>
|
</p>

## 新闻
- [2026-01-20] 🚀🚀🚀我们开源了基于MiniCPM4.1-8B构建的AgentCPM-Report，能够在报告生成领域比肩顶尖的闭源商业系统如Gemini-2.5-pro-DeepResearch
## 概述
AgentCPM-Report是由[THUNLP](https://nlp.csai.tsinghua.edu.cn)、中国人民大学[RUCBM] (https://github.com/RUCBM)和[ModelBest](https://modelbest.cn/en)联合开发的开源大语言模型智能体。它基于[MiniCPM4.1](https://github.com/OpenBMB/MiniCPM4.1) 80亿参数基座模型，接受用户指令作为输入，自主生成长篇报告。其有以下亮点：

- 洞察力和全面性的显著优势：首个在深度调研报告生成任务上赶超闭源DeepResearch系统的8B端侧模型，重新定义小规模智能体系统性能的天花板，尤其是在洞察力（Insight）这个指标上取得SOTA结果。
- 轻量化和本地化部署：支持本地进行敏捷部署，基于UltraRAG等框架实现规模化的知识库构建，完成甚至比大模型更加专业、深入的报告生成。轻量级的模型和本地知识库的支持使得可以在个人计算机上部署深度调研报告写作系统成为可能，为基于个人隐私数据或私域数据的报告写作提供了基础。

## 演示案例：
`这里是视频的油管链接或bilibili链接`

## 快速开始
### Docker部署
我们实现了一个最简单的docker-compose一键部署，集成进了UltraRAG，包含RAG框架UltraRAG2.0，模型推理框架vllm与向量数据库milvus；如果您想使用cpu推理，我们也有使用llama.cpp对gguf格式文件模型的版本，将`docker-compose.yml`转成`docker-compose.cpu.yml`即可。

``` bash
git clone git@github.com:OpenBMB/UltraRAG.git
cd UltraRAG
git checkout agentcpm-report-demo
cd agentcpm-report-demo
cp env.example .env
docker-compose -f docker-compose.yml up -d --build
docker-compose -f docker-compose.yml logs -f ultrarag-ui
``` 
第一次启动需要拉取镜像，下载模型并配环境，需要稍等约30分钟左右
之后您打开http:/localhost:5050 ，如果能看到图形界面，则说明您部署成功。
您可以遵循界面指示，上传本地文件，并进行切片，建索引；之后在Chat板块pipeline选择AgentCPM-Report开始您的流程！
（可选）您可以从https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark/tree/master/corpus/wiki24 导入Wiki2024作为写作数据库。
您可以从https://ultrarag.openbmb.cn/pages/cn/pipeline/agentcpm-report 中阅读更多关于AgentCPM-Report的教程。

### 代码结构
```
AgentCPM-Report/
├── agentcpm-report-demo/  # 包含一键部署的docker-compose配置
├── examples/              # 包含AgentCPM-Report的配置文件示例
├── prompts/               # 包含报告生成所需的Prompt模板
├── servers/               # 自定义服务实现（主要是AgentCPM-Report管线）
└── UltraRAG/              # 部署框架UltraRAG，已经融合进部署相关内容
```

## 方法：
AgentCPM-Report主要特性包括：
- 更符合人类认知的写作模式：提出“写作即推理”（Writing as Reasoning）的执行模式，智能体根据写作内容自主决策是否调整写作计划，真正做到像人一样“边写作，边规划”，在写作过程中不断获得新的洞察。
- 自主决策和深化：赋予智能体更多的自主性，其可以根据当前的写作结果自主决策是正式提交或继续深化。
- 多阶段强化学习：将报告生成目标分解为规划(planning)、检索(retrieval)、写作(write)和决策(decision-making)四项原子能力。**我们采取三阶段训练策略**：首先通过SFT进行冷启动；其次设计特定奖励函数（如利用“轨迹剪枝”优化决策、利用“召回率”优化检索）对各原子能力进行独立强化（Atomic Skill RL），保证训练效率与稳定性；最后以整体报告质量为目标进行全流程强化学习（Pipeline RL），确保各模块协同工作达到最优。
## 评估
| DeepResearch Bench            | Overall | Comprehensiveness. | Insight | Instruction Following | Readability |
|-------------------------------|---------|--------------------|---------|-----------------------|-------------|
| Doubao-research               | 44.34   | 44.84              | 40.56   | 47.95                 | 44.69       |
| Claude-research               | 45      | 45.34              | 42.79   | 47.58                 | 44.66       |
| OpenAI-deepresearch           | 46.45   | 46.46              | 43.73   | 49.39                 | 47.22       |
| Gemini-2.5-Pro-deepresearch   | 49.71   | 49.51              | 49.45   | 50.12                 | 50          |
| WebWeaver(Qwen3-30B-A3B)      | 46.77   | 45.15              | 45.78   | 49.21                 | 47.34       |
| WebWeaver(Claude-Sonnet-4)    | 50.58   | 51.45              | 50.02   | 50.81                 | 49.79       |
| Enterprise-DR(Gemini-2.5-Pro) | 49.86   | 49.01              | 50.28   | 50.03                 | 49.98       |
| RhinoInsigh(Gemini-2.5-Pro)   | 50.92   | 50.51              | 51.45   | 51.72                 | 50          |
| AgentCPM-Report               | 50.11   | 50.54              | 52.64   | 48.87                 | 44.17       |

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


我们的评测数据集包括DeepResearch Bench， DeepConsult和DeepResearch Gym，写作时知识库包括约270万Arxiv论文（https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv）以及内部的约20万条网页总结。

## 引用

如果 **AgentCPM-Report** 对您的研究有所帮助，您可以按照如下方式进行引用：

```bibtex
@software{AgentCPMReport2026,
  title  = {AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch},
  author = {Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Cong Xin, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```