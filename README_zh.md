

<div align="center">
  <img src="./assets/light.svg" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【中文 | <a href="README.md">English</a>】
</p>



# 最新消息

* [2026-01-20] 🚀🚀🚀我们开源了基于MiniCPM4.1-8B构建的AgentCPM-Report，能够在报告生成领域比肩顶尖的闭源商业系统如Gemini-2.5-pro-DeepResearch。

* [2026-01-12] 🚀🚀🚀我们开源了基于全量仅**4B参数**的智能体大模型AgentCPM-Explore及其所有训练、推理、工具沙盒环境代码，成功闯入GAIA、HLE、BrowseComp等8个经典长难智能体任务榜单，同级别SOTA的表现带来更长行为链路、更准确的深度调研能力，由此突破端侧智能体的性能壁垒。

## 目录

- [概述](#概述)
- [模型列表](#模型列表)
- [AgentCPM-Explore](#agentcpm-explore)
  - [示例视频](#示例视频)
  - [快速开始](#快速开始)
- [AgentCPM-Report](#agentcpm-report)
  - [示例视频](#示例视频)
  - [快速开始](#快速开始-1)
- [开源协议](#开源协议)
- [引用](#引用)
- [更多项目](#更多项目)


# 概述
AgentCPM 是由[清华大学自然语言处理实验室（THUNLP）](https://nlp.csai.tsinghua.edu.cn)、[中国人民大学](http://ai.ruc.edu.cn/)、[面壁智能](https://modelbest.cn/en)以及[OpenBMB社区](https://www.openbmb.cn/home)联合开发的一系列开源大语言模型智能体。针对智能体在真实世界应用时所面临的长程性、自主性、泛化性不足的问题，提出一系列模型构建方案。团队近期聚焦于先对智能体的深度研究能力进行全方位构建，发布[AgentCPM-Explore](./AgentCPM-Explore/README_zh.md)深度搜索大语言模型智能体与[AgentCPM-Report](./AgentCPM-Explore/README_zh.md)深度调研大语言模型智能体。

# 模型列表

| 模型            | 下载链接                                                                                                                                | 开源内容 | 技术报告 | 如何使用 |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-----------|
| [AgentCPM-Explore](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Explore/README_zh.md)          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Explore)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Explore/)                  |  [AgentDock](./AgentCPM-Explore/AgentDock/README_zh.md): 工具沙盒环境统一管理调度平台  <br> [AgentRL](./AgentCPM-Explore/AgentRL/README_zh.md): 全异步智能体强化学习框架  <br> [AgentToLeaP](./AgentCPM-Explore/AgentToLeaP/README_zh.md): 智能体工具学习能力一键测评框架 | 即将发布 | [README_zh.md](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Explore/README_zh.md)
| [AgentCPM-Report](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Report/README_zh.md)          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Report)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Report/)                  |  [UltraRAG](https://github.com/OpenBMB/UltraRAG): 低代码RAG框架   | 即将发布 | [README_zh.md](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Report/README_zh.md)


## AgentCPM-Explore

### 简介
**AgentCPM-Explore** 拥有 40 亿参数，取得同尺寸模型SOTA、越级赶上甚至超越两倍大参数量（8B级）SOTA模型、比肩部分30B级以上和闭源大模型的效果，实现了更长的行为链路和更准确的深度调研（Deep Research）能力，真正让大模型的长程任务处理能力有望部署于端侧，加速私有化智能助手的普及。AgentCPM-Explore的亮点包括：

- 首个以 4B 全量参数登上 GAIA、HLE、BrowseComp 等 8 个长程复杂智能体任务榜单的端侧智能体模型。

- 可实现超过 100 轮的连续环境交互，支持多源信息交叉验证、搜索策略动态调整、实时核验最新信息，持续深度探索直至任务完成。

- 全流程开源，包括智能体全异步强化学习训练框架AgentRL、工具沙盒统一管理调度平台AgentDock、智能体工具学习能力一键测评平台AgentToLeaP，支持社区共建与自定义扩展。


### 示例视频
演示案例（倍速）：


https://github.com/user-attachments/assets/f8487889-d17a-447e-9aef-2608f4c84a83


### 快速开始

- **多模型多工具协作环境部署**：首先启动 AgentDock 工具沙盒平台，提供统一的 MCP 工具服务。和 API 模型协作时，配置模型的 `BASE_URL` 和 `API_KEY`；和本地 host 的模型协作时，确保模型服务可访问。在 `config.toml` 文件中配置工具所需的使用参数。

- **启动环境**：开箱即用，一键启动。AgentDock 统一工具沙盒管理平台支持 `docker compose up -d` 一键启动所有服务，包括管理面板、数据库和工具节点。

- **启动执行**：通过 QuickStart 脚本快速体验框架的核心能力，无需繁琐配置即可运行一个完整的 Agent 任务。

0. **准备评测环境 (推荐)**：
   我们提供了一个预装好所有评测依赖的 Docker 镜像，建议直接拉取镜像并在容器内运行：
   
   ```bash
   # 1. 进入项目目录
   cd AgentCPM-Explore
   
   # 2. 拉取镜像
   docker pull yuyangfu/agenttoleap-eval:v1.0
   
   # 3. 启动容器 (请根据实际路径修改 -v 参数)
   docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval:v1.0
   
   # 4. 进入容器
   docker exec -it agenttoleap /bin/bash
   cd /workspace
   ```

1. **配置与运行**：
   打开 `quickstart.py`，在 `[USER CONFIGURATION]` 区域进行简单配置：
   
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

## AgentCPM-Report

### 简介
**AgentCPM-Report**基于[MiniCPM4.1](https://github.com/OpenBMB/MiniCPM) 80亿参数基座模型，接受用户指令作为输入，自主生成长篇报告。其有以下亮点：

- 极致效能，以小博大：通过平均40轮的深度检索与近100轮的思维链推演，实现对信息的全方位挖掘与重组，让端侧模型也能产出逻辑严密、洞察深刻的万字长文，在深度调研任务上以8B参数规模达成与顶级闭源系统的性能对标。
- 物理隔绝，本地安全：专为高隐私场景设计，支持完全离线的本地化敏捷部署，彻底杜绝云端泄密风险。基于我们的 UltraRAG 框架，它能高效挂载并理解您的本地私有知识库，让核心机密数据在“不出域”的前提下，安全地转化为极具价值的专业决策报告。

### 示例视频
<div align="center">
  <a href="https://www.bilibili.com/video/BV1DYkLBNE6f"><img src="https://i0.hdslb.com/bfs/archive/05f18d5914b8691316161021298a5b63da54eaeb.jpg", width=70%></a>
</div>

### 快速开始
#### Docker部署
<div align="center">
  <a href="https://www.bilibili.com/video/BV1Kfk5BtEbG"><img src="http://i1.hdslb.com/bfs/archive/614883b2cf7ada53ade878e4baaad821c5f25a8c.jpg", width=70%></a>
</div>


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
之后您打开`http:/localhost:5050` ，如果能看到图形界面，则说明您部署成功。
您可以遵循界面指示，上传本地文件，并进行切片，建索引；之后在Chat板块pipeline选择AgentCPM-Report开始您的流程！

（可选）您可以从[Wiki2024](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark/tree/master/corpus/wiki24) 导入Wiki2024作为写作数据库。

您可以从[教程](https://ultrarag.openbmb.cn/pages/cn/demo/deepresearch) 中阅读更多关于AgentCPM-Report的教程。

# 开源协议

* 本仓库开源的代码遵照 [Apache-2.0](./LICENSE) 协议。


# 引用
如果 **AgentCPM-Explore** 对您的研究有帮助，您可以按照如下方式进行引用

```bibtex
@software{AgentCPMExplore2026,
  title  = {AgentCPM-Explore: An End-to-End Infrastructure for Training and Evaluating LLM Agents},
  author = {Haotian Chen and Xin Cong and Shengda Fan and Yuyang Fu and Ziqin Gong and Yaxi Lu and Yishan Li and Boye Niu and Chengjun Pan and Zijun Song and Huadong Wang and Yesai Wu and Yueying Wu and Zihao Xie and Yukun Yan and Zhong Zhang and Yankai Lin and Zhiyuan Liu and Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```

如果 **AgentCPM-Report** 对您的研究有所帮助，您可以按照如下方式进行引用：

```bibtex
@software{AgentCPMReport2026,
  title  = {AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch},
  author = {Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```


# 更多项目

- [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- [UltraRAG](https://github.com/OpenBMB/UltraRAG)



