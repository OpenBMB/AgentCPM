

<div align="center">
  <img src="./assets/light.svg" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【中文 | <a href="README.md">English</a>】
</p>



# 最新消息

* [2026-01-12] 🚀🚀🚀我们开源了基于全量仅**4B参数**的智能体大模型AgentCPM-Explore及其所有训练、推理、工具沙盒环境代码，成功闯入GAIA、HLE、BrowseComp等8个经典长难智能体任务榜单，同级别SOTA的表现带来更长行为链路、更准确的深度调研能力，由此突破端侧智能体的性能壁垒。


# 概述
AgentCPM 是由[清华大学自然语言处理实验室（THUNLP）](https://nlp.csai.tsinghua.edu.cn)、[中国人民大学](http://ai.ruc.edu.cn/)、[面壁智能](https://modelbest.cn/en)以及[OpenBMB社区](https://www.openbmb.cn/home)联合开发的一系列开源大语言模型智能体。针对智能体在真实世界应用时所面临的长程性、自主性、泛化性不足的问题，提出一系列模型构建方案。团队近期聚焦于先对智能体的深度研究能力进行全方位构建，发布[AgentCPM-Explore](./AgentCPM-Explore/README_zh.md)深度搜索大语言模型智能体与AgentCPM-Report深度调研大语言模型智能体。

# 模型列表

| 模型            | 下载链接                                                                                                                                | 开源内容 | 技术报告 | 如何使用 |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-----------|
| [AgentCPM-Explore](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Explore/README_zh.md)          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Explore)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Explore/)                  |  [AgentDock](./AgentCPM-Explore/AgentDock/README_zh.md): 工具沙盒环境统一管理调度平台  <br> [AgentRL](./AgentCPM-Explore/AgentRL/README_zh.md): 全异步智能体强化学习框架  <br> [AgentToLeaP](./AgentCPM-Explore/AgentToLeaP/README_zh.md): 智能体工具学习能力一键测评框架 | 即将发布 | [README_zh.md](https://github.com/OpenBMB/AgentCPM/blob/main/AgentCPM-Explore/README_zh.md)


## AgentCPM-Explore

### 简介
**AgentCPM-Explore** 拥有 40 亿参数，取得同尺寸模型SOTA、越级赶上甚至超越两倍大参数量（8B级）SOTA模型、比肩部分30B级以上和闭源大模型的效果，实现了更长的行为链路和更准确的深度调研（Deep Research）能力，真正让大模型的长程任务处理能力有望部署于端侧，加速私有化智能助手的普及。AgentCPM-Explore的亮点包括：

- 首个以 4B 全量参数登上 GAIA、HLE、BrowseComp 等 8 个长程复杂智能体任务榜单的端侧智能体模型。

- 可实现超过 100 轮的连续环境交互，支持多源信息交叉验证、搜索策略动态调整、实时核验最新信息，持续深度探索直至任务完成。

- 全流程开源，包括智能体全异步强化学习训练框架AgentRL、工具沙盒统一管理调度平台AgentDock、智能体工具学习能力一键测评平台AgentToLeaP，支持社区共建与自定义扩展。


### 持续深度探索
演示案例（倍速）：


https://github.com/user-attachments/assets/f8487889-d17a-447e-9aef-2608f4c84a83


### QuickStart

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



# 更多项目

- [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- [UltraRAG](https://github.com/OpenBMB/UltraRAG)



