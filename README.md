

<div align="center">
  <img src="./assets/light.svg" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【<a href="README_zh.md">中文</a> | English】
</p>



# Latest News

* [2026-01-12] 🚀🚀🚀 We have open-sourced **AgentCPM-Explore**, the first open-source agent model with 4B parameters to appear on 8 widely used long-horizon agent benchmarks.

# Overview
AgentCPM is a series of open-source large language model agents jointly developed by the [THUNLP](https://nlp.csai.tsinghua.edu.cn), [Renmin University of China](http://ai.ruc.edu.cn/), [ModelBest](https://modelbest.cn/en), and [OpenBMB](https://www.openbmb.cn/en/home). 

# Model List

| Model            | Download Links                                                                                                                                | Resources | Technical Report | How to Use |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-----------|
| AgentCPM-Explore          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Explore)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Explore/)                  |  [AgentDock](./AgentCPM-Explore/AgentDock): An unified tool sandbox management and scheduling platform  <br> [AgentRL](./AgentCPM-Explore/AgentRL): An asynchronous agent reinforcement learning training framework  <br> [AgentToLeaP](./AgentCPM-Explore/AgentToLeaP): An one-click evaluation platform for agent tool-learning capabilities | Coming Soon | [README.md](./AgentCPM-Explore) |


## AgentCPM-Explore

The AgentCPM team has focused on systematically building agents’ deep research capabilities and released **AgentCPM-Explore**, a deep-search LLM agent. **AgentCPM-Explore** is the first open-source agent model with 4B parameters to appear on eight widely used long-horizon agent benchmarks such as GAIA, XBench, etc.

Key highlights:

- **SOTA at 4B Scale**: Best-in-class among same-size models, matches or surpasses 8B models, rivals some 30B+ and closed-source LLMs.

- **Deep Exploration**: 100+ turns of continuous interaction with multi-source cross-validation and dynamic strategy adjustment.

- **End-to-End Open Source**: Complete training and evaluation infrastructure for community development and custom extensions.


### Demo

Demo examples (speed up):

https://github.com/user-attachments/assets/f2b3bb20-ccd5-4b61-8022-9f6e90992baa



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

# Explore More

- [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- [UltraRAG](https://github.com/OpenBMB/UltraRAG)
