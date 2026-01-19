# AgentCPM-Report：Gemini-2.5-pro-DeepResearch水平的本地DeepResearch

<p align="center">
<a href='https://huggingface.co/openbmb/AgentCPM-Report'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report-yellow'>
<a href='https://huggingface.co/openbmb/AgentCPM-Report-GGUF'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report--GGUF-yellow'>
<a href='https://github.com/OpenBMB/UltraRAG'><img src='https://img.shields.io/badge/GitHub-UltraRAG-blue?logo=github'>
</p>

<p align="center">
    【中文 | <a href="./README.md"><b>English</b></a>】
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
之后您打开`http:/localhost:5050` ，如果能看到图形界面，则说明您部署成功。
您可以遵循界面指示，上传本地文件，并进行切片，建索引；之后在Chat板块pipeline选择AgentCPM-Report开始您的流程！

（可选）您可以从[Wiki2024](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark/tree/master/corpus/wiki24) 导入Wiki2024作为写作数据库。

您可以从[教程](https://ultrarag.openbmb.cn/pages/cn/pipeline/agentcpm-report) 中阅读更多关于AgentCPM-Report的教程。

### 代码结构
```
AgentCPM-Report/
├── agentcpm-report-demo/  # 包含一键部署的docker-compose配置
├── examples/              # 包含AgentCPM-Report的配置文件示例
├── prompts/               # 包含报告生成所需的Prompt模板
├── servers/               # 自定义服务实现（主要是AgentCPM-Report管线）
└── UltraRAG/              # 部署框架UltraRAG，已经融合进部署相关内容
```

## 评估
<table align="center">
  <thead>
    <tr>
      <th align="center">DeepResearch Bench</th>
      <th align="center">Overall</th>
      <th align="center">Comprehensiveness.</th>
      <th align="center">Insight</th>
      <th align="center">Instruction Following</th>
      <th align="center">Readability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Doubao-research</td>
      <td align="center">44.34</td>
      <td align="center">44.84</td>
      <td align="center">40.56</td>
      <td align="center">47.95</td>
      <td align="center">44.69</td>
    </tr>
    <tr>
      <td align="center">Claude-research</td>
      <td align="center">45</td>
      <td align="center">45.34</td>
      <td align="center">42.79</td>
      <td align="center">47.58</td>
      <td align="center">44.66</td>
    </tr>
    <tr>
      <td align="center">OpenAI-deepresearch</td>
      <td align="center">46.45</td>
      <td align="center">46.46</td>
      <td align="center">43.73</td>
      <td align="center">49.39</td>
      <td align="center">47.22</td>
    </tr>
    <tr>
      <td align="center">Gemini-2.5-Pro-deepresearch</td>
      <td align="center">49.71</td>
      <td align="center">49.51</td>
      <td align="center">49.45</td>
      <td align="center">50.12</td>
      <td align="center">50</td>
    </tr>
    <tr>
      <td align="center">WebWeaver(Qwen3-30B-A3B)</td>
      <td align="center">46.77</td>
      <td align="center">45.15</td>
      <td align="center">45.78</td>
      <td align="center">49.21</td>
      <td align="center">47.34</td>
    </tr>
    <tr>
      <td align="center">WebWeaver(Claude-Sonnet-4)</td>
      <td align="center">50.58</td>
      <td align="center">51.45</td>
      <td align="center">50.02</td>
      <td align="center">50.81</td>
      <td align="center">49.79</td>
    </tr>
    <tr>
      <td align="center">Enterprise-DR(Gemini-2.5-Pro)</td>
      <td align="center">49.86</td>
      <td align="center">49.01</td>
      <td align="center">50.28</td>
      <td align="center">50.03</td>
      <td align="center">49.98</td>
    </tr>
    <tr>
      <td align="center">RhinoInsigh(Gemini-2.5-Pro)</td>
      <td align="center">50.92</td>
      <td align="center">50.51</td>
      <td align="center">51.45</td>
      <td align="center">51.72</td>
      <td align="center">50</td>
    </tr>
    <tr>
      <td align="center">AgentCPM-Report</td>
      <td align="center">50.11</td>
      <td align="center">50.54</td>
      <td align="center">52.64</td>
      <td align="center">48.87</td>
      <td align="center">44.17</td>
    </tr>
  </tbody>
</table>



<table align="center">
  <thead>
    <tr>
      <th align="center">DeepResearch Gym</th>
      <th align="center">Avg.</th>
      <th align="center">Clarity</th>
      <th align="center">Depth</th>
      <th align="center">Balance</th>
      <th align="center">Breadth</th>
      <th align="center">Support</th>
      <th align="center">Insightfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Doubao-research</td>
      <td align="center">84.46</td>
      <td align="center">68.85</td>
      <td align="center">93.12</td>
      <td align="center">83.96</td>
      <td align="center">93.33</td>
      <td align="center">84.38</td>
      <td align="center">83.12</td>
    </tr>
    <tr>
      <td align="center">Claude-research</td>
      <td align="center">80.25</td>
      <td align="center">86.67</td>
      <td align="center">96.88</td>
      <td align="center">84.41</td>
      <td align="center">96.56</td>
      <td align="center">26.77</td>
      <td align="center">90.22</td>
    </tr>
    <tr>
      <td align="center">OpenAI-deepresearch</td>
      <td align="center">91.27</td>
      <td align="center">84.90</td>
      <td align="center">98.10</td>
      <td align="center">89.80</td>
      <td align="center">97.40</td>
      <td align="center">88.40</td>
      <td align="center">89.00</td>
    </tr>
    <tr>
      <td align="center">Gemini-2.5-pro-deepresearch</td>
      <td align="center">96.02</td>
      <td align="center">90.71</td>
      <td align="center">99.90</td>
      <td align="center">93.37</td>
      <td align="center">99.69</td>
      <td align="center">95.00</td>
      <td align="center">97.45</td>
    </tr>
    <tr>
      <td align="center">WebWeaver (Qwen3-30b-a3b)</td>
      <td align="center">77.27</td>
      <td align="center">71.88</td>
      <td align="center">85.51</td>
      <td align="center">75.80</td>
      <td align="center">84.78</td>
      <td align="center">63.77</td>
      <td align="center">81.88</td>
    </tr>
    <tr>
      <td align="center">WebWeaver (Claude-sonnet-4)</td>
      <td align="center">96.77</td>
      <td align="center">90.50</td>
      <td align="center">99.87</td>
      <td align="center">94.30</td>
      <td align="center">100.00</td>
      <td align="center">98.73</td>
      <td align="center">97.22</td>
    </tr>
    <tr>
      <td align="center">AgentCPM-Report</td>
      <td align="center">98.48</td>
      <td align="center">95.1</td>
      <td align="center">100.0</td>
      <td align="center">98.5</td>
      <td align="center">100.0</td>
      <td align="center">97.3</td>
      <td align="center">100.0</td>
    </tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr>
      <th align="center">DeepConsult</th>
      <th align="center">Avg.</th>
      <th align="center">Win</th>
      <th align="center">Tie</th>
      <th align="center">Lose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Doubao-research</td>
      <td align="center">5.42</td>
      <td align="center">29.95</td>
      <td align="center">40.35</td>
      <td align="center">29.7</td>
    </tr>
    <tr>
      <td align="center">Claude-research</td>
      <td align="center">4.6</td>
      <td align="center">25</td>
      <td align="center">38.89</td>
      <td align="center">36.11</td>
    </tr>
    <tr>
      <td align="center">OpenAI-deepresearch</td>
      <td align="center">5</td>
      <td align="center">0</td>
      <td align="center">100</td>
      <td align="center">0</td>
    </tr>
    <tr>
      <td align="center">Gemini-2.5-Pro-deepresearch</td>
      <td align="center">6.7</td>
      <td align="center">61.27</td>
      <td align="center">31.13</td>
      <td align="center">7.6</td>
    </tr>
    <tr>
      <td align="center">WebWeaver(Qwen3-30B-A3B)</td>
      <td align="center">4.57</td>
      <td align="center">28.65</td>
      <td align="center">34.9</td>
      <td align="center">36.46</td>
    </tr>
    <tr>
      <td align="center">WebWeaver(Claude-Sonnet-4)</td>
      <td align="center">6.96</td>
      <td align="center">66.86</td>
      <td align="center">10.47</td>
      <td align="center">22.67</td>
    </tr>
    <tr>
      <td align="center">Enterprise-DR(Gemini-2.5-Pro)</td>
      <td align="center">6.82</td>
      <td align="center">71.57</td>
      <td align="center">19.12</td>
      <td align="center">9.31</td>
    </tr>
    <tr>
      <td align="center">RhinoInsigh(Gemini-2.5-Pro)</td>
      <td align="center">6.82</td>
      <td align="center">68.51</td>
      <td align="center">11.02</td>
      <td align="center">20.47</td>
    </tr>
    <tr>
      <td align="center">AgentCPM-Report</td>
      <td align="center">6.6</td>
      <td align="center">57.6</td>
      <td align="center">13.73</td>
      <td align="center">28.68</td>
    </tr>
  </tbody>
</table>


我们的评测数据集包括DeepResearch Bench， DeepConsult和DeepResearch Gym，写作时知识库包括约270万[Arxiv论文](https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv)以及内部的约20万条网页总结。

## 致谢
本项目的实现离不开开源社区的支持与贡献。我们在开发过程中参考并使用了多个优秀的开源框架、模型和数据资源，包括[verl](https://github.com/volcengine/verl)、[UltraRAG](https://github.com/OpenBMB/UltraRAG)、[MiniCPM4.1](https://github.com/OpenBMB/MiniCPM4.1)、[SurveyGo](https://surveygo.modelbest.cn/)

## 贡献

项目负责人：李奕杉，陈文通

项目贡献者：李奕杉，陈文通，闫宇坤，李明蔚，梅森，王晓荣，刘鲲鹏，从鑫，王硕，张众，卢雅西，刘正皓，林衍凯，刘知远，孙茂松

项目指导人：闫宇坤，林衍凯，刘知远，孙茂松

## 引用

如果 **AgentCPM-Report** 对您的研究有所帮助，您可以按照如下方式进行引用：

```bibtex
@software{AgentCPMReport2026,
  title  = {AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch},
  author = {Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```