<div align="center">
  <img src="../assets/AgentCPM-Report-logo.png" alt="AgentCPM-Report Logo" width="400em"></img>
</div>


<p align="center">
<a href='https://huggingface.co/openbmb/AgentCPM-Report'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report-yellow'>
<a href='https://huggingface.co/openbmb/AgentCPM-Report-GGUF'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AgentCPM--Report--GGUF-yellow'>
<a href='https://github.com/OpenBMB/UltraRAG'><img src='https://img.shields.io/badge/GitHub-UltraRAG-blue?logo=github'>
</p>

<p align="center">
    [<a href="./README_zh.md"><b>中文</b></a> | <b>English</b>]
</p>

## News
- [2026-01-20] 🚀🚀🚀 We open-sourced AgentCPM-Report built on MiniCPM4.1-8B, capable of matching top closed-source commercial systems like Gemini-2.5-pro-DeepResearch in report generation.

## Overview
AgentCPM-Report is an open-source large language model agent jointly developed by [THUNLP](https://nlp.csai.tsinghua.edu.cn), Renmin University of China [RUCBM](https://github.com/RUCBM), and [ModelBest](https://modelbest.cn/en). It is based on the [MiniCPM4.1](https://github.com/OpenBMB/MiniCPM) 8B-parameter base model. It accepts user instructions as input and autonomously generates long-form reports. Key highlights:

- **Extreme Performance, Minimal Footprint**: Through an average of 40 rounds of deep retrieval and nearly 100 rounds of chain-of-thought reasoning, it achieves comprehensive information mining and restructuring, enabling edge-side models to produce logically rigorous, deeply insightful long-form articles with tens of thousands of words. With just 8 billion parameters, it delivers performance on par with top-tier closed-source systems in deep research tasks.  
- **Physical Isolation, Local Security**: Specifically designed for high-privacy scenarios, it supports fully offline and agile local deployment, completely eliminating the risk of cloud data leaks. Leveraging our UltraRAG framework, it efficiently mounts and understands your local private knowledge base, securely transforming core confidential data into highly valuable professional decision-making reports without ever leaving its domain.

## Demo
`YouTube link or Bilibili link for the video`

## Quick Start
### Docker Deployment
We provide a minimal one-click `docker-compose` deployment integrated with UltraRAG, including the RAG framework UltraRAG2.0, the model inference framework vllm, and the vector database milvus. If you want CPU inference, we also provide a llama.cpp-based version for gguf models—just switch `docker-compose.yml` to `docker-compose.cpu.yml`.

``` bash
git clone git@github.com:OpenBMB/UltraRAG.git
cd UltraRAG
git checkout agentcpm-report-demo
cd agentcpm-report-demo
cp env.example .env
docker-compose -f docker-compose.yml up -d --build
docker-compose -f docker-compose.yml logs -f ultrarag-ui
``` 
The first startup pulls images, downloads the model, and configures the environment, which takes about 30 minutes.
Then open `http://localhost:5050`. If you can see the UI, your deployment is successful.
Follow the UI instructions to upload local files, chunk them, and build indexes; then in the Chat section, select AgentCPM-Report in the pipeline to start your workflow.

(Optional) You can import [Wiki2024](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark/tree/master/corpus/wiki24) as the writing database.

You can read more tutorials about AgentCPM-Report in the [documentation](https://ultrarag.openbmb.cn/pages/cn/pipeline/agentcpm-report).

### Code Structure
```
AgentCPM-Report/
├── agentcpm-report-demo/  # docker-compose configuration for one-click deployment
├── examples/              # configuration examples for AgentCPM-Report
├── prompts/               # prompt templates required for report generation
├── servers/               # custom service implementation (mainly the AgentCPM-Report pipeline)
└── UltraRAG/              # deployment framework UltraRAG (deployment-related content integrated)
```

## Evaluation
<table align="center">
  <thead>
    <tr>
      <th align="center">DeepResearch Bench</th>
      <th align="center">Overall</th>
      <th align="center">Comprehensiveness</th>
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

Our evaluation datasets include DeepResearch Bench, DeepConsult, and DeepResearch Gym. The writing-time knowledge base includes about 2.7 million [Arxiv papers](https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv) and about 200,000 internal webpage summaries.

## Acknowledgements
This project would not be possible without the support and contributions of the open-source community. During development, we referred to and used multiple excellent open-source frameworks, models, and data resources, including [verl](https://github.com/volcengine/verl), [UltraRAG](https://github.com/OpenBMB/UltraRAG), [MiniCPM4.1](https://github.com/OpenBMB/MiniCPM), and [SurveyGo](https://surveygo.modelbest.cn/).

## Contributions
Project leads: Yishan Li, Wentong Chen

Contributors: Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun

Advisors: Yukun Yan, Yankai Lin, Zhiyuan Liu, Maosong Sun

## Citation

If **AgentCPM-Report** is helpful for your research, please cite it as follows:

```bibtex
@software{AgentCPMReport2026,
  title  = {AgentCPM-Report: Gemini-2.5-pro-DeepResearch Level Local DeepResearch},
  author = {Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```
