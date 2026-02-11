# 电商评论观点挖掘 - 天池赛道六

[![Competition](https://img.shields.io/badge/Competition-TianChi-orange)](https://tianchi.aliyun.com/competition/entrance/532421)
[![Season1](https://img.shields.io/badge/Season1-F1%200.81%20Rank3-blue)](https://tianchi.aliyun.com/forum/post/936514)
[![Season2](https://img.shields.io/badge/Season2-F1%200.81%20Rank1-red)](https://tianchi.aliyun.com/forum/post/949436)

## 📋 项目简介

本项目是针对天池经典打榜赛「赛道六-评论观点挖掘赛」的完整解决方案。任务目标是从电商化妆品评论中抽取结构化的观点四元组信息：**（属性特征词、观点词、情感极性、属性种类）**，即 **ACOS 四元组抽取任务**。

该任务属于**基于方面的情感分析（ABSA, Aspect-Based Sentiment Analysis）**领域，对于电商平台的舆情监控、产品优化和营销决策具有重要的应用价值。

## 🏆 比赛成绩

| 赛季 | F1 分数 | 排名 | 方案 | 方案分享文章 |
|------|---------|------|------|-------------|
| 第一赛季 | **0.81** | 🥉 第3名 | [MethodsV1](./MethodsV1/) | [方案分析](https://tianchi.aliyun.com/forum/post/936514) |
| 第二赛季 | **0.81** | 🥇 第1名 | [MethodsV2](./MethodsV2/) | [方案分享](https://tianchi.aliyun.com/forum/post/949436) |

## 🎯 任务定义

### 四元组结构

每个观点四元组包含以下四个字段：

- **AspectTerm（属性特征词）**：商品的某个属性特征，如"价格"、"包装"、"质地"等
- **OpinionTerm（观点词）**：用户对该属性的评价，如"很便宜"、"精致"、"水润"等
- **Category（属性种类）**：属性所属的大类，如"价格"、"包装"、"功效"等
- **Polarity（情感极性）**：情感倾向，取值为 {正面, 中性, 负面}

### 示例

```
评论：价格很便宜，但是包装太随便了
四元组：
1. (价格, 很便宜, 价格, 正面)
2. (包装, 太随便了, 包装, 负面)
```

## 📊 数据说明

### 训练数据
- `Train_reviews.csv`：包含评论ID和评论原文
- `Train_labels.csv`：包含标注的四元组信息

### 测试数据
- `Test_reviews.csv`：待预测的评论数据
- `Result(example).csv`：提交格式示例

### 评分标准

采用 **F1-score** 作为最终评测指标：

$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

其中：
- **Precision（精确率）** = 正确预测的四元组数 / 预测的四元组总数
- **Recall（召回率）** = 正确预测的四元组数 / 真实标注的四元组总数

**判定标准**：AspectTerm、OpinionTerm、Category、Polarity 四个字段**全部正确**才算一个正确的四元组。

## 🚀 方案概述

### 方案一：LoRA 微调方案（MethodsV1）

**核心思路**：对 Qwen 系列大语言模型进行 LoRA 微调，使其学习结构化输出能力。

**技术栈**：
- 基础模型：Qwen2.5-7B-Instruct / Qwen2.5-32B-Instruct
- 微调方法：LoRA (Low-Rank Adaptation)
- 训练平台：
  - 阿里云 PAI + LLaMA Factory (Qwen3-4B, F1=0.75)
  - 硅基流动算力 (Qwen2.5-7B, F1=0.78)
  - 硅基流动算力 (Qwen2.5-32B, F1=0.81)

**关键参数**：
- Learning Rate: 0.0001 - 0.0002
- Epochs: 3
- Batch Size: 8 - 16
- LoRA Rank: 8 - 16
- LoRA Alpha: 32
- Max Tokens: 32768

**详细文档**：
- [方法说明](./MethodsV1/README.md)
- [微调配置](./MethodsV1/模型微调与数据转换.md)

### 方案二：LoRA 微调 + RAG 检索增强（MethodsV2）

**核心思路**：在方案一基础上，增加检索增强生成（RAG）模块，为模型提供相似样本参考。

**技术架构**：
1. **模型微调**：LoRA 微调 Qwen2.5-32B-Instruct
2. **检索模块**：
   - 第一阶段：BM25 粗召回（Top 20）
   - 第二阶段：bge-reranker-v2-m3 精排（Top 5）
3. **Few-shot Prompting**：将检索到的相似样本作为示例注入 Prompt
4. **结构化输出**：通过 Pydantic 校验确保输出合法性

**技术优势**：
- ✅ 提升长尾表达和隐式属性的识别能力
- ✅ 通过相似样本引导模型理解复杂情感表达
- ✅ 并发推理 + 断点续跑，保证工程稳定性

**详细文档**：
- [方案分享](./MethodsV2/方案分享.md)

## 📁 项目结构

```
TianChiTrack6_ABSA/
├── data/                           # 数据目录
│   ├── TRAIN/                      # 训练数据
│   │   ├── Train_reviews.csv
│   │   └── Train_labels.csv
│   └── TEST/                       # 测试数据
│       ├── Test_reviews.csv
│       └── Result(example).csv
│
├── MethodsV1/                      # 方案一：LoRA微调
│   ├── README.md                   # 方案说明
│   ├── 模型微调与数据转换.md       # 微调详细配置
│   ├── convert_absa_to_jsonl.py   # 训练数据构建
│   └── predict_acos_in_code.py    # 预测脚本
│
├── MethodsV2/                      # 方案二：微调+RAG
│   ├── 方案分享.md                 # 方案说明
│   ├── bm25_retriever.py          # BM25检索
│   ├── predict_multi_cot.py       # 多阶段推理
│   └── predict_multi_agent_langgraph.py  # Multi-Agent方案
│
├── llamafactory_aliyun_lora_qwen3-4b/  # 阿里云微调实验
│   └── qwen3_4b_llama_factory.ipynb
│
├── papers_survey/                  # 相关论文调研
├── tecnology_doc/                  # 技术文档
├── des.md                          # 比赛任务描述
└── README.md                       # 本文件
```

## 🛠️ 使用方法

### 环境配置

```bash
pip install openai pydantic jieba rank-bm25
```

### 数据准备

1. 将训练数据转换为 JSONL 格式：
```bash
cd MethodsV1
python convert_absa_to_jsonl.py \
    --reviews ../data/TRAIN/Train_reviews.csv \
    --labels ../data/TRAIN/Train_labels.csv \
    --output train.jsonl
```

### 模型微调

**方案一（推荐硅基流动平台）**：
1. 访问 [硅基流动](https://cloud.siliconflow.cn/)
2. 上传 `train.jsonl`
3. 选择 Qwen2.5-32B-Instruct
4. 配置 LoRA 参数并启动微调

**方案二（阿里云 PAI）**：
参考 [qwen3_4b_llama_factory.ipynb](./llamafactory_aliyun_lora_qwen3-4b/qwen3_4b_llama_factory.ipynb)

### 推理预测

**基础预测**（方案一）：
```bash
cd MethodsV1
python predict_acos_in_code.py
```

**RAG 增强预测**（方案二）：
```bash
cd MethodsV2
python predict_multi_cot.py
```

## 🔍 关键经验

1. **模型规模与效果正相关**：在 LoRA 微调框架下，模型越大（4B → 7B → 32B），F1 分数越高
2. **隐式属性处理**：训练数据中混入大量 `aspect="_"` 样本，提升隐式属性识别能力
3. **结构化输出**：通过 System Prompt 严格约束输出格式，避免模型生成多余解释
4. **RAG 检索增强**：BM25 + Reranker 两阶段检索，为模型提供高质量参考样本
5. **工程化实践**：并发推理 + 断点续跑 + 日志记录，确保大规模推理的稳定性

## 🧱 工程复盘（踩坑与对策）

这部分来自参赛过程中的真实踩坑记录，核心结论是：**严格匹配类抽取任务，工程稳定性与输出治理决定下限**。

### 1) 线上推理的稳定性：断网/限流是“必然事件”

**现象**：长时间批量调用 API 时，偶发 `SSLEOFError` / 超时 / 429 限流，会导致部分 ID 直接空结果，最终评分直接受损。

**对策**：
- 请求层：指数退避重试（含抖动）、可配置最大重试次数、区分可重试错误与不可重试错误
- 并发层：控制并发（例如 ThreadPoolExecutor 的 worker 数），避免把限流当成模型问题
- 落盘层：每条样本预测后立刻写入 `jsonl`（增量写），支持断点续跑与失败重跑
- 回退策略：多次失败后回退到更保守的解码参数/更短 prompt，至少保证“ID 不缺失”

### 2) 输出治理：把“解析/校验/去重/排序”当成模型的一部分

**易错点**：
- JSON 被污染（夹杂解释文本/多余换行）
- 字段越界（Category 不在集合、Polarity 非 {正面, 中性, 负面}）
- 重复四元组、或同一观点被拆/合导致错配
- ID 缺失、ID 未升序、空评论未按规则输出 `_`

**对策**：
- 解析：只截取首个可解析 JSON 对象；失败则记录并进入回退
- 校验：用 Pydantic/自定义校验器做字段合法性检查；非法条目直接丢弃
- 去重：以 `(aspect, opinion, category, polarity)` 作为 key 去重
- 排序：按 `opinion` 在原文起始位置升序（缺失位置放后），保证 determinism

### 3) 可复现与对照实验：一次只改一个变量

**现象**：温度、Self-Consistency、多阶段/CoT、RAG、后处理等同时调整时，很难定位涨跌原因。

**对策**：
- 固化配置：把所有可变参数集中在一个配置文件（模型名、温度、top_p、topk、检索 k、并发度等）
- Ablation：一次只改一个因素，保留“最稳 baseline”（低温、短 prompt、最强校验）随时回滚
- 记录指标：除了最终 F1，还记录非法输出比例、平均四元组数、失败 ID 数、重试次数分布

### 4) Self-Consistency（SC）在抽取任务里，投票粒度要“结构级”

**现象**：对整段 JSON 字符串做 majority，往往因为顺序/细微差异导致“都不一致”，SC 收益不明显。

**对策**：
- 先把每次采样解析成四元组集合
- 对四元组做频次统计（结构级投票），保留达到阈值的四元组
- 最后统一去重与排序，输出稳定可控

### 5) 训练-推理格式一致性：不要让模型“学 A、用 B”

**现象**：微调学的是严格 JSON 模板，但推理阶段加入多阶段解释/CoT，容易让输出变脏并降低一致性。

**对策**：
- 若要“带推理过程”，需要在训练数据中也包含同风格的输出（否则建议在日志里保留推理，而不是要求模型输出推理）
- 优先追求结构化输出的稳定，再做复杂策略叠加

## 📚 参考资源

- **比赛链接**：https://tianchi.aliyun.com/competition/entrance/532421
- **Baseline**：[OpinioNet](https://github.com/eguilg/OpinioNet)
- **相关论文**：见 [papers_survey](./papers_survey/)







