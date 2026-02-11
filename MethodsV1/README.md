## 1 引言

了解比赛题目后，可将本次竞赛任务可归类为基于方面的情感分析（ABSA, Aspect-Based Sentiment Analysis）。我因偶然机会参与本次竞赛，核心思路以高效便捷为导向，未开展大量文献调研，方案核心围绕 LoRA 微调 Qwen系列模型与结构化输出展开，具体分为以下两个阶段：

方案一：基于阿里云免费算力的初步尝试。受限于初始算力条件，我首先关注到比赛官网 “算力与工具” 栏目，随后借助阿里云新人优惠获取了免费计算资源，并采用官网推荐的 Notebook（https://gallery.pai-ml.com/#/preview/deepLearning/nlp/qwen3_4b_llama_factory）开展实验。该 Notebook 基于 Qwen3 4B 模型与 LLaMA Factory 工具链，我按操作指引完成 LoRA 微调后，模型最终 F1 分数为 0.75。此成绩虽达到基础效果，但这个分数还达不到我参加评比的好成绩。要想获得更高的F1，就需要微调更大的模型。

方案二：基于硅基流动算力的进阶优化。在资源受限的情况下，我开始寻找免费算力的平台，于是发现了硅基流动。通过邀请多位好友注册，我积累了足量免费算力，随后基于该平台分别对 Qwen/Qwen2.5-7B-Instruct 与 Qwen/Qwen2.5-32B-Instruct 模型进行 LoRA 微调，对应的 F1 分数分别提升至 0.78 与 0.81，看了下排名，这个分数应该已经足够我参加评比了。

从上述实验结果可见，在 LoRA 微调框架下，模型参数规模与任务效果呈明显正相关：模型越大，情感分析的精准度越高。如果你的朋友更多，可以微调72b，如果你有H100，通过LLaMA Factory 或者unsloth提供的notebook微调qwen3系列更大的模型，我猜可以稳拿第一了。

下文将完整提供本方案的训练数据构建脚本与预测脚本，供参考实践。

## 2 方法

### 2.1 构建train数据

构建训练数据是为了将相关数据转换为LLM微调所需的 JSONL 格式数据，为模型学习特定的 ABSA 任务知识提供数据基础。

```python
#!/usr/bin/env python

\# -*- coding: utf-8 -*-

"""

将 ABSA 四元组标注数据 (Train_reviews.csv + Train_labels.csv)

转换为 SiliconFlow 微调所需 .jsonl 格式：

每行一个 JSON，包含 messages: [system, user, assistant]



用法示例：

python convert_absa_to_jsonl.py 

  --reviews Train_reviews.csv 

  --labels Train_labels.csv 

  --output train.jsonl 

  --add_missing_ids



可选参数:

--system-prompt-file 若提供，将使用外部文件中的 system prompt 替换默认内置版本。

--encoding 默认为 utf-8-sig (兼容可能的 BOM)。

--add_missing_ids 如果存在 labels 中的 id 在 reviews 中缺失，会跳过；

​         若提供此参数，会为缺失评论的 id 生成一条空 quadruples（不推荐一般训练）。

"""



import argparse

import csv

import json

import sys

from collections import defaultdict

from pathlib import Path



DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。

请严格遵守以下规范：

\1. 四元组定义

  \- aspect: 属性词（若评论中未出现显式属性词，用 "_" 标记）

  \- opinion: 观点情感词（必须与原文中的字面内容完全一致，不得增删或形态改写；如果无观点则不输出该条四元组）

  \- category: 必须从以下 13 个预定义类别中选择（逐字匹配）：

   {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

  \- polarity: 情感极性，取值仅限：{正面, 中性, 负面}

\2. 当一条评论包含多条独立观点时，输出多个四元组；不得合并。

\3. aspect = "_" 仅表示“隐式属性”而非未知；隐式属性仍需给出正确的 category。

\4. opinion 必须是评论原文中连续的片段，保持原始顺序与字符。

\5. 不要产生重复四元组；同一 (aspect, opinion, category, polarity) 只保留一份。

\6. 不要臆造评论中不存在的观点；如果确实没有可抽取的观点，返回空数组。

\7. 输出格式：严格输出一个 JSON 对象字符串：

  {"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

  \- quadruples 的值为一个列表

  \- 列表为空则输出 {"quadruples":[]}

  \- 键名必须是 aspect / opinion / category / polarity（全部小写）

  \- 字符串使用原始中文，不转义除非 JSON 需要

\8. 不添加任何额外解释、注释、换行提示或自然语言说明。仅输出 JSON。

\9. 保持输出 determinism：同一输入始终输出相同顺序的四元组。排序规则：按 opinion 在原文出现的起始位置升序；如 opinion 相同则按 aspect 起始位置；若 aspect 为 "_" 视其开始位置为 +∞。

\10. 若同一个 opinion 对不同 category 的确合理，可分别输出；但请避免不必要的多类别解释。"""



def read_reviews(path, encoding="utf-8-sig"):

  reviews = {}

  with open(path, "r", encoding=encoding, newline="") as f:

​    reader = csv.DictReader(f)

​    \# Expect columns: id, Reviews (case sensitive per sample)

​    for row in reader:

​      rid = row.get("id")

​      if rid is None:

​        continue

​      rid = rid.strip()

​      reviews[rid] = row.get("Reviews", "").strip()

  return reviews



def read_labels(path, encoding="utf-8-sig"):

  \# Return dict: id -> list of label dicts

  grouped = defaultdict(list)

  with open(path, "r", encoding=encoding, newline="") as f:

​    reader = csv.DictReader(f)

​    required = {"id","AspectTerms","OpinionTerms","Categories","Polarities"}

​    missing = required - set(reader.fieldnames or [])

​    if missing:

​      raise ValueError(f"Labels file missing columns: {missing}")



​    for row in reader:

​      rid = row["id"].strip()

​      aspect = row["AspectTerms"].strip()

​      opinion = row["OpinionTerms"].strip()

​      cat = row["Categories"].strip()

​      pol = row["Polarities"].strip()

​      \# positions (may be empty)

​      try:

​        o_start = int(row.get("O_start","").strip()) if row.get("O_start","").strip() != "" else None

​      except:

​        o_start = None

​      try:

​        a_start = int(row.get("A_start","").strip()) if row.get("A_start","").strip() != "" else None

​      except:

​        a_start = None



​      grouped[rid].append({

​        "aspect": aspect if aspect else "_",

​        "opinion": opinion if opinion else "",

​        "category": cat,

​        "polarity": pol,

​        "o_start": o_start,

​        "a_start": a_start

​      })

  return grouped



def normalize_and_sort(labels_for_one):

  \# Deduplicate exact quadruples

  seen = set()

  cleaned = []

  for item in labels_for_one:

​    aspect = item["aspect"] if item["aspect"] else "_"

​    opinion = item["opinion"]

​    category = item["category"]

​    polarity = item["polarity"]

​    if not opinion:

​      \# 没有 opinion 则不构造该条（任务要求 opinion 必须存在）

​      continue

​    key = (aspect, opinion, category, polarity)

​    if key in seen:

​      continue

​    seen.add(key)

​    cleaned.append({

​      "aspect": aspect,

​      "opinion": opinion,

​      "category": category,

​      "polarity": polarity,

​      "o_start": item["o_start"],

​      "a_start": item["a_start"]

​    })



  \# 排序

  def sort_key(x):

​    o_start = x["o_start"]

​    a_start = x["a_start"] if x["aspect"] != "_" else float("inf")

​    \# 缺失时放后

​    if o_start is None:

​      o_start = float("inf")

​    if a_start is None:

​      if x["aspect"] == "_":

​        a_start = float("inf")

​      else:

​        a_start = float("inf") - 1

​    return (o_start, a_start)



  cleaned.sort(key=sort_key)

  \# Drop position fields for final output

  final_list = [{

​    "aspect": c["aspect"],

​    "opinion": c["opinion"],

​    "category": c["category"],

​    "polarity": c["polarity"]

  } for c in cleaned]

  return final_list



def build_jsonl(reviews, labels_grouped, system_prompt, output_path, add_missing_ids=False, encoding="utf-8"):

  all_ids = list(reviews.keys())

  \# Optionally include ids appearing only in labels (not typical)

  if add_missing_ids:

​    more = [i for i in labels_grouped.keys() if i not in reviews]

​    if more:

​      print(f"[WARN] IDs in labels but not in reviews: {more}. Will add empty review placeholders.", file=sys.stderr)

​      for mid in more:

​        reviews[mid] = ""  # placeholder

​        all_ids.append(mid)



  \# Stable sort by numeric id if possible

  try:

​    all_ids.sort(key=lambda x: int(x))

  except:

​    all_ids.sort()



  with open(output_path, "w", encoding=encoding) as out_f:

​    for rid in all_ids:

​      review_text = reviews[rid]

​      label_items = labels_grouped.get(rid, [])

​      quadruples = normalize_and_sort(label_items) if label_items else []

​      assistant_obj = {"quadruples": quadruples}



​      user_content = f"请从下面这条评论中抽取所有 ACOS 观点四元组。\n评论ID: {rid}\n评论文本: {review_text}"



​      messages = [

​        {"role": "system", "content": system_prompt},

​        {"role": "user", "content": user_content},

​        {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False, separators=(',', ':'))}

​      ]

​      line_obj = {"messages": messages}

​      out_f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")



def main():

  parser = argparse.ArgumentParser(description="Convert ABSA labeled data to SiliconFlow fine-tune JSONL format.")

  parser.add_argument("--reviews", required=True, help="Path to Train_reviews.csv")

  parser.add_argument("--labels", required=True, help="Path to Train_labels.csv")

  parser.add_argument("--output", required=True, help="Output JSONL file path")

  parser.add_argument("--system-prompt-file", help="External system prompt file (override default)")

  parser.add_argument("--encoding", default="utf-8-sig", help="Input CSV encoding (default utf-8-sig)")

  parser.add_argument("--add_missing_ids", action="store_true", help="Add ids found only in labels")

  args = parser.parse_args()



  if args.system_prompt_file:

​    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

  else:

​    system_prompt = DEFAULT_SYSTEM_PROMPT



  reviews = read_reviews(args.reviews, encoding=args.encoding)

  labels_grouped = read_labels(args.labels, encoding=args.encoding)



  if not reviews:

​    print("No reviews loaded. Check reviews file.", file=sys.stderr)

​    sys.exit(1)



  build_jsonl(

​    reviews=reviews,

​    labels_grouped=labels_grouped,

​    system_prompt=system_prompt,

​    output_path=args.output,

​    add_missing_ids=args.add_missing_ids,

​    encoding="utf-8"

  )

  print(f"Done. Wrote JSONL to {args.output}")



if __name__ == "__main__":

  main()
```



### 2.2 阿里云notebook的Qwen3 4B × LLaMA Factory参数设置

直接在LLaMA Factory的webUI选择设置。

```
top.booster: auto

top.checkpoint_path: []

top.finetuning_type: lora

top.model_name: Qwen3-4B-Instruct-2507

top.quantization_bit: none

top.quantization_method: bnb

top.rope_scaling: none

top.template: qwen3_nothink

train.additional_target: ''

train.apollo_rank: 16

train.apollo_scale: 32

train.apollo_target: all

train.apollo_update_interval: 200

train.badam_mode: layer

train.badam_switch_interval: 50

train.badam_switch_mode: ascending

train.badam_update_ratio: 0.05

train.batch_size: 4

train.compute_type: bf16

train.create_new_adapter: false

train.cutoff_len: 2048

train.dataset:

\- tianchi6_train

\- tianchi6_eval

train.dataset_dir: /mnt/workspace/demos/qwen3_4b_llama_factory/LLaMA-Factory/data/tianchi6

train.ds_offload: false

train.ds_stage: none

train.enable_thinking: true

train.extra_args: "{\n  \"optim\": \"adamw_torch\",\n  \"lr_scheduler_kwargs\": {\n\

 \   \"min_lr_rate\": 0.05 \n  }\n}"

train.freeze_extra_modules: ''

train.freeze_language_model: false

train.freeze_multi_modal_projector: true

train.freeze_trainable_layers: 2

train.freeze_trainable_modules: all

train.freeze_vision_tower: true

train.galore_rank: 16

train.galore_scale: 2

train.galore_target: all

train.galore_update_interval: 200

train.gradient_accumulation_steps: 8

train.image_max_pixels: 768*768

train.image_min_pixels: 32*32

train.learning_rate: 1e-4

train.logging_steps: 5

train.lora_alpha: 32

train.lora_dropout: 0.05

train.lora_rank: 16

train.lora_target: ''

train.loraplus_lr_ratio: 0

train.lr_scheduler_type: cosine_warmup_with_min_lr

train.mask_history: false

train.max_grad_norm: '1.0'

train.max_samples: '100000'

train.neat_packing: false

train.neftune_alpha: 0

train.num_train_epochs: '3.0'

train.packing: false

train.ppo_score_norm: false

train.ppo_whiten_rewards: false

train.pref_beta: 0.1

train.pref_ftx: 0

train.pref_loss: sigmoid

train.report_to: none

train.resize_vocab: false

train.reward_model: []

train.save_steps: 50

train.swanlab_api_key: ''

train.swanlab_link: null

train.swanlab_mode: cloud

train.swanlab_project: llamafactory

train.swanlab_run_name: ''

train.swanlab_workspace: ''

train.train_on_prompt: false

train.training_stage: Supervised Fine-Tuning

train.use_apollo: false

train.use_badam: false

train.use_dora: false

train.use_galore: false

train.use_llama_pro: false

train.use_pissa: false

train.use_rslora: false

train.use_swanlab: false

train.val_size: 0

train.video_max_pixels: 256*256

train.video_min_pixels: 16*16

train.warmup_steps: 25
```

### 2.3 LoRA微调Qwen/Qwen2.5-7B-Instruct参数设置

（1）Learning Rate（学习率）

配置值：0.0001，取值范围：(0, 0.1]。

（2）Number of Epochs（训练轮次）

配置值：3，取值范围：\[1, 10]。

（3）Batch Size（批次大小）

配置值：8。

（4）LoRA 相关参数

*   **LoRA Rank**：配置值 8。LoRA 的秩，秩的大小影响低秩矩阵的表达能力，秩越大，可学习的参数表达能力越强，但训练成本也会相应增加；设置为 8，在参数效率与表达能力间取得较好平衡。

*   **LoRA Alpha**：配置值 32。LoRA 的缩放因子，与 LoRA Rank 共同作用，影响 LoRA 模块对模型整体输出的贡献程度，助力模型更高效地学习任务相关知识。

*   **LoRA Dropout**：配置值 0.05，取值范围：\[0, 1.0)。Dropout 是一种正则化技术，用于防止过拟合。LoRA Dropout 设置为 0.05，可随机丢弃部分 LoRA 层的连接，增强模型的泛化能力。

（5）Max Tokens（最大 tokens 数）

配置值：32768，取值范围：(0, 32768]。

最大 tokens 数决定了模型在一次输入中能处理的文本长度上限。设置为 32768，能让模型处理较长篇幅的文本数据，满足训练数据中文本内容长度的需求，确保长文本信息能被模型有效学习。

### 2.3 LoRA微调Qwen/Qwen2.5-32B-Instruct参数设置

（1）Learning Rate（学习率）

配置值：0.0002，取值范围：(0, 0.1]。

（2）Number of Epochs（训练轮次）

配置值：3，取值范围：\[1, 10]。

（3）Batch Size（批次大小）

配置值：16。

（4）LoRA 相关参数

*   **LoRA Rank**：配置值 16。LoRA 的秩，秩的大小影响低秩矩阵的表达能力，秩越大，可学习的参数表达能力越强，但训练成本也会相应增加；设置为 16，在参数效率与表达能力间取得较好平衡。

*   **LoRA Alpha**：配置值 32。LoRA 的缩放因子，与 LoRA Rank 共同作用，影响 LoRA 模块对模型整体输出的贡献程度，助力模型更高效地学习任务相关知识。

*   **LoRA Dropout**：配置值 0.05，取值范围：\[0, 1.0)。Dropout 是一种正则化技术，用于防止过拟合。LoRA Dropout 设置为 0.05，可随机丢弃部分 LoRA 层的连接，增强模型的泛化能力。

（5）Max Tokens（最大 tokens 数）

配置值：32768，取值范围：(0, 32768]。

最大 tokens 数决定了模型在一次输入中能处理的文本长度上限。设置为 32768，能让模型处理较长篇幅的文本数据，满足训练数据中文本内容长度的需求，确保长文本信息能被模型有效学习。

### 2.4 预测脚本

需要使用结构化输出。

```python
\#!/usr/bin/env python

\# -*- coding: utf-8 -*-

"""

预测



步骤：

\1. 修改下方 CONFIG 部分的路径 / 模型名 / API Key 等

\2. python predict_acos_in_code.py

\3. 生成：

  \- Result.csv  (提交文件，无表头：id,AspectTerm,OpinionTerm,Category,Polarity)

  \- raw_predictions.jsonl  (结构化解析后的结果备份)



"""



import os

import json

import time

import re

import requests

from typing import List, Tuple, Set

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError, constr

from typing import Literal



\# ======================== CONFIG（直接修改即可） ========================



TEST_CSV_PATH   = "Test_reviews.csv"  # 测试集 CSV 路径（含列：id,Reviews）

OUTPUT_CSV_PATH  = "Result.csv"     # 预测结果输出路径（提交格式）

RAW_JSONL_PATH   = "raw_predictions.jsonl"  # 保存结构化结果（便于排查）

MODEL_NAME     = "LoRA/Qwen/Qwen2.5-7B-Instruct"

API_KEY      = ""  # 尽量用环境变量

USE_RESPONSE_FORMAT = False  # 如果将来 API 支持严格 JSON，可改 True

TEMPERATURE    = 0.0

MAX_TOKENS     = 512

MAX_RETRIES    = 1      # 每条样本解析失败重试次数

SLEEP_BETWEEN   = 0.6     # 每次调用间隔秒数（防限流）

BATCH_SIZE     = 1      # 当前实现是串行，如要并发需自行改造



\# =====================================================================



API_URL = "https://api.siliconflow.cn/v1/chat/completions"



CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]

POLARITIES = ["正面","中性","负面"]



DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。

严格遵守：

\1. 输出唯一 JSON：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

\2. quadruples 为数组；若没有观点，输出 {"quadruples":[]}

\3. aspect：若无显式属性词用 "_"；原文中出现的需与原文一致；不添加空格。

\4. opinion：必须是原文中连续片段；保持原字符；无观点不输出该条。

\5. category 取值必须在 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

\6. polarity 取值必须在 {正面, 中性, 负面}

\7. 不得出现与原文无关或臆造的词；不得输出解释文字；不得添加多余字段。

\8. 去重：同一 (aspect, opinion, category, polarity) 只保留一个。

\9. 排序：按 opinion 在原文首次出现位置升序；若相同再按 aspect（"_" 视为 +∞）。

\10. 只输出 JSON，不输出其它任何文本。

"""



class Quadruple(BaseModel):

  aspect: constr(strip_whitespace=True)

  opinion: constr(strip_whitespace=True, min_length=1)

  category: Literal["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]

  polarity: Literal["正面","中性","负面"]



class QuadrupleList(BaseModel):

  quadruples: List[Quadruple]



@dataclass

class Prediction:

  review_id: str

  quadruples: List[Quadruple]





def read_test_csv(path: str) -> List[Tuple[str,str]]:

  import csv

  result = []

  with open(path, "r", encoding="utf-8-sig", newline="") as f:

​    reader = csv.DictReader(f)

​    for row in reader:

​      rid = row.get("id")

​      txt = row.get("Reviews","")

​      if rid is None:

​        continue

​      result.append((str(rid).strip(), txt.strip()))

  return result





def call_model(model: str,

​        system_prompt: str,

​        review_id: str,

​        review_text: str,

​        api_key: str,

​        use_response_format: bool=False,

​        temperature: float=0.0,

​        max_tokens: int=512) -> str:

  """

  调用 SiliconFlow Chat API，返回 assistant 内容。

  """

  user_content = f"请抽取 ACOS 四元组。\n评论ID: {review_id}\n评论文本: {review_text}\n严格只输出一个 JSON 对象。"

  messages = [

​    {"role": "system", "content": system_prompt},

​    {"role": "user", "content": user_content}

  ]

  payload = {

​    "model": model,

​    "messages": messages,

​    "temperature": temperature,

​    "max_tokens": max_tokens

  }

  if use_response_format:

​    payload["response_format"] = {"type": "json_object"}



  headers = {

​    "Authorization": f"Bearer {api_key}",

​    "Content-Type": "application/json"

  }

  resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)

  if resp.status_code != 200:

​    raise RuntimeError(f"API错误 status={resp.status_code} body={resp.text}")

  data = resp.json()

  try:

​    content = data["choices"][0]["message"]["content"]

  except Exception:

​    raise RuntimeError(f"未找到模型输出字段：{data}")

  return content





def extract_json_block(text: str) -> str:

  """

  若模型额外输出杂项文本，尝试抽取第一个 { ... }。

  """

  text = text.strip()

  if text.startswith("{") and text.endswith("}"):

​    return text

  m = re.search(r'\{.*\}', text, flags=re.DOTALL)

  if m:

​    return m.group(0)

  return text





def validate_and_repair(json_str: str, original_text: str) -> List[Quadruple]:

  try:

​    obj = json.loads(json_str)

  except Exception:

​    return []

  if not isinstance(obj, dict):

​    return []

  items = obj.get("quadruples", [])

  if not isinstance(items, list):

​    return []



  cleaned = []

  seen: Set[Tuple[str,str,str,str]] = set()

  for it in items:

​    if not isinstance(it, dict):

​      continue

​    aspect = (it.get("aspect") or "").strip()

​    if aspect == "":

​      aspect = "_"

​    opinion = (it.get("opinion") or "").strip()

​    category = (it.get("category") or "").strip()

​    polarity = (it.get("polarity") or "").strip()

​    if not opinion:

​      continue

​    if category not in CATEGORIES:

​      continue

​    if polarity not in POLARITIES:

​      continue

​    key = (aspect, opinion, category, polarity)

​    if key in seen:

​      continue

​    seen.add(key)

​    cleaned.append({

​      "aspect": aspect,

​      "opinion": opinion,

​      "category": category,

​      "polarity": polarity

​    })



  def first_pos(opinion: str) -> int:

​    idx = original_text.find(opinion)

​    return idx if idx >= 0 else 10**9



  def aspect_pos(aspect: str) -> int:

​    if aspect == "_":

​      return 10**9

​    idx = original_text.find(aspect)

​    return idx if idx >= 0 else 10**9 - 1



  cleaned.sort(key=lambda x: (first_pos(x["opinion"]), aspect_pos(x["aspect"])))



  try:

​    qlist = QuadrupleList(quadruples=[Quadruple(**c) for c in cleaned])

​    return qlist.quadruples

  except ValidationError:

​    return []





def process_one(review_id: str,

​        review_text: str,

​        model: str,

​        api_key: str,

​        system_prompt: str,

​        max_retries: int,

​        use_response_format: bool=False,

​        sleep: float=0.6) -> Prediction:

  last_error = None

  for attempt in range(max_retries + 1):

​    try:

​      raw = call_model(

​        model=model,

​        system_prompt=system_prompt,

​        review_id=review_id,

​        review_text=review_text,

​        api_key=api_key,

​        use_response_format=use_response_format,

​        temperature=TEMPERATURE,

​        max_tokens=MAX_TOKENS

​      )

​      json_candidate = extract_json_block(raw)

​      quadruples = validate_and_repair(json_candidate, review_text)

​      return Prediction(review_id=review_id, quadruples=quadruples)

​    except Exception as e:

​      last_error = e

​      time.sleep(sleep)

  print(f"[WARN] review_id={review_id} 解析失败: {last_error}")

  return Prediction(review_id=review_id, quadruples=[])





def write_result_csv(predictions: List[Prediction], path: str):

  lines = []

  for pred in predictions:

​    if not pred.quadruples:

​      lines.append(f"{pred.review_id},_,_,_,_")

​      continue

​    for q in pred.quadruples:

​      lines.append(f"{pred.review_id},{q.aspect},{q.opinion},{q.category},{q.polarity}")

  with open(path, "w", encoding="utf-8") as f:

​    f.write("\n".join(lines))





def run():

  if not API_KEY or API_KEY.startswith("REPLACE_WITH_YOUR_NEW_KEY"):

​    raise ValueError("请在脚本顶部配置 API_KEY（或设置环境变量 SILICONFLOW_API_KEY）后再运行。")



  tests = read_test_csv(TEST_CSV_PATH)

  print(f"加载测试样本数: {len(tests)}")



  \# # ====== 只测试前 10 条 ======

  \# tests = tests[:20]

  \# print(f"仅测试前 {len(tests)} 条样本")



  predictions: List[Prediction] = []

  with open(RAW_JSONL_PATH, "w", encoding="utf-8") as raw_f:

​    for idx, (rid, text) in enumerate(tests, 1):

​      pred = process_one(

​        review_id=rid,

​        review_text=text,

​        model=MODEL_NAME,

​        api_key=API_KEY,

​        system_prompt=DEFAULT_SYSTEM_PROMPT,

​        max_retries=MAX_RETRIES,

​        use_response_format=USE_RESPONSE_FORMAT,

​        sleep=SLEEP_BETWEEN

​      )

​      predictions.append(pred)

​      raw_f.write(json.dumps({

​        "id": rid,

​        "review": text,

​        "quadruples": [q.model_dump() for q in pred.quadruples]

​      }, ensure_ascii=False) + "\n")

​      print(f"[{idx}/{len(tests)}] id={rid} 预测 {len(pred.quadruples)} 条")



  write_result_csv(predictions, OUTPUT_CSV_PATH)

  print(f"完成：提交文件 -> {OUTPUT_CSV_PATH}")

  print(f"结构化备份 -> {RAW_JSONL_PATH}")





if __name__ == "__main__":

  run()
```



