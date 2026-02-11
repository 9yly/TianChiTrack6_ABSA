#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-CoT 3模型×2路径抽取 + 多策略聚合

改进版：使用3个LoRA checkpoint模型，每个模型走2条不同的CoT路径。
这样既有模型多样性，又有路径多样性，投票更有意义。

分配策略：
- 模型1(最终版) → A→O→C + O→A→C
- 模型2(step364) → C→A→O + A→C→O
- 模型3(step182) → O→C→A + C→O→A

总调用次数：6次（与原方案相同，但获得了模型多样性）

策略：
1. 3模型×2路径并行调用
2. 投票聚合：统计四元组出现次数，>=3次为高置信度
3. 聚类聚合：对候选四元组按(aspect, opinion)聚类，每簇取最常见的
4. Judge仲裁：对冲突和低置信度四元组调用Thinking模型进行验证

步骤：
1. 修改下方 CONFIG 部分的路径 / 模型名 / API Key 等
2. python predict_multi_cot.py
3. 生成：
   - Result.csv  (提交文件)
   - raw_predictions_multi_cot.jsonl  (结构化结果备份)
"""

import os
import json
import time
import re
import requests
import csv
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError, validator
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter, defaultdict
from bm25_retriever import BM25Retriever, RerankerClient, retrieve_similar_examples

# ======================== CONFIG ========================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 数据路径配置
TRAIN_CSV_PATH    = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_reviews.csv"
TRAIN_CSV_LABELS  = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_labels.csv"
TEST_CSV_PATH     = "D:/python-ECI/competitions/tianchi/data/TEST/Test_reviews.csv"
OUTPUT_CSV_PATH   = "D:/python-ECI/competitions/tianchi/MethodsV2/Result_multi_cot.csv"
RAW_JSONL_PATH    = "D:/python-ECI/competitions/tianchi/MethodsV2/raw_predictions_multi_cot.jsonl"

# 3个LoRA checkpoint模型
EXTRACTOR_MODELS = [
    "ft:LoRA/Qwen/Qwen2.5-32B-Instruct:clzc5301v000e12rx905dx6mo:qwen-32b:buzvweycucvunqozahym",           # 最终版
    "ft:LoRA/Qwen/Qwen2.5-32B-Instruct:clzc5301v000e12rx905dx6mo:qwen-32b:buzvweycucvunqozahym-ckpt_step_364",  # step364
    "ft:LoRA/Qwen/Qwen2.5-32B-Instruct:clzc5301v000e12rx905dx6mo:qwen-32b:buzvweycucvunqozahym-ckpt_step_182",  # step182
]
EXTRACTOR_API_KEY = "sk-dxfzwvjzdjtcxruqduruesrxpjljxgccxodrloqnfbgtwvak"

# Judge模型配置（Thinking模型）
JUDGE_MODEL       = "Qwen/Qwen3-30B-A3B-Thinking-2507"
JUDGE_API_KEY     = "sk-dxfzwvjzdjtcxruqduruesrxpjljxgccxodrloqnfbgtwvak"

# Embedding模型配置（用于聚类）
EMBEDDING_MODEL   = "Pro/BAAI/bge-m3"
EMBEDDING_API_KEY = "sk-znalsgiyebbitavkonhjlhmrzwufhgzndfhmhrgxwwpqncof"

# 运行参数
TEMPERATURE       = 0.5
MAX_TOKENS        = 4096
MAX_RETRIES       = 3
SLEEP_BETWEEN     = 0           # 样本间不需要等待
MAX_WORKERS       = 6          # 并发线程数（6路径并行）
VOTE_THRESHOLD    = 3          # 投票阈值，>=3次为高置信度
ENABLE_JUDGE      = True       # 是否启用Judge仲裁
ENABLE_CLUSTERING = True       # 是否启用聚类聚合
USE_BM25_EXAMPLES = True       # 是否使用BM25检索示例
ENABLE_CHECKPOINT = True       # 是否启用断点续传
CHECKPOINT_INTERVAL = 5        # 每处理N条保存一次检查点
JUDGE_MAX_RETRIES = 10         # Judge调用最大重试次数
JUDGE_RETRY_DELAY = 5.0        # Judge调用重试间隔（增加到5秒）
JUDGE_TIMEOUT = 180            # Judge调用超时时间（Thinking模型需要更长）

# =====================================================================

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"

CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITIES = ["正面","中性","负面"]

# =====================================================================
# 六种CoT路径的System Prompt
# A=Aspect, O=Opinion, C=Category
# =====================================================================

# 路径1: A→O→C (先找Aspect，再找Opinion，最后确定Category)
PROMPT_A_O_C = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略（A→O→C思维链）
请按以下步骤分析：

**Step 1: 识别Aspect词**
- 扫描评论，找出所有产品特征/属性词
- 显式Aspect示例：物流、价格、包装、味道、活动、赠品、速度、服务等
- 注意：约80%的评论没有显式Aspect，此时Aspect设为"_"

**Step 2: 为每个Aspect找Opinion词**
- 在Aspect上下文中找修饰它的情感评价词
- Opinion示例：不错、很好、好用、快、喜欢、满意、差、慢等

**Step 3: 确定Category和Polarity**
- Category取值：{包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
- Polarity取值：{正面, 中性, 负面}

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# 路径2: A→C→O (先找Aspect，确定Category，再找Opinion)
PROMPT_A_C_O = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略（A→C→O思维链）
请按以下步骤分析：

**Step 1: 识别Aspect词**
- 找出所有产品特征/属性词，无显式词则设为"_"

**Step 2: 确定每个Aspect的Category**
- 根据Aspect语义判断所属类别
- Category取值：{包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

**Step 3: 找Opinion词并确定Polarity**
- 找与该Aspect-Category相关的情感评价词
- Polarity取值：{正面, 中性, 负面}

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# 路径3: O→A→C (先找Opinion，回溯Aspect，再确定Category)
PROMPT_O_A_C = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略（O→A→C思维链）
请按以下步骤分析：

**Step 1: 识别所有Opinion词**
- 扫描评论，找出所有表达主观情感/评价的词语
- Opinion示例：不错、很好、好用、快、喜欢、满意、特别快、很好闻、太随便了等

**Step 2: 为每个Opinion回溯找Aspect**
- 在Opinion上下文中找它所修饰的属性词
- 若无明确修饰对象，Aspect设为"_"

**Step 3: 确定Category和Polarity**
- Category取值：{包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
- Polarity取值：{正面, 中性, 负面}

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# 路径4: O→C→A (先找Opinion，确定Category，再回溯Aspect)
PROMPT_O_C_A = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略（O→C→A思维链）
请按以下步骤分析：

**Step 1: 识别所有Opinion词**
- 找出所有情感评价词，并判断其Polarity

**Step 2: 确定每个Opinion的Category**
- 根据Opinion的语义上下文判断它描述的是哪个类别
- Category取值：{包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

**Step 3: 回溯找Aspect**
- 根据Category和Opinion，回溯找对应的属性词
- 若无显式属性词，设为"_"

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# 路径5: C→A→O (先假设Category，找Aspect，再找Opinion)
PROMPT_C_A_O = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略（C→A→O思维链）
请按以下步骤分析：

**Step 1: 扫描可能的Category**
- 逐一检查评论是否涉及以下类别：
  {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

**Step 2: 为每个涉及的Category找Aspect**
- 找该类别对应的属性词
- 若无显式属性词，设为"_"

**Step 3: 找Opinion词并确定Polarity**
- 找与该Category-Aspect相关的情感词
- Polarity取值：{正面, 中性, 负面}

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# 路径6: C→O→A (先假设Category，找Opinion，再回溯Aspect)
PROMPT_C_O_A = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略（C→O→A思维链）
请按以下步骤分析：

**Step 1: 扫描可能的Category**
- 逐一检查评论是否涉及以下类别：
  {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

**Step 2: 为每个涉及的Category找Opinion词**
- 找与该类别相关的情感评价词
- 确定Polarity：{正面, 中性, 负面}

**Step 3: 回溯找Aspect**
- 根据Category和Opinion回溯对应的属性词
- 若无显式属性词，设为"_"

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# 3模型×2路径配置
# 每个模型走2条不同的CoT路径
COT_PATHS = [
    # 模型1(最终版) → 2条路径
    {"name": "M1-A→O→C", "prompt": PROMPT_A_O_C, "model_idx": 0},
    {"name": "M1-O→A→C", "prompt": PROMPT_O_A_C, "model_idx": 0},
    # 模型2(step364) → 2条路径
    {"name": "M2-C→A→O", "prompt": PROMPT_C_A_O, "model_idx": 1},
    {"name": "M2-A→C→O", "prompt": PROMPT_A_C_O, "model_idx": 1},
    # 模型3(step182) → 2条路径
    {"name": "M3-O→C→A", "prompt": PROMPT_O_C_A, "model_idx": 2},
    {"name": "M3-C→O→A", "prompt": PROMPT_C_O_A, "model_idx": 2},
]

# Judge Prompt
JUDGE_SYSTEM_PROMPT = """你是一个专业的ACOS四元组验证专家。

## 任务
给定一条评论和多条路径抽取的候选四元组，验证并选择正确的结果。

## 验证规则
1. **Aspect验证**：若不为"_"，必须是原文连续子串
2. **Opinion验证**：必须是原文连续子串，且确实表达情感
3. **Category验证**：必须属于{包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
4. **Polarity验证**：必须准确反映Opinion的情感倾向

## 输入说明
你会收到：
- 原始评论文本
- 各路径的抽取结果和出现次数
- 需要验证的候选四元组

## 输出格式
严格按JSON输出：{"verified_quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

只输出验证通过的四元组，不需要解释。
"""

# =====================================================================
# 数据模型
# =====================================================================

class Quadruple(BaseModel):
    aspect: str
    opinion: str
    category: str
    polarity: str
    
    @validator('aspect')
    def validate_aspect(cls, v):
        if v is None:
            return "_"
        return str(v).strip()
    
    @validator('opinion')
    def validate_opinion(cls, v):
        if not v or not str(v).strip():
            raise ValueError('opinion cannot be empty')
        return str(v).strip()
    
    @validator('category')
    def validate_category(cls, v):
        if v not in CATEGORIES:
            raise ValueError(f'category must be one of {CATEGORIES}')
        return v
    
    @validator('polarity')
    def validate_polarity(cls, v):
        if v not in POLARITIES:
            raise ValueError(f'polarity must be one of {POLARITIES}')
        return v
    
    def to_tuple(self) -> Tuple[str, str, str, str]:
        return (self.aspect, self.opinion, self.category, self.polarity)
    
    def to_dict(self) -> Dict:
        return {"aspect": self.aspect, "opinion": self.opinion, 
                "category": self.category, "polarity": self.polarity}


@dataclass
class PathResult:
    """单条路径的抽取结果"""
    path_name: str
    quadruples: List[Quadruple]
    raw_output: str = ""


@dataclass
class AggregatedResult:
    """聚合后的结果"""
    high_confidence: List[Quadruple]  # 高置信度（投票>=阈值）
    low_confidence: List[Quadruple]   # 低置信度（需要Judge验证）
    conflicts: List[Dict]             # 冲突项（同一opinion不同category/polarity）


@dataclass 
class Prediction:
    review_id: str
    quadruples: List[Quadruple]
    path_results: List[PathResult] = field(default_factory=list)


# =====================================================================
# 工具函数
# =====================================================================

def read_test_csv(path: str) -> List[Tuple[str, str]]:
    """读取测试集"""
    result = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            txt = row.get("Reviews", "")
            if rid is None:
                continue
            result.append((str(rid).strip(), txt.strip()))
    return result


def call_model(model: str,
               system_prompt: str,
               user_content: str,
               api_key: str,
               api_url: str = API_URL,
               temperature: float = 0.0,
               max_tokens: int = 4096,
               max_retries: int = 3,
               retry_delay: float = 2.0,
               timeout: int = 120) -> str:
    """调用模型API
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 重试间隔秒数
        timeout: 请求超时时间（秒）
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    return content
                except Exception:
                    raise RuntimeError(f"未找到模型输出字段：{data}")
            elif resp.status_code == 429:  # 限流
                last_error = f"API限流 status=429"
                time.sleep(retry_delay * (attempt + 1))  # 递增延迟
                continue
            else:
                raise RuntimeError(f"API错误 status={resp.status_code} body={resp.text}")
        except requests.exceptions.Timeout:
            last_error = "请求超时"
            time.sleep(retry_delay)
            continue
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise
    
    raise RuntimeError(f"API调用失败，已重试{max_retries}次: {last_error}")


def get_embeddings(texts: List[str], api_key: str = EMBEDDING_API_KEY) -> List[List[float]]:
    """获取文本的embedding向量"""
    if not texts:
        return []
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "encoding_format": "float"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            print(f"Embedding API错误: {resp.status_code}")
            return []
        
        data = resp.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return embeddings
    except Exception as e:
        print(f"Embedding调用失败: {e}")
        return []


def extract_json_block(text: str) -> str:
    """提取JSON块"""
    text = text.strip()
    # 处理Thinking模型的<think>标签
    if "<think>" in text:
        # 移除think标签内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
    
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text


def validate_and_repair(json_str: str, original_text: str) -> List[Quadruple]:
    """验证并修复JSON输出"""
    try:
        obj = json.loads(json_str)
    except Exception:
        return []
    
    if not isinstance(obj, dict):
        return []
    
    # 尝试多种可能的key
    items = obj.get("quadruples", []) or obj.get("verified_quadruples", [])
    if not isinstance(items, list):
        return []

    cleaned = []
    seen: Set[Tuple[str, str, str, str]] = set()
    
    for it in items:
        if not isinstance(it, dict):
            continue
        
        aspect = (it.get("aspect") or "").strip()
        if aspect == "":
            aspect = "_"
        opinion = (it.get("opinion") or "").strip()
        category = (it.get("category") or "").strip()
        polarity = (it.get("polarity") or "").strip()
        
        if not opinion:
            continue
        if category not in CATEGORIES:
            continue
        if polarity not in POLARITIES:
            continue
        
        key = (aspect, opinion, category, polarity)
        if key in seen:
            continue
        seen.add(key)
        
        try:
            quad = Quadruple(
                aspect=aspect,
                opinion=opinion,
                category=category,
                polarity=polarity
            )
            cleaned.append(quad)
        except ValidationError:
            continue

    # 按opinion在原文位置排序
    def first_pos(opinion: str) -> int:
        idx = original_text.find(opinion)
        return idx if idx >= 0 else 10**9

    def aspect_pos(aspect: str) -> int:
        if aspect == "_":
            return 10**9
        idx = original_text.find(aspect)
        return idx if idx >= 0 else 10**9 - 1

    cleaned.sort(key=lambda x: (first_pos(x.opinion), aspect_pos(x.aspect)))
    
    return cleaned


# =====================================================================
# 核心处理逻辑
# =====================================================================

def call_single_path(path_config: Dict,
                     review_id: str,
                     review_text: str,
                     similar_examples: str = "") -> PathResult:
    """调用单条CoT路径（使用对应的checkpoint模型）"""
    path_name = path_config["name"]
    system_prompt = path_config["prompt"]
    model_idx = path_config.get("model_idx", 0)  # 获取模型索引
    model = EXTRACTOR_MODELS[model_idx]  # 选择对应的模型
    
    # 构建用户内容
    if similar_examples:
        user_content = f"""**参考示例**：
{similar_examples}

**待完成任务**：
评论ID: {review_id}
评论文本: {review_text}

严格只输出一个JSON对象。"""
    else:
        user_content = f"""**待完成任务**：
评论ID: {review_id}
评论文本: {review_text}

严格只输出一个JSON对象。"""
    
    try:
        raw_output = call_model(
            model=model,  # 使用对应的checkpoint模型
            system_prompt=system_prompt,
            user_content=user_content,
            api_key=EXTRACTOR_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        json_str = extract_json_block(raw_output)
        quadruples = validate_and_repair(json_str, review_text)
        
        return PathResult(
            path_name=path_name,
            quadruples=quadruples,
            raw_output=raw_output
        )
    except Exception as e:
        print(f"[WARN] 路径 {path_name} 调用失败: {e}")
        return PathResult(path_name=path_name, quadruples=[], raw_output=str(e))


def aggregate_by_voting(path_results: List[PathResult], 
                        threshold: int = VOTE_THRESHOLD) -> AggregatedResult:
    """投票聚合策略"""
    # 统计每个四元组的出现次数
    quad_counter = Counter()
    quad_objects = {}  # 存储Quadruple对象
    
    for result in path_results:
        for quad in result.quadruples:
            key = quad.to_tuple()
            quad_counter[key] += 1
            quad_objects[key] = quad
    
    high_confidence = []
    low_confidence = []
    
    for key, count in quad_counter.items():
        quad = quad_objects[key]
        if count >= threshold:
            high_confidence.append(quad)
        else:
            low_confidence.append(quad)
    
    # 检测冲突（同一opinion但不同category或polarity）
    opinion_groups = defaultdict(list)
    for key, quad in quad_objects.items():
        opinion_groups[quad.opinion].append((key, quad_counter[key], quad))
    
    conflicts = []
    for opinion, items in opinion_groups.items():
        if len(items) > 1:
            # 检查是否有category或polarity冲突
            categories = set(q.category for _, _, q in items)
            polarities = set(q.polarity for _, _, q in items)
            if len(categories) > 1 or len(polarities) > 1:
                conflicts.append({
                    "opinion": opinion,
                    "candidates": [(k, c, q.to_dict()) for k, c, q in items]
                })
    
    return AggregatedResult(
        high_confidence=high_confidence,
        low_confidence=low_confidence,
        conflicts=conflicts
    )


def cluster_quadruples(quadruples: List[Quadruple], n_clusters: int = None) -> List[Quadruple]:
    """
    对四元组进行聚类，每簇取出现次数最多的
    使用(aspect, opinion)的embedding进行聚类
    """
    if not quadruples or not ENABLE_CLUSTERING:
        return quadruples
    
    if len(quadruples) <= 3:
        return quadruples
    
    # 构建聚类文本
    texts = [f"{q.aspect} {q.opinion}" for q in quadruples]
    
    try:
        embeddings = get_embeddings(texts)
        if not embeddings or len(embeddings) != len(quadruples):
            return quadruples
        
        # 简单的基于余弦相似度的聚类
        embeddings_np = np.array(embeddings)
        
        # 归一化
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_np = embeddings_np / norms
        
        # 计算相似度矩阵
        sim_matrix = np.dot(embeddings_np, embeddings_np.T)
        
        # 简单的合并策略：相似度>0.9的视为同一簇
        merged = [False] * len(quadruples)
        clusters = []
        
        for i in range(len(quadruples)):
            if merged[i]:
                continue
            cluster = [i]
            merged[i] = True
            for j in range(i + 1, len(quadruples)):
                if not merged[j] and sim_matrix[i][j] > 0.9:
                    cluster.append(j)
                    merged[j] = True
            clusters.append(cluster)
        
        # 每簇取第一个（保持原有顺序）
        result = []
        for cluster in clusters:
            result.append(quadruples[cluster[0]])
        
        return result
    
    except Exception as e:
        print(f"[WARN] 聚类失败: {e}")
        return quadruples


def call_judge(review_text: str,
               candidates: List[Quadruple],
               conflicts: List[Dict],
               path_results: List[PathResult]) -> List[Quadruple]:
    """调用Judge模型进行验证"""
    if not candidates and not conflicts:
        return []
    
    # 构建候选信息
    candidate_info = []
    for quad in candidates:
        candidate_info.append(quad.to_dict())
    
    # 构建路径结果摘要
    path_summary = []
    for result in path_results:
        quads_str = json.dumps([q.to_dict() for q in result.quadruples], ensure_ascii=False)
        path_summary.append(f"路径{result.path_name}: {quads_str}")
    
    # 构建用户内容
    user_content = f"""## 原始评论
{review_text}

## 各路径抽取结果
{chr(10).join(path_summary)}

## 需要验证的候选四元组
{json.dumps(candidate_info, ensure_ascii=False, indent=2)}

## 冲突信息
{json.dumps(conflicts, ensure_ascii=False, indent=2) if conflicts else "无"}

请验证以上候选四元组，输出验证通过的结果。严格按JSON格式输出。
"""
    
    # Judge模型调用，重试直到成功
    last_error = None
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            raw_output = call_model(
                model=JUDGE_MODEL,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_content=user_content,
                api_key=JUDGE_API_KEY,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                max_retries=2,  # 内部重试2次
                retry_delay=3.0,
                timeout=JUDGE_TIMEOUT  # Thinking模型需要更长超时
            )
            
            json_str = extract_json_block(raw_output)
            verified = validate_and_repair(json_str, review_text)
            return verified
            
        except Exception as e:
            last_error = e
            if attempt < JUDGE_MAX_RETRIES - 1:
                print(f"[WARN] Judge调用失败(第{attempt+1}次): {e}，{JUDGE_RETRY_DELAY}秒后重试...")
                time.sleep(JUDGE_RETRY_DELAY)
            else:
                print(f"[WARN] Judge调用失败(已重试{JUDGE_MAX_RETRIES}次): {e}")
    
    # 所有重试都失败后，返回出现次数>=2的低置信度四元组
    print(f"[WARN] Judge最终失败，使用fallback策略")
    quad_counter = Counter()
    for result in path_results:
        for quad in result.quadruples:
            quad_counter[quad.to_tuple()] += 1
    
    fallback = []
    seen = set()
    for quad in candidates:
        key = quad.to_tuple()
        if key not in seen and quad_counter[key] >= 2:
            fallback.append(quad)
            seen.add(key)
    return fallback


def process_one_multi_cot(review_id: str,
                          review_text: str,
                          bm25_retriever: Optional[BM25Retriever] = None,
                          reranker_client: Optional[RerankerClient] = None) -> Prediction:
    """
    Multi-CoT处理单个样本
    
    流程：
    1. 检索相似示例（可选）
    2. 六路径并行抽取
    3. 投票聚合
    4. 聚类去重（可选）
    5. Judge仲裁（可选）
    """
    # Step 1: 检索相似示例
    similar_examples = ""
    if USE_BM25_EXAMPLES and bm25_retriever and reranker_client:
        try:
            similar_examples = retrieve_similar_examples(
                query_text=review_text,
                bm25_retriever=bm25_retriever,
                reranker_client=reranker_client,
                top_k_bm25=20,
                top_k_rerank=5
            )
        except Exception as e:
            print(f"[WARN] review_id={review_id} 检索示例失败: {e}")
    
    # Step 2: 六路径并行抽取
    path_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                call_single_path,
                path_config,
                review_id,
                review_text,
                similar_examples
            ): path_config["name"]
            for path_config in COT_PATHS
        }
        
        for future in as_completed(futures):
            path_name = futures[future]
            try:
                result = future.result()
                path_results.append(result)
            except Exception as e:
                print(f"[ERROR] 路径 {path_name} 执行失败: {e}")
                path_results.append(PathResult(path_name=path_name, quadruples=[]))
    
    # Step 3: 投票聚合
    aggregated = aggregate_by_voting(path_results, threshold=VOTE_THRESHOLD)
    
    # Step 4: 准备最终结果
    final_quadruples = list(aggregated.high_confidence)
    
    # Step 5: 处理低置信度和冲突
    if ENABLE_JUDGE and (aggregated.low_confidence or aggregated.conflicts):
        # 对低置信度进行聚类去重
        candidates_to_verify = aggregated.low_confidence
        if ENABLE_CLUSTERING and len(candidates_to_verify) > 3:
            candidates_to_verify = cluster_quadruples(candidates_to_verify)
        
        # 调用Judge验证
        verified = call_judge(
            review_text=review_text,
            candidates=candidates_to_verify,
            conflicts=aggregated.conflicts,
            path_results=path_results
        )
        final_quadruples.extend(verified)
    else:
        # 不启用Judge时，将出现>=2次的低置信度也加入
        quad_counter = Counter()
        for result in path_results:
            for quad in result.quadruples:
                quad_counter[quad.to_tuple()] += 1
        
        for quad in aggregated.low_confidence:
            if quad_counter[quad.to_tuple()] >= 2:
                final_quadruples.append(quad)
    
    # 去重并排序
    seen = set()
    unique_quads = []
    for quad in final_quadruples:
        key = quad.to_tuple()
        if key not in seen:
            seen.add(key)
            unique_quads.append(quad)
    
    # 按原文位置排序
    def sort_key(quad):
        op_pos = review_text.find(quad.opinion)
        op_pos = op_pos if op_pos >= 0 else 10**9
        asp_pos = 10**9 if quad.aspect == "_" else review_text.find(quad.aspect)
        asp_pos = asp_pos if asp_pos >= 0 else 10**9 - 1
        return (op_pos, asp_pos)
    
    unique_quads.sort(key=sort_key)
    
    return Prediction(
        review_id=review_id,
        quadruples=unique_quads,
        path_results=path_results
    )


def write_result_csv(predictions: List[Prediction], path: str):
    """写入结果CSV"""
    lines = []
    for pred in predictions:
        if not pred.quadruples:
            lines.append(f"{pred.review_id},_,_,_,_")
            continue
        for q in pred.quadruples:
            lines.append(f"{pred.review_id},{q.aspect},{q.opinion},{q.category},{q.polarity}")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_jsonl(predictions: List[Prediction], path: str):
    """写入JSONL备份"""
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            record = {
                "id": pred.review_id,
                "quadruples": [q.to_dict() for q in pred.quadruples],
                "path_details": [
                    {
                        "path": r.path_name,
                        "count": len(r.quadruples),
                        "quadruples": [q.to_dict() for q in r.quadruples]
                    }
                    for r in pred.path_results
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =====================================================================
# 断点续传功能
# =====================================================================

def load_checkpoint(jsonl_path: str) -> Dict[str, Prediction]:
    """
    从JSONL文件加载已完成的预测结果
    返回: {review_id: Prediction} 字典
    """
    completed = {}
    if not os.path.exists(jsonl_path):
        return completed
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    rid = record.get("id", "")
                    if not rid:
                        continue
                    
                    # 重建Prediction对象
                    quads = []
                    for q in record.get("quadruples", []):
                        try:
                            quads.append(Quadruple(
                                aspect=q.get("aspect", "_"),
                                opinion=q.get("opinion", ""),
                                category=q.get("category", ""),
                                polarity=q.get("polarity", "")
                            ))
                        except:
                            continue
                    
                    # 重建PathResult（简化版，不含raw_output）
                    path_results = []
                    for pr in record.get("path_details", []):
                        pr_quads = []
                        for q in pr.get("quadruples", []):
                            try:
                                pr_quads.append(Quadruple(
                                    aspect=q.get("aspect", "_"),
                                    opinion=q.get("opinion", ""),
                                    category=q.get("category", ""),
                                    polarity=q.get("polarity", "")
                                ))
                            except:
                                continue
                        path_results.append(PathResult(
                            path_name=pr.get("path", ""),
                            quadruples=pr_quads
                        ))
                    
                    completed[rid] = Prediction(
                        review_id=rid,
                        quadruples=quads,
                        path_results=path_results
                    )
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[WARN] 加载检查点失败: {e}")
    
    return completed


def save_checkpoint(predictions: List[Prediction], jsonl_path: str):
    """
    保存检查点到JSONL文件
    """
    write_jsonl(predictions, jsonl_path)


# =====================================================================
# 主函数
# =====================================================================

def run():
    """主运行函数（支持断点续传）"""
    print("=" * 60)
    print("Multi-CoT 六路径抽取 + 多策略聚合")
    print("=" * 60)
    print(f"抽取模型: {EXTRACTOR_MODELS}")
    print(f"Judge模型: {JUDGE_MODEL}")
    print(f"Embedding模型: {EMBEDDING_MODEL}")
    print(f"投票阈值: {VOTE_THRESHOLD}")
    print(f"启用Judge: {ENABLE_JUDGE}")
    print(f"启用聚类: {ENABLE_CLUSTERING}")
    print(f"使用BM25示例: {USE_BM25_EXAMPLES}")
    print(f"断点续传: {ENABLE_CHECKPOINT}")
    print("=" * 60)
    
    # 初始化BM25检索器（可选）
    bm25_retriever = None
    reranker_client = None
    
    if USE_BM25_EXAMPLES:
        print("\n正在初始化BM25检索器...")
        bm25_retriever = BM25Retriever()
        bm25_retriever.build_index(TRAIN_CSV_PATH, TRAIN_CSV_LABELS)
        print("BM25检索器初始化完成")
        
        print("初始化Reranker客户端...")
        reranker_client = RerankerClient(EXTRACTOR_API_KEY)
        print("Reranker客户端初始化完成")
    
    # 读取测试集
    print(f"\n正在读取测试集: {TEST_CSV_PATH}")
    test_samples = read_test_csv(TEST_CSV_PATH)
    print(f"测试集样本数: {len(test_samples)}")
    
    # ========== 断点续传：加载已完成的结果 ==========
    completed_map: Dict[str, Prediction] = {}
    if ENABLE_CHECKPOINT:
        print(f"\n正在检查断点文件: {RAW_JSONL_PATH}")
        completed_map = load_checkpoint(RAW_JSONL_PATH)
        if completed_map:
            print(f"✓ 从断点恢复，已完成 {len(completed_map)} 条")
        else:
            print("✓ 无断点记录，从头开始")
    
    # 分离已完成和待处理的样本
    pending_samples = []
    all_predictions = []
    
    for rid, text in test_samples:
        if rid in completed_map:
            all_predictions.append(completed_map[rid])
        else:
            pending_samples.append((rid, text))
    
    print(f"待处理样本数: {len(pending_samples)}")
    
    if not pending_samples:
        print("\n所有样本已处理完成！")
    else:
        print(f"\n开始处理，并发数: {MAX_WORKERS}")
        start_time = time.time()
        
        # 处理待处理的样本
        for idx, (rid, text) in enumerate(pending_samples, 1):
            try:
                pred = process_one_multi_cot(
                    review_id=rid,
                    review_text=text,
                    bm25_retriever=bm25_retriever,
                    reranker_client=reranker_client
                )
                all_predictions.append(pred)
                
                quad_count = len(pred.quadruples)
                path_counts = [len(r.quadruples) for r in pred.path_results]
                
                # 计算进度和预估时间
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(pending_samples) - idx) * avg_time
                
                total_done = len(completed_map) + idx
                print(f"[{total_done}/{len(test_samples)}] id={rid} 最终{quad_count}条 "
                      f"(各路径: {path_counts}) "
                      f"[剩余约{remaining/60:.1f}分钟]")
                
                # 定期保存检查点
                if ENABLE_CHECKPOINT and idx % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(all_predictions, RAW_JSONL_PATH)
                    print(f"  ✓ 已保存检查点 ({len(all_predictions)}条)")
                
            except Exception as e:
                print(f"[ERROR] 处理样本 {rid} 失败: {e}")
                all_predictions.append(Prediction(review_id=rid, quadruples=[]))
                # 出错也保存检查点
                if ENABLE_CHECKPOINT:
                    save_checkpoint(all_predictions, RAW_JSONL_PATH)
    
    # 按ID排序
    all_predictions.sort(key=lambda p: int(p.review_id) if p.review_id.isdigit() else p.review_id)
    
    # 写入最终结果
    write_result_csv(all_predictions, OUTPUT_CSV_PATH)
    write_jsonl(all_predictions, RAW_JSONL_PATH)
    
    # 统计
    total_quads = sum(len(p.quadruples) for p in all_predictions)
    valid_count = sum(1 for p in all_predictions if p.quadruples)
    empty_count = len(all_predictions) - valid_count
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"提交文件: {OUTPUT_CSV_PATH}")
    print(f"详细备份: {RAW_JSONL_PATH}")
    print(f"总样本数: {len(all_predictions)}")
    print(f"有效预测: {valid_count}")
    print(f"空预测: {empty_count}")
    print(f"总四元组数: {total_quads}")
    print("=" * 60)


if __name__ == "__main__":
    run()
