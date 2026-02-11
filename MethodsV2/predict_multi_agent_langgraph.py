#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph Multi-Agent ACOS抽取

基于LangGraph框架实现真正的Multi-Agent方案，采用Evaluator-Optimizer循环模式。

架构：
┌─────────────────────────────────────────────────────────────────┐
│                        ACOS Agent Graph                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐      ┌───────────────────┐      ┌─────────────┐  │
│   │  START  │─────▶│  aspect_extractor │─────▶│   router    │  │
│   └─────────┘      └───────────────────┘      └──────┬──────┘  │
│                                                       │         │
│         ┌─────────────────────────────────────────────┤         │
│         │                                             │         │
│         ▼                                             ▼         │
│  ┌──────────────┐                           ┌──────────────┐   │
│  │opinion_valid │                           │category_class│   │
│  └──────┬───────┘                           └──────┬───────┘   │
│         │                                           │           │
│         └───────────────┬───────────────────────────┘           │
│                         ▼                                       │
│                  ┌─────────────┐                                │
│                  │  assembler  │                                │
│                  └──────┬──────┘                                │
│                         ▼                                       │
│                  ┌─────────────┐      ┌─────────────┐          │
│                  │  validator  │─────▶│   refiner   │──┐       │
│                  └──────┬──────┘      └─────────────┘  │       │
│                         │                     ▲         │       │
│                         │                     └─────────┘       │
│                         ▼                                       │
│                  ┌─────────────┐                                │
│                  │     END     │                                │
│                  └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘

Agent角色：
1. AspectExtractor: 使用LoRA模型抽取初始四元组
2. OpinionValidator: 验证Opinion是否为原文子串
3. CategoryClassifier: 重新分类Category
4. QuadrupleAssembler: 组装最终四元组
5. Validator: 验证四元组质量，生成反馈
6. Refiner: 根据反馈修正错误

执行-判断-观察 循环：
- 执行: 各Agent执行抽取/验证任务
- 判断: Validator评估结果质量
- 观察: 根据feedback决定是否需要重新处理

使用方法：
1. 安装依赖: pip install langgraph langchain langchain-core
2. 修改CONFIG部分的配置
3. python predict_multi_agent_langgraph.py
"""

import os
import json
import time
import re
import requests
import csv
from typing import List, Tuple, Set, Dict, Any, Literal, Annotated
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import operator

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ======================== CONFIG ========================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 数据路径配置
TRAIN_CSV_PATH    = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_reviews.csv"
TRAIN_CSV_LABELS  = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_labels.csv"
TEST_CSV_PATH     = "D:/python-ECI/competitions/tianchi/data/TEST/Test_reviews.csv"
OUTPUT_CSV_PATH   = "D:/python-ECI/competitions/tianchi/MethodsV2/Result_multi_agent_lg.csv"
RAW_JSONL_PATH    = "D:/python-ECI/competitions/tianchi/MethodsV2/raw_predictions_multi_agent_lg.jsonl"

# 主抽取模型配置（LoRA微调模型）
EXTRACTOR_MODEL   = "ft:LoRA/Qwen/Qwen2.5-32B-Instruct:clzc5301v000e12wen-32b:buzvweycucvunqozahym"
EXTRACTOR_API_KEY = "sk-dxfzwvjzdjjljxgccxodrloqnfbgtwvak"

# Thinking模型配置（用于验证和修正）
THINKING_MODEL    = "Qwen/Qwen3-30B-A3B-Thinking-2507"
THINKING_API_KEY  = "sk-avfabhrvxcunxysmffsgwvarvgyrnhe"

# 运行参数
TEMPERATURE       = 0.0
MAX_TOKENS        = 4096
MAX_RETRIES       = 3
RETRY_DELAY       = 2.0
API_DELAY         = 0.5        # API调用间隔
MAX_ITERATIONS    = 2          # 最大迭代次数（执行-判断-观察循环）
ENABLE_BM25       = True       # 是否使用BM25检索示例

# =====================================================================

API_URL = "https://api.siliconflow.cn/v1/chat/completions"

CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITIES = ["正面","中性","负面"]

# =====================================================================
# Agent Prompts
# =====================================================================

# Agent 1: AspectExtractor - 初始抽取
EXTRACTOR_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。

## 任务
给定一条化妆品电商评论，抽取所有观点四元组（AspectTerm, OpinionTerm, Category, Polarity）。

## 抽取策略
1. 先找Opinion词（情感评价词）
2. 回溯找Aspect词（被评价的属性）
3. 确定Category和Polarity

## Category取值
{包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

## Polarity取值
{正面, 中性, 负面}

## 输出格式
严格按JSON输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

## 规则
1. aspect无显式词用"_"；有则必须是原文连续片段
2. opinion必须是原文连续片段
3. 去重，只输出JSON
"""

# Agent 2: OpinionValidator - 验证Opinion
OPINION_VALIDATOR_SYSTEM_PROMPT = """你是一个Opinion验证专家。

## 任务
验证抽取的Opinion是否为原文的连续子串，并检查其情感表达是否正确。

## 输入
- 原文
- 待验证的四元组列表

## 验证规则
1. Opinion必须是原文的**连续子串**（精确匹配）
2. Opinion必须确实表达主观情感或评价
3. Polarity必须与Opinion的情感倾向一致

## 输出格式
```json
{
  "validated": [
    {"aspect":"...","opinion":"...","category":"...","polarity":"...","valid":true},
    {"aspect":"...","opinion":"...","category":"...","polarity":"...","valid":false,"reason":"Opinion不在原文中"}
  ]
}
```

对每个四元组标记valid=true/false，无效的需给出reason。
"""

# Agent 3: CategoryClassifier - 重新分类Category  
CATEGORY_CLASSIFIER_SYSTEM_PROMPT = """你是一个Category分类专家。

## 任务
根据Aspect和Opinion的语义，重新确认或修正Category分类。

## Category取值及含义
- 包装: 产品外包装、盒子、瓶子设计等
- 成分: 产品成分、配方、添加物等
- 尺寸: 产品大小、容量、规格等
- 服务: 客服、售后服务态度等
- 功效: 产品效果、功能、使用后变化等
- 价格: 产品价格、性价比、优惠活动等
- 气味: 产品香味、味道等
- 使用体验: 使用感受、质地、肤感等
- 物流: 快递、配送速度、发货等
- 新鲜度: 产品保质期、生产日期、新鲜程度等
- 真伪: 正品、假货、仿品等
- 整体: 对产品的整体评价
- 其他: 不属于以上类别的

## 输入
- 原文
- 待分类的四元组列表

## 输出格式
```json
{
  "classified": [
    {"aspect":"...","opinion":"...","category":"...","polarity":"...","new_category":"..."}
  ]
}
```

new_category为重新确认后的分类（可与原category相同）。
"""

# Agent 4: Validator - 综合验证
VALIDATOR_SYSTEM_PROMPT = """你是一个ACOS四元组综合验证专家。

## 任务
对抽取的四元组进行最终验证，判断整体质量，并给出改进反馈。

## 验证规则
1. **Aspect验证**：若不为"_"，必须是原文连续子串
2. **Opinion验证**：必须是原文连续子串
3. **Category验证**：必须属于指定类别，且语义匹配
4. **Polarity验证**：必须准确反映Opinion的情感倾向
5. **完整性验证**：是否遗漏了评论中的重要观点

## 输入
- 原文
- 待验证的四元组列表

## 输出格式
```json
{
  "is_valid": true/false,
  "score": 0.0-1.0,
  "verified_quadruples": [...],
  "errors": [{"quad_index":0, "error_type":"...", "description":"..."}],
  "missing_opinions": ["可能遗漏的观点词"],
  "feedback": "改进建议"
}
```

- is_valid: 整体是否通过验证（score>=0.8视为通过）
- score: 质量分数
- verified_quadruples: 验证通过的四元组
- errors: 错误列表
- missing_opinions: 可能遗漏的观点
- feedback: 具体改进建议
"""

# Agent 5: Refiner - 根据反馈修正
REFINER_SYSTEM_PROMPT = """你是一个ACOS四元组修正专家。

## 任务
根据验证反馈，修正错误的四元组，补充遗漏的观点。

## 输入
- 原文
- 当前四元组列表
- 验证反馈（包含错误信息和遗漏的观点）

## 修正策略
1. 修正Opinion不在原文中的问题：找到原文中最接近的子串
2. 修正Category错误：根据Aspect/Opinion语义重新分类
3. 修正Polarity错误：根据Opinion情感倾向重新判断
4. 补充遗漏的观点：根据feedback提示添加新四元组

## 输出格式
```json
{
  "refined_quadruples": [
    {"aspect":"...","opinion":"...","category":"...","polarity":"..."}
  ],
  "changes": ["修改了xxx", "添加了xxx"]
}
```
"""

# =====================================================================
# State Definition
# =====================================================================

class QuadrupleDict(BaseModel):
    """四元组字典格式"""
    aspect: str = "_"
    opinion: str
    category: str
    polarity: str
    valid: bool = True
    reason: str = ""
    
    def to_tuple(self) -> Tuple[str, str, str, str]:
        return (self.aspect, self.opinion, self.category, self.polarity)


class ValidationResult(BaseModel):
    """验证结果"""
    is_valid: bool = False
    score: float = 0.0
    verified_quadruples: List[Dict] = Field(default_factory=list)
    errors: List[Dict] = Field(default_factory=list)
    missing_opinions: List[str] = Field(default_factory=list)
    feedback: str = ""


# 使用TypedDict定义状态，支持Annotated并发更新
class ACOSState(TypedDict, total=False):
    """Agent状态定义 - LangGraph核心
    
    使用TypedDict + Annotated支持并发节点更新同一字段
    """
    # 输入
    review_id: str
    review_text: str
    similar_examples: str
    
    # 抽取结果
    raw_quadruples: List[Dict]  # 初始抽取
    opinion_validated: List[Dict]  # Opinion验证后
    category_classified: List[Dict]  # Category分类后
    assembled_quadruples: List[Dict]  # 组装后
    
    # 验证状态
    validation_result: Dict
    is_valid: bool
    
    # 迭代控制
    iteration: int
    max_iterations: int
    
    # 最终输出
    final_quadruples: List[Dict]
    
    # 执行日志 - 使用Annotated支持并发追加
    agent_logs: Annotated[List[str], operator.add]


# =====================================================================
# API调用工具
# =====================================================================

def call_model(model: str,
               system_prompt: str,
               user_content: str,
               api_key: str,
               api_url: str = API_URL,
               temperature: float = TEMPERATURE,
               max_tokens: int = MAX_TOKENS) -> str:
    """调用模型API（带重试机制）"""
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
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            elif resp.status_code == 429:
                last_error = "API限流 status=429"
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                raise RuntimeError(f"API错误 status={resp.status_code} body={resp.text}")
        except requests.exceptions.Timeout:
            last_error = "请求超时"
            time.sleep(RETRY_DELAY)
            continue
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise
    
    raise RuntimeError(f"API调用失败，已重试{MAX_RETRIES}次: {last_error}")


def extract_json_block(text: str) -> str:
    """提取JSON块"""
    text = text.strip()
    # 处理Thinking模型的<think>标签
    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
    
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text


def parse_quadruples(json_str: str, review_text: str) -> List[Dict]:
    """解析四元组JSON"""
    try:
        obj = json.loads(json_str)
    except Exception:
        return []
    
    if not isinstance(obj, dict):
        return []
    
    # 尝试多种key
    items = (obj.get("quadruples") or 
             obj.get("verified_quadruples") or 
             obj.get("refined_quadruples") or 
             obj.get("classified") or
             obj.get("validated") or [])
    
    if not isinstance(items, list):
        return []
    
    cleaned = []
    seen = set()
    
    for it in items:
        if not isinstance(it, dict):
            continue
        
        aspect = (it.get("aspect") or it.get("new_aspect") or "").strip()
        if aspect == "":
            aspect = "_"
        opinion = (it.get("opinion") or "").strip()
        category = (it.get("new_category") or it.get("category") or "").strip()
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
        
        cleaned.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category,
            "polarity": polarity
        })
    
    return cleaned


# =====================================================================
# Agent Node Functions
# =====================================================================

def aspect_extractor_node(state: ACOSState) -> Dict:
    """
    Agent 1: AspectExtractor
    执行初始抽取，使用LoRA微调模型
    """
    review_id = state.get("review_id", "")
    review_text = state.get("review_text", "")
    similar_examples = state.get("similar_examples", "")
    iteration = state.get("iteration", 0)
    agent_logs = state.get("agent_logs", [])
    
    log_prefix = f"[Iter {iteration}][AspectExtractor]"
    
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
    
    # 如果是重试迭代，加入之前的反馈
    validation_result = state.get("validation_result", {})
    if iteration > 0 and validation_result:
        feedback = validation_result.get("feedback", "")
        missing = validation_result.get("missing_opinions", [])
        if feedback or missing:
            user_content += f"""

**上一轮反馈**：
{feedback}
可能遗漏的观点词：{', '.join(missing) if missing else '无'}

请根据反馈改进抽取结果。"""
    
    try:
        raw_output = call_model(
            model=EXTRACTOR_MODEL,
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            user_content=user_content,
            api_key=EXTRACTOR_API_KEY
        )
        
        json_str = extract_json_block(raw_output)
        quadruples = parse_quadruples(json_str, review_text)
        
        log_msg = f"{log_prefix} 抽取到 {len(quadruples)} 个四元组"
        print(log_msg)
        
        return {
            "raw_quadruples": quadruples,
            "agent_logs": [log_msg]  # Annotated会自动追加
        }
    except Exception as e:
        log_msg = f"{log_prefix} 抽取失败: {e}"
        print(log_msg)
        return {
            "raw_quadruples": [],
            "agent_logs": [log_msg]
        }


def opinion_validator_node(state: ACOSState) -> Dict:
    """
    Agent 2: OpinionValidator
    验证Opinion是否为原文子串
    """
    review_text = state.get("review_text", "")
    quadruples = state.get("raw_quadruples", [])
    iteration = state.get("iteration", 0)
    
    log_prefix = f"[Iter {iteration}][OpinionValidator]"
    
    if not quadruples:
        return {
            "opinion_validated": [],
            "agent_logs": [f"{log_prefix} 无四元组待验证"]
        }
    
    # 串行调用Thinking模型
    time.sleep(API_DELAY)
    
    user_content = f"""## 原文
{review_text}

## 待验证的四元组
{json.dumps(quadruples, ensure_ascii=False, indent=2)}

请验证每个Opinion是否为原文的连续子串。"""
    
    try:
        raw_output = call_model(
            model=THINKING_MODEL,
            system_prompt=OPINION_VALIDATOR_SYSTEM_PROMPT,
            user_content=user_content,
            api_key=THINKING_API_KEY
        )
        
        json_str = extract_json_block(raw_output)
        
        try:
            result = json.loads(json_str)
            validated = result.get("validated", [])
        except:
            validated = quadruples  # 解析失败时使用原始数据
        
        # 过滤有效的四元组
        valid_quads = []
        for item in validated:
            if isinstance(item, dict):
                if item.get("valid", True):
                    valid_quads.append({
                        "aspect": item.get("aspect", "_"),
                        "opinion": item.get("opinion", ""),
                        "category": item.get("category", ""),
                        "polarity": item.get("polarity", "")
                    })
        
        # 如果验证后为空，保留原始结果
        if not valid_quads and quadruples:
            valid_quads = quadruples
        
        log_msg = f"{log_prefix} 验证通过 {len(valid_quads)}/{len(quadruples)} 个"
        print(log_msg)
        
        return {
            "opinion_validated": valid_quads,
            "agent_logs": [log_msg]
        }
    except Exception as e:
        log_msg = f"{log_prefix} 验证失败: {e}"
        print(log_msg)
        return {
            "opinion_validated": quadruples,  # 失败时保留原始
            "agent_logs": [log_msg]
        }


def category_classifier_node(state: ACOSState) -> Dict:
    """
    Agent 3: CategoryClassifier  
    重新分类Category
    """
    review_text = state.get("review_text", "")
    quadruples = state.get("raw_quadruples", [])
    iteration = state.get("iteration", 0)
    
    log_prefix = f"[Iter {iteration}][CategoryClassifier]"
    
    if not quadruples:
        return {
            "category_classified": [],
            "agent_logs": [f"{log_prefix} 无四元组待分类"]
        }
    
    # 串行调用Thinking模型
    time.sleep(API_DELAY)
    
    user_content = f"""## 原文
{review_text}

## 待分类的四元组
{json.dumps(quadruples, ensure_ascii=False, indent=2)}

请确认或修正每个四元组的Category分类。"""
    
    try:
        raw_output = call_model(
            model=THINKING_MODEL,
            system_prompt=CATEGORY_CLASSIFIER_SYSTEM_PROMPT,
            user_content=user_content,
            api_key=THINKING_API_KEY
        )
        
        json_str = extract_json_block(raw_output)
        classified = parse_quadruples(json_str, review_text)
        
        # 如果分类后为空，保留原始结果
        if not classified and quadruples:
            classified = quadruples
        
        log_msg = f"{log_prefix} 分类完成 {len(classified)} 个"
        print(log_msg)
        
        return {
            "category_classified": classified,
            "agent_logs": [log_msg]
        }
    except Exception as e:
        log_msg = f"{log_prefix} 分类失败: {e}"
        print(log_msg)
        return {
            "category_classified": quadruples,  # 失败时保留原始
            "agent_logs": [log_msg]
        }


def assembler_node(state: ACOSState) -> Dict:
    """
    Agent 4: QuadrupleAssembler
    组装最终四元组（合并Opinion验证和Category分类的结果）
    """
    opinion_validated = state.get("opinion_validated", [])
    category_classified = state.get("category_classified", [])
    review_text = state.get("review_text", "")
    iteration = state.get("iteration", 0)
    
    log_prefix = f"[Iter {iteration}][Assembler]"
    
    # 构建opinion -> 验证结果的映射
    opinion_map = {}
    for q in opinion_validated:
        key = (q.get("aspect", "_"), q.get("opinion", ""))
        opinion_map[key] = q
    
    # 构建(aspect, opinion) -> 分类结果的映射
    category_map = {}
    for q in category_classified:
        key = (q.get("aspect", "_"), q.get("opinion", ""))
        category_map[key] = q.get("category", "")
    
    # 合并结果：以opinion_validated为基准，更新category
    assembled = []
    seen = set()
    
    for q in opinion_validated:
        key = (q.get("aspect", "_"), q.get("opinion", ""))
        
        # 获取分类结果
        category = category_map.get(key, q.get("category", ""))
        if category not in CATEGORIES:
            category = q.get("category", "整体")  # 默认使用原category或"整体"
        
        quad = {
            "aspect": q.get("aspect", "_"),
            "opinion": q.get("opinion", ""),
            "category": category,
            "polarity": q.get("polarity", "")
        }
        
        quad_key = (quad["aspect"], quad["opinion"], quad["category"], quad["polarity"])
        if quad_key not in seen and quad["opinion"]:
            seen.add(quad_key)
            assembled.append(quad)
    
    # 补充category_classified中独有的四元组
    for q in category_classified:
        quad_key = (q.get("aspect", "_"), q.get("opinion", ""), 
                    q.get("category", ""), q.get("polarity", ""))
        if quad_key not in seen and q.get("opinion"):
            seen.add(quad_key)
            assembled.append(q)
    
    log_msg = f"{log_prefix} 组装完成 {len(assembled)} 个四元组"
    print(log_msg)
    
    return {
        "assembled_quadruples": assembled,
        "agent_logs": [log_msg]
    }


def validator_node(state: ACOSState) -> Dict:
    """
    Agent 5: Validator
    综合验证四元组质量
    
    这是Evaluator-Optimizer循环的核心判断节点
    """
    review_text = state.get("review_text", "")
    quadruples = state.get("assembled_quadruples", [])
    iteration = state.get("iteration", 0)
    
    log_prefix = f"[Iter {iteration}][Validator]"
    
    if not quadruples:
        log_msg = f"{log_prefix} 无四元组待验证"
        print(log_msg)
        return {
            "validation_result": {"is_valid": False, "score": 0.0, "feedback": "无抽取结果"},
            "is_valid": False,
            "agent_logs": [log_msg]
        }
    
    # 串行调用Thinking模型
    time.sleep(API_DELAY)
    
    user_content = f"""## 原文
{review_text}

## 待验证的四元组
{json.dumps(quadruples, ensure_ascii=False, indent=2)}

请进行综合验证，判断整体质量。"""
    
    try:
        raw_output = call_model(
            model=THINKING_MODEL,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            user_content=user_content,
            api_key=THINKING_API_KEY
        )
        
        json_str = extract_json_block(raw_output)
        
        try:
            result = json.loads(json_str)
        except:
            result = {}
        
        is_valid = result.get("is_valid", True)
        score = result.get("score", 0.8)
        verified_quads = result.get("verified_quadruples", quadruples)
        errors = result.get("errors", [])
        missing = result.get("missing_opinions", [])
        feedback = result.get("feedback", "")
        
        # 如果score >= 0.8 且没有严重错误，视为通过
        if score >= 0.8 and len(errors) == 0:
            is_valid = True
        
        validation_result = {
            "is_valid": is_valid,
            "score": score,
            "verified_quadruples": verified_quads if verified_quads else quadruples,
            "errors": errors,
            "missing_opinions": missing,
            "feedback": feedback
        }
        
        log_msg = f"{log_prefix} 验证完成 score={score:.2f} is_valid={is_valid}"
        print(log_msg)
        
        return {
            "validation_result": validation_result,
            "is_valid": is_valid,
            "agent_logs": [log_msg]
        }
    except Exception as e:
        log_msg = f"{log_prefix} 验证失败: {e}"
        print(log_msg)
        # 验证失败时，默认通过
        return {
            "validation_result": {"is_valid": True, "score": 0.8, "verified_quadruples": quadruples},
            "is_valid": True,
            "agent_logs": [log_msg]
        }


def refiner_node(state: ACOSState) -> Dict:
    """
    Agent 6: Refiner
    根据验证反馈修正错误
    
    这是Evaluator-Optimizer循环的优化节点
    """
    review_text = state.get("review_text", "")
    quadruples = state.get("assembled_quadruples", [])
    validation_result = state.get("validation_result", {})
    iteration = state.get("iteration", 0)
    
    log_prefix = f"[Iter {iteration}][Refiner]"
    
    errors = validation_result.get("errors", [])
    missing = validation_result.get("missing_opinions", [])
    feedback = validation_result.get("feedback", "")
    
    # 如果没有错误和遗漏，直接返回
    if not errors and not missing:
        log_msg = f"{log_prefix} 无需修正"
        print(log_msg)
        return {
            "iteration": iteration + 1,
            "agent_logs": [log_msg]
        }
    
    # 串行调用Thinking模型
    time.sleep(API_DELAY)
    
    user_content = f"""## 原文
{review_text}

## 当前四元组
{json.dumps(quadruples, ensure_ascii=False, indent=2)}

## 验证反馈
错误列表: {json.dumps(errors, ensure_ascii=False)}
遗漏的观点词: {missing}
改进建议: {feedback}

请根据反馈修正错误并补充遗漏。"""
    
    try:
        raw_output = call_model(
            model=THINKING_MODEL,
            system_prompt=REFINER_SYSTEM_PROMPT,
            user_content=user_content,
            api_key=THINKING_API_KEY
        )
        
        json_str = extract_json_block(raw_output)
        
        try:
            result = json.loads(json_str)
            refined = result.get("refined_quadruples", [])
        except:
            refined = quadruples
        
        # 验证refined结果
        refined_parsed = parse_quadruples(json.dumps({"quadruples": refined}), review_text)
        
        if not refined_parsed:
            refined_parsed = quadruples
        
        log_msg = f"{log_prefix} 修正完成 {len(quadruples)} -> {len(refined_parsed)}"
        print(log_msg)
        
        return {
            "raw_quadruples": refined_parsed,  # 更新raw_quadruples以便重新验证
            "iteration": iteration + 1,
            "agent_logs": [log_msg]
        }
    except Exception as e:
        log_msg = f"{log_prefix} 修正失败: {e}"
        print(log_msg)
        return {
            "iteration": iteration + 1,
            "agent_logs": [log_msg]
        }


def finalize_node(state: ACOSState) -> Dict:
    """
    最终化节点
    输出最终四元组
    """
    validation_result = state.get("validation_result", {})
    assembled = state.get("assembled_quadruples", [])
    review_text = state.get("review_text", "")
    
    # 优先使用验证通过的四元组
    final_quads = validation_result.get("verified_quadruples", assembled)
    
    if not final_quads:
        final_quads = assembled
    
    # 去重并排序
    seen = set()
    unique_quads = []
    for q in final_quads:
        key = (q.get("aspect", "_"), q.get("opinion", ""), 
               q.get("category", ""), q.get("polarity", ""))
        if key not in seen and q.get("opinion"):
            seen.add(key)
            unique_quads.append(q)
    
    # 按原文位置排序
    def sort_key(q):
        op_pos = review_text.find(q.get("opinion", ""))
        op_pos = op_pos if op_pos >= 0 else 10**9
        asp = q.get("aspect", "_")
        asp_pos = 10**9 if asp == "_" else review_text.find(asp)
        asp_pos = asp_pos if asp_pos >= 0 else 10**9 - 1
        return (op_pos, asp_pos)
    
    unique_quads.sort(key=sort_key)
    
    return {
        "final_quadruples": unique_quads,
        "agent_logs": [f"[Finalize] 最终输出 {len(unique_quads)} 个四元组"]
    }


# =====================================================================
# Routing Functions
# =====================================================================

def router_after_extractor(state: ACOSState) -> List[str]:
    """
    路由：AspectExtractor之后
    并行分发到OpinionValidator和CategoryClassifier
    """
    if not state.get("raw_quadruples"):
        return ["finalize"]  # 无结果直接结束
    return ["opinion_validator", "category_classifier"]


def should_continue_or_end(state: ACOSState) -> Literal["refiner", "finalize"]:
    """
    条件路由：Validator之后
    判断是否需要继续迭代（执行-判断-观察循环的核心）
    """
    # 检查是否通过验证
    if state.get("is_valid"):
        return "finalize"
    
    # 检查是否达到最大迭代次数
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    if iteration >= max_iterations:
        print(f"[Router] 达到最大迭代次数 {max_iterations}，结束")
        return "finalize"
    
    # 需要继续优化
    print(f"[Router] 验证未通过，进入Refiner (iteration={iteration})")
    return "refiner"


def after_refiner(state: ACOSState) -> str:
    """
    路由：Refiner之后
    返回AspectExtractor重新抽取
    """
    return "aspect_extractor"


# =====================================================================
# Build LangGraph
# =====================================================================

def build_acos_graph() -> StateGraph:
    """
    构建ACOS Agent Graph
    
    图结构：
    START -> aspect_extractor -> [opinion_validator, category_classifier] (并行)
          -> assembler -> validator -> (refiner -> aspect_extractor) 或 finalize -> END
    """
    # 创建状态图
    workflow = StateGraph(ACOSState)
    
    # 添加节点
    workflow.add_node("aspect_extractor", aspect_extractor_node)
    workflow.add_node("opinion_validator", opinion_validator_node)
    workflow.add_node("category_classifier", category_classifier_node)
    workflow.add_node("assembler", assembler_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("finalize", finalize_node)
    
    # 添加边
    # START -> aspect_extractor
    workflow.add_edge(START, "aspect_extractor")
    
    # aspect_extractor -> opinion_validator AND category_classifier (并行)
    workflow.add_edge("aspect_extractor", "opinion_validator")
    workflow.add_edge("aspect_extractor", "category_classifier")
    
    # opinion_validator, category_classifier -> assembler
    workflow.add_edge("opinion_validator", "assembler")
    workflow.add_edge("category_classifier", "assembler")
    
    # assembler -> validator
    workflow.add_edge("assembler", "validator")
    
    # validator -> conditional edge (refiner or finalize)
    workflow.add_conditional_edges(
        "validator",
        should_continue_or_end,
        {
            "refiner": "refiner",
            "finalize": "finalize"
        }
    )
    
    # refiner -> aspect_extractor (循环)
    workflow.add_edge("refiner", "aspect_extractor")
    
    # finalize -> END
    workflow.add_edge("finalize", END)
    
    return workflow


# =====================================================================
# BM25 Retriever (复用现有实现)
# =====================================================================

try:
    from bm25_retriever import BM25Retriever, RerankerClient, retrieve_similar_examples
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[WARN] bm25_retriever not found, BM25 examples disabled")


# =====================================================================
# Main Processing Functions
# =====================================================================

def process_one_review(review_id: str,
                       review_text: str,
                       graph,
                       similar_examples: str = "") -> Dict:
    """
    处理单个评论
    
    Args:
        review_id: 评论ID
        review_text: 评论文本
        graph: 编译后的LangGraph
        similar_examples: BM25检索的相似示例
    
    Returns:
        处理结果字典
    """
    # 初始化状态 - TypedDict直接使用字典
    initial_state: ACOSState = {
        "review_id": review_id,
        "review_text": review_text,
        "similar_examples": similar_examples,
        "raw_quadruples": [],
        "opinion_validated": [],
        "category_classified": [],
        "assembled_quadruples": [],
        "validation_result": {},
        "is_valid": False,
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "final_quadruples": [],
        "agent_logs": []
    }
    
    # 运行图
    try:
        final_state = graph.invoke(initial_state)
        
        return {
            "review_id": review_id,
            "quadruples": final_state.get("final_quadruples", []),
            "is_valid": final_state.get("is_valid", False),
            "iteration": final_state.get("iteration", 0),
            "logs": final_state.get("agent_logs", [])
        }
    except Exception as e:
        print(f"[ERROR] 处理 {review_id} 失败: {e}")
        return {
            "review_id": review_id,
            "quadruples": [],
            "is_valid": False,
            "iteration": 0,
            "logs": [f"Error: {e}"]
        }


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


def write_result_csv(results: List[Dict], path: str):
    """写入结果CSV"""
    lines = []
    for res in results:
        rid = res["review_id"]
        quads = res["quadruples"]
        if not quads:
            lines.append(f"{rid},_,_,_,_")
            continue
        for q in quads:
            lines.append(f"{rid},{q['aspect']},{q['opinion']},{q['category']},{q['polarity']}")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_jsonl(results: List[Dict], path: str):
    """写入JSONL备份"""
    with open(path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


# =====================================================================
# Main Function
# =====================================================================

def run():
    """主运行函数"""
    print("=" * 70)
    print("LangGraph Multi-Agent ACOS抽取")
    print("=" * 70)
    print(f"抽取模型: {EXTRACTOR_MODEL}")
    print(f"Thinking模型: {THINKING_MODEL}")
    print(f"最大迭代次数: {MAX_ITERATIONS}")
    print(f"使用BM25示例: {ENABLE_BM25 and BM25_AVAILABLE}")
    print("=" * 70)
    
    # 构建并编译图
    print("\n正在构建Agent Graph...")
    workflow = build_acos_graph()
    graph = workflow.compile()
    print("Agent Graph构建完成")
    
    # 可视化图结构
    try:
        print("\nGraph结构:")
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"[WARN] 无法绘制Graph: {e}")
    
    # 初始化BM25检索器
    bm25_retriever = None
    reranker_client = None
    
    if ENABLE_BM25 and BM25_AVAILABLE:
        print("\n正在初始化BM25检索器...")
        try:
            bm25_retriever = BM25Retriever()
            bm25_retriever.build_index(TRAIN_CSV_PATH, TRAIN_CSV_LABELS)
            reranker_client = RerankerClient(EXTRACTOR_API_KEY)
            print("BM25检索器初始化完成")
        except Exception as e:
            print(f"[WARN] BM25初始化失败: {e}")
            bm25_retriever = None
    
    # 读取测试集
    print(f"\n正在读取测试集: {TEST_CSV_PATH}")
    test_samples = read_test_csv(TEST_CSV_PATH)
    print(f"测试集样本数: {len(test_samples)}")
    
    # 处理所有样本
    all_results = []
    
    print("\n开始处理...")
    for idx, (rid, text) in enumerate(test_samples, 1):
        # 获取相似示例
        similar_examples = ""
        if bm25_retriever and reranker_client:
            try:
                similar_examples = retrieve_similar_examples(
                    query_text=text,
                    bm25_retriever=bm25_retriever,
                    reranker_client=reranker_client,
                    top_k_bm25=20,
                    top_k_rerank=5
                )
            except Exception as e:
                print(f"[WARN] 检索示例失败: {e}")
        
        # 处理单个评论
        result = process_one_review(
            review_id=rid,
            review_text=text,
            graph=graph,
            similar_examples=similar_examples
        )
        all_results.append(result)
        
        quad_count = len(result["quadruples"])
        print(f"[{idx}/{len(test_samples)}] id={rid} 抽取到 {quad_count} 个四元组 "
              f"(迭代{result['iteration']}次, valid={result['is_valid']})")
        
        # 每10条保存一次中间结果
        if idx % 10 == 0:
            write_jsonl(all_results, RAW_JSONL_PATH)
        
        time.sleep(API_DELAY)
    
    # 按ID排序
    all_results.sort(key=lambda r: int(r["review_id"]) if r["review_id"].isdigit() else r["review_id"])
    
    # 写入结果
    write_result_csv(all_results, OUTPUT_CSV_PATH)
    write_jsonl(all_results, RAW_JSONL_PATH)
    
    # 统计
    total_quads = sum(len(r["quadruples"]) for r in all_results)
    valid_count = sum(1 for r in all_results if r["quadruples"])
    empty_count = len(all_results) - valid_count
    avg_iterations = sum(r["iteration"] for r in all_results) / len(all_results)
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print(f"提交文件: {OUTPUT_CSV_PATH}")
    print(f"详细备份: {RAW_JSONL_PATH}")
    print(f"总样本数: {len(all_results)}")
    print(f"有效预测: {valid_count}")
    print(f"空预测: {empty_count}")
    print(f"总四元组数: {total_quads}")
    print(f"平均迭代次数: {avg_iterations:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
