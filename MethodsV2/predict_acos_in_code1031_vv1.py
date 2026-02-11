#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测

步骤：
1. 修改下方 CONFIG 部分的路径 / 模型名 / API Key 等
2. python predict_acos_in_code.py
3. 生成：
   - Result.csv  (提交文件，无表头：id,AspectTerm,OpinionTerm,Category,Polarity)
   - raw_predictions.jsonl  (结构化解析后的结果备份)

   

   https://aclanthology.org/2025.acl-long.41.pdf

   According to the following sentiment elements definition:- The “aspect term” refers to a specific feature, attribute, or aspect
of a product or service on which a user can express an opinion.
Aspect terms appear explicitly as a substring of the given text.- The “sentiment polarity” refers to the degree of positivity,
negativity or neutrality expressed in the opinion towards a
particular aspect or feature of a product or service, and the
available polarities include: “POS”, “NEG” and “NEU”. “NEU”
means mildly positive or mildly negative. Tuples with objective
sentiment polarity should be ignored.Please carefully follow the instructions. Ensure that aspect terms
are recognized as exact matches in the review Ensure that
sentiment polarities are from the available polarities.Recognize all sentiment elements with their corresponding aspect
terms and sentiment polarity in the given input text (review).
Provide your response in the format of a Python list of tuples:
‘Sentiment elements: [(“aspect term”, “sentiment polarity”), ...]’.
Note that “, ...” indicates that there might be more tuples in the list
if applicable and must not occur in the answer. Ensure there is no
additional text in the response.Input: “““La comida divina .”””


https://aclanthology.org/2025.acl-long.913.pdf
gressive nature of the transformer (Vaswani*Corresponding author.*Our implementation is publicly available at https://
github.com/imsongpasimin/DOT

"""

import os
import json
import time
import re
import requests
import csv
from typing import List, Tuple, Set, Union, Optional
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, validator
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from bm25_retriever import BM25Retriever, RerankerClient, retrieve_similar_examples

# ======================== CONFIG（直接修改即可） ========================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

TRAIN_CSV_PATH    = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_reviews.csv"  # 训练集 CSV 路径（含列：id,Reviews）
TRAIN_CSV_LABELS    = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_labels.csv"  # 训练集 CSV 路径（含列：id,Reviews,Category,Polarity）
TEST_CSV_PATH     = "D:/python-ECI/competitions/tianchi/data/TEST/Test_reviews.csv"   # 测试集 CSV 路径（含列：id,Reviews）
OUTPUT_CSV_PATH   = "D:/python-ECI/competitions/tianchi/choose/Result_1031vv1.csv"         # 预测结果输出路径（提交格式）
RAW_JSONL_PATH    = "D:/python-ECI/competitions/tianchi/choose/raw_predictions_1031vv1.jsonl"  # 保存结构化结果（便于排查）
MODEL_NAME        = "ft:LoRA/Qwen/Qwen2.5-32B-Instruct:clzc5301v000e12rx905d:buzvweycucvunqozahym"   
API_KEY           = "sk-slhfvynmzmwpwnhlivzkfkhb"  # 尽量用环境变量
USE_RESPONSE_FORMAT = False  # 如果将来 API 支持严格 JSON，可改 True
TEMPERATURE       = 0.0
MAX_TOKENS        = 4096
MAX_RETRIES       = 1          # 每条样本解析失败重试次数
SLEEP_BETWEEN     = 0.6        # 每次调用间隔秒数（防限流）
MAX_WORKERS       = 6         # 并发线程数

# =====================================================================

API_URL = "https://api.siliconflow.cn/v1/chat/completions"

CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITIES = ["正面","中性","负面"]

# DEFAULT_SYSTEM_PROMPT = """
# 你是一个专业的中文电商化妆品评论观点四元组抽取助手。


# **任务描述**
# 给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。

# **四元组定义**
#    - aspect: 属性词（若评论中未出现显式属性词，用 "_" 标记）
#    - opinion: 观点情感词（必须与原文中的字面内容完全一致，不得增删或形态改写；如果无观点则不输出该条四元组）
#    - category: 必须从以下 13 个预定义类别中选择（逐字匹配）：
#      {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
#    - polarity: 情感极性，取值仅限：{正面, 中性, 负面}

# **严格遵守规则**

# 1. quadruples 为数组；若没有观点，输出 {"quadruples":[]}
# 2. aspect：若无显式属性词用 "_"；原文中出现的需与原文一致；不添加空格。
# 3. opinion：必须是原文中连续片段；保持原字符；无观点不输出该条。
# 4. category 取值必须在 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
# 5. polarity 取值必须在 {正面, 中性, 负面}
# 正面：表达满意或积极情绪。
# 负面：表达不满或消极情绪。
# 中性：既非明确积极也非明确消极，或上下文未提供足够信息以判断明确情感

# 6. 不得出现与原文无关或臆造的词；不得输出解释文字；不得添加多余字段。
# 7. 去重：同一 (aspect, opinion, category, polarity) 只保留一个。
# 8. 排序：按 opinion 在原文首次出现位置升序；若相同再按 aspect（"_" 视为 +∞）。
# 9. 只输出 JSON，不输出其它任何文本。


# **输出格式要求**
# 输出唯一 JSON：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

# """
DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论观点四元组抽取助手。
任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。
严格遵守：
1. 输出唯一 JSON：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}
2. quadruples 为数组；若没有观点，输出 {"quadruples":[]}
3. aspect：若无显式属性词用 "_"；原文中出现的需与原文一致；不添加空格。
4. opinion：必须是原文中连续片段；保持原字符；无观点不输出该条。
5. category 取值必须在 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
6. polarity 取值必须在 {正面, 中性, 负面}
7. 不得出现与原文无关或臆造的词；不得输出解释文字；不得添加多余字段。
8. 去重：同一 (aspect, opinion, category, polarity) 只保留一个。
9. 排序：按 opinion 在原文首次出现位置升序；若相同再按 aspect（"_" 视为 +∞）。
10. 只输出 JSON，不输出其它任何文本。
"""
# DEFAULT_SYSTEM_PROMPT = """你是一个专业的中文电商化妆品评论ACOS四元组抽取专家。基于大量训练数据分析，需要精确识别属性词-观点词的语义关联配对，并进行准确的类别和情感分类。

# ## 核心任务
# 给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。

# ## ACOS四元组抽取规则

# **属性词（AspectTerm）识别规则**：
# - **产品特征词**：具体的产品组成部分或特征（如：物流、价格、包装、味道、活动、赠品、速度、补水、服务等）
# - **功能词**：产品的具体功效（如：遮暇功能、上妆效果、持久情况）
# - **服务相关**：购买体验相关词汇（如：快递员、客服、活动价）
# - **大多数情况为隐式**：约80%的观点没有明确属性词，用"_"表示
# - **必须原文精确**：若有属性词，则必须是原文中连续的片段，保持原字符。原文中出现的需与原文一致；不添加空格

# **观点词（OpinionTerm）识别规则**：
# - **情感评价词**：表达主观评价的词汇（如：不错、很好、很好用、还不错、还可以、好用、挺好的、很快、很喜欢、快、好评）
# - **程度描述词**：带有情感倾向的描述（如：特别快、很好闻、水水嫩嫩的、太随便了）
# - **态度表达词**：喜好或厌恶（如：大爱、喜欢、满意、不满意）
# - **必须原文精确**：必须是原文中连续片段；保持原字符；不能修改、添加或省略任何字符

# **类别（Category）判断规则**：
# - 取值必须在 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}

# **情感极性（Polarity）判断规则**：
# - 取值必须在 {正面, 中性, 负面}

# **配对策略与约束**：
# 1. **语义邻近原则**：在句子中寻找语义相关的属性词-观点词配对
# 2. **语法结构分析**：识别"属性+观点"的语法模式（如：味道很好闻、物流特别快）
# 3. **隐式属性处理**：当观点词无明确修饰对象时，属性设为"_"
# 4. **完整性保证**：确保每个表达情感态度的观点词都被捕获
# 5. **关键约束**：AspectTerm和OpinionTerm不可能同时为"_"，至少有一个必须是有效的原文片段
# 6. **关键约束**：四元组（AspectTerm, OpinionTerm, Category, Polarity）不可能同时为空，也就是不可能同时为"_"


# ## 输出格式要求
# 严格按JSON格式输出：{"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}

# ## 严格遵守规则
# 1. quadruples 为数组；若没有观点，输出 {"quadruples":[]}
# 2. aspect：若无显式属性词用 "_"；若有属性词，则必须是原文中连续的片段，保持原字符。原文中出现的需与原文一致；不添加空格
# 3. opinion：若有观点，必须是原文中连续片段；保持原字符；不能修改、添加或省略任何字符
# 4. **关键约束**：AspectTerm和OpinionTerm不可能同时为"_"，至少有一个必须是有效的原文片段
# 5. category 取值必须在 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
# 6. polarity 取值必须在 {正面, 中性, 负面}
# 7. 不得出现与原文无关或臆造的词；不得输出解释文字；不得添加多余字段
# 8. 去重：同一 (aspect, opinion, category, polarity) 只保留一个
# 9. 排序：按 opinion 在原文首次出现位置升序；若相同再按 aspect（"_" 视为 +∞）
# 10. 只输出 JSON，不输出其它任何文本


# **关键要求**：必须结合原始评论的完整语义上下文进行分析，确保分类决策基于原文的真实表达意图。
# """

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
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            txt = row.get("Reviews","")
            if rid is None:
                continue
            result.append((str(rid).strip(), txt.strip()))
    return result


def call_model(model: str,
               system_prompt: str,
               review_id: str,
               review_text: str,
               api_key: str,
               similar_examples: str = "",
               use_response_format: bool=False,
               temperature: float=0.0,
               max_tokens: int=512) -> str:
    """
    调用 SiliconFlow Chat API，返回 assistant 内容。
    
    Args:
        model: 模型名称
        system_prompt: 系统提示词
        review_id: 评论ID
        review_text: 评论文本
        api_key: API密钥
        similar_examples: 相似示例文本
        use_response_format: 是否使用response_format
        temperature: 温度参数
        max_tokens: 最大token数
    """
    # 构建用户内容
    if similar_examples:
        user_content = f"""
**参考示例**：
{similar_examples}

**待完成任务**：
\n评论ID: {review_id}\n评论文本: {review_text}\n
    
严格只输出一个 JSON 对象。"""
    else:
        user_content = f"""
**待完成任务**：
\n评论ID: {review_id}\n评论文本: {review_text}\n
    
严格只输出一个 JSON 对象。"""
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
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"API错误 status={resp.status_code} body={resp.text}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"未找到模型输出字段：{data}")
    return content


def extract_json_block(text: str) -> str:
    """
    若模型额外输出杂项文本，尝试抽取第一个 { ... }。
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text


def validate_and_repair(json_str: str, original_text: str) -> List[Quadruple]:
    try:
        obj = json.loads(json_str)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    items = obj.get("quadruples", [])
    if not isinstance(items, list):
        return []

    cleaned = []
    seen: Set[Tuple[str,str,str,str]] = set()
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
        
        # 尝试创建 Quadruple 对象进行验证
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


def process_one(review_id: str,
                review_text: str,
                model: str,
                api_key: str,
                system_prompt: str,
                max_retries: int,
                bm25_retriever: Optional[BM25Retriever] = None,
                reranker_client: Optional[RerankerClient] = None,
                use_response_format: bool=False,
                sleep: float=0.6) -> Prediction:
    """
    处理单个样本
    
    Args:
        review_id: 评论ID
        review_text: 评论文本
        model: 模型名称
        api_key: API密钥
        system_prompt: 系统提示词
        max_retries: 最大重试次数
        bm25_retriever: BM25检索器（可选）
        reranker_client: Reranker客户端（可选）
        use_response_format: 是否使用response_format
        sleep: 休眠时间
    """
    # 检索相似示例
    similar_examples = ""
    if bm25_retriever and reranker_client:
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
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = call_model(
                model=model,
                system_prompt=system_prompt,
                review_id=review_id,
                review_text=review_text,
                api_key=api_key,
                similar_examples=similar_examples,
                use_response_format=use_response_format,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            json_candidate = extract_json_block(raw)
            quadruples = validate_and_repair(json_candidate, review_text)
            return Prediction(review_id=review_id, quadruples=quadruples)
        except Exception as e:
            last_error = e
            time.sleep(sleep)
    print(f"[WARN] review_id={review_id} 解析失败: {last_error}")
    return Prediction(review_id=review_id, quadruples=[])


def write_result_csv(predictions: List[Prediction], path: str):
    lines = []
    for pred in predictions:
        if not pred.quadruples:
            lines.append(f"{pred.review_id},_,_,_,_")
            continue
        for q in pred.quadruples:
            lines.append(f"{pred.review_id},{q.aspect},{q.opinion},{q.category},{q.polarity}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run():
    if not API_KEY or API_KEY.startswith("REPLACE_WITH_YOUR_NEW_KEY"):
        raise ValueError("请在脚本顶部配置 API_KEY（或设置环境变量 SILICONFLOW_API_KEY）后再运行。")

    # 初始化BM25检索器
    print("正在初始化BM25检索器...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(TRAIN_CSV_PATH, TRAIN_CSV_LABELS)
    print("BM25检索器初始化完成")
    
    # 初始化Reranker客户端
    print("初始化Reranker客户端...")
    reranker_client = RerankerClient(API_KEY)
    print("Reranker客户端初始化完成")
    
    tests = read_test_csv(TEST_CSV_PATH)
    print(f"加载测试样本数: {len(tests)}")

    # # # # ====== 只测试前 10 条 ======
    # tests = tests[:10]
    # print(f"仅测试前 {len(tests)} 条样本")

    print(f"使用 {MAX_WORKERS} 个并发线程处理...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(RAW_JSONL_PATH), exist_ok=True)
    
    # 用于线程安全的文件写入
    file_lock = Lock()
    predictions_dict = {}  # 使用字典存储，key为index，保证顺序
    completed_count = 0
    
    # 定义处理单个样本的包装函数
    def process_wrapper(idx_rid_text):
        idx, (rid, text) = idx_rid_text
        try:
            pred = process_one(
                review_id=rid,
                review_text=text,
                model=MODEL_NAME,
                api_key=API_KEY,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_retries=MAX_RETRIES,
                bm25_retriever=bm25_retriever,
                reranker_client=reranker_client,
                use_response_format=USE_RESPONSE_FORMAT,
                sleep=SLEEP_BETWEEN
            )
            return idx, rid, text, pred
        except Exception as e:
            print(f"[ERROR] 处理样本 {rid} 时发生异常: {e}")
            return idx, rid, text, Prediction(review_id=rid, quadruples=[])
    
    # 使用线程池并发处理
    with open(RAW_JSONL_PATH, "w", encoding="utf-8") as raw_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_wrapper, (idx, (rid, text))): idx
                for idx, (rid, text) in enumerate(tests, 1)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_idx):
                try:
                    idx, rid, text, pred = future.result()
                    
                    # 存储结果
                    predictions_dict[idx] = pred
                    
                    # 线程安全地写入文件
                    with file_lock:
                        raw_f.write(json.dumps({
                            "id": rid,
                            "review": text,
                            "quadruples": [q.dict() for q in pred.quadruples]
                        }, ensure_ascii=False) + "\n")
                        raw_f.flush()  # 立即刷新到磁盘
                        
                        completed_count += 1
                        print(f"[{completed_count}/{len(tests)}] id={rid} 预测 {len(pred.quadruples)} 条")
                
                except Exception as e:
                    print(f"[ERROR] 处理结果时发生异常: {e}")
    
    # 按顺序整理预测结果
    predictions = [predictions_dict[idx] for idx in sorted(predictions_dict.keys())]
    
    # 写入结果CSV
    write_result_csv(predictions, OUTPUT_CSV_PATH)
    print(f"\n完成：提交文件 -> {OUTPUT_CSV_PATH}")
    print(f"结构化备份 -> {RAW_JSONL_PATH}")


if __name__ == "__main__":
    run()