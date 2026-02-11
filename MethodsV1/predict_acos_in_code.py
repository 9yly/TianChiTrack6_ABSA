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

# ======================== CONFIG（直接修改即可） ========================

TEST_CSV_PATH     = "D:\python-ECI\competitions\tianchi\data\TEST\Test_reviews.csv"   # 测试集 CSV 路径（含列：id,Reviews）
OUTPUT_CSV_PATH   = "D:\python-ECI\competitions\tianchi\MethodsV1\1229Result.csv"         # 预测结果输出路径（提交格式）
RAW_JSONL_PATH    = "D:\python-ECI\competitions\tianchi\MethodsV1\1229raw_predictions.jsonl"  # 保存结构化结果（便于排查）
MODEL_NAME        = "LoRA/Qwen/Qwen2.5-7B-Instruct"
API_KEY           = "sk-avfabhrvxcuwwrlxpvhaezijzarvgyrnhe"  # 尽量用环境变量
USE_RESPONSE_FORMAT = False  # 如果将来 API 支持严格 JSON，可改 True
TEMPERATURE       = 0.0
MAX_TOKENS        = 512
MAX_RETRIES       = 1          # 每条样本解析失败重试次数
SLEEP_BETWEEN     = 0.6        # 每次调用间隔秒数（防限流）
BATCH_SIZE        = 1          # 当前实现是串行，如要并发需自行改造

# =====================================================================

API_URL = "https://api.siliconflow.cn/v1/chat/completions"

CATEGORIES = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITIES = ["正面","中性","负面"]

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
               use_response_format: bool=False,
               temperature: float=0.0,
               max_tokens: int=512) -> str:
    """
    调用 SiliconFlow Chat API，返回 assistant 内容。
    """
    user_content = f"请抽取 ACOS 四元组。\n评论ID: {review_id}\n评论文本: {review_text}\n严格只输出一个 JSON 对象。"
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
        cleaned.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category,
            "polarity": polarity
        })

    def first_pos(opinion: str) -> int:
        idx = original_text.find(opinion)
        return idx if idx >= 0 else 10**9

    def aspect_pos(aspect: str) -> int:
        if aspect == "_":
            return 10**9
        idx = original_text.find(aspect)
        return idx if idx >= 0 else 10**9 - 1

    cleaned.sort(key=lambda x: (first_pos(x["opinion"]), aspect_pos(x["aspect"])))

    try:
        qlist = QuadrupleList(quadruples=[Quadruple(**c) for c in cleaned])
        return qlist.quadruples
    except ValidationError:
        return []


def process_one(review_id: str,
                review_text: str,
                model: str,
                api_key: str,
                system_prompt: str,
                max_retries: int,
                use_response_format: bool=False,
                sleep: float=0.6) -> Prediction:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = call_model(
                model=model,
                system_prompt=system_prompt,
                review_id=review_id,
                review_text=review_text,
                api_key=api_key,
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

    tests = read_test_csv(TEST_CSV_PATH)
    print(f"加载测试样本数: {len(tests)}")

    # # ====== 只测试前 10 条 ======
    # tests = tests[:20]
    # print(f"仅测试前 {len(tests)} 条样本")

    predictions: List[Prediction] = []
    with open(RAW_JSONL_PATH, "w", encoding="utf-8") as raw_f:
        for idx, (rid, text) in enumerate(tests, 1):
            pred = process_one(
                review_id=rid,
                review_text=text,
                model=MODEL_NAME,
                api_key=API_KEY,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_retries=MAX_RETRIES,
                use_response_format=USE_RESPONSE_FORMAT,
                sleep=SLEEP_BETWEEN
            )
            predictions.append(pred)
            raw_f.write(json.dumps({
                "id": rid,
                "review": text,
                "quadruples": [q.model_dump() for q in pred.quadruples]
            }, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(tests)}] id={rid} 预测 {len(pred.quadruples)} 条")

    write_result_csv(predictions, OUTPUT_CSV_PATH)
    print(f"完成：提交文件 -> {OUTPUT_CSV_PATH}")
    print(f"结构化备份 -> {RAW_JSONL_PATH}")


if __name__ == "__main__":
    run()