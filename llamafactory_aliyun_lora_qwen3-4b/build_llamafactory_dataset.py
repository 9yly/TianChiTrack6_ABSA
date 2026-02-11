#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a LLaMA-Factory-ready training dataset for ACOS extraction (Qwen3-4B).

Inputs: Train_reviews.csv + Train_labels.csv
Output: JSON array file with either OpenAI messages format (default) or ShareGPT format.

Key features:
- Minimal, strict system prompt with 13 fixed categories and JSON-only rule
- Synonym → canonical category mapping + keyword fallback → 13 categories
- Polarity normalization to {正面, 中性, 负面}
- Opinion strict substring check (must be a contiguous substring of the review)
- Deduplicate and stable sorting by opinion start, then aspect start ("_" as +inf)

Usage (examples):
  python llamafactory_aliyun/build_llamafactory_dataset.py \
    --reviews choose/tianchi/data/TRAIN/Train_reviews.csv \
    --labels  choose/tianchi/data/TRAIN/Train_labels.csv \
    --output  cot_train_output/train_llamafactory_format.json \
    --format messages --include_empty

  python llamafactory_aliyun/build_llamafactory_dataset.py \
    --reviews choose/tianchi/data/TRAIN/Train_reviews.csv \
    --labels  choose/tianchi/data/TRAIN/Train_labels.csv \
    --output  cot_train_output/train_llamafactory_sharegpt.json \
    --format sharegpt --include_empty

The produced file can be referenced in dataset_info.json or used directly with LLaMA-Factory.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re
import sys


# ----- Canonical categories and polarity -----
CATEGORIES = [
    "包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜", "真伪", "整体", "其他"
]

POLARITY_SET = {"正面", "中性", "负面"}


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Normalize common invisible/space chars and collapse whitespace
    s = s.replace("\u00A0", " ").replace("\u3000", " ")
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    # Remove common garbled placeholder
    s = s.replace("�", "")
    return s.strip()


def normalize_polarity(p: str) -> str:
    p = (p or "").strip()
    if p in POLARITY_SET:
        return p
    if "正" in p:
        return "正面"
    if "中" in p or "中立" in p:
        return "中性"
    if "负" in p or "差" in p:
        return "负面"
    return "中性"


# Synonyms to canonical categories (surface cues)
SYNONYM_TO_CATEGORY = {
    # 使用体验
    "颜色": "使用体验", "色号": "使用体验", "显白": "使用体验", "显色": "使用体验", "假白": "使用体验",
    "卡粉": "使用体验", "不油腻": "使用体验", "很油": "使用体验", "清爽": "使用体验",
    "厚重": "使用体验", "轻薄": "使用体验", "搓泥": "使用体验",
    # 功效
    "持久": "功效", "持久度": "功效", "遮瑕": "功效", "保湿": "功效", "补水": "功效",
    "控油": "功效", "祛痘": "功效", "美白": "功效", "抗皱": "功效", "提亮": "功效",
    "淡斑": "功效", "收缩毛孔": "功效", "去角质": "功效", "去黄": "功效", "祛黄": "功效",
    # 价格
    "性价比": "价格", "价位": "价格", "价钱": "价格",
    # 物流
    "物流速度": "物流", "发货速度": "物流", "快递": "物流", "到货": "物流", "送货": "物流",
    # 服务
    "客服": "服务", "售后": "服务", "服务态度": "服务",
    # 气味
    "香味": "气味", "味道": "气味", "香": "气味", "臭": "气味", "清香": "气味",
    # 包装
    "包装质量": "包装", "外观": "包装", "瓶身": "包装", "盒": "包装", "封口": "包装", "破损": "包装",
    # 成分
    "配方": "成分", "酒精": "成分", "香精": "成分", "防腐剂": "成分", "激素": "成分",
    # 尺寸
    "容量": "尺寸", "规格": "尺寸", "大小": "尺寸", "分量": "尺寸", "克": "尺寸",
    "毫升": "尺寸", "ml": "尺寸", "g": "尺寸",
    # 新鲜
    "新鲜度": "新鲜", "生产日期": "新鲜", "保质期": "新鲜",
    # 真伪
    "真假": "真伪", "正品": "真伪", "假货": "真伪", "验真": "真伪", "授权": "真伪", "专柜": "真伪",
}


def normalize_category(raw: str, text: str, aspect: str, opinion: str) -> str:
    raw = (raw or "").strip()
    if raw in CATEGORIES:
        return raw
    if raw in SYNONYM_TO_CATEGORY:
        return SYNONYM_TO_CATEGORY[raw]

    # Chinese keyword fallback using combined context
    zh = raw + (aspect or "") + (opinion or "") + (text or "")
    if any(k in zh for k in ["价", "便宜", "贵", "性价比"]):
        return "价格"
    if any(k in zh for k in ["物流", "发货", "快递", "到货", "送货"]):
        return "物流"
    if any(k in zh for k in ["包装", "盒", "瓶", "盖", "封口", "破损", "外观"]):
        return "包装"
    if any(k in zh for k in ["香", "味道", "清香", "臭"]):
        return "气味"
    if any(k in zh for k in ["成分", "配方", "酒精", "香精", "防腐", "激素"]):
        return "成分"
    if any(k in zh for k in ["持久", "遮瑕", "保湿", "补水", "控油", "祛痘", "美白", "抗皱", "提亮", "收缩毛孔", "淡斑", "去角质", "去黄", "祛黄"]):
        return "功效"
    if any(k in zh for k in ["容量", "规格", "大小", "分量", "克", "毫升", "ml", "g"]):
        return "尺寸"
    if any(k in zh for k in ["客服", "售后", "服务"]):
        return "服务"
    if any(k in zh for k in ["新鲜", "日期", "保质"]):
        return "新鲜"
    if any(k in zh for k in ["正品", "假货", "验真", "授权", "专柜", "真伪", "真假"]):
        return "真伪"
    if any(k in zh for k in ["好评", "差评", "一般", "不错", "很差", "满意", "失望", "喜欢", "不喜欢", "推荐", "还行"]):
        return "整体"
    return "其他"


MINIMAL_SYSTEM_PROMPT = (
    "你是中文化妆品评论的 ACOS 四元组抽取助手。只输出一个 JSON 对象："
    "{\"quadruples\":[{\"aspect\":\"...\",\"opinion\":\"...\",\"category\":\"...\",\"polarity\":\"...\"}, ...]}。"
    "规则：1) quadruples 为数组；无观点输出 {\"quadruples\":[]}。"
    "2) aspect：无显式属性时用 \"_\"。3) opinion：必须为原文中连续片段。"
    "4) category 仅可取 {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜, 真伪, 整体, 其他}。"
    "5) polarity 仅可取 {正面, 中性, 负面}。6) 去重，同一 (aspect,opinion,category,polarity) 保留一条。"
    "7) 排序：按 opinion 在原文首次出现位置升序；若相同按 aspect（\"_\" 视为 +∞）。"
    "8) 不输出任何解释或多余文本。"
)


def read_reviews(path: str, encoding: str = "utf-8-sig"):
    reviews = {}
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            reviews[rid] = normalize_text(row.get("Reviews", ""))
    return reviews


def read_labels(path: str, encoding: str = "utf-8-sig"):
    grouped = defaultdict(list)
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "AspectTerms", "OpinionTerms", "Categories", "Polarities"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels file missing columns: {missing}")
        for row in reader:
            rid = (row.get("id") or "").strip()
            aspect = normalize_text(row.get("AspectTerms", "")) or "_"
            opinion = normalize_text(row.get("OpinionTerms", ""))
            category = normalize_text(row.get("Categories", ""))
            polarity = normalize_polarity(normalize_text(row.get("Polarities", "")))
            try:
                o_start = int((row.get("O_start", "") or "").strip()) if (row.get("O_start", "") or "").strip() != "" else None
            except Exception:
                o_start = None
            try:
                a_start = int((row.get("A_start", "") or "").strip()) if (row.get("A_start", "") or "").strip() != "" else None
            except Exception:
                a_start = None
            grouped[rid].append({
                "aspect": aspect,
                "opinion": opinion,
                "category": category,
                "polarity": polarity,
                "o_start": o_start,
                "a_start": a_start,
            })
    return grouped


def normalize_and_sort(labels_for_one, review_text: str):
    text = normalize_text(review_text)
    seen = set()
    cleaned = []
    for item in labels_for_one:
        aspect = item.get("aspect") or "_"
        opinion = item.get("opinion") or ""
        if not opinion:
            continue
        # Strict substring check in review text
        if opinion not in text:
            continue
        category = normalize_category(item.get("category"), text=text, aspect=aspect, opinion=opinion)
        polarity = normalize_polarity(item.get("polarity"))
        key = (aspect, opinion, category, polarity)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category,
            "polarity": polarity,
            "o_start": item.get("o_start"),
            "a_start": item.get("a_start"),
        })

    # Sorting by opinion start, then aspect start ("_" as +inf)
    def sort_key(x):
        o_start = x.get("o_start")
        a_start = x.get("a_start") if x.get("aspect") != "_" else float("inf")
        if o_start is None:
            # attempt to derive from text index if possible
            idx = text.find(x.get("opinion", ""))
            o_start = idx if idx >= 0 else float("inf")
        if a_start is None:
            a_start = float("inf") if x.get("aspect") == "_" else float("inf") - 1
        return (o_start, a_start)

    cleaned.sort(key=sort_key)
    # Drop position fields for final content
    return [{
        "aspect": c["aspect"],
        "opinion": c["opinion"],
        "category": c["category"],
        "polarity": c["polarity"],
    } for c in cleaned]


def build_records(reviews, labels_grouped, system_prompt: str, include_empty: bool = True):
    ids = list(reviews.keys())
    try:
        ids.sort(key=lambda x: int(x))
    except Exception:
        ids.sort()
    records = []
    for rid in ids:
        text = reviews[rid]
        label_items = labels_grouped.get(rid, [])
        quads = normalize_and_sort(label_items, text) if label_items else []
        if not quads and not include_empty:
            continue
        assistant_obj = {"quadruples": quads}
        user_content = f"请抽取 ACOS 四元组。\n评论ID: {rid}\n评论文本: {text}"
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False, separators=(",", ":"))},
            ]
        }
        records.append(record)
    return records


def to_sharegpt(records):
    """Convert OpenAI messages records to ShareGPT format with conversations [{from,value}]."""
    converted = []
    for r in records:
        msgs = r.get("messages", [])
        conv = []
        for m in msgs:
            role = m.get("role")
            if role == "system":
                conv.append({"from": "system", "value": m.get("content", "")})
            elif role == "user":
                conv.append({"from": "human", "value": m.get("content", "")})
            elif role == "assistant":
                conv.append({"from": "gpt", "value": m.get("content", "")})
        converted.append({"conversations": conv})
    return converted


def main():
    parser = argparse.ArgumentParser(description="Build LLaMA-Factory dataset from ABSA labels.")
    parser.add_argument("--reviews", required=True, help="Path to Train_reviews.csv")
    parser.add_argument("--labels", required=True, help="Path to Train_labels.csv")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    parser.add_argument("--include_empty", action="store_true", help="Include empty quadruples samples")
    parser.add_argument("--format", choices=["messages", "sharegpt"], default="messages", help="Output dataset format")
    parser.add_argument("--system_prompt_file", help="Optional external minimal system prompt file")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    else:
        system_prompt = MINIMAL_SYSTEM_PROMPT

    reviews = read_reviews(args.reviews, encoding=args.encoding)
    labels = read_labels(args.labels, encoding=args.encoding)
    if not reviews:
        print("No reviews loaded. Check --reviews.", file=sys.stderr)
        sys.exit(1)

    records = build_records(reviews, labels, system_prompt, include_empty=args.include_empty or True)
    if args.format == "messages":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        sharegpt_records = to_sharegpt(records)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sharegpt_records, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {args.format} dataset to {out_path}")


if __name__ == "__main__":
    main()

