#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
                  若提供此参数，会为缺失评论的 id 生成一条空 quadruples（不推荐一般训练）。
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
1. 四元组定义
   - aspect: 属性词（若评论中未出现显式属性词，用 "_" 标记）
   - opinion: 观点情感词（必须与原文中的字面内容完全一致，不得增删或形态改写；如果无观点则不输出该条四元组）
   - category: 必须从以下 13 个预定义类别中选择（逐字匹配）：
     {包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他}
   - polarity: 情感极性，取值仅限：{正面, 中性, 负面}
2. 当一条评论包含多条独立观点时，输出多个四元组；不得合并。
3. aspect = "_" 仅表示“隐式属性”而非未知；隐式属性仍需给出正确的 category。
4. opinion 必须是评论原文中连续的片段，保持原始顺序与字符。
5. 不要产生重复四元组；同一 (aspect, opinion, category, polarity) 只保留一份。
6. 不要臆造评论中不存在的观点；如果确实没有可抽取的观点，返回空数组。
7. 输出格式：严格输出一个 JSON 对象字符串：
   {"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}, ...]}
   - quadruples 的值为一个列表
   - 列表为空则输出 {"quadruples":[]}
   - 键名必须是 aspect / opinion / category / polarity（全部小写）
   - 字符串使用原始中文，不转义除非 JSON 需要
8. 不添加任何额外解释、注释、换行提示或自然语言说明。仅输出 JSON。
9. 保持输出 determinism：同一输入始终输出相同顺序的四元组。排序规则：按 opinion 在原文出现的起始位置升序；如 opinion 相同则按 aspect 起始位置；若 aspect 为 "_" 视其开始位置为 +∞。
10. 若同一个 opinion 对不同 category 的确合理，可分别输出；但请避免不必要的多类别解释。"""

def read_reviews(path, encoding="utf-8-sig"):
    reviews = {}
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns: id, Reviews (case sensitive per sample)
        for row in reader:
            rid = row.get("id")
            if rid is None:
                continue
            rid = rid.strip()
            reviews[rid] = row.get("Reviews", "").strip()
    return reviews

def read_labels(path, encoding="utf-8-sig"):
    # Return dict: id -> list of label dicts
    grouped = defaultdict(list)
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        required = {"id","AspectTerms","OpinionTerms","Categories","Polarities"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels file missing columns: {missing}")

        for row in reader:
            rid = row["id"].strip()
            aspect = row["AspectTerms"].strip()
            opinion = row["OpinionTerms"].strip()
            cat = row["Categories"].strip()
            pol = row["Polarities"].strip()
            # positions (may be empty)
            try:
                o_start = int(row.get("O_start","").strip()) if row.get("O_start","").strip() != "" else None
            except:
                o_start = None
            try:
                a_start = int(row.get("A_start","").strip()) if row.get("A_start","").strip() != "" else None
            except:
                a_start = None

            grouped[rid].append({
                "aspect": aspect if aspect else "_",
                "opinion": opinion if opinion else "",
                "category": cat,
                "polarity": pol,
                "o_start": o_start,
                "a_start": a_start
            })
    return grouped

def normalize_and_sort(labels_for_one):
    # Deduplicate exact quadruples
    seen = set()
    cleaned = []
    for item in labels_for_one:
        aspect = item["aspect"] if item["aspect"] else "_"
        opinion = item["opinion"]
        category = item["category"]
        polarity = item["polarity"]
        if not opinion:
            # 没有 opinion 则不构造该条（任务要求 opinion 必须存在）
            continue
        key = (aspect, opinion, category, polarity)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category,
            "polarity": polarity,
            "o_start": item["o_start"],
            "a_start": item["a_start"]
        })

    # 排序
    def sort_key(x):
        o_start = x["o_start"]
        a_start = x["a_start"] if x["aspect"] != "_" else float("inf")
        # 缺失时放后
        if o_start is None:
            o_start = float("inf")
        if a_start is None:
            if x["aspect"] == "_":
                a_start = float("inf")
            else:
                a_start = float("inf") - 1
        return (o_start, a_start)

    cleaned.sort(key=sort_key)
    # Drop position fields for final output
    final_list = [{
        "aspect": c["aspect"],
        "opinion": c["opinion"],
        "category": c["category"],
        "polarity": c["polarity"]
    } for c in cleaned]
    return final_list

def build_jsonl(reviews, labels_grouped, system_prompt, output_path, add_missing_ids=False, encoding="utf-8"):
    all_ids = list(reviews.keys())
    # Optionally include ids appearing only in labels (not typical)
    if add_missing_ids:
        more = [i for i in labels_grouped.keys() if i not in reviews]
        if more:
            print(f"[WARN] IDs in labels but not in reviews: {more}. Will add empty review placeholders.", file=sys.stderr)
            for mid in more:
                reviews[mid] = ""  # placeholder
                all_ids.append(mid)

    # Stable sort by numeric id if possible
    try:
        all_ids.sort(key=lambda x: int(x))
    except:
        all_ids.sort()

    with open(output_path, "w", encoding=encoding) as out_f:
        for rid in all_ids:
            review_text = reviews[rid]
            label_items = labels_grouped.get(rid, [])
            quadruples = normalize_and_sort(label_items) if label_items else []
            assistant_obj = {"quadruples": quadruples}

            user_content = f"请从下面这条评论中抽取所有 ACOS 观点四元组。\n评论ID: {rid}\n评论文本: {review_text}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False, separators=(',', ':'))}
            ]
            line_obj = {"messages": messages}
            out_f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

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
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    reviews = read_reviews(args.reviews, encoding=args.encoding)
    labels_grouped = read_labels(args.labels, encoding=args.encoding)

    if not reviews:
        print("No reviews loaded. Check reviews file.", file=sys.stderr)
        sys.exit(1)

    build_jsonl(
        reviews=reviews,
        labels_grouped=labels_grouped,
        system_prompt=system_prompt,
        output_path=args.output,
        add_missing_ids=args.add_missing_ids,
        encoding="utf-8"
    )
    print(f"Done. Wrote JSONL to {args.output}")

if __name__ == "__main__":
    main()