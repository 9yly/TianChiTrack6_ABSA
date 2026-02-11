#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate and repair ACOS predictions before CSV conversion.

Reads a generated_predictions.jsonl file (each line is a JSON object with at
least ``prompt`` and ``predict`` fields), parses the quadruples, enforces the
task specification (JSON-only, 13 categories, opinions as contiguous substrings
of the review, deduplication, ordering, etc.), and writes a cleaned JSONL that
can be consumed by ``convert_generated_to_result_csv.py``.

Optionally, the script can also emit the final CSV directly so the usual
conversion step can be chained in one command.

Usage examples::

    # Clean predictions and overwrite the input JSONL
    python llamafactory_aliyun/validate_predictions.py \
        --input llamafactory_aliyun/generated_predictions.jsonl \
        --output llamafactory_aliyun/generated_predictions_validated.jsonl

    # Clean predictions and immediately produce Result.csv
    python llamafactory_aliyun/validate_predictions.py \
        --input llamafactory_aliyun/generated_predictions.jsonl \
        --output llamafactory_aliyun/generated_predictions_validated.jsonl \
        --csv llamafactory_aliyun/Result.csv

The script prints a small summary so you can see how many quadruples were kept,
adjusted, or dropped.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ALLOWED_CATEGORIES = {
    "包装",
    "成分",
    "尺寸",
    "服务",
    "功效",
    "价格",
    "气味",
    "使用体验",
    "物流",
    "新鲜",
    "真伪",
    "整体",
    "其他",
}

ALLOWED_POLARITIES = {"正面", "中性", "负面"}


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.replace("\u00A0", " ").replace("\u3000", " ")).strip()


def normalize_polarity(raw: str) -> str:
    raw = normalize_text(raw)
    if raw in ALLOWED_POLARITIES:
        return raw
    if "正" in raw:
        return "正面"
    if "中" in raw:
        return "中性"
    if "负" in raw or "差" in raw:
        return "负面"
    return "中性"


def normalize_category(raw: str) -> str:
    raw = normalize_text(raw)
    if raw in ALLOWED_CATEGORIES:
        return raw
    # Simple keyword fallback heuristic
    if any(k in raw for k in ["价", "便宜", "贵", "性价比"]):
        return "价格"
    if any(k in raw for k in ["物流", "发货", "快递", "到货", "送货"]):
        return "物流"
    if any(k in raw for k in ["包装", "盒", "瓶", "盖", "封口", "破损", "外观"]):
        return "包装"
    if any(k in raw for k in ["香", "味道", "气味", "清香", "臭"]):
        return "气味"
    if any(k in raw for k in ["客服", "售后", "服务"]):
        return "服务"
    if any(k in raw for k in ["持久", "遮瑕", "保湿", "补水", "控油", "提亮", "美白", "祛痘", "抗皱", "淡斑", "收缩毛孔"]):
        return "功效"
    if any(k in raw for k in ["容量", "规格", "大小", "分量", "克", "毫升", "ml", "g"]):
        return "尺寸"
    if any(k in raw for k in ["成分", "配方", "酒精", "香精", "防腐", "激素"]):
        return "成分"
    if any(k in raw for k in ["新鲜", "日期", "保质"]):
        return "新鲜"
    if any(k in raw for k in ["正品", "假货", "验真", "授权", "专柜", "真伪", "真假"]):
        return "真伪"
    if any(k in raw for k in ["好评", "差评", "一般", "不错", "满意", "失望", "喜欢", "推荐", "还行"]):
        return "整体"
    return "其他"


def parse_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def parse_predict_field(field: Any) -> Optional[Dict[str, Any]]:
    if field is None:
        return None
    if isinstance(field, dict):
        return field
    if isinstance(field, str):
        return parse_first_json_block(field)
    return parse_first_json_block(str(field))


PROMPT_RE = re.compile(r"评论文本[:：]\s*(.*)", re.S)


def extract_review_text(prompt: str) -> str:
    match = PROMPT_RE.search(prompt or "")
    if not match:
        return ""
    return normalize_text(match.group(1))


def clean_quadruples(
    quadruples: Iterable[Dict[str, Any]], review: str
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats = {"total": 0, "kept": 0, "dropped": 0, "adjusted": 0}
    cleaned: List[Dict[str, str]] = []
    seen = set()
    review_lower = review.lower()

    for item in quadruples:
        stats["total"] += 1
        if not isinstance(item, dict):
            stats["dropped"] += 1
            continue
        aspect_raw = item.get("aspect", "")
        opinion_raw = item.get("opinion", "")
        category_raw = item.get("category", "")
        polarity_raw = item.get("polarity", "")

        aspect = normalize_text(aspect_raw) or "_"
        opinion = normalize_text(opinion_raw)
        if not opinion:
            stats["dropped"] += 1
            continue

        # Ensure opinion is a contiguous substring; drop if not found.
        if opinion not in review:
            # Try a lowercase match as fall-back.
            if opinion.lower() not in review_lower:
                stats["dropped"] += 1
                continue
            else:
                opinion = review[review_lower.index(opinion.lower()) : review_lower.index(opinion.lower()) + len(opinion)]
                stats["adjusted"] += 1

        category = normalize_category(category_raw)
        polarity = normalize_polarity(polarity_raw)

        key = (aspect, opinion, category, polarity)
        if key in seen:
            stats["dropped"] += 1
            continue
        seen.add(key)

        cleaned.append(
            {
                "aspect": aspect,
                "opinion": opinion,
                "category": category,
                "polarity": polarity,
            }
        )
        if (
            aspect != normalize_text(aspect_raw) or opinion != normalize_text(opinion_raw) or category != normalize_text(category_raw) or polarity != normalize_text(polarity_raw)
        ):
            stats["adjusted"] += 1
        stats["kept"] += 1

    # Sorting by opinion start index, then aspect start index ("_" as +inf)
    def sort_key(item: Dict[str, str]) -> Tuple[int, int]:
        opinion_index = review.find(item["opinion"])
        if opinion_index == -1:
            opinion_index = sys.maxsize
        if item["aspect"] == "_":
            aspect_index = sys.maxsize
        else:
            aspect_index = review.find(item["aspect"])
            if aspect_index == -1:
                aspect_index = sys.maxsize - 1
        return (opinion_index, aspect_index)

    cleaned.sort(key=sort_key)
    return cleaned, stats


def write_clean_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(records: List[List[Dict[str, str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for idx, quads in enumerate(records, start=1):
            if not quads:
                out.write(f"{idx},_,_,_,_\n")
                continue
            for q in quads:
                aspect = q.get("aspect", "_").replace("\n", " ")
                opinion = q.get("opinion", "_").replace("\n", " ")
                category = q.get("category", "_").replace("\n", " ")
                polarity = q.get("polarity", "_").replace("\n", " ")
                out.write(f"{idx},{aspect},{opinion},{category},{polarity}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and repair generated ACOS predictions.")
    parser.add_argument("--input", required=True, help="Path to generated_predictions.jsonl")
    parser.add_argument("--output", required=True, help="Path to cleaned JSONL output")
    parser.add_argument("--csv", help="Optional path to CSV output (same format as Result.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_path = Path(args.csv) if args.csv else None

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    cleaned_records: List[Tuple[Dict[str, Any], List[Dict[str, str]]]] = []
    stats_total = {"samples": 0, "quad_total": 0, "quad_kept": 0, "quad_dropped": 0, "quad_adjusted": 0}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt", "")
            review = extract_review_text(prompt)
            predict_raw = parse_predict_field(obj.get("predict")) or {}
            quads_raw = predict_raw.get("quadruples", []) if isinstance(predict_raw, dict) else []

            cleaned_quads, stats = clean_quadruples(quads_raw, review)

            sanitized_obj = dict(obj)
            sanitized_obj["predict"] = json.dumps({"quadruples": cleaned_quads}, ensure_ascii=False, separators=(",", ":"))
            cleaned_records.append((sanitized_obj, cleaned_quads))

            stats_total["samples"] += 1
            stats_total["quad_total"] += stats["total"]
            stats_total["quad_kept"] += stats["kept"]
            stats_total["quad_dropped"] += stats["dropped"]
            stats_total["quad_adjusted"] += stats["adjusted"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_clean_jsonl([rec for rec, _ in cleaned_records], output_path)

    if csv_path:
        write_csv([quads for _, quads in cleaned_records], csv_path)

    print(
        "Processed {samples} samples | quads: kept {quad_kept} / dropped {quad_dropped} / total {quad_total} | adjustments {quad_adjusted}".format(
            **stats_total
        )
    )


if __name__ == "__main__":
    main()
