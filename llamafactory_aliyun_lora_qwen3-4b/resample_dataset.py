#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create resampled or curriculum-focused datasets for ACOS training.

This utility reads a LLaMA-Factory messages-style dataset (e.g.
``cot_train_output/train_llamafactory_format.json``), analyses category
distribution, and produces:

1. A resampled dataset where under-represented categories are upsampled to at
   least ``--min_count`` per category (bounded by ``--max_multiplier``).
2. An optional "difficult samples" dataset that collects reviews containing
   at least ``--difficulty_min_quads`` quadruples or any category listed in
   ``--focus_categories``.

Usage example::

    python llamafactory_aliyun/resample_dataset.py \
        --input cot_train_output/train_llamafactory_format.json \
        --resampled-output llamafactory_aliyun/data/train_resampled.json \
        --difficulty-output llamafactory_aliyun/data/train_difficult.json \
        --min-count 800 --max-multiplier 4 \
        --difficulty-min-quads 3 \
        --focus-categories 使用体验 功效 价格

The output retains the original message structure so it can be referenced
directly from ``dataset_info.json``.
"""

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

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


def load_dataset(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of records")
    return data


def extract_quadruples(record: Dict) -> List[Dict]:
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                obj = json.loads(content)
            except Exception:
                return []
            quads = obj.get("quadruples")
            if isinstance(quads, list):
                # Only keep dict items
                return [q for q in quads if isinstance(q, dict)]
            return []
    return []


def collect_categories(quads: Iterable[Dict]) -> Set[str]:
    cats = set()
    for q in quads:
        cat = q.get("category")
        if cat in ALLOWED_CATEGORIES:
            cats.add(cat)
    return cats


def upsample_dataset(
    data: List[Dict],
    min_count: int,
    max_multiplier: int,
) -> List[Dict]:
    category_counts = collections.Counter()
    per_record_categories: List[Set[str]] = []

    for rec in data:
        quads = extract_quadruples(rec)
        cats = collect_categories(quads)
        per_record_categories.append(cats)
        category_counts.update(cats)

    duplicated_records: List[Dict] = []
    for rec, cats in zip(data, per_record_categories):
        if not cats:
            duplicated_records.append(rec)
            continue
        multiplier = 1
        for cat in cats:
            current = category_counts[cat]
            if current == 0:
                continue
            needed = math.ceil(min_count / current)
            multiplier = max(multiplier, min(needed, max_multiplier))
        duplicated_records.extend([rec] * multiplier)

    return duplicated_records


def select_difficult_samples(
    data: List[Dict],
    min_quads: int,
    focus_categories: Optional[Set[str]],
) -> List[Dict]:
    difficult: List[Dict] = []
    for rec in data:
        quads = extract_quadruples(rec)
        if len(quads) >= min_quads:
            difficult.append(rec)
            continue
        cats = collect_categories(quads)
        if focus_categories and (cats & focus_categories):
            difficult.append(rec)
    return difficult


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample ACOS datasets and build curriculum subsets.")
    parser.add_argument("--input", required=True, help="Path to messages-format JSON dataset")
    parser.add_argument("--resampled-output", help="Where to write the upsampled dataset")
    parser.add_argument("--difficulty-output", help="Where to write difficult-sample subset")
    parser.add_argument("--min-count", type=int, default=600, help="Minimum target count per category after upsampling")
    parser.add_argument("--max-multiplier", type=int, default=4, help="Upper bound on duplication factor per sample")
    parser.add_argument("--difficulty-min-quads", type=int, default=3, help="Minimum quadruples to mark sample as difficult")
    parser.add_argument("--focus-categories", nargs="*", help="Additional categories that define difficult samples")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")

    data = load_dataset(input_path)

    if args.resampled_output:
        resampled = upsample_dataset(data, min_count=max(args.min_count, 1), max_multiplier=max(args.max_multiplier, 1))
        Path(args.resampled_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.resampled_output).open("w", encoding="utf-8") as f:
            json.dump(resampled, f, ensure_ascii=False, indent=2)
        print(f"Resampled dataset written to {args.resampled_output} (size {len(resampled)})")

    if args.difficulty_output:
        focus = None
        if args.focus_categories:
            focus = set(cat for cat in args.focus_categories if cat in ALLOWED_CATEGORIES)
        difficult = select_difficult_samples(data, min_quads=max(args.difficulty_min_quads, 1), focus_categories=focus)
        Path(args.difficulty_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.difficulty_output).open("w", encoding="utf-8") as f:
            json.dump(difficult, f, ensure_ascii=False, indent=2)
        print(f"Difficult subset written to {args.difficulty_output} (size {len(difficult)})")


if __name__ == "__main__":
    main()

