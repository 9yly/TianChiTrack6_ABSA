#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate synonym-based augmented samples while preserving opinions.

Reads a messages-format dataset whose assistant response is ACOS JSON. For each
sample, the script produces up to ``--num-variants`` augmented records by
replacing selected context words in the review with synonyms, without touching
opinion substrings. The assistant JSON stays unchanged.

Example::

    python llamafactory_aliyun/augment_synonyms.py \
        --input cot_train_output/train_llamafactory_format.json \
        --output llamafactory_aliyun/data/train_syn_aug.json \
        --num-variants 2 --seed 42

The resulting JSON array can be merged into the main training set or referenced
as a separate dataset entry.
"""

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROMPT_RE = re.compile(r"评论文本[:：]\s*(.*)", re.S)

# Synonym dictionary (keys are target phrases in context, values are replacement choices)
SYNONYMS: Dict[str, Sequence[str]] = {
    "物流": ["发货", "快递", "配送"],
    "物流速度": ["发货速度", "配送速度"],
    "送货速度": ["物流速度", "配送速度"],
    "客服": ["售后", "客服人员"],
    "性价比": ["价位", "价格实惠", "价钱优势"],
    "价格": ["售价", "价钱"],
    "香味": ["味道", "香气", "清香"],
    "气味": ["味道", "气息"],
    "保湿": ["滋润", "补水"],
    "补水": ["保湿", "滋润"],
    "控油": ["去油", "控油效果"],
    "发货": ["出货", "寄出"],
    "包装": ["外包装", "包装盒"],
    "整体": ["总体", "整体感受"],
    "功效": ["效果", "功能"],
    "使用体验": ["使用感", "体验感"],
}


def extract_review_text(user_content: str) -> str:
    match = PROMPT_RE.search(user_content or "")
    return match.group(1).strip() if match else ""


def rebuild_user_content(user_content: str, new_review: str) -> str:
    # Replace the text after "评论文本:" with new_review
    if "评论文本" not in user_content:
        return user_content
    return re.sub(r"(评论文本[:：]).*", r"\1 " + new_review, user_content, count=1, flags=re.S)


def find_spans(text: str, fragment: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not fragment:
        return spans
    start = 0
    while True:
        idx = text.find(fragment, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(fragment)))
        start = idx + len(fragment)
    return spans


def spans_overlap(span: Tuple[int, int], protected: List[Tuple[int, int]]) -> bool:
    a, b = span
    for p_start, p_end in protected:
        if max(a, p_start) < min(b, p_end):
            return True
    return False


def replace_safe(text: str, target: str, replacement: str, protected: List[Tuple[int, int]]) -> Tuple[str, bool]:
    if not target or target not in text:
        return text, False
    result_parts: List[str] = []
    idx = 0
    changed = False
    while idx < len(text):
        pos = text.find(target, idx)
        if pos == -1:
            result_parts.append(text[idx:])
            break
        span = (pos, pos + len(target))
        if spans_overlap(span, protected):
            result_parts.append(text[idx : pos + len(target)])
            idx = pos + len(target)
            continue
        result_parts.append(text[idx:pos])
        result_parts.append(replacement)
        idx = pos + len(target)
        changed = True
    return "".join(result_parts), changed


def collect_opinion_spans(review: str, assistant_content: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    try:
        obj = json.loads(assistant_content)
    except Exception:
        return spans
    quads = obj.get("quadruples", [])
    if not isinstance(quads, list):
        return spans
    for item in quads:
        if not isinstance(item, dict):
            continue
        opinion = item.get("opinion")
        if not opinion:
            continue
        spans.extend(find_spans(review, opinion))
    return spans


def load_dataset(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    return data


def augment_record(record: Dict, num_variants: int, rng: random.Random) -> List[Dict]:
    messages = record.get("messages", [])
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if not user_msg or not assistant_msg:
        return []

    review = extract_review_text(user_msg.get("content", ""))
    if not review:
        return []

    protected_spans = collect_opinion_spans(review, assistant_msg.get("content", ""))
    if not protected_spans:
        return []

    augmented_records: List[Dict] = []

    for variant_id in range(num_variants):
        new_review = review
        changed_any = False
        keys = list(SYNONYMS.keys())
        rng.shuffle(keys)
        for key in keys:
            if key not in new_review:
                continue
            options = SYNONYMS[key]
            if not options:
                continue
            replacement = options[(variant_id + keys.index(key)) % len(options)]
            new_review, changed = replace_safe(new_review, key, replacement, protected_spans)
            changed_any = changed_any or changed

        if not changed_any or new_review == review:
            continue

        new_record = deepcopy(record)
        for msg in new_record.get("messages", []):
            if msg.get("role") == "user":
                msg["content"] = rebuild_user_content(msg.get("content", ""), new_review)
        augmented_records.append(new_record)

    return augmented_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Synonym-based augmentation for ACOS dataset")
    parser.add_argument("--input", required=True, help="Path to original messages dataset")
    parser.add_argument("--output", required=True, help="Path to write augmented dataset")
    parser.add_argument("--num-variants", type=int, default=1, help="Number of variants per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_path = Path(args.input)
    if not base_path.exists():
        raise SystemExit(f"Input dataset not found: {base_path}")

    data = load_dataset(base_path)
    rng = random.Random(args.seed)

    augmented: List[Dict] = []
    for record in data:
        augmented.extend(augment_record(record, max(args.num_variants, 1), rng))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(augmented, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated {len(augmented)} augmented samples -> {out_path}")


if __name__ == "__main__":
    main()

