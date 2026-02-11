#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construct ORPO preference pairs via procedural perturbations.

Reads a messages-format dataset (e.g. ``cot_train_output/train_llamafactory_format.json``)
where the assistant response is a strict ACOS JSON. For each sample, the script
creates one or more *bad* variants by introducing common failure patterns
(wrong category, polarity flip, non-substring opinion, duplicate entries,
extra explanations, etc.). These form (chosen, rejected) pairs suitable for
``--stage orpo`` in LLaMA-Factory.

Output format (JSON array): each item contains ``messages`` (system+user only),
the gold assistant ``chosen`` string, and a list of ``rejected`` responses.

Example::

    python llamafactory_aliyun/build_orpo_pairs.py \
        --input cot_train_output/train_llamafactory_format.json \
        --output llamafactory_aliyun/data/orpo_pairs.json \
        --max-pairs-per-sample 3

Recommended training command (after updating ``dataset_info.json``)::

    llamafactory-cli train --stage orpo ... --dataset tianchi6_orpo_pairs
"""

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ALLOWED_CATEGORIES = [
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
]

ALLOWED_POLARITIES = ["正面", "中性", "负面"]

PROMPT_RE = re.compile(r"评论文本[:：]\s*(.*)", re.S)


def extract_review_text(user_content: str) -> str:
    match = PROMPT_RE.search(user_content or "")
    return (match.group(1).strip() if match else "").replace("\u00A0", " ")


def load_dataset(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    return data


def get_messages(record: Dict) -> Dict[str, str]:
    system = ""
    user = ""
    assistant = ""
    for msg in record.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system = msg.get("content", "")
        elif role == "user":
            user = msg.get("content", "")
        elif role == "assistant":
            assistant = msg.get("content", "")
    return {"system": system, "user": user, "assistant": assistant}


def parse_assistant_json(content: str) -> Optional[Dict]:
    try:
        return json.loads(content)
    except Exception:
        return None


def perturb_wrong_category(quads: List[Dict]) -> Optional[List[Dict]]:
    candidates = [i for i, q in enumerate(quads) if q.get("category") in ALLOWED_CATEGORIES]
    if not candidates:
        return None
    idx = random.choice(candidates)
    bad = deepcopy(quads)
    current = bad[idx]["category"]
    choices = [c for c in ALLOWED_CATEGORIES if c != current]
    if not choices:
        return None
    bad[idx]["category"] = random.choice(choices)
    return bad


def perturb_wrong_polarity(quads: List[Dict]) -> Optional[List[Dict]]:
    if not quads:
        return None
    idx = random.randrange(len(quads))
    bad = deepcopy(quads)
    current = bad[idx].get("polarity", "")
    choices = [p for p in ALLOWED_POLARITIES if p != current]
    if not choices:
        return None
    bad[idx]["polarity"] = random.choice(choices)
    return bad


def perturb_non_substring(quads: List[Dict]) -> Optional[List[Dict]]:
    if not quads:
        return None
    idx = random.randrange(len(quads))
    bad = deepcopy(quads)
    opinion = bad[idx].get("opinion", "")
    if not opinion:
        return None
    bad[idx]["opinion"] = opinion + "!"
    return bad


def perturb_duplicate(quads: List[Dict]) -> Optional[List[Dict]]:
    if not quads:
        return None
    bad = deepcopy(quads)
    bad.append(deepcopy(bad[-1]))
    return bad


def perturb_extra_text(quads: List[Dict]) -> Optional[str]:
    return json.dumps({"quadruples": quads}, ensure_ascii=False, separators=(",", ":")) + "\n请注意严格遵守以上规范。"


def build_bad_responses(quads: List[Dict], max_pairs: int) -> List[str]:
    bad_responses: List[str] = []

    generators = [
        perturb_wrong_category,
        perturb_wrong_polarity,
        perturb_non_substring,
        perturb_duplicate,
    ]

    random.shuffle(generators)
    for gen in generators:
        if len(bad_responses) >= max_pairs:
            break
        mutated = gen(quads)
        if mutated is None:
            continue
        bad_json = json.dumps({"quadruples": mutated}, ensure_ascii=False, separators=(",", ":"))
        if bad_json not in bad_responses:
            bad_responses.append(bad_json)

    if len(bad_responses) < max_pairs:
        extra = perturb_extra_text(quads)
        if extra and extra not in bad_responses:
            bad_responses.append(extra)

    return bad_responses[:max_pairs]


def build_orpo_pairs(
    data: List[Dict],
    max_pairs_per_sample: int,
    seed: int,
) -> List[Dict]:
    random.seed(seed)
    pairs: List[Dict] = []

    for record in data:
        msgs = get_messages(record)
        assistant_obj = parse_assistant_json(msgs["assistant"])
        if not assistant_obj or not isinstance(assistant_obj.get("quadruples"), list):
            continue

        quads = assistant_obj["quadruples"]
        bad_list = build_bad_responses(quads, max_pairs_per_sample)
        if not bad_list:
            continue

        entry = {
            "messages": [
                {"role": "system", "content": msgs["system"]},
                {"role": "user", "content": msgs["user"]},
            ],
            "chosen": json.dumps({"quadruples": quads}, ensure_ascii=False, separators=(",", ":")),
            "rejected": bad_list,
        }
        pairs.append(entry)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build procedural ORPO preference pairs")
    parser.add_argument("--input", required=True, help="Path to messages-format dataset JSON")
    parser.add_argument("--output", required=True, help="Output JSON file for ORPO pairs")
    parser.add_argument("--max-pairs-per-sample", type=int, default=3, help="Maximum rejected variants per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")

    data = load_dataset(input_path)
    pairs = build_orpo_pairs(data, max(args.max_pairs_per_sample, 1), args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(pairs)} preference pairs to {output_path}")


if __name__ == "__main__":
    main()

