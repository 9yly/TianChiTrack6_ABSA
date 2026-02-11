#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert generated_predictions.jsonl -> submission CSV compatible with predict_acos_in_code.py

Reads the file `generated_predictions.jsonl` in the same folder, expects each line to be a JSON object
with a `predict` field which is itself a JSON string like '{"quadruples":[...]}'.

Output CSV format (no header) lines are:
    {index},{aspect},{opinion},{category},{polarity}

Indexing: rows are numbered sequentially starting from 1 following the order in the jsonl file.

If a prediction has no quadruples, a single row with underscores is emitted: `{index},_,_,_,_`

Usage:
    python convert_generated_to_result_csv.py 

"""
import json
from pathlib import Path
import sys


# Use the absolute paths requested by the user (as Path objects)
INPUT_PATH = Path(r"D:\python-ECI\compitation\tianchi\llamafactory_aliyun\generated_predictions.jsonl")
# Match the output path used in predict_acos_in_code.py
OUTPUT_CSV_PATH = Path(r"D:\python-ECI\compitation\tianchi\llamafactory_aliyun\Result.csv")
BACKUP_JSONL = Path(r"D:\python-ECI\compitation\tianchi\llamafactory_aliyun\generated_predictions_parsed.jsonl")


def parse_predict_field(predict_field):
    """Ensure predict_field is parsed into a dict with key 'quadruples'.
    predict_field may be already a dict or a JSON-string. Return list of quadruples (possibly empty).
    """
    if predict_field is None:
        return []
    if isinstance(predict_field, dict):
        obj = predict_field
    else:
        # some entries have predict as a JSON string
        try:
            obj = json.loads(predict_field)
        except Exception:
            # fallback: try to find first {...} block
            s = str(predict_field)
            start = s.find('{')
            end = s.rfind('}')
            if start >= 0 and end > start:
                try:
                    obj = json.loads(s[start:end+1])
                except Exception:
                    return []
            else:
                return []

    quads = obj.get('quadruples') if isinstance(obj, dict) else None
    if not isinstance(quads, list):
        return []
    # keep as list of dicts with keys aspect, opinion, category, polarity
    cleaned = []
    for it in quads:
        if not isinstance(it, dict):
            continue
        aspect = it.get('aspect', '')
        opinion = it.get('opinion', '')
        category = it.get('category', '')
        polarity = it.get('polarity', '')
        cleaned.append({
            'aspect': aspect if aspect is not None and aspect != '' else '_',
            'opinion': opinion if opinion is not None and opinion != '' else '_',
            'category': category if category is not None and category != '' else '_',
            'polarity': polarity if polarity is not None and polarity != '' else '_',
        })
    return cleaned


def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        sys.exit(1)

    # Ensure output directory exists
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    parsed_backup_lines = []

    with open(INPUT_PATH, 'r', encoding='utf-8') as inf, open(OUTPUT_CSV_PATH, 'w', encoding='utf-8') as outf:
        idx = 1
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # skip invalid lines
                continue

            predict_field = rec.get('predict')
            quads = parse_predict_field(predict_field)

            # write backup parsed form
            parsed_backup_lines.append({'index': idx, 'raw': rec, 'quadruples': quads})

            if not quads:
                outf.write(f"{idx},_,_,_,_\n")
                idx += 1
                continue

            for q in quads:
                aspect = q.get('aspect', '_')
                opinion = q.get('opinion', '_')
                category = q.get('category', '_')
                polarity = q.get('polarity', '_')
                # ensure no newlines or commas break CSV; replace newlines with spaces
                aspect = str(aspect).replace('\n', ' ').replace('\r', ' ')
                opinion = str(opinion).replace('\n', ' ').replace('\r', ' ')
                category = str(category).replace('\n', ' ').replace('\r', ' ')
                polarity = str(polarity).replace('\n', ' ').replace('\r', ' ')
                outf.write(f"{idx},{aspect},{opinion},{category},{polarity}\n")

            idx += 1

    # write backup parsed JSONL for debugging
    try:
        with BACKUP_JSONL.open('w', encoding='utf-8') as bf:
            for item in parsed_backup_lines:
                bf.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception:
        pass

    print(f"Wrote CSV -> {OUTPUT_CSV_PATH}")
    print(f"Wrote parsed backup -> {BACKUP_JSONL}")


if __name__ == '__main__':
    main()
