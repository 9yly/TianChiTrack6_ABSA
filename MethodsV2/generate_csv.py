#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 JSONL 生成 CSV 结果文件"""

import json

jsonl_path = 'raw_predictions_1229v2yml.jsonl'
csv_path = 'Result_1229v2yml.csv'

# 读取JSONL
records = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            records.append(data)

# 按ID排序
records.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else x['id'])

# 生成CSV
lines = []
for rec in records:
    rid = rec['id']
    quads = rec.get('quadruples', [])
    if not quads:
        lines.append(f'{rid},_,_,_,_')
    else:
        for q in quads:
            aspect = q.get('aspect', '_') or '_'
            opinion = q.get('opinion', '')
            category = q.get('category', '')
            polarity = q.get('polarity', '')
            lines.append(f'{rid},{aspect},{opinion},{category},{polarity}')

with open(csv_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'CSV生成完成: {csv_path}')
print(f'总记录数: {len(records)}')
print(f'总行数: {len(lines)}')
