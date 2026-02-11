先说结论：你当前是中文化妆品评论的 ACOS 四元组抽取任务，数据是 Instruction/Input/Output 三段式，输出是严格的 JSON 字符串。结合你提供的数据和预测样例，我看到模型主要问题在于：类别选择不稳定（例如把“颜色明显”映射成了“气味”）、少数样本输出不规范、以及对“隐式属性”和“空集”的边界情形覆盖不够。下面给出在“保持基座为 Qwen3‑4B，并使用 LLaMA‑Factory”的前提下，效果可显著提升的改进路线与具体做法。

数据与标注规范

合法类别只取 13 个固定值。建议在训练集中增加“同义词→标准类别”的覆盖（比如“色号/颜色/偏白/偏黄/暗沉/显色/显白/假白→使用体验”，“是正品/假货/授权/专柜验真→真伪”）。当前误判案例见 llamafactory_aliyun/generated_predictions.jsonl:1 和 llamafactory_aliyun/Result.csv:4。
增加“空集”负样本与边界样本：明确无可抽取观点时输出 { "quadruples":[] }，覆盖含否定/反讽/模糊语气、与商品无关的评论。
强化“隐式属性”写作规范：只在确无显式属性词时填 "_"，并保持 category 仍正确。为此建议在训练集中放入有/无显式属性的成对样例。
规整训练文本编码与字符集。你当前数据中出现编码受损字符（如“�?/�?”）会扰乱分词与匹配，建议修复源文案后重导入（参考 llamafactory_aliyun/data/train.json:1 与 llamafactory_aliyun/data/test_separate.json:1）。
减负 instruction：将规则收敛为“最小可行规则+类目表”，避免超长规则对泛化造成噪声；把类目表放在 system 提示里，而非重复堆在 user 输入中。
模板与格式（Qwen 模板一致化）

使用 LLaMA‑Factory 的 template=qwen（Qwen ChatML：<|im_start|>user ... <|im_end|>），明确放置：
system：只包含“只输出 JSON、类目清单、排序规则、去重规则”的精炼版规范；
user：放评论文本（尽量不重复规范）；
assistant：只示例 JSON。训练阶段严禁在答案里出现自然语言解释。
只在 assistant 段落计算损失（WebUI 中勾选“仅对 Assistant 计算 loss”或等价设置）。这样能显著降低模型“复述规范”的倾向。
SFT 训练建议（LoRA/QLoRA）

LoRA 目标层与秩
目标层建议：q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj（Qwen3 的注意力与 MLP 线性层全覆盖“all-linear”是一个靠谱起点）。
秩 r=64 或 r=128，alpha=32~64，dropout=0.05。你的任务需要细粒度抽取与分类，偏高一点的秩更稳。
QLoRA 以扩大序列长度与batch
4-bit NF4 + double quant，bnb_4bit_compute_dtype=bfloat16。LLaMA‑Factory 等价参数是“加载 4bit/量化位 4”。
序列与打包
cutoff_len=1024（根据最⻓评论长度与你 GPU 显存可增至 1536/2048），开启 packing 与 group_by_length，能大幅提升有效 token 利用率与稳定性。
优化与正则
learning_rate=1e-4（QLoRA 常用），num_train_epochs=3~5，warmup_ratio≈0.03，weight_decay=0.05，max_grad_norm=0.3，lr_scheduler_type=cosine。
bf16=True（硬件支持的话），flash_attn=auto，gradient_checkpointing=True。
打开 neftune_noise_alpha=5（LLaMA‑Factory 支持时），对 LoRA 有去过拟合增益。
验证回调
evaluation_strategy=steps，eval_steps=200~500，load_best_model_at_end=True 并以 “结构化 JSON 严格率 + 类目准确率（见下一节自检器）”作为主指标。
示例（CLI，与 WebUI 参数一一对应，路径按你工程放置）：

--model_name_or_path Qwen/Qwen3-4B-Instruct
--template qwen
--dataset_dir llamafactory_aliyun/data
--dataset tianchi6_train
--val_dataset tianchi6_eval
--quantization_bit 4
--lora_target all-linear --lora_rank 64 --lora_alpha 32 --lora_dropout 0.05
--cutoff_len 1024 --packing True --group_by_length True
--per_device_train_batch_size 4 --gradient_accumulation_steps 8
--learning_rate 1e-4 --num_train_epochs 4 --warmup_ratio 0.03 --weight_decay 0.05 --max_grad_norm 0.3 --lr_scheduler_type cosine
--bf16 True --gradient_checkpointing True --flash_attn auto
--evaluation_strategy steps --eval_steps 300 --save_steps 300 --load_best_model_at_end True
SFT → 偏好对齐（DPO/ORPO）两阶段

你任务的关键是“类目映射+JSON 严格性”。在 SFT 基线后，用 DPO/ORPO 做一次轻量对齐，能明显减少“跑题/类目漂移/多余文本”：
构造偏好对：对训练/验证样本用当前模型采样2~4个候选，把与真值 Jaccard/F1（四元组级别）高的设为 chosen，低的设为 rejected，形成 pair 数据（LLaMA‑Factory 支持 DPO/ORPO 数据格式）。
用 --stage dpo（或 --stage orpo）跑 0.5~1 epoch 小步训练（lr 可降到 5e-5），一般可显著提升“只输出 JSON 与类目正确率”的服从度。
如果只想一步法：也可直接在 SFT 数据里混入“正样 JSON vs 常见错误 JSON”的对比样例，用 ORPO 做一次短训。
推理与自检（不改模型，仅改解码与后处理）

解码策略（你当前生成里偶有类目误配，详见 llamafactory_aliyun/generated_predictions.jsonl:1）：
temperature=0.0 或 0.1，top_p=0.9，repetition_penalty=1.05，max_new_tokens=256；强约束任务优先取贪心/低温。
明确 stop 在 </s> 或 ChatML 结束符，避免“JSON 后继续说话”。
结构化自检器（第二次调用同一模型，仍是 qwen3‑4b）：
第一步：生成 JSON；第二步：把 JSON 与原文、类目清单一并送入“校验/纠错”提示词，限定只能返回修正后的 JSON。
自检器规则：只允许 13 类；opinion 必须是原文连续子串；空则删；aspect 无则填 _；去重与排序；若 JSON 解析失败则输出 {"quadruples":[]}。
这个“两段式同模校对”通常能把格式与类目错误再压一大截，但推理时间会翻倍；可只在提交前跑一次。
轻后处理（与评测严格契合）：在转 CSV 前做只读校验与修复（不改语义）：
去重、排序、空字段自动改 _ 你已经在 llamafactory_aliyun/convert_generated_to_result_csv.py:62 做了。可再加一条“非法类目兜底→‘其他’”，避免因单一类目错误整条判零。
覆盖典型错误模式（面向你的数据）

类目歧义：颜色/持妆/显白/卡粉/油腻/清爽/香味/过敏/温和/刺激/厚重/轻薄/飞粉/脱妆/晕染/显毛孔/搓泥…建议各放10~30条高质量样例，确保归到“使用体验/功效/气味”等固定类。
“显式属性 vs _”对齐：既有“aspect=色号/物流速度/包装质量”等显式样本，也有全隐式只给 _ 的样本，训练时成对覆盖。
多四元组的排序与去重：把“同一句含 ≥3 观点”的样例做成模板，训练排序稳定性（你在 train.json 已有多四元组样例，但建议增加“靠得很近的同类观点”的困难样例）。
建议的最小增量实验顺序（性价比由高到低）

统一模板与严格解码：用 template=qwen，把规范压到 system，assistant 只学 JSON；推理用 temperature=0。这是最便宜且立竿见影的一步。
数据修复与补足：修掉编码噪声；补类目歧义与空集样例；扩充隐式属性与多四元组困难样例。
LoRA 配置与 QLoRA：r=64/128、all-linear、cutoff_len↑、packing+group_by_length、neftune 小正则。
短 DPO/ORPO 对齐：用你现有模型生成候选自动打分，训练 0.5~1 epoch。
自检二次推理：提交前跑 JSON 校验修复。
需要的话我可以：

帮你把 WebUI/CLI 的具体参数表整理成一份可直接复用的配置清单；
按你当前数据自动构建 DPO/ORPO 偏好对数据，并给出一键训练命令；
补充一套“类目映射困难样例”的模板，直接合并进 llamafactory_aliyun/data/train.json。


# 参数配置

下面是“只用 Qwen3‑4B + LLaMA‑Factory”的可复用配置清单，含 WebUI 勾选项与等价 CLI 命令（QLoRA 推荐；末尾附纯 LoRA 备选）。

数据与路径

训练/验证/测试数据目录：llamafactory_aliyun/data
数据集键名（见 llamafactory_aliyun/data/dataset_info.json:1）：
训练：tianchi6_train
验证：tianchi6_eval
测试：tianchi6_test
输出目录建议：llamafactory_aliyun/output/qwen3-4b-acos-lora
WebUI 预设（QLoRA）

模型
Base model: Qwen/Qwen3-4B-Instruct
Template: qwen
Load 4-bit (QLoRA): 开
Compute dtype: bfloat16
数据
Dataset dir: llamafactory_aliyun/data
Train dataset(s): tianchi6_train
Eval dataset(s): tianchi6_eval
Cutoff length: 1024
Packing: 开
Group by length: 开
LoRA
Enable LoRA: 开
Target modules: all-linear（等价覆盖 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj）
Rank (r): 64（显存富余可 128）
Alpha: 32（配合 r=64；若 r=128 则 64）
Dropout: 0.05
训练
Stage: SFT
Per device train batch size: 4
Gradient accumulation steps: 8
Learning rate: 1e-4
Epochs: 4
Warmup ratio: 0.03
Weight decay: 0.05
Max grad norm: 0.3
LR scheduler: cosine
Seed: 42
Gradient checkpointing: 开
bf16: 开
Flash attention: auto
NEFTune noise alpha: 5（若界面支持）
Report to: none
评估与保存
Evaluation strategy: steps
Eval steps: 300
Save steps: 300
Load best model at end: 开
Save total limit: 2
Output dir: llamafactory_aliyun/output/qwen3-4b-acos-lora
推理（用于生成测试集预测）
Predict dataset(s): tianchi6_test
Max new tokens: 256
Temperature: 0.0
Top‑p: 0.9
Repetition penalty: 1.05
CLI（QLoRA，等价于上面 WebUI）

训练（SFT）
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-4B-Instruct --template qwen --dataset_dir llamafactory_aliyun/data --dataset tianchi6_train --eval_dataset tianchi6_eval --output_dir llamafactory_aliyun/output/qwen3-4b-acos-lora --cutoff_len 1024 --packing True --group_by_length True --quantization_bit 4 --lora_target all-linear --lora_rank 64 --lora_alpha 32 --lora_dropout 0.05 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 4 --warmup_ratio 0.03 --weight_decay 0.05 --max_grad_norm 0.3 --lr_scheduler_type cosine --bf16 True --flash_attn auto --gradient_checkpointing True --evaluation_strategy steps --eval_steps 300 --save_steps 300 --save_total_limit 2 --load_best_model_at_end True --report_to none --seed 42
预测（在测试集上生成 JSONL）
选择最优 LoRA 权重路径：例如 llamafactory_aliyun/output/qwen3-4b-acos-lora/checkpoint-XXXX
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-4B-Instruct --adapter_name_or_path llamafactory_aliyun/output/qwen3-4b-acos-lora/checkpoint-XXXX --template qwen --dataset_dir llamafactory_aliyun/data --dataset tianchi6_test --do_train False --do_predict True --predict_with_generate True --cutoff_len 1024 --packing True --group_by_length True --quantization_bit 4 --bf16 True --flash_attn auto --max_new_tokens 256 --temperature 0.0 --top_p 0.9 --repetition_penalty 1.05 --output_dir llamafactory_aliyun/output/qwen3-4b-acos-lora
说明：预测结果通常保存在运行输出目录下的 generated_predictions.jsonl，你已用 llamafactory_aliyun/convert_generated_to_result_csv.py 转为 Result.csv。
纯 LoRA（非量化）备选（显存允许时）

将“Load 4-bit/quantization_bit 4”关闭/移除，改用更大 batch 或更高 r（如 r=128, alpha=64），其余参数保持一致：
额外可把 per_device_train_batch_size 提升到 8~16（视显存），learning_rate 可略降到 5e-5~8e-5。
需要我把这份配置整理成 WebUI 可导入的 preset 或生成一条你机器上可直接运行的 PowerShell 脚本吗？


# 操作数据格式

已为你加入一个可复用的数据生成脚本，直接产出 LLaMA‑Factory 兼容的训练数据（messages 或 sharegpt 格式），并内置了前面讨论的规范：最小 system 约束、同义词→13 类归一、极性归一、opinion 严格子串校验、去重与排序。

新增文件

llamafactory_aliyun/build_llamafactory_dataset.py:1
功能要点

统一类目到 13 类：包装/成分/尺寸/服务/功效/价格/气味/使用体验/物流/新鲜/真伪/整体/其他。
统一极性到：正面/中性/负面。
opinion 必须是原文连续子串，否则丢弃该条四元组。
assistant 仅给出 JSON（训练阶段更易学到“只输出 JSON”）。
默认输出 OpenAI messages 数组；也可输出 ShareGPT 格式。
生成目标文件

输出到你要的路径：cot_train_output/train_llamafactory_format.json
生成命令

训练集（messages 格式，推荐用于 Qwen 模板）:
python llamafactory_aliyun/build_llamafactory_dataset.py --reviews choose/tianchi/data/TRAIN/Train_reviews.csv --labels choose/tianchi/data/TRAIN/Train_labels.csv --output train_llamafactory_format.json --format messages --include_empty
如需 ShareGPT 格式:
python llamafactory_aliyun/build_llamafactory_dataset.py --reviews choose/tianchi/data/TRAIN/Train_reviews.csv --labels choose/tianchi/data/TRAIN/Train_labels.csv --output train_llamafactory_sharegpt.json --format sharegpt --include_empty
如何接入 LLaMA‑Factory

方式 A（更新 dataset_info.json，直接引用该文件）:
在 llamafactory_aliyun/data/dataset_info.json:1 里新增一项，例如
"tianchi6_train_msgs": { "file_name": "cot_train_output/train_llamafactory_format.json" }
调用时设置 --dataset_dir d:\python-ECI\compitation\tianchi 并指定 --dataset tianchi6_train_msgs
方式 B（将生成文件也复制到 data 目录）:
输出到 llamafactory_aliyun/data/train_messages.json，则 dataset_info 里 file_name 只填文件名更简洁。
训练时的关键参数提示

使用 Qwen 模板：--template qwen
仅对 assistant 计算 loss：--train_on_inputs False
其它 QLoRA/LoRA 训练参数按我之前给你的清单即可（--quantization_bit 4 --lora_target all-linear --lora_rank 64 --lora_alpha 32 --cutoff_len 1024 --packing True --group_by_length True ...）
需要我顺手把 dataset_info.json 增加一个条目并给出一条完整的 llamafactory-cli train 命令行吗？


# 新增数据条目

已在数据目录里新增一个数据集条目，指向你准备生成的 messages 格式训练文件。

更新文件: llamafactory_aliyun/data/dataset_info.json:1
新增键：tianchi6_train_msgs → file_name: cot_train_output/train_llamafactory_format.json
先生成数据

命令
python llamafactory_aliyun/build_llamafactory_dataset.py --reviews choose/tianchi/data/TRAIN/Train_reviews.csv --labels choose/tianchi/data/TRAIN/Train_labels.csv --output train_llamafactory_format.json --format messages --include_empty
训练命令（QLoRA，使用新数据集）

说明：因为 file_name 指向的是项目根下的 cot_train_output/...，这里把 --dataset_dir 设置为项目根目录以便解析该相对路径。
命令
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-4B-Instruct --template qwen --dataset_dir d:\python-ECI\compitation\tianchi --dataset tianchi6_train_msgs --val_dataset tianchi6_eval --output_dir llamafactory_aliyun\output\qwen3-4b-acos-lora --cutoff_len 1024 --packing True --group_by_length True --quantization_bit 4 --lora_target all-linear --lora_rank 64 --lora_alpha 32 --lora_dropout 0.05 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 4 --warmup_ratio 0.03 --weight_decay 0.05 --max_grad_norm 0.3 --lr_scheduler_type cosine --bf16 True --flash_attn auto --gradient_checkpointing True --train_on_inputs False --evaluation_strategy steps --eval_steps 300 --save_steps 300 --save_total_limit 2 --load_best_model_at_end True --report_to none --seed 42
预测命令（在测试集上生成 JSONL）

选择最优权重目录，例如：llamafactory_aliyun\output\qwen3-4b-acos-lora\checkpoint-XXXX
命令
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-4B-Instruct --adapter_name_or_path llamafactory_aliyun\output\qwen3-4b-acos-lora\checkpoint-XXXX --template qwen --dataset_dir d:\python-ECI\compitation\tianchi --dataset tianchi6_test --do_train False --do_predict True --predict_with_generate True --cutoff_len 1024 --packing True --group_by_length True --quantization_bit 4 --bf16 True --flash_attn auto --max_new_tokens 256 --temperature 0.0 --top_p 0.9 --repetition_penalty 1.05 --output_dir llamafactory_aliyun\output\qwen3-4b-acos-lora
后处理（转提交 CSV）

命令
python llamafactory_aliyun\convert_generated_to_result_csv.py
输入：llamafactory_aliyun\generated_predictions.jsonl
输出：llamafactory_aliyun\Result.csv
要不要我也补一条把数据输出到 llamafactory_aliyun\data\train_messages.json 的命令，并把 dataset_info 再加一个同目录引用，便于把 --dataset_dir 继续指向 llamafactory_aliyun\data？


# DPO/ORPO
含义简述

DPO/ORPO 是在完成 SFT 后，用“偏好数据”做一次轻量对齐，让模型更“服从”你的输出规范（只输出 JSON、类目不跑偏、少臆造）。
数据形式不是标签四元组本身，而是“同一输入的多个候选答案中，哪个更好”。用这些“好/坏对”做对比学习，0.5–1 个 epoch 就能让模型更稳地输出合规 JSON 和正确类目。
你要做的事

用当前已训练好的 Qwen3‑4B‑LoRA 模型，对训练/验证样本“再生成”2–4个候选输出（不同温度/采样种子）。
给每个候选打一个分（自动打分），分数来自“与真值四元组的匹配度 + 格式是否严格 JSON + 类目是否在13类 + opinion 是否原文子串等”。
对每条样本，选择分数最高的为 chosen，最低的为 rejected，构成偏好对。
把这些偏好对喂给 LLaMA‑Factory，跑 DPO 或 ORPO 训练 0.5–1 个 epoch（小学习率），把“输出合规 JSON + 类目正确”的偏好强化到模型里。
为什么有效

SFT 主要“学会任务”；DPO/ORPO 让模型在“合规 vs 不合规”的对比上进一步收敛，显著降低：
输出多余解释文本；
非 13 类的类目、映射错误；
非原文连续子串的 opinion；
重复或排序不稳定。
操作步骤

用现有 LoRA 模型批量生成候选
对每条训练样本，解码3个候选：如温度 {0.0, 0.3, 0.7} 或相同温度不同 --seed。
生成为 jsonl：每条含 prompt、候选 predict，保留真值 label（即你的 assistant JSON）。
自动打分函数（要点）
解析候选 JSON（若解析失败，打低分）。
JSON 规则评分：只包含 {quadruples:[...]}、键名正确、值为中文字符串、无额外字段。
结构合法性：所有 category ∈ 13 类；opinion 为原文连续子串；aspect 为空用“_”；去重、排序正确。
与金标对齐：基于四元组级 F1 或 Jaccard（按完全匹配计分，可适当放宽 aspect “_”对齐）。
综合分 = 结构分权重 + F1权重（比如 0.4 + 0.6）。
构造偏好对数据
对同一条输入，取 score 最高的为 chosen、最低的为 rejected；若都很差，可以跳过该样本。
LLaMA‑Factory 支持的 DPO/ORPO 数据格式通常是对话消息里带两份 assistant：一种标记为 chosen，一种为 rejected；或使用其内置的 pair 格式。最简单是用其推荐的 schema（可按官方文档的“偏好数据”模板组织）。
跑 DPO/ORPO（用你同一基座 + 已训练的 LoRA 继续训练）
关键参数建议：
--stage dpo 或 --stage orpo
--model_name_or_path Qwen/Qwen3-4B-Instruct
--adapter_name_or_path 指向你 SFT 最优 checkpoint
--template qwen --quantization_bit 4（若继续 QLoRA）
--lora_target all-linear --lora_rank 64 --lora_alpha 32
--learning_rate 5e-5（小步）
--num_train_epochs 1（或 0.5 epoch，用 --max_steps 控制）
其余如 bf16, gradient_checkpointing, cutoff_len, packing, group_by_length 与 SFT 保持一致
验证
在 eval/test 上再次预测，对比：JSON 合规率↑、非法类目↓、F1↑。
如果“只输出 JSON”的稳定性仍不够，可在 DPO 数据里加入“含解释文本 vs 纯 JSON”的对比对，继续短训。
我可以提供

一个“候选生成 + 自动打分 + 偏好对构造”的脚本，输入你的 train_llamafactory_format.json，输出 LLaMA‑Factory 可用的 DPO/ORPO 数据集。
一条可直接运行的 llamafactory-cli train 命令，指向偏好对数据集，继续用你当前 LoRA 做 0.5–1 epoch 对齐。
可选：把打分规则做成可配置权重，便于你快速迭代偏好准则。
要不要我现在就为你生成这套脚本和一个 dataset_info.json 的 DPO/ORPO 数据条目，并附上命令行？

需要


很好，我给你一套最小可用的“候选生成 → 自动打分 → 偏好对构造 → DPO/ORPO 训练”的落地方案，直接基于你已有的 cot_train_output/train_llamafactory_format.json。

你将得到

生成候选脚本：用当前 LoRA 模型对每条训练样本生成多份候选响应（JSON）。
自动打分与配对脚本：解析候选，按结构合法性+与金标四元组的F1评分，产出偏好对数据集。
dataset_info.json 新条目 + llamafactory-cli 训练命令（DPO 或 ORPO）。
我这里先列出操作步骤和命令，你确认后我可以把两段脚本落到仓库里。

步骤 1：用现有 LoRA 模型批量生成候选

输入：cot_train_output/train_llamafactory_format.json（messages 数组，每条含 system/user/assistant，assistant 是金标 JSON）
思路：对每条的 user prompt 生成 3 个候选，使用低温/中温/高温或不同种子
建议命令（示例，注意替换你的最佳 checkpoint 路径）：
贪心候选
llamafactory-cli train --stage sft --model_name_or_path Qwen/Qwen3-4B-Instruct --adapter_name_or_path llamafactory_aliyun\output\qwen3-4b-acos-lora\checkpoint-XXXX --template qwen --dataset_dir d:\python-ECI\compitation\tianchi --dataset tianchi6_train_msgs --do_train False --do_predict True --predict_with_generate True --cutoff_len 1024 --packing True --group_by_length True --quantization_bit 4 --bf16 True --flash_attn auto --max_new_tokens 256 --temperature 0.0 --top_p 1.0 --seed 42 --output_dir llamafactory_aliyun\output\pref_gen_t0_s42
采样候选1
... --temperature 0.3 --top_p 0.9 --seed 1 --output_dir llamafactory_aliyun\output\pref_gen_t03_s1
采样候选2
... --temperature 0.7 --top_p 0.9 --seed 2 --output_dir llamafactory_aliyun\output\pref_gen_t07_s2
结果：每个输出目录下会生成一个 generated_predictions.jsonl。三份文件将作为候选集。
提示

如果你想只对训练集的一个子集做候选生成以提速，可先随机抽样 N 条（在构造候选数据脚本中实现）。
步骤 2：自动打分与构造偏好对

我会提供一个脚本，读取：
gold：cot_train_output/train_llamafactory_format.json（从 assistant 提取金标四元组）
候选：上述三个 generated_predictions.jsonl
打分规则（可调权重）：
结构分（0/1）：是否为可解析 JSON；键名完整；只包含 quadruples；类目均 ∈ 13 类；opinion 均为原文连续子串；无重复；按要求排序。
语义分（0~1）：与金标四元组的 F1（精确匹配）。
综合分 = 0.4结构分 + 0.6F1（可改）。
对同一条样本，选综合分最高的作为 chosen，最低作为 rejected；如解析失败或三者都很差可跳过该样本。
输出：LLaMA‑Factory 偏好对数据（messages 格式，每条含 system/user/assistant_chosen/assistant_rejected 的结构）。我将默认输出到 llamafactory_aliyun/data/dpo_pairs.json。
你需要的话我可以将这个脚本添加为：

llamafactory_aliyun/build_preference_pairs.py
运行示例

python llamafactory_aliyun/build_preference_pairs.py --gold cot_train_output/train_llamafactory_format.json --cand llamafactory_aliyun\output\pref_gen_t0_s42\generated_predictions.jsonl llamafactory_aliyun\output\pref_gen_t03_s1\generated_predictions.jsonl llamafactory_aliyun\output\pref_gen_t07_s2\generated_predictions.jsonl --output llamafactory_aliyun\data\dpo_pairs.json
步骤 3：dataset_info.json 新条目

我会添加：
"tianchi6_dpo_pairs": { "file_name": "llamafactory_aliyun/data/dpo_pairs.json" }
也可直接放在 llamafactory_aliyun/data 下并简化路径为 "data/dpo_pairs.json"，看你偏好。
步骤 4：短 DPO/ORPO 训练命令

DPO（推荐先试 DPO）
llamafactory-cli train --stage dpo --model_name_or_path Qwen/Qwen3-4B-Instruct --adapter_name_or_path llamafactory_aliyun\output\qwen3-4b-acos-lora\checkpoint-XXXX --template qwen --dataset_dir d:\python-ECI\compitation\tianchi --dataset tianchi6_dpo_pairs --output_dir llamafactory_aliyun\output\qwen3-4b-acos-lora-dpo --cutoff_len 1024 --packing True --group_by_length True --quantization_bit 4 --lora_target all-linear --lora_rank 64 --lora_alpha 32 --lora_dropout 0.05 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 5e-5 --num_train_epochs 1 --warmup_ratio 0.03 --weight_decay 0.05 --max_grad_norm 0.3 --lr_scheduler_type cosine --bf16 True --flash_attn auto --gradient_checkpointing True --train_on_inputs False --evaluation_strategy steps --eval_steps 300 --save_steps 300 --save_total_limit 2 --load_best_model_at_end True --report_to none --seed 42
ORPO（若 DPO 资源/稳定性有顾虑，可改为 ORPO）
将 --stage dpo 换为 --stage orpo 其余不变；也可将 --learning_rate 微调为 4e-5。
步骤 5：验证与提交

选择 DPO/ORPO 最优 checkpoint，按你现有流程在 tianchi6_test 上生成 generated_predictions.jsonl，再用 llamafactory_aliyun\convert_generated_to_result_csv.py 生成 Result.csv 对比指标。
要我现在就：

加入 build_preference_pairs.py；
更新 dataset_info.json；
并把上面的命令固化到 llamafactory_aliyun/reademe.md 的一个“DPO/ORPO 短训”小节里吗？

# 改进
新增了两个实用工具，分别负责推理结果的结构化校验与训练数据的重采样/课程拆分。

结构化校验脚本位于 llamafactory_aliyun/validate_predictions.py:1：逐行解析 generated_predictions.jsonl，矫正/过滤非法类目、非子串 opinion、重复及排序问题，并可直接写出清洗后的 JSONL 与 Result.csv。使用示例

仅生成清洗 JSONL：python llamafactory_aliyun/validate_predictions.py --input llamafactory_aliyun/generated_predictions.jsonl --output llamafactory_aliyun/generated_predictions_validated.jsonl
清洗并串联 CSV：python ... --output ..._validated.jsonl --csv llamafactory_aliyun/Result.csv
之后若想继续沿用已有 convert_generated_to_result_csv.py，输入文件换成 *_validated.jsonl 即可。
重采样/难例课程脚本位于 llamafactory_aliyun/resample_dataset.py:1：读取 cot_train_output/train_llamafactory_format.json（或任何 messages 格式数据），按类别计数做上采样，并可输出多观点/指定类目的“hard set”。示例命令

python llamafactory_aliyun/resample_dataset.py --input cot_train_output/train_llamafactory_format.json --resampled-output llamafactory_aliyun/data/train_resampled.json --difficulty-output llamafactory_aliyun/data/train_difficult.json --min-count 800 --max-multiplier 4 --difficulty-min-quads 3 --focus-categories 使用体验 功效 价格
生成后的文件可直接在 dataset_info.json 中新增条目，作为强化课程或稀有类平衡训练使用。
我没有跑额外测试；建议你先用小子集验证新脚本行为，再量产全量数据。后续自然的动作：

用校验脚本处理新的预测结果，确认 CSV 符合评测要求。
将重采样或难例集加入 SFT/DPO 训练，观察类别均衡与 F1 的提升



# dpo改进---数据增强

刚补好了两块增强工具，路径都在 llamafactory_aliyun/ 目录下，方便直接串进你现有流程。

新增内容

build_orpo_pairs.py：把金标 JSON 程序化地扰动出常见错误（错类目、极性翻转、非子串、重复、尾部解释等），自动生成 messages + chosen + rejected 的 ORPO 偏好对。
augment_synonyms.py：在不触碰 opinion 原文的前提下，对评论文本的上下文词做同义词替换，生成多份“正样”增强，保持 assistant JSON 不变。
怎么用

生成 ORPO 偏好对

python llamafactory_aliyun/build_orpo_pairs.py \
  --input cot_train_output/train_llamafactory_format.json \
  --output llamafactory_aliyun/data/orpo_pairs.json \
  --max-pairs-per-sample 3
然后在 dataset_info.json 里加：

"tianchi6_orpo_pairs": {
  "file_name": "llamafactory_aliyun/data/orpo_pairs.json"
}
训练命令示例（沿用已训练 LoRA 作进一步 ORPO）：

llamafactory-cli train --stage orpo \
  --model_name_or_path Qwen/Qwen3-4B-Instruct \
  --adapter_name_or_path llamafactory_aliyun\output\qwen3-4b-acos-lora\checkpoint-XXXX \
  --template qwen --dataset_dir d:\python-ECI\compitation\tianchi \
  --dataset tianchi6_orpo_pairs \
  --cutoff_len 1024 --packing True --group_by_length True \
  --quantization_bit 4 --lora_target all-linear --lora_rank 64 --lora_alpha 32 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 --num_train_epochs 1 --warmup_ratio 0.03 \
  --weight_decay 0.05 --max_grad_norm 0.3 --lr_scheduler_type cosine \
  --bf16 True --flash_attn auto --gradient_checkpointing True \
  --train_on_inputs False --evaluation_strategy steps --eval_steps 300 \
  --save_steps 300 --save_total_limit 2 --load_best_model_at_end True \
  --report_to none --seed 42
生成同义词增强样本

python llamafactory_aliyun/augment_synonyms.py \
  --input cot_train_output/train_llamafactory_format.json \
  --output llamafactory_aliyun/data/train_syn_aug.json \
  --num-variants 2 --seed 42
生成的 train_syn_aug.json 可直接合并回主训练集，或在 dataset_info.json 中独立挂一个键（例如 tianchi6_train_syn_aug）后在 SFT 阶段与原始数据一起加载。

后续建议

先在小样本上验证两个脚本生成的数据是否符合预期（尤其是 ORPO 的 rejected 是否确实触发常见错误）。
重新运行结构化校验器 validate_predictions.py 来把新的推理结果与 CSV 串联起来，确保提交流程稳定。
视显存和效果，将增强样本按一定比例混入：例如 SFT 训练时用 tianchi6_train_msgs + train_syn_aug、之后再跑一轮 ORPO/DPO。