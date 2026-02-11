# 比赛来源
【天池经典打榜赛】赛道六-评论观点挖掘赛
https://tianchi.aliyun.com/competition/entrance/532421

# 比赛成绩

- 第一赛季
    - 成绩：
        0.81（排名第三）
    - 方案：
        MethodsV1
    - 方案分析文章：
        https://tianchi.aliyun.com/forum/post/936514

- 第二赛季
    - 成绩：
        0.81（排名第一）
    - 方案：
        MethodsV2
    - 方案分享文章：
        https://tianchi.aliyun.com/forum/post/949436


# 比赛记录

## sc

### base_sc

也就是直接一条prompt。

分数：0.78

结果："D:\python-ECI\compitation\tianchi\SCoutput\Result2.csv"

base_sc.py



ENABLE_SC     = True    # 是否启用 Self-Consistency

SC_SAMPLES     = 5     # SC 采样次数（建议3-5次）

SC_TEMPERATURES  = [0.3, 0.5, 0.7, 1.0, 1.3]  # SC 使用的不同温度参数

SC_VOTING_METHOD  = "majority"  # 投票方法：majority（多数投票）或 weighted（加权投票）

SC_MIN_CONSENSUS  = 3  

**分析**

估计是温度调的不够好。和一次的分数没区别。

### clear_sc

通过clear框架提示词

比0.78低

结果："D:\python-ECI\compitation\tianchi\SCoutput\Result1.csv"





## 微调

### base微调

提交结果



### 推理cot微调

train数据构建

D:\python-ECI\compitation\tianchi\train_data_enhancement.py

结果

D:\python-ECI\compitation\tianchi\cot_train_output\enhanced_train_data.jsonl

注意，这里对于train的数据推理链生成缺少一些数据，因为断网。所以需要检验并增加。


## 通过base prompt ，然后使用lora微调的364模型
0.7556
微调设置为0.3

D:\python-ECI\compitation\tianchi\choose\tianchi\output\Result_step_364.csv

分析： 估计是温度的影响，毕竟是微调了，温度太大就类似于直接调原生模型的api，没有使用的微调。所以，还是得0.0的温度


## reason cot 微调后调用

使用cot的数据训练，推理结果更低

分析：使用的模型不是微调后的完整模型，然后prompt不够好，不太符合训练阶段生成结构。
所以，需要测试温度为0和0.1等

0.6568



断网记录：
[1768/2237] 处理 ID=1768
评论: 物流给力，也很喜欢这牌子面膜，化妆品也是这牌子，值得信赖
[WARN] review_id=1768 处理失败: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1129)')))
预测结果: 0 个四元组

[1769/2237] 处理 ID=1769
评论: 还好吧，主要便宜，物流贼快
[WARN] review_id=1769 处理失败: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1129)')))
预测结果: 0 个四元组


# 多阶段的结果hi\choose\tianchi\predict_acos_two_stage.py

D:\python-ECI\compitation\tianc
很戏剧，从阶段一到阶段四，结果在逐步下降

日期:2025-09-15 14:51:49

分数:0.7628

日期:2025-09-15 14:48:46

分数:0.7603

日期:2025-09-15 14:39:27

分数:0.7494

日期:2025-09-15 09:37:51

分数:0.7249

分析结论：要想提升结果，只能从该模型，从训练入手了


# 改进
从训练入手，也就是数据增强
参考
1.
https://aclanthology.org/2024.acl-long.26.pdf

2.
https://aclanthology.org/2023.eacl-main.145.pdf
https://github.com/YoumiMa/dreeam   


# baselines
来自于钉钉群分享

https://github.com/eguilg/OpinioNet/tree/master




对于现在的代码D:\python-ECI\competitions\tianchi\choose\predict_acos_in_code copy.py

我需要为其添加**参考示例**：
{some_example}，也就是代码的232行

如何构造：
更加test样本在test中选择相似样本，

样本示例：

评论文本: 很好，遮暇功能差一些，总体还不错"
{\"quadruples\":[{\"aspect\":\"_\",\"opinion\":\"很好\",\"category\":\"整体\",\"polarity\":\"正面\"},{\"aspect\":\"遮暇功能\",\"opinion\":\"差一些\",\"category\":\"功效\",\"polarity\":\"负面\"},{\"aspect\":\"_\",\"opinion\":\"还不错\",\"category\":\"整体\",\"polarity\":\"正面\"}]}

评论文本: 包装太随便了，连个包装盒都没有，第一感觉很不好"
{\"quadruples\":[{\"aspect\":\"包装\",\"opinion\":\"太随便了\",\"category\":\"包装\",\"polarity\":\"负面\"},{\"aspect\":\"包装盒\",\"opinion\":\"没有\",\"category\":\"包装\",\"polarity\":\"负面\"},{\"aspect\":\"_\",\"opinion\":\"很不好\",\"category\":\"整体\",\"polarity\":\"负面\"}]}

对于如何构造这个格式，可以参考代码D:\python-ECI\competitions\tianchi\choose\tianchi\convert_absa_to_jsonl.py

要进行相似度的数据：
每条数据的列 Reviews进行相似度计算

检索：分两阶段

阶段一：
对于样本的检索，使用bm25算法检索

阶段二：
对于相似度的检索，使用reranker模型进行相似度计算
相似比较模型使用：
import requests

url = "https://api.siliconflow.cn/v1/rerank"

payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Apple",
    "documents": ["apple", "banana", "fruit", "vegetable"]
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())

选择前5个最相似的train样本作为样本，放在代码的{some_example}



# 1226

## 1

当前脚本testv3.py中，使用的模型MODEL_NAME        = "ft:LoRA/Qwen/Qwen2.5-32B-Instruct:clzc5301v000e12rx905dx6mo:qwen-32b:buzvweycucvunqozahym-ckpt_step_182" ，是通过lora微调的Qwen2.5-7B-Instruct，对于这个任务，我需要进行改进，模型已经固定了。

基于当前的方案testv3.py，还可以有哪些更好的改进方案，你进行深度的分析。除了给你的文件，不需要读取别的文件数据。

给你的des.md是对这个任务的介绍


## 1229

对于testv3.py的方案，最后在输出的时候，只是一次的输出，直接得到结果，我希望模型有相关的推理过程，也就是think-step-by-step，这样才能更好的理解模型的推理过程。从而提升模型的预测效果。

另外，对于检索相似案例部分，为了得到更好的检索结果，我希望通过混合检索来实现好的检索效果，比如使用bm25算法检索，然后使用reranker模型进行相似度计算。对于embedding，使用模型Pro/BAAI/bge-m3。
请你给出具体的改进方案和代码实现。






