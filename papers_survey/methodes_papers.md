## 任务类别

Aspect-Category-Opinion-Sentiment Quadruples





## cot prompt



https://arxiv.org/pdf/2512.19651

OATS: Opinion Aspect Target Sentiment Quadruple Extraction via Retrieval-Augmented LLMs with Triplet Decomposition



- method

we propose a novel Chainof-Thought (CoT) prompting technique that utilises an intermediate Unified Meaning Representation (UMR) to structure the reasoning process for the ACSA task. 



- review

直接使用prompt进行验证





## prompt 11

https://arxiv.org/pdf/2508.17258



> 设计结构

论文提出一套**零样本、多链式思维（CoT）代理+token 级不确定性聚合**框架，将 ACSA 任务转化为**无需任何标注即可部署**的流水线。关键步骤与对应设计如下：

1. 构建六种 CoT 代理
   - 对 Aspect（A）、Category（C）、Opinion（O）三元素做全排列，得到 6 条不同推理顺序（如 O→C→A、C→A→O 等）。
   - 每条顺序封装成**单条枚举式提示**，强制模型按指定步骤输出中间结果，最后统一生成 Python 列表形式的 `(category, polarity)` 元组。
   - 采用角色扮演系统指令+严格输出格式，降低冗余与解析误差。
2. Token 级置信度提取
   - 采用**贪心解码**保证可复现的 token 条件概率。
   - 对每对 `(category, polarity)`，取组成词的平均对数概率再转线性概率，作为**pair-confidence**。
3. 五种聚合策略（零标注融合）
   1. **Highest probability list**
      选六条代理中**整条列表置信度最高**的那一份作为最终输出。
   2. **Most common list**
      若某份列表在代理间出现次数最多则直接采用；无多数时退回策略 1。
   3. **Highest probability pairs**
      跨代理收集全部候选对，按置信度排序后取前 n 个；n 用数据无关的 α 偏置或均值/最大启发式估计。
   4. **Clustered pairs**
      用 RoBERTa-Sentence 嵌入对所有候选对的 category 做 K-means 聚成 n 簇，每簇取置信度最高者，保证类别不冗余。
   5. **Most confident agent**
      整体验证集上累加代理给出的列表置信度，选**最自信的那个代理**统一输出。
4. 顺序敏感性消融
   - 在 4 个公开数据集（Laptop16、Restaurant16、MAMS、Shoes）上分别运行 6 条顺序，发现**最优顺序随数据集变化**，打破“方面优先”传统假设；大模型与小模型在相同数据集上呈现一致顺序偏好，为后续无标注场景提供经验先验。



> 结论

论文核心主张：**在完全零样本场景下，利用多 CoT 代理 + token 置信度聚合，可显著超越现有零样本基线，且置信度方差可作为无标注情况下的自诊断工具。**



- summary

prompt策略可以进行使用





## 最相关的研究---mvp

https://arxiv.org/pdf/2305.12627

https://github.com/ZubinGou/multi-view-prompting



> 要求

需要深度探索其中的数据集，以及代码，方法研究，搞定这种任务，

同时，该paper是2023的，应该找数据集的榜单，然后看最新的方法，

或则通过该paper的引用paper查看

要深度调研



## prompt方法有显著的提升---有code可以直接使用



https://arxiv.org/pdf/2502.13044



https://github.com/NilsHellwig/llm-prompting-asqp

- 总结

将这些prompt策略使用上，看是否有改进



## 开源的2024的acl，适合考虑直接使用看结果--ASQP，很符合场景



https://github.com/HITSZ-HLT/ST-w-Scorer-ABSA/tree/main

https://arxiv.org/pdf/2406.18078

不是llm的，偏传统方法





## 参考学习，acl2024，还是ASQP方向

https://arxiv.org/pdf/2406.07365

这个方法不是llm，有点难理解，有空再仔细看

https://github.com/byinhao/BvSP



## prompt方法改进

https://arxiv.org/pdf/2310.06502



## some survey

https://arxiv.org/pdf/2512.19537

https://github.com/unikcc/AwesomeEventExtraction