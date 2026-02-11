#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25 检索器：从训练集中检索与测试样本相似的示例

使用方法：
1. 初始化检索器并构建索引
2. 对每个测试样本检索最相似的训练样本
3. 使用reranker模型进行二次排序
"""

import csv
import json
import jieba
import math
import requests
from typing import List, Dict, Tuple
from collections import defaultdict, Counter


class BM25Retriever:
    """基于BM25算法的文本检索器"""
    
    def __init__(self, k1=1.5, b=0.75):
        """
        初始化BM25检索器
        
        Args:
            k1: BM25参数，控制词频饱和度
            b: BM25参数，控制文档长度归一化
        """
        self.k1 = k1
        self.b = b
        self.corpus = []  # 存储所有文档
        self.corpus_ids = []  # 存储文档ID
        self.corpus_labels = []  # 存储标签信息
        self.tokenized_corpus = []  # 分词后的语料库
        self.doc_freqs = Counter()  # 文档频率
        self.idf = {}  # IDF值
        self.doc_len = []  # 文档长度
        self.avgdl = 0  # 平均文档长度
        
    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        return list(jieba.cut(text))
    
    def build_index(self, train_reviews_path: str, train_labels_path: str, encoding="utf-8-sig"):
        """
        构建BM25索引
        
        Args:
            train_reviews_path: 训练集评论文件路径
            train_labels_path: 训练集标签文件路径
            encoding: 文件编码
        """
        # 读取训练集评论
        reviews = {}
        with open(train_reviews_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get("id")
                if rid is None:
                    continue
                rid = rid.strip()
                reviews[rid] = row.get("Reviews", "").strip()
        
        # 读取训练集标签
        labels_dict = defaultdict(list)
        with open(train_labels_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["id"].strip()
                aspect = row["AspectTerms"].strip() if row["AspectTerms"].strip() else "_"
                opinion = row["OpinionTerms"].strip()
                category = row["Categories"].strip()
                polarity = row["Polarities"].strip()
                
                if not opinion:  # 跳过没有观点词的标签
                    continue
                    
                labels_dict[rid].append({
                    "aspect": aspect,
                    "opinion": opinion,
                    "category": category,
                    "polarity": polarity
                })
        
        # 构建语料库
        self.corpus = []
        self.corpus_ids = []
        self.corpus_labels = []
        
        for rid, review_text in reviews.items():
            if rid in labels_dict and labels_dict[rid]:  # 只保留有标签的样本
                self.corpus.append(review_text)
                self.corpus_ids.append(rid)
                self.corpus_labels.append(labels_dict[rid])
        
        print(f"构建索引：共 {len(self.corpus)} 条训练样本")
        
        # 对语料库进行分词
        self.tokenized_corpus = [self.tokenize(doc) for doc in self.corpus]
        
        # 计算文档长度和平均长度
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # 计算文档频率（包含某个词的文档数）
        self.doc_freqs = Counter()
        for doc in self.tokenized_corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # 计算IDF值
        N = len(self.corpus)
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_scores(self, query: str) -> List[float]:
        """
        计算查询与所有文档的BM25分数
        
        Args:
            query: 查询文本
            
        Returns:
            分数列表，与语料库对应
        """
        query_tokens = self.tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.tokenized_corpus):
            score = 0
            doc_len = self.doc_len[idx]
            
            # 计算查询中每个词的贡献
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # 计算词频
                tf = doc.count(token)
                
                # BM25公式
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        检索最相似的文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            结果列表，每个结果包含：id, text, labels, score
        """
        scores = self.get_scores(query)
        
        # 获取top_k个最高分的索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": self.corpus_ids[idx],
                "text": self.corpus[idx],
                "labels": self.corpus_labels[idx],
                "score": scores[idx]
            })
        
        return results


class RerankerClient:
    """Reranker模型客户端"""
    
    def __init__(self, api_key: str, model: str = "Pro/BAAI/bge-reranker-v2-m3", timeout: int = 60, max_retries: int = 3):
        """
        初始化Reranker客户端
        
        Args:
            api_key: API密钥
            model: 模型名称
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model = model
        self.url = "https://api.siliconflow.cn/v1/rerank"
        self.timeout = timeout
        self.max_retries = max_retries
    
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """
        使用reranker模型对文档重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前n个结果
            
        Returns:
            重排序后的结果，包含index和relevance_score
        """
        if not documents:
            return []
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents))
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        import time
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                return result.get("results", [])
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"Reranker超时，正在重试 ({attempt + 1}/{self.max_retries})...")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                print(f"Reranker调用超时，已重试{self.max_retries}次")
                return []
            except Exception as e:
                print(f"Reranker调用失败: {e}")
                return []


def format_example(review_text: str, labels: List[Dict]) -> str:
    """
    将样本格式化为示例文本
    
    Args:
        review_text: 评论文本
        labels: 标签列表
        
    Returns:
        格式化后的示例文本
    """
    # 构建quadruples数组
    quadruples = []
    for label in labels:
        quadruples.append({
            "aspect": label["aspect"],
            "opinion": label["opinion"],
            "category": label["category"],
            "polarity": label["polarity"]
        })
    
    # 转为JSON字符串
    json_str = json.dumps({"quadruples": quadruples}, ensure_ascii=False)
    
    # 格式化为示例
    example = f'评论文本: {review_text}"\n{json_str}'
    
    return example


def retrieve_similar_examples(
    query_text: str,
    bm25_retriever: BM25Retriever,
    reranker_client: RerankerClient,
    top_k_bm25: int = 20,
    top_k_rerank: int = 5
) -> str:
    """
    检索相似示例并格式化
    
    Args:
        query_text: 查询文本（测试样本）
        bm25_retriever: BM25检索器
        reranker_client: Reranker客户端
        top_k_bm25: BM25检索的top_k
        top_k_rerank: Reranker重排序的top_k
        
    Returns:
        格式化后的示例文本
    """
    # 阶段一：BM25检索
    bm25_results = bm25_retriever.retrieve(query_text, top_k=top_k_bm25)
    
    if not bm25_results:
        return ""
    
    # 阶段二：Reranker重排序
    documents = [r["text"] for r in bm25_results]
    rerank_results = reranker_client.rerank(query_text, documents, top_n=top_k_rerank)
    
    if not rerank_results:
        # 如果rerank失败，使用BM25的前top_k个结果
        rerank_results = [{"index": i} for i in range(min(top_k_rerank, len(bm25_results)))]
    
    # 格式化示例
    examples = []
    for result in rerank_results:
        idx = result["index"]
        if idx < len(bm25_results):
            bm25_result = bm25_results[idx]
            example = format_example(bm25_result["text"], bm25_result["labels"])
            examples.append(example)
    
    # 用换行分隔多个示例
    return "\n\n".join(examples)


if __name__ == "__main__":
    # 测试代码
    import os
    
    # 设置代理
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    
    # 初始化检索器
    retriever = BM25Retriever()
    
    # 构建索引
    train_reviews = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_reviews.csv"
    train_labels = "D:/python-ECI/competitions/tianchi/data/TRAIN/Train_labels.csv"
    retriever.build_index(train_reviews, train_labels)
    
    # 初始化Reranker
    api_key = "sk-dxfzwvjzdjtcxruqduruesrxpjlgtwvak"
    reranker = RerankerClient(api_key)
    
    # 测试查询
    test_query = "物流很快，包装也不错，用着挺好的"
    examples = retrieve_similar_examples(test_query, retriever, reranker)
    
    print("检索到的相似示例：")
    print(examples)
