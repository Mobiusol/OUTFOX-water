import numpy as np
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from config.settings import similarity_config

class SimilarityCalculatorInterface(ABC):
    """相似度计算器接口"""
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本之间的相似度"""
        pass
    
    @abstractmethod
    def find_similar_texts(self, query: str, corpus: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """在语料库中查找最相似的文本"""
        pass

class BERTSemanticSimilarity(SimilarityCalculatorInterface):
    """基于BERT的语义相似度计算器 - 用于质量评估"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._embedding_cache = {}
        print(f"BERT模型 {model_name} 加载完成")
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """获取文本的BERT嵌入"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # 截断过长文本
        if len(text) > 500:
            text = text[:500]
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]token的嵌入作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        self._embedding_cache[text] = embedding
        return embedding
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度（0-1之间）"""
        emb1 = self._get_bert_embedding(text1).reshape(1, -1)
        emb2 = self._get_bert_embedding(text2).reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0, 0]
        # 将cosine相似度从[-1,1]转换到[0,1]
        return (similarity + 1) / 2
    
    def find_similar_texts(self, query: str, corpus: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """查找语义相似的文本"""
        query_emb = self._get_bert_embedding(query).reshape(1, -1)
        
        similarities = []
        for i, text in enumerate(corpus):
            text_emb = self._get_bert_embedding(text).reshape(1, -1)
            sim = cosine_similarity(query_emb, text_emb)[0, 0]
            similarities.append((i, (sim + 1) / 2))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class TFIDFRetrievalSimilarity(SimilarityCalculatorInterface):
    """基于TF-IDF的检索相似度计算器 - 用于案例检索"""
    
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            lowercase=True
        )
        self.corpus_vectors = None
        self.corpus_texts = None
        print(f"TF-IDF向量化器初始化完成，最大特征数: {max_features}")
    
    def fit_corpus(self, corpus: List[str]):
        """在语料库上训练TF-IDF向量化器"""
        self.corpus_texts = corpus
        if corpus:
            self.corpus_vectors = self.vectorizer.fit_transform(corpus)
            print(f"TF-IDF向量化器已在{len(corpus)}个文档上训练")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算TF-IDF相似度"""
        try:
            vectors = self.vectorizer.transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]
            return max(0, similarity)  # 确保非负
        except:
            return 0.0
    
    def find_similar_texts(self, query: str, corpus: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """快速检索相似文本"""
        if not corpus:
            return []
        
        # 如果语料库发生变化，重新训练
        if self.corpus_texts != corpus:
            self.fit_corpus(corpus)
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.corpus_vectors)[0]
            
            # 获取top_k个最相似的索引和分数
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [(idx, max(0, similarities[idx])) for idx in top_indices]
            
            return results
        except:
            return [(i, 0.0) for i in range(min(top_k, len(corpus)))]

class HybridSimilarityManager:
    """混合相似度管理器 - 统一管理不同用途的相似度计算"""
    
    def __init__(self):
        # 语义相似度 - 用于质量评估
        self.semantic_calculator = BERTSemanticSimilarity(
            model_name=similarity_config.bert_model_name
        )
        
        # 检索相似度 - 用于案例检索
        self.retrieval_calculator = TFIDFRetrievalSimilarity(
            max_features=similarity_config.tfidf_max_features
        )
        
        print("混合相似度管理器初始化完成")
    
    def calculate_semantic_similarity(self, original: str, watermarked: str) -> float:
        """计算语义相似度 - 用于评估水印质量"""
        return self.semantic_calculator.calculate_similarity(original, watermarked)
    
    def find_similar_cases(self, query_text: str, cases: List[Dict], top_k: int = 10) -> List[Dict]:
        """基于TF-IDF快速检索相似案例"""
        if not cases:
            return []
        
        # 提取原始文本用于检索
        corpus = [case.get('original', '') for case in cases]
        
        # 使用TF-IDF进行快速检索
        similar_indices = self.retrieval_calculator.find_similar_texts(
            query_text, corpus, top_k
        )
        
        # 返回相似的案例
        return [cases[idx] for idx, score in similar_indices if score > similarity_config.retrieval_threshold]
    
    def evaluate_watermark_quality(self, original: str, watermarked: str) -> Dict[str, float]:
        """综合评估水印质量"""
        # 语义保持度（BERT相似度）
        semantic_similarity = self.calculate_semantic_similarity(original, watermarked)
        
        # 文本变化度（长度比例等）
        length_ratio = min(len(watermarked), len(original)) / max(len(watermarked), len(original))
        
        # 词汇重叠度（简单统计）
        original_words = set(original.lower().split())
        watermarked_words = set(watermarked.lower().split())
        word_overlap = len(original_words & watermarked_words) / len(original_words | watermarked_words) if original_words | watermarked_words else 0
        
        return {
            'semantic_similarity': semantic_similarity,
            'length_ratio': length_ratio,
            'word_overlap': word_overlap,
            'overall_quality': (
                similarity_config.semantic_weight * semantic_similarity +
                similarity_config.length_weight * length_ratio +
                similarity_config.overlap_weight * word_overlap
            )
        }
