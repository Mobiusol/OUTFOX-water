"""嵌入和相似度计算模块"""
import re
import numpy as np
from collections import Counter
from typing import Dict, Optional

from .config import USE_SENTENCE_TRANSFORMERS, DEFAULT_SIMILARITY_WEIGHTS

# 可选导入sentence_transformers
SENTENCE_TRANSFORMERS_AVAILABLE = False
sentence_model = None

if USE_SENTENCE_TRANSFORMERS:
    try:
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        print("✅ sentence-transformers loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: sentence-transformers not available: {e}")
        SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingService:
    """嵌入服务类"""
    
    def __init__(self):
        self.cache = {}
    
    def get_text_embedding(self, text: str, dimension: int = 100) -> np.ndarray:
        """获取文本嵌入"""
        if not text:
            return np.zeros(dimension)

        # 优先使用sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE and sentence_model is not None:
            try:
                embedding = sentence_model.encode([text])[0]
                if len(embedding) != dimension:
                    # 调整维度
                    if len(embedding) > dimension:
                        embedding = embedding[:dimension]
                    else:
                        embedding = np.pad(embedding, (0, dimension - len(embedding)))
                return embedding
            except Exception as e:
                print(f"Warning: sentence-transformers encoding failed: {e}")

        # 回退到简单方法
        return self._simple_embedding(text, dimension)
    
    def _simple_embedding(self, text: str, dimension: int) -> np.ndarray:
        """简单嵌入方法"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return np.zeros(dimension)

        word_counts = Counter(words)
        total_words = len(words)
        vector = np.zeros(dimension)

        for word, count in word_counts.items():
            tf = count / total_words
            for i in range(3):
                hash_val = (hash(word + str(i)) % dimension)
                vector[hash_val] += tf * (1.0 / (i + 1))

        # 添加bigram特征
        for i in range(len(words) - 1):
            bigram = words[i] + "_" + words[i + 1]
            hash_val = hash(bigram) % dimension
            vector[hash_val] += 0.5

        # 归一化
        norm_val = np.linalg.norm(vector)
        if norm_val > 0:
            vector = vector / norm_val

        return vector
    
    def get_cached_embedding(self, text: str, dimension: int = 100) -> np.ndarray:
        """获取缓存的嵌入"""
        if text not in self.cache:
            self.cache[text] = self.get_text_embedding(text, dimension)
        return self.cache[text]


class SimilarityCalculator:
    """相似度计算器"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def get_comprehensive_similarity(
        self, 
        text1: str, 
        text2: str, 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """计算综合相似度"""
        if not text1 or not text2:
            return 0.0

        if weights is None:
            weights = DEFAULT_SIMILARITY_WEIGHTS.copy()

        try:
            # 1. 词汇相似度
            lexical_sim = self._calculate_lexical_similarity(text1, text2)
            
            # 2. 语义相似度
            semantic_sim = self._calculate_semantic_similarity(text1, text2)
            
            # 3. 结构相似度
            structure_sim = self._calculate_structure_similarity(text1, text2)
            
            # 4. 长度相似度
            length_sim = self._calculate_length_similarity(text1, text2)

            # 综合相似度
            comprehensive_sim = (
                weights['lexical'] * lexical_sim +
                weights['semantic'] * semantic_sim +
                weights['structure'] * structure_sim +
                weights['length'] * length_sim
            )

            return min(1.0, max(0.0, comprehensive_sim))

        except Exception as e:
            print(f"Warning: Error in comprehensive similarity calculation: {e}")
            return self._fallback_similarity(text1, text2)
    
    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """计算词汇相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        try:
            emb1 = self.embedding_service.get_cached_embedding(text1[:500])
            emb2 = self.embedding_service.get_cached_embedding(text2[:500])

            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0
            else:
                return max(0.0, dot_product / (norm1 * norm2))
        except Exception:
            return self._calculate_lexical_similarity(text1, text2)
    
    def _calculate_structure_similarity(self, text1: str, text2: str) -> float:
        """计算结构相似度"""
        sentences1 = text1.split('.')
        sentences2 = text2.split('.')

        sent_count_ratio = min(len(sentences1), len(sentences2)) / max(len(sentences1), len(sentences2), 1)

        avg_len1 = sum(len(s.split()) for s in sentences1) / max(len(sentences1), 1)
        avg_len2 = sum(len(s.split()) for s in sentences2) / max(len(sentences2), 1)
        avg_len_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2, 1)

        return (sent_count_ratio + avg_len_ratio) / 2
    
    def _calculate_length_similarity(self, text1: str, text2: str) -> float:
        """计算长度相似度"""
        len1, len2 = len(text1), len(text2)
        return min(len1, len2) / max(len1, len2, 1)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """回退相似度计算"""
        return self._calculate_lexical_similarity(text1, text2)


# 全局实例
embedding_service = EmbeddingService()
similarity_calculator = SimilarityCalculator(embedding_service)

# 兼容性函数
def get_cached_embedding(text: str, dimension: int = 100) -> np.ndarray:
    """兼容性函数"""
    return embedding_service.get_cached_embedding(text, dimension)

def get_comprehensive_similarity(text1: str, text2: str, weights: Optional[Dict[str, float]] = None) -> float:
    """兼容性函数"""
    return similarity_calculator.get_comprehensive_similarity(text1, text2, weights)
