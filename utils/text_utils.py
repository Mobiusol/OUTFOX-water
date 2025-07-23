import numpy as np
from torch import cosine_similarity
import tiktoken
from typing import List
from config.settings import watermark_config

class TextProcessor:
    """文本处理器 - 专注于文本预处理功能"""
    
    def __init__(self):
        self.qwen_encoding = tiktoken.get_encoding("cl100k_base")
        self.embedding_cache = {}  # 添加缺失的缓存
    
    def truncate_text(self, text: str, truncated_length: int, truncate_type: str = 'token') -> str:
        """截断文本"""
        if truncate_type == 'token':
            truncated_encodes = self.qwen_encoding.encode(text)[:truncated_length]
            return self.qwen_encoding.decode(truncated_encodes)
        elif truncate_type == 'word':
            return ' '.join(text.split(' ')[:truncated_length])
        else:
            raise ValueError(f"不支持的截断类型: {truncate_type}")
    
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        return len(self.qwen_encoding.encode(text))
    
    def remove_common_prefixes(self, text: str) -> str:
        """移除常见前缀"""
        processed_text = text.strip()
        for prefix in watermark_config.common_prefixes:
            if processed_text.startswith(prefix):
                processed_text = processed_text.replace(prefix, "", 1).strip()
                break
        return processed_text
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 基本清理
        text = text.strip()
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def extract_content_words(self, text: str) -> List[str]:
        """提取内容词（去除停用词后的词汇）"""
        import re
        # 简单的停用词列表
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # 提取单词（只保留字母）
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # 过滤停用词和短词
        content_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return content_words
    
    def get_text_embedding(self, text: str, dimension: int = None) -> np.ndarray:
        """获取文本嵌入（简单实现）"""
        # 简单的词汇统计向量
        words = self.extract_content_words(text)
        if not words:
            return np.zeros(dimension or 100)
        
        # 创建简单的特征向量
        vector = np.random.normal(0, 1, dimension or 100)
        return vector / np.linalg.norm(vector)
    
    def get_cached_embedding(self, text: str, dimension: int = None) -> np.ndarray:
        """获取缓存的文本嵌入"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.get_text_embedding(text, dimension)
        return self.embedding_cache[text]
    
    def find_similar_cases(self, query_text: str, cases: List, top_k: int = 10) -> List:
        """基于语义相似度查找最相似的案例"""
        if not cases or len(cases) == 0:
            return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_embed = self.get_cached_embedding(query_text).reshape(1, -1)
        
        corpus_embeds = []
        for case in cases:
            if isinstance(case, dict) and 'original' in case:
                case_embed = self.get_cached_embedding(case['original'])
                corpus_embeds.append(case_embed)
            else:
                corpus_embeds.append(np.zeros_like(query_embed[0]))
        
        if not corpus_embeds:
            return []
            
        corpus_embeds = np.array(corpus_embeds)
        similarities = cosine_similarity(query_embed, corpus_embeds)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [cases[i] for i in top_indices]
