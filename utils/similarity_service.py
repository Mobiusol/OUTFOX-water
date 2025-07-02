"""相似性服务模块"""
import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_service import embedding_service


def find_similar_cases(query_text: str, cases: List[Any], top_k: int = 10) -> List[Any]:
    """
    基于语义相似度查找最相似的案例
    """
    if not cases or len(cases) == 0:
        return []

    # 获取查询文本的嵌入
    query_embed = embedding_service.get_cached_embedding(query_text).reshape(1, -1)

    # 获取所有案例原始文本的嵌入
    corpus_embeds = []
    valid_cases = []

    for case in cases:
        try:
            # 兼容不同的数据结构
            if hasattr(case, 'original'):
                case_text = case.original
            elif isinstance(case, dict):
                case_text = case.get('original', '')
            else:
                continue

            if case_text:
                case_embed = embedding_service.get_cached_embedding(case_text)
                corpus_embeds.append(case_embed)
                valid_cases.append(case)
        except Exception as e:
            print(f"处理案例时出错: {e}")
            continue

    if not corpus_embeds:
        return []

    # 转换为numpy数组
    corpus_embeds = np.array(corpus_embeds)

    # 计算相似度
    similarities = cosine_similarity(query_embed, corpus_embeds)[0]

    # 获取相似度最高的索引
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # 返回相似度最高的案例
    return [valid_cases[i] for i in top_indices if i < len(valid_cases)]
