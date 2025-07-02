"""基础工具模块 - 文件操作和数据处理"""
import os
import json
import pickle
import random
import time
import tiktoken
from collections import Counter

# 初始化tokenizer
try:
    qwen_encoding = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"Error loading tiktoken tokenizer: {e}")
    qwen_encoding = None


def load_pkl(path: str):
    """加载pickle文件"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path: str):
    """保存pickle文件"""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def json2dict(path: str) -> dict:
    """加载JSON文件为字典"""
    with open(path, mode="rt", encoding="utf-8") as f:
        return json.load(f)


def truncate_text(text: str, truncated_length: int, truncate_type: str) -> str:
    """截断文本"""
    if truncate_type == 'token' and qwen_encoding:
        truncated_encodes = qwen_encoding.encode(text)[:truncated_length]
        return qwen_encoding.decode(truncated_encodes)
    elif truncate_type == 'word':
        return ' '.join(text.split(' ')[:truncated_length])
    return text


def string2token_nums(string: str) -> int:
    """计算字符串的token数量"""
    if qwen_encoding:
        return len(qwen_encoding.encode(string))
    return len(string.split())  # 简单估算


def make_mixed_data(human_path: str, lm_path: str, ps_path: str):
    """创建混合数据集"""
    random.seed(42)
    humans, lms, pss = load_pkl(human_path), load_pkl(lm_path), load_pkl(ps_path)
    humans_with_label_ps = [(human, '0', ps) for human, ps in zip(humans, pss)]
    lms_with_label_ps = [(lm, '1', ps) for lm, ps in zip(lms, pss)]
    all_with_label_ps = humans_with_label_ps + lms_with_label_ps
    random.shuffle(all_with_label_ps)
    
    data = [t[0] for t in all_with_label_ps]
    labels = [t[1] for t in all_with_label_ps]
    pss = [t[2] for t in all_with_label_ps]
    return (data, labels, pss)
