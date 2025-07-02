#!/usr/bin/env python3
"""
演示如何禁用 sentence-transformers 警告的脚本
"""

import os

# 方法1: 通过环境变量禁用sentence-transformers
os.environ['USE_SENTENCE_TRANSFORMERS'] = 'false'

# 现在导入utils - 不会显示警告
from utils.utils import get_text_embedding, get_comprehensive_similarity

# 测试嵌入功能
text1 = "This is a sample text for testing."
text2 = "This is another sample text for comparison."

print("🧪 测试改进的文本嵌入系统")
print("=" * 50)

# 测试嵌入
embedding1 = get_text_embedding(text1)
embedding2 = get_text_embedding(text2)

print(f"文本1: {text1}")
print(f"嵌入维度: {len(embedding1)}")
print(f"嵌入向量示例: {embedding1[:5]}...")

print(f"\n文本2: {text2}")
print(f"嵌入维度: {len(embedding2)}")
print(f"嵌入向量示例: {embedding2[:5]}...")

# 测试相似度
similarity = get_comprehensive_similarity(text1, text2)
print(f"\n综合相似度: {similarity:.3f}")

print("\n✅ 测试完成！使用改进的简单嵌入方法正常工作。")
