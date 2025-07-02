"""系统配置模块"""
import os

# API配置
DASHSCOPE_API_KEY = "sk-7418f3fcffb64927b6400f213e1d4561"
DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 模型配置
USE_T5 = os.getenv('USE_T5', 'false').lower() == 'true'
USE_SENTENCE_TRANSFORMERS = os.getenv('USE_SENTENCE_TRANSFORMERS', 'true').lower() == 'true'

# 常见前缀配置
COMMON_PREFIXES = [
    "Here's the watermarked text embedding detectable semantic constructs:",
    "Here is the watermarked text:",
    "Watermarked:",
    "Certainly! Here's the watermarked text:",
    "---",
    "Here is the watermarked text embedding with detectable semantic constructs:",
    "Certainly, here's the watermarked version of the provided text:",
    "Certainly! Here's the watermarked version of the text:",
    "Of course. Below is the watermarked version:",
    "Certainly! Here's a watermarked version of your text:",
]

# 文本相似度权重配置
DEFAULT_SIMILARITY_WEIGHTS = {
    'lexical': 0.3,
    'semantic': 0.4,
    'structure': 0.2,
    'length': 0.1
}

# 质量评估配置
QUALITY_CONFIG = {
    'label_weight': 0.8,
    'confidence_weight': 0.2,
    'threshold': 0.6,
    'similarity_threshold': 0.7
}
