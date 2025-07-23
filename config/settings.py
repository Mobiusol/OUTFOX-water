import os
from dataclasses import dataclass
from typing import List

@dataclass
class APIConfig:
    """API配置"""
    dashscope_api_key: str = " "#填自己的
    dashscope_endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    model_name: str = "qwen-turbo"
    default_temperature: float = 1.2
    default_top_p: float = 0.9
    default_max_tokens: int = 2048
    max_retries: int = 3
    retry_delay: float = 2.0

@dataclass
class WatermarkConfig:
    """水印相关配置"""
    common_prefixes: List[str] = None
    max_examples_size: int = 200  # 保留的最大示例数量
    max_prompt_examples: int = 10  # prompt中使用的最大示例数
    quality_threshold: float = 0.6
    truncate_type: str = 'token'
    embedding_dimension: int = 100
    
    def __post_init__(self):
        if self.common_prefixes is None:
            self.common_prefixes = [
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
                "Watermarked text:",
                "Watermarked Text:",
                "Watermarked version:",
                "Watermarked Version:",
                "**Watermarked**:",
                "**Watermarked version:**",
                "**Watermarked Version:**",
                "**Watermarked text:**",
                "**Watermarked Text:**",
            ]

@dataclass
class SystemConfig:
    """系统配置 - 统一管理所有系统级参数"""
    # 路径配置
    data_dir: str = "../../data/"
    output_dir: str = "../results/"
    
    # 训练配置
    max_epochs: int = 3  # 默认最大训练轮次
    random_seed: int = 42
    
    # 运行时配置
    checkpoint_interval: int = 50
    display_interval_percent: float = 0.05
    test_original_interval: int = 5
    
    # 数据集配置
    max_train_samples: int = 200  # 训练样本数量限制
    
    # 模型选择
    available_models: List[str] = None
    default_model: str = "qwen-turbo"
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = ['qwen-turbo', 'gpt-3.5-turbo']

@dataclass
class SimilarityConfig:
    """相似度计算配置"""
    bert_model_name: str = "bert-base-uncased"
    semantic_similarity_threshold: float = 0.8
    tfidf_max_features: int = 10000
    retrieval_threshold: float = 0.1
    semantic_weight: float = 0.6
    length_weight: float = 0.2
    overlap_weight: float = 0.2
    enable_embedding_cache: bool = True
    max_cache_size: int = 10000

# 全局配置实例
api_config = APIConfig()
watermark_config = WatermarkConfig()
system_config = SystemConfig()
similarity_config = SimilarityConfig()

def update_config_from_args(args):
    """根据命令行参数更新配置"""
    # 更新系统配置
    if hasattr(args, 'data_dir'):
        system_config.data_dir = args.data_dir
    if hasattr(args, 'output_dir'):
        system_config.output_dir = args.output_dir
    if hasattr(args, 'max_epochs'):
        system_config.max_epochs = args.max_epochs
    
    # 更新水印配置
    if hasattr(args, 'max_examples'):
        watermark_config.max_examples_size = args.max_examples
    if hasattr(args, 'prompt_examples'):
        watermark_config.max_prompt_examples = args.prompt_examples
    
    # 更新API配置
    if hasattr(args, 'model'):
        api_config.model_name = args.model
