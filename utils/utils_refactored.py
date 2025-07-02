"""重构后的简化工具模块 - 保持向后兼容"""

# 导入所有新模块的主要功能
from .base_utils import (
    load_pkl, save_pkl, json2dict, truncate_text, 
    string2token_nums, make_mixed_data
)

from .api_service import (
    completions_with_backoff, generation_by_qwen, 
    call_model_api, process_reply_from_qwen, remove_common_prefixes
)

from .embedding_service import (
    get_cached_embedding, get_comprehensive_similarity
)

from .metrics import (
    compute_three_recalls, compute_metrics
)

from .prompt_generator import (
    make_prompt_for_detection, make_prompt_for_watermark_generation,
    convert_to_detector_examples
)

from .confidence_calculators import (
    get_confidence_voting
)

from .watermark_validator import (
    identify_watermark_effectiveness
)

from .similarity_service import (
    find_similar_cases
)

from .config import COMMON_PREFIXES

# 保持原有的全局变量和常量
embedding_cache = {}  # 向后兼容

# 重新导出重要的函数和类，保持API兼容性
__all__ = [
    # 基础工具
    'load_pkl', 'save_pkl', 'json2dict', 'truncate_text', 
    'string2token_nums', 'make_mixed_data',
    
    # API服务
    'completions_with_backoff', 'generation_by_qwen', 
    'call_model_api', 'process_reply_from_qwen', 'remove_common_prefixes',
    
    # 嵌入和相似度
    'get_cached_embedding', 'get_comprehensive_similarity',
    
    # 指标计算
    'compute_three_recalls', 'compute_metrics',
    
    # Prompt生成
    'make_prompt_for_detection', 'make_prompt_for_watermark_generation',
    'convert_to_detector_examples',
    
    # 置信度计算
    'get_confidence_voting',
    
    # 水印验证
    'identify_watermark_effectiveness',
    
    # 相似性服务
    'find_similar_cases',
    
    # 常量
    'COMMON_PREFIXES'
]
