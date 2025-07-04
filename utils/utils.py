"""
重构后的工具模块 - 保持向后兼容
本文件现在主要作为重构后模块的统一入口点
保持 make_prompt_for_detection 和 make_prompt_for_watermark_generation 不变
"""

# 重要函数保持不变，确保向后兼容
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

# 向后兼容的全局变量
embedding_cache = {}

# 重新导出所有重要函数，确保向后兼容
__all__ = [
    'load_pkl', 'save_pkl', 'json2dict', 'truncate_text', 
    'string2token_nums', 'make_mixed_data',
    'completions_with_backoff', 'generation_by_qwen', 
    'call_model_api', 'process_reply_from_qwen', 'remove_common_prefixes',
    'get_cached_embedding', 'get_comprehensive_similarity',
    'compute_three_recalls', 'compute_metrics',
    'make_prompt_for_detection', 'make_prompt_for_watermark_generation',
    'convert_to_detector_examples',
    'get_confidence_voting',
    'identify_watermark_effectiveness',
    'find_similar_cases',
    'COMMON_PREFIXES'
]
