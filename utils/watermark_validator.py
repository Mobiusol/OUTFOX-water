"""水印检测效果验证模块"""
from typing import Tuple, Optional, Dict, Any, List
from .prompt_generator import make_prompt_for_detection, convert_to_detector_examples
from .api_service import call_model_api, remove_common_prefixes


def identify_watermark_effectiveness(
    original_text: str, 
    watermarked_text: str, 
    examples: Dict[str, List]
) -> Tuple[str, str, str]:
    """
    检验水印效果
    
    Args:
        original_text: 原始文本
        watermarked_text: 水印文本
        examples: 示例数据
        
    Returns:
        (original_result, watermarked_result, label)
    """
    # 转换示例格式
    detector_examples = convert_to_detector_examples(examples) if isinstance(examples, dict) else examples

    # 预处理水印文本
    processed_watermarked = remove_common_prefixes(watermarked_text)

    # 检测水印文本
    prompt_watermarked = make_prompt_for_detection(processed_watermarked, detector_examples[:5])
    res_watermarked = call_model_api(prompt_watermarked)

    # 检测原始文本
    prompt_original = make_prompt_for_detection(original_text, detector_examples[:5])
    res_original = call_model_api(prompt_original)

    # 判断逻辑
    is_watermarked_detected = "watermarked" in res_watermarked.lower()
    is_original_correct = "original" in res_original.lower()

    label = 'Good' if (is_watermarked_detected and is_original_correct) else 'Bad'
    return (res_original, res_watermarked, label)
