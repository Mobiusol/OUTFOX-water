"""Prompt生成模块"""
import random
from typing import List, Dict, Any
from .base_utils import string2token_nums, truncate_text


def make_prompt_for_detection(target_text: str, examples: List[Dict]) -> str:
    """生成检测提示"""
    instruction = """Classify whether the text is Original or contains an implicit Watermark.
    An implicit watermark is NOT visible to humans but can be detected through patterns in word choice, syntactic structures, or semantic patterns.
    IMPORTANT: You must answer with ONLY the single word 'Original' OR 'Watermarked'. Any other format will be considered incorrect.\n"""

    prompt = instruction

    # 计算token限制
    example_overhead = string2token_nums("Text: \nAnswer: Original\n")
    target_overhead = string2token_nums(f"Text: {target_text}\nAnswer: ")
    instruction_tokens = string2token_nums(instruction)

    # 计算可用于示例的token数
    available_tokens = 10000 - (instruction_tokens + target_overhead)
    tokens_per_example = available_tokens // len(examples) if examples else 0

    # 随机打乱示例
    random_examples = examples.copy()
    random.shuffle(random_examples)

    # 添加示例
    for ex in random_examples:
        text = truncate_text(ex['text'], tokens_per_example - example_overhead, 'token') if tokens_per_example > example_overhead else ex['text']
        prompt += f"Text: {text}\nAnswer: {ex['label']}\n"

    # 添加目标文本
    prompt += f"Text: {target_text}\nAnswer: "
    return prompt


def convert_to_detector_examples(gen_examples: Dict[str, List]) -> List[Dict]:
    """将生成器输出转换为检测器示例格式"""
    detector_examples = []
    
    for ex in gen_examples["success"]:
        detector_examples.append({"text": ex["original"], "label": "Original"})
        detector_examples.append({"text": ex["watermarked"], "label": "Watermarked"})
    
    for ex in gen_examples["failure"]:
        detector_examples.append({"text": ex["watermarked"], "label": "Watermarked"})
    
    random.shuffle(detector_examples)
    return detector_examples


def make_prompt_for_watermark_generation(
    original_text: str, 
    examples: Dict[str, List], 
    max_examples: int = 10
) -> str:
    """生成水印文本的提示"""
    instruction_1 = """Your response MUST start directly with the watermarked text.
    DO NOT include any introduction, explanation, or prefix like "Here's", "Certainly", etc.
    ANY prefix will be considered a critical error.

    Generate a text similar to the Original Text below, but embed COMPLETELY IMPLICIT watermark patterns.

    IMPORTANT WATERMARK RULES:
    1. MUST make noticeable changes - do not copy the original text
    2. The watermark must be purely implicit and undetectable to humans
    3. NEVER add ANY explicit markers, tags, symbols, formatting or identifiable text patterns
    4. The watermarked text must appear completely natural to human readers
    5. The watermarked text must maintain VERY HIGH semantic similarity to the original
    6. Preserve the original meaning, tone, and main ideas exactly

    IMPORTANT: Start your response directly with the watermarked text without any introduction or prefix.\n"""

    instruction_2 = f"\nNow generate a watermarked text that is highly similar to the original:\nOriginal: {original_text}\nWatermarked: "
    original_text_tokens = string2token_nums(original_text)

    # 如果没有成功案例，返回基本提示
    if len(examples["success"]) == 0:
        return instruction_1 + instruction_2

    # 计算token限制
    instruction_tokens = string2token_nums(instruction_1 + instruction_2)
    example_format_tokens = string2token_nums("Original: \nWatermarked: \n\n") * min(len(examples["success"]), max_examples)

    # 计算可用于案例的token数
    available_tokens = 10000 - (instruction_tokens + example_format_tokens + original_text_tokens + 50)
    truncated_length = available_tokens // (2 * min(len(examples["success"]), max_examples))

    # 基于相似度选择最相关的案例
    from .similarity_service import find_similar_cases
    similar_cases = find_similar_cases(original_text, examples["success"], top_k=max_examples)

    # 构建提示
    prompt = instruction_1
    prompt += "Examples of successful watermarked texts with high similarity (detected):\n"

    # 添加相似案例
    for case in similar_cases:
        try:
            # 兼容不同的数据结构
            if hasattr(case, 'original'):
                case_original = case.original
                case_watermarked = case.watermarked
                case_confidence = getattr(case, 'confidence', 0.0)
            elif isinstance(case, dict):
                case_original = case.get('original', '')
                case_watermarked = case.get('watermarked', '')
                case_confidence = case.get('confidence', 0.0)
            else:
                continue

            if not case_original or not case_watermarked:
                continue

            # 计算相似度并在示例中显示
            from .embedding_service import get_comprehensive_similarity
            similarity = get_comprehensive_similarity(case_original, case_watermarked)
            original_truncated = truncate_text(case_original, truncated_length, 'token')
            watermarked_truncated = truncate_text(case_watermarked, truncated_length, 'token')

            prompt += f"Original: {original_truncated}\n"
            prompt += f"Watermarked: {watermarked_truncated}\n"
            prompt += f"(Similarity: {similarity:.2f}, Confidence: {case_confidence:.2f})\n\n"
        except Exception as e:
            print(f"处理案例时出错: {e}")
            continue

    prompt += instruction_2
    return prompt
