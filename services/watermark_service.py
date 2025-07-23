import time
from typing import Dict, List, Tuple
from services.api_service import APIServiceInterface
from utils.text_utils import TextProcessor
from utils.similarity_utils import HybridSimilarityManager
from config.settings import watermark_config

class WatermarkGenerator:
    """水印生成器 - 集成智能案例检索"""
    
    def __init__(self, api_service: APIServiceInterface, text_processor: TextProcessor,
                 similarity_manager: HybridSimilarityManager):
        self.api_service = api_service
        self.text_processor = text_processor
        self.similarity_manager = similarity_manager
    
    def make_prompt_for_watermark_generation(self, original_text: str, examples: Dict, max_examples: int = 10) -> Tuple[str, int]:
        """创建水印生成提示"""
        instruction_1 = """Your task is to create a version of the given text that contains subtle, undetectable patterns while preserving the original meaning, vocabulary, and structure as closely as possible.

        IMPORTANT REQUIREMENTS:
        1. Maintain not less than 80% vocabulary overlap with the original text
        2. Keep the text length within 95-105% of the original length
        3. Preserve the exact same arguments, facts, and core information
        4. Use synonyms sparingly and only when they maintain the same tone and register
        5. Keep the same sentence structure and paragraph organization where possible

        Generate a natural-sounding text that is nearly identical to the original but contains detectable patterns for watermark purposes.

        """
        
        instruction_2 = f"\nOriginal Text: {original_text}\n\nDirectly generate your version (maintain high similarity while adding subtle watermark patterns):"
        
        # 如果没有任何案例，使用基础提示
        if len(examples["success"]) == 0 and len(examples["failure"]) == 0:
            return instruction_1 + instruction_2, self.text_processor.count_tokens(original_text)
        
        # 构建完整提示
        prompt = instruction_1 + "Learning from examples:\n\n"
        
        # 使用智能检索获取相似的成功案例
        similar_success_cases = self.similarity_manager.find_similar_cases(
            original_text, examples["success"], top_k=max_examples
        )
        
        # 添加成功案例
        if similar_success_cases:
            prompt += "SUCCESSFUL WATERMARKING EXAMPLES (follow these patterns):\n"
            for i, case in enumerate(similar_success_cases[:4]):  # 减少到4个为失败案例腾出空间
                prompt += f"\nSuccess Example {i+1}:\n"
                prompt += f"Original: {case['original'][:500]}...\n"
                prompt += f"Watermarked: {case['watermarked'][:500]}...\n"
                
        # 使用智能检索获取相似的失败案例
        similar_failure_cases = self.similarity_manager.find_similar_cases(
            original_text, examples["failure"], top_k=max_examples
        )
        
        # 添加失败案例作为反面教材
        if similar_failure_cases:
            prompt += "\nFAILED WATERMARKING EXAMPLES (learn from these mistakes):\n"
            for i, case in enumerate(similar_failure_cases[:3]):  # 使用3个失败案例
                prompt += f"\nFailure Example {i+1} (AVOID this pattern):\n"
                prompt += f"Original: {case['original'][:500]}...\n"
                prompt += f"Failed Watermarked: {case['watermarked'][:500]}...\n"

                # 详细的失败原因分析
                failure_analysis = case.get('failure_analysis', 'Unknown failure reason')
                prompt += f"Why it failed: {failure_analysis}\n"
                
                # 检测结果信息
                orig_result = case.get('original_detection_result', 'unknown')
                wm_result = case.get('watermark_detection_result', 'unknown')
                prompt += f"Detection Results: Original->'{orig_result}', Watermarked->'{wm_result}'\n"
        
        prompt += instruction_2
        return prompt, self.text_processor.count_tokens(original_text)
    
    def generate_watermarked_text(self, prompt: str, original_text_tokens: int) -> str:
        """生成水印文本"""
        lm_essay = ''
        while True:
            try:
                lm_essay = self.api_service.call_api(
                    prompt,
                    temperature=1.3,
                    top_p=1,
                    max_tokens=original_text_tokens + 50
                )
                break
            except Exception as e:
                print(f"生成失败: {e}, 重试中...")
                time.sleep(3)
        
        processed_text = self.text_processor.remove_common_prefixes(lm_essay)
        return processed_text

class WatermarkDetector:
    """水印检测器 - 集成智能案例检索"""
    
    def __init__(self, api_service: APIServiceInterface, text_processor: TextProcessor, 
                 similarity_manager: HybridSimilarityManager = None):
        self.api_service = api_service
        self.text_processor = text_processor
        self.similarity_manager = similarity_manager
    
    def make_prompt_for_detection(self, target_text: str, gen_examples: Dict = None, max_examples: int = 10) -> str:
        """创建检测提示"""
        instruction = """Classify whether the text is Original or contains an implicit Watermark. 
        An implicit watermark is NOT visible to humans but can be detected through patterns in word choice, syntactic structures, or semantic patterns.
        IMPORTANT: You must answer with ONLY the single word 'Original' OR 'Watermarked'. Any other format will be considered incorrect.\n"""
        
        instruction_2 = f"Text: {target_text}\nAnswer: "
        
        # 如果没有生成器案例或相似度管理器，使用基础提示
        if not gen_examples or not self.similarity_manager or (len(gen_examples.get("success", [])) == 0 and len(gen_examples.get("failure", [])) == 0):
            return instruction + instruction_2
        
        # 使用智能检索构建结构化提示
        prompt = instruction + "Learning from examples:\n\n"
        
        # 获取相似的成功案例
        similar_success_cases = self.similarity_manager.find_similar_cases(
            target_text, gen_examples["success"], top_k=max_examples
        )
        
        # 成功案例部分也统一格式
        if similar_success_cases:
            prompt += "SUCCESSFUL DETECTION EXAMPLES (follow these patterns):\n"
            for i, case in enumerate(similar_success_cases[:4]):
                prompt += f"\nSuccess Example {i+1}:\n"
                
                # 原文示例 - 修复格式错误
                orig_result = case.get('original_detection_result', 'original')
                prompt += f"Text: {case['original'][:500]}...\n"
                prompt += f"Answer: Original\n"
                
                # 水印文本示例 - 修复格式错误
                wm_result = case.get('watermark_detection_result', 'watermarked')
                prompt += f"Text: {case['watermarked'][:500]}...\n"
                prompt += f"Answer: Watermarked\n"

        # 获取相似的失败案例
        similar_failure_cases = self.similarity_manager.find_similar_cases(
            target_text, gen_examples["failure"], top_k=max_examples
        )
        
        # 失败案例部分保持详细分析
        if similar_failure_cases:
            prompt += "\nFAILED DETECTION EXAMPLES (learn from these mistakes):\n"
            for i, case in enumerate(similar_failure_cases[:3]):
                prompt += f"\nFailure Example {i+1} (AVOID this pattern):\n"
                
                # 原文
                orig_result = case.get('original_detection_result', 'original')
                prompt += f"Text: {case['original'][:500]}...\n"
                prompt += f"Answer: Original\n"
                
                # 失败的水印文本
                wm_result = case.get('watermark_detection_result', 'original')
                correct_label = "Watermarked" if case.get('watermark_detected', False) else "Original"
                is_correct = correct_label.lower() == wm_result.lower()
                
                prompt += f"Text: {case['watermarked'][:500]}...\n"
                prompt += f"Answer: {correct_label} (Previously detected as: '{wm_result}' - this was {'correct' if is_correct else 'incorrect'})\n"
        
        # 添加待检测文本
        prompt += f"\n{instruction_2}"
        return prompt
    
    def detect_text(self, text: str, gen_examples: Dict = None) -> str:
        """检测文本"""
        prompt = self.make_prompt_for_detection(text, gen_examples)
        return self.api_service.call_api(
            prompt,
            temperature=1,
            top_p=1,
            max_tokens=20
        ).strip().lower()