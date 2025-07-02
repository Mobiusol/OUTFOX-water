"""置信度计算抽象接口"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ConfidenceCalculator(ABC):
    """置信度计算器抽象基类"""
    
    @abstractmethod
    def calculate_confidence(self, text: str, examples: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        计算置信度
        
        Args:
            text: 待检测文本
            examples: 检测示例
            **kwargs: 其他参数
            
        Returns:
            包含置信度和预测结果的字典
        """
        pass


class SimpleConfidenceCalculator(ConfidenceCalculator):
    """简单置信度计算器"""
    
    def __init__(self, api_caller):
        self.api_caller = api_caller
    
    def calculate_confidence(self, text: str, examples: List[Dict], **kwargs) -> Dict[str, Any]:
        """简单置信度计算"""
        from .prompt_generator import make_prompt_for_detection
        
        prompt = make_prompt_for_detection(text, examples)
        response = self.api_caller(prompt).lower()
        
        is_watermarked = 'watermarked' in response
        confidence = 1.0 if is_watermarked else 0.0
        
        return {
            'prediction': 'watermarked' if is_watermarked else 'original',
            'confidence': confidence,
            'method_used': 'simple'
        }
