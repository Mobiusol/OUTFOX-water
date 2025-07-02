"""案例管理模块 - 简化的案例处理"""
from typing import Dict, Any, Optional, List
from .embedding_service import get_cached_embedding, get_comprehensive_similarity
from .confidence_calculators import VotingConfidenceCalculator
from .config import QUALITY_CONFIG


class WatermarkCase:
    """简化的水印案例类"""
    
    def __init__(self, original: str, watermarked: str, label: Optional[str] = None, confidence: float = 0.0):
        self.original = original
        self.watermarked = watermarked
        self.label = label
        self.confidence = confidence
        self.quality_score = 0.0
        self._embedding = None
    
    @property
    def embedding(self):
        """延迟计算嵌入"""
        if self._embedding is None:
            self._embedding = get_cached_embedding(self.original)
        return self._embedding
    
    def calculate_quality_score(self) -> float:
        """计算案例质量分数"""
        # 水印强度得分
        watermark_strength = self.confidence
        
        # 相似度得分
        try:
            similarity_score = get_comprehensive_similarity(self.original, self.watermarked)
        except Exception:
            similarity_score = 0.5
        
        # 标签得分
        label_score = 1.0 if self.label == 'Good' else 0.0
        
        # 加权计算
        quality_score = (
            QUALITY_CONFIG['label_weight'] * label_score +
            QUALITY_CONFIG['confidence_weight'] * watermark_strength +
            0.4 * similarity_score
        )
        
        self.quality_score = quality_score
        return quality_score
    
    def update_labels(self, detector_examples: List[Dict], use_confidence: bool = True):
        """更新标签和置信度"""
        if use_confidence:
            calculator = VotingConfidenceCalculator(rounds=5)
            
            # 检测水印文本
            watermark_result = calculator.calculate_confidence(self.watermarked, detector_examples)
            self.confidence = watermark_result['confidence']
            
            # 检测原文
            original_result = calculator.calculate_confidence(self.original, detector_examples)
            original_confidence = original_result['confidence']
            
            # 判断标签
            watermark_detected = self.confidence > 0.6
            original_correct = original_confidence < 0.3
            
            self.label = 'Good' if (watermark_detected and original_correct) else 'Bad'
        else:
            # 简单模式
            from .prompt_generator import make_prompt_for_detection
            from .api_service import call_model_api
            
            # 检测水印文本
            prompt = make_prompt_for_detection(self.watermarked, detector_examples)
            watermark_response = call_model_api(prompt).lower()
            
            # 检测原文
            prompt_orig = make_prompt_for_detection(self.original, detector_examples)
            original_response = call_model_api(prompt_orig).lower()
            
            watermark_detected = 'watermarked' in watermark_response
            original_correct = 'original' in original_response
            
            self.label = 'Good' if (watermark_detected and original_correct) else 'Bad'
            self.confidence = 1.0 if watermark_detected else 0.0
        
        # 更新质量分数
        self.calculate_quality_score()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'original': self.original,
            'watermarked': self.watermarked,
            'label': self.label,
            'confidence': self.confidence,
            'quality_score': self.quality_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WatermarkCase':
        """从字典创建实例"""
        case = cls(
            data['original'],
            data['watermarked'],
            data.get('label'),
            data.get('confidence', 0.0)
        )
        case.quality_score = data.get('quality_score', 0.0)
        return case


class CaseManager:
    """案例管理器"""
    
    def __init__(self, max_cases: int = 200):
        self.max_cases = max_cases
        self.success_cases: List[WatermarkCase] = []
        self.failure_cases: List[WatermarkCase] = []
    
    def add_case(self, case: WatermarkCase):
        """添加案例"""
        if case.quality_score >= QUALITY_CONFIG['threshold']:
            self.success_cases.append(case)
            # 保持数量限制，只保留质量最高的案例
            if len(self.success_cases) > self.max_cases:
                self.success_cases.sort(key=lambda x: x.quality_score, reverse=True)
                self.success_cases = self.success_cases[:self.max_cases]
        else:
            self.failure_cases.append(case)
    
    def get_examples_dict(self) -> Dict[str, List[Dict]]:
        """获取示例字典格式"""
        return {
            "success": [case.to_dict() for case in self.success_cases],
            "failure": [case.to_dict() for case in self.failure_cases]
        }
    
    def get_high_quality_cases(self, threshold: float = 0.8) -> List[WatermarkCase]:
        """获取高质量案例"""
        return [case for case in self.success_cases if case.quality_score >= threshold]
