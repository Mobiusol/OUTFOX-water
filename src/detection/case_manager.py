"""
案例管理器 - 单一职责原则
负责案例的创建、评估和管理
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import random

from utils.utils import (
    get_cached_embedding, get_comprehensive_similarity,
    make_prompt_for_detection, call_model_api, get_confidence_voting
)
from .config import DetectionConfig


@dataclass
class WatermarkCase:
    """简化的水印案例类"""
    original: str
    watermarked: str
    label: str = "Unknown"
    confidence: float = 0.0
    quality_score: float = 0.0
    
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
        """从字典创建"""
        return cls(
            original=data['original'],
            watermarked=data['watermarked'],
            label=data.get('label', 'Unknown'),
            confidence=data.get('confidence', 0.0),
            quality_score=data.get('quality_score', 0.0)
        )


class CaseManager:
    """案例管理器 - 开放封闭原则"""
    
    def __init__(self):
        self.success_cases: List[WatermarkCase] = []
        self.failure_cases: List[WatermarkCase] = []
    
    def evaluate_case(self, case: WatermarkCase, detector_examples: List[Dict]) -> WatermarkCase:
        """评估案例质量 - 单一职责"""
        try:
            if DetectionConfig.USE_CONFIDENCE:
                # 使用置信度评估
                watermark_confidence = get_confidence_voting(
                    case.watermarked, detector_examples, rounds=3
                )
                original_confidence = get_confidence_voting(
                    case.original, detector_examples, rounds=2
                )
                
                case.confidence = watermark_confidence
                watermark_detected = watermark_confidence > 0.6
                original_correct = original_confidence < 0.3
                
            else:
                # 简单评估
                watermark_response = self._simple_detection(case.watermarked, detector_examples)
                original_response = self._simple_detection(case.original, detector_examples)
                
                watermark_detected = 'watermarked' in watermark_response.lower()
                original_correct = 'original' in original_response.lower()
                case.confidence = 1.0 if watermark_detected and original_correct else 0.0
            
            # 设置标签
            case.label = 'Good' if (watermark_detected and original_correct) else 'Bad'
            
            # 计算质量分数
            case.quality_score = self._calculate_quality_score(case)
            
        except Exception as e:
            print(f"案例评估失败: {e}")
            case.label = 'Bad'
            case.confidence = 0.0
            case.quality_score = 0.0
        
        return case
    
    def add_case(self, case: WatermarkCase, detector_examples: List[Dict]):
        """添加案例 - 自动评估和分类"""
        evaluated_case = self.evaluate_case(case, detector_examples)
        
        if evaluated_case.quality_score >= DetectionConfig.QUALITY_THRESHOLD:
            self.success_cases.append(evaluated_case)
            # 保持成功案例按质量排序
            self.success_cases.sort(key=lambda x: x.quality_score, reverse=True)
            # 限制数量
            if len(self.success_cases) > DetectionConfig.MAX_EXAMPLES_SIZE:
                self.success_cases = self.success_cases[:DetectionConfig.MAX_EXAMPLES_SIZE]
        else:
            self.failure_cases.append(evaluated_case)
    
    def get_examples_dict(self) -> Dict[str, List[Dict]]:
        """获取兼容格式的示例字典"""
        return {
            "success": [case.to_dict() for case in self.success_cases],
            "failure": [case.to_dict() for case in self.failure_cases]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_success = len(self.success_cases)
        high_quality = len([c for c in self.success_cases if c.quality_score > 0.8])
        avg_quality = sum(c.quality_score for c in self.success_cases) / total_success if total_success > 0 else 0.0
        
        return {
            'success_count': total_success,
            'failure_count': len(self.failure_cases),
            'high_quality_count': high_quality,
            'average_quality': avg_quality,
            'high_quality_ratio': high_quality / total_success if total_success > 0 else 0.0
        }
    
    def _simple_detection(self, text: str, examples: List[Dict]) -> str:
        """简单检测方法"""
        if not examples:
            return "original"
        
        sample_examples = random.sample(examples, min(len(examples), 5))
        prompt = make_prompt_for_detection(text, sample_examples)
        return call_model_api(prompt)
    
    def _calculate_quality_score(self, case: WatermarkCase) -> float:
        """计算案例质量分数"""
        try:
            # 基础评分
            label_score = 1.0 if case.label == 'Good' else 0.0
            confidence_score = case.confidence
            
            # 相似度评分
            similarity_score = get_comprehensive_similarity(case.original, case.watermarked)
            
            # 综合评分
            quality = (
                DetectionConfig.LABEL_WEIGHT * label_score +
                DetectionConfig.CONFIDENCE_WEIGHT * confidence_score +
                0.2 * similarity_score  # 相似度权重
            )
            
            return min(1.0, max(0.0, quality))
            
        except Exception:
            return 0.0
