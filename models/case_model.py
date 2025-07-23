import random
from typing import Dict, List, Optional
from utils.text_utils import TextProcessor
from utils.similarity_utils import HybridSimilarityManager
from services.watermark_service import WatermarkDetector
from config.settings import similarity_config

def calculate_case_quality_score(case, similarity_manager: HybridSimilarityManager) -> float:
    """
    基于语义相似度和文本差异性的质量评分
    :param case: EnhancedCase实例
    :param similarity_manager: 相似度管理器
    :return: 0.0-1.0之间的质量分数
    """
    # 基础标签得分
    label_score = 1.0 if case.label == 'Good' else 0.3
    
    # 原文正确分类得分
    original_detection_score = 1.0 if getattr(case, 'original_correctly_identified', False) else 0.0
    
    # 语义质量评估
    quality_metrics = similarity_manager.evaluate_watermark_quality(case.original, case.watermarked)
    semantic_quality = quality_metrics['overall_quality']
    
    # 文本差异性评估（新增）
    text_similarity = case._calculate_text_difference()
    difference_score = max(0, (0.95 - text_similarity))  # 差异性越大分数越高
    
    # 最终加权得分 - 添加差异性权重
    w1, w2, w3, w4 = 0.4, 0.2, 0.3, 0.1  # 标签权重, 原文识别权重, 语义质量权重, 差异性权重
    quality_score = (w1 * label_score) + (w2 * original_detection_score) + (w3 * semantic_quality) + (w4 * difference_score)
    
    return min(1.0, quality_score)  # 确保不超过1.0

class EnhancedCase:
    """增强案例类 - 集成新的相似度系统"""
    
    def __init__(self, original: str, watermarked: str, label: Optional[str] = None,
                 text_processor: Optional[TextProcessor] = None,
                 detector: Optional[WatermarkDetector] = None,
                 similarity_manager: Optional[HybridSimilarityManager] = None):
        self.original = original
        self.watermarked = watermarked
        self.label = label
        self.quality_score = 0.0
        self.text_processor = text_processor
        self.detector = detector
        self.similarity_manager = similarity_manager
        
        # 存储详细的质量指标
        self.quality_metrics = {}
        
        # 添加详细检测结果存储
        self.watermark_detection_result = ""  # 水印文本的检测结果
        self.original_detection_result = ""   # 原始文本的检测结果
        self.original_correctly_identified = False
        self.watermark_detected = False
        
        # 添加生成器案例存储
        self.gen_examples = None
    
        # 如果有相似度管理器，立即计算质量指标
        if self.similarity_manager:
            self.quality_metrics = self.similarity_manager.evaluate_watermark_quality(
                self.original, self.watermarked
            )
    
    def update_label(self, gen_examples: Dict = None):
        """更新标签，基于检测结果 - 使用智能检索"""
        if not self.detector:
            print("未设置检测器，无法更新标签")
            return
        
        # 存储生成器案例供后续使用
        self.gen_examples = gen_examples
        
        # 检测水印文本和原始文本，使用智能检索
        self.watermark_detection_result = self._detect_text(self.watermarked, gen_examples)
        self.original_detection_result = self._detect_text(self.original, gen_examples)
        
        self.original_correctly_identified = 'original' in self.original_detection_result.lower()
        self.watermark_detected = 'watermarked' in self.watermark_detection_result.lower()
        
        # 计算文本差异性（仅用于质量分数，不影响标签）
        text_similarity = self._calculate_text_difference()
        
        # 简化标签判断逻辑 - 完全基于检测结果
        if (self.watermark_detected and 
            self.original_correctly_identified):
            self.label = 'Good'
        else:
            self.label = 'Bad'
        
        # 更新质量分数（文本差异性在这里发挥作用）
        if self.similarity_manager:
            self.quality_score = calculate_case_quality_score(self, self.similarity_manager)
        else:
            # 简化的质量分数计算，包含差异性奖励
            base_score = 1.0 if self.label == 'Good' else 0.3
            difference_bonus = max(0, (0.95 - text_similarity)) * 0.2  # 降低差异性权重
            self.quality_score = min(1.0, base_score + difference_bonus)
    
    def _detect_text(self, text: str, gen_examples: Dict = None) -> str:
        """单次文本检测 - 使用智能检索"""
        try:
            return self.detector.detect_text(text, gen_examples=gen_examples)
        except Exception as e:
            print(f"检测失败: {e}")
            return "original"
    
    def _calculate_text_difference(self) -> float:
        """计算文本差异性（相似度越低，差异性越高）"""
        original_words = set(self.original.lower().split())
        watermarked_words = set(self.watermarked.lower().split())
        
        if not original_words and not watermarked_words:
            return 1.0
        
        intersection = original_words & watermarked_words
        union = original_words | watermarked_words
        
        return len(intersection) / len(union) if union else 1.0
    
    def get_semantic_similarity(self) -> float:
        """获取语义相似度"""
        return self.quality_metrics.get('semantic_similarity', 0.0)
    
    def is_high_semantic_quality(self) -> bool:
        """判断是否为高语义质量"""
        return self.get_semantic_similarity() >= similarity_config.semantic_similarity_threshold
    
    def get_detection_summary(self) -> str:
        """获取检测结果摘要"""
        summary = f"标签分类: {self.label}\n"
        summary += f"原始文本检测: '{self.original_detection_result.strip()}' ({'✓正确' if self.original_correctly_identified else '✗错误'})\n"
        summary += f"水印文本检测: '{self.watermark_detection_result.strip()}' ({'✓正确' if self.watermark_detected else '✗错误'})"
        return summary
    
    def get_failure_analysis(self) -> str:
        """获取失败分析（新增方法）"""
        if self.label == 'Good':
            return "案例成功"
        
        analysis = "失败原因分析:\n"
        if not self.watermark_detected:
            analysis += f"- 水印文本未被识别为Watermarked (检测结果: '{self.watermark_detection_result.strip()}')\n"
        if not self.original_correctly_identified:
            analysis += f"- 原始文本未被识别为Original (检测结果: '{self.original_detection_result.strip()}')\n"
        
        return analysis
    
    def to_dict(self) -> Dict:
        """转换为字典，包含详细的质量指标"""
        return {
            'original': self.original,
            'watermarked': self.watermarked,
            'label': self.label,
            'quality_score': self.quality_score,
            'quality_metrics': self.quality_metrics,
            'watermark_detection_result': self.watermark_detection_result,
            'original_detection_result': self.original_detection_result,
            'original_correctly_identified': self.original_correctly_identified,
            'watermark_detected': self.watermark_detected,
            'failure_analysis': self.get_failure_analysis()  # 新增失败分析
        }
    
    @classmethod
    def from_dict(cls, data: Dict, text_processor=None, detector=None, similarity_manager=None) -> 'EnhancedCase':
        """从字典创建案例对象"""
        case = cls(
            data['original'], 
            data['watermarked'], 
            data.get('label', None),
            text_processor=text_processor,
            detector=detector,
            similarity_manager=similarity_manager
        )
        case.quality_score = data.get('quality_score', 0.0)
        case.quality_metrics = data.get('quality_metrics', {})
        case.watermark_detection_result = data.get('watermark_detection_result', '')
        case.original_detection_result = data.get('original_detection_result', '')
        case.original_correctly_identified = data.get('original_correctly_identified', False)
        case.watermark_detected = data.get('watermark_detected', False)
        return case