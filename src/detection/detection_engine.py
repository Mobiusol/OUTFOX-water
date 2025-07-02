"""
检测引擎 - 依赖倒置原则
核心检测逻辑，依赖于抽象接口
"""

from typing import List, Dict, Any, Tuple
import random
from tqdm import tqdm

from utils.utils import (
    make_prompt_for_watermark_generation, generation_by_qwen,
    convert_to_detector_examples, string2token_nums, compute_metrics
)
from .config import DetectionConfig
from .case_manager import CaseManager, WatermarkCase


class DetectionEngine:
    """检测引擎 - 里氏替换原则"""
    
    def __init__(self, case_manager: CaseManager):
        self.case_manager = case_manager
        self.performance_log: List[Dict] = []
    
    def generate_watermarked_text(self, original_text: str) -> str:
        """生成水印文本"""
        examples_dict = self.case_manager.get_examples_dict()
        
        # 生成水印提示
        prompt = make_prompt_for_watermark_generation(
            original_text, examples_dict, max_examples=DetectionConfig.MAX_PROMPT_EXAMPLES
        )
        
        # 计算token数
        orig_tokens = string2token_nums(original_text)
        
        # 生成水印文本
        watermarked_text = generation_by_qwen(prompt, orig_tokens)
        
        return watermarked_text
    
    def process_single_text(self, original_text: str, idx: int = 0) -> Tuple[str, str]:
        """处理单个文本 - 单一职责"""
        # 生成水印文本
        watermarked_text = self.generate_watermarked_text(original_text)
        
        # 创建案例
        case = WatermarkCase(original_text, watermarked_text)
        
        # 获取检测器示例
        examples_dict = self.case_manager.get_examples_dict()
        detector_examples = convert_to_detector_examples(examples_dict)
        
        # 添加到案例管理器（自动评估）
        self.case_manager.add_case(case, detector_examples)
        
        return original_text, watermarked_text
    
    def evaluate_performance(self, labels: List[str], predictions: List[str]) -> Dict[str, float]:
        """评估性能指标"""
        if not labels or not predictions:
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "watermarked_recall": 0.0,
                "original_recall": 0.0,
                "average_recall": 0.0
            }
        
        try:
            human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(labels, predictions)
            
            return {
                "accuracy": acc,
                "f1": f1,
                "watermarked_recall": machine_rec / 100,
                "original_recall": human_rec / 100,
                "average_recall": avg_rec / 100
            }
        except Exception as e:
            print(f"性能评估失败: {e}")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "watermarked_recall": 0.0,
                "original_recall": 0.0,
                "average_recall": 0.0
            }
    
    def run_epoch(self, texts: List[str], epoch: int) -> Dict[str, Any]:
        """运行单个epoch - 接口隔离原则"""
        labels, predictions = [], []
        
        # 进度条
        pbar = tqdm(enumerate(texts), total=len(texts), desc=f"第{epoch+1}轮")
        
        for idx, original_text in pbar:
            try:
                # 处理水印文本
                _, watermarked_text = self.process_single_text(original_text, idx)
                
                # 获取案例的预测结果
                case = WatermarkCase(original_text, watermarked_text)
                examples_dict = self.case_manager.get_examples_dict()
                detector_examples = convert_to_detector_examples(examples_dict)
                
                evaluated_case = self.case_manager.evaluate_case(case, detector_examples)
                
                # 记录结果用于评估
                labels.append("1")  # 水印文本标签
                predictions.append("1" if evaluated_case.label == "Good" else "0")
                
                # 每5个样本测试一个原始文本
                if idx % 5 == 0 and len(texts) > 1:
                    test_idx = random.randint(0, len(texts)-1)
                    while test_idx == idx:
                        test_idx = random.randint(0, len(texts)-1)
                    
                    test_original = texts[test_idx]
                    test_case = WatermarkCase(test_original, test_original)  # 原始文本作为"水印文本"
                    test_evaluated = self.case_manager.evaluate_case(test_case, detector_examples)
                    
                    labels.append("0")  # 原始文本标签
                    predictions.append("0" if test_evaluated.confidence < 0.5 else "1")
                
                # 更新进度条
                if len(labels) > 0:
                    acc = sum(1 for l, p in zip(labels, predictions) if l == p) / len(labels)
                    stats = self.case_manager.get_statistics()
                    pbar.set_postfix({
                        'acc': f"{acc:.2%}",
                        'cases': stats['success_count'],
                        'qual': f"{stats['average_quality']:.2%}"
                    })
                
            except Exception as e:
                print(f"处理文本 {idx} 时出错: {e}")
                # 记录失败案例
                labels.append("1")
                predictions.append("0")
        
        pbar.close()
        
        # 计算性能指标
        metrics = self.evaluate_performance(labels, predictions)
        stats = self.case_manager.get_statistics()
        
        # 记录性能
        epoch_result = {
            "epoch": epoch + 1,
            **{k: f"{v:.2%}" for k, v in metrics.items()},
            **stats
        }
        
        self.performance_log.append(epoch_result)
        
        return epoch_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_log:
            return {}
        
        latest = self.performance_log[-1]
        stats = self.case_manager.get_statistics()
        
        return {
            "latest_metrics": latest,
            "case_statistics": stats,
            "performance_history": self.performance_log,
            "config_used": DetectionConfig.get_status()
        }
