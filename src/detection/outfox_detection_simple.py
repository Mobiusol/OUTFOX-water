"""
简化的水印检测系统 - 基于SOLID原则重构
保持 make_prompt_for_detection 和 make_prompt_for_watermark_generation 不变
"""
import os
import sys
import argparse
import json
import random
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入重构后的模块
from utils.base_utils import load_pkl, save_pkl, string2token_nums
from utils.api_service import generation_by_qwen
from utils.prompt_generator import (
    make_prompt_for_watermark_generation, 
    make_prompt_for_detection,
    convert_to_detector_examples
)
from utils.case_manager import WatermarkCase, CaseManager
from utils.metrics import compute_metrics
from utils.config import QUALITY_CONFIG

# 设置随机种子
random.seed(42)

class WatermarkDetectionSystem:
    """水印检测系统主类"""
    
    def __init__(self, use_confidence: bool = False, max_epochs: int = 2, max_examples: int = 200):
        self.use_confidence = use_confidence
        self.max_epochs = max_epochs
        self.max_examples = max_examples
        self.case_manager = CaseManager(max_examples)
        self.performance_log = []
    
    def process_single_text(self, original_text: str) -> WatermarkCase:
        """处理单个文本"""
        # 获取当前示例
        examples_dict = self.case_manager.get_examples_dict()
        
        # 生成水印文本
        prompt_gen = make_prompt_for_watermark_generation(
            original_text, examples_dict, max_examples=10
        )
        orig_tokens = string2token_nums(original_text)
        watermarked_text = generation_by_qwen(prompt_gen, orig_tokens)
        
        # 创建案例
        case = WatermarkCase(original_text, watermarked_text)
        
        # 更新标签和置信度
        detector_examples = convert_to_detector_examples(examples_dict)
        case.update_labels(detector_examples, self.use_confidence)
        
        return case
    
    def train_epoch(self, original_texts: list, epoch: int) -> dict:
        """训练一个epoch"""
        print(f"\n=== 第 {epoch+1}/{self.max_epochs} 轮训练 ===")
        
        labels, preds = [], []
        epoch_metrics = {"success": 0, "failure": 0}
        
        pbar = tqdm(original_texts, desc=f"第{epoch+1}轮")
        
        for idx, original_text in enumerate(pbar):
            # 处理文本
            case = self.process_single_text(original_text)
            
            # 添加到案例管理器
            self.case_manager.add_case(case)
            
            # 记录指标
            if case.label == 'Good':
                labels.append("1")
                preds.append("1")
                epoch_metrics["success"] += 1
            else:
                labels.append("1")
                preds.append("0")
                epoch_metrics["failure"] += 1
            
            # 定期测试原始文本检测
            if idx % 5 == 0 and len(self.case_manager.success_cases) > 0:
                self._test_original_detection(original_texts, labels, preds, idx)
            
            # 更新进度条
            if len(labels) > 0:
                acc = sum([1 for l, p in zip(labels, preds) if l == p]) / len(labels)
                high_q = len(self.case_manager.get_high_quality_cases())
                pbar.set_postfix({'acc': f"{acc:.2%}", 'high_q': high_q})
            
            # 保存检查点
            if (idx + 1) % 50 == 0:
                self._save_checkpoint(epoch)
        
        pbar.close()
        return self._calculate_epoch_metrics(labels, preds, epoch_metrics, len(original_texts))
    
    def _test_original_detection(self, original_texts: list, labels: list, preds: list, current_idx: int):
        """测试原始文本检测"""
        # 随机选择一个不同的文本
        test_idx = random.randint(0, len(original_texts)-1)
        while test_idx == current_idx:
            test_idx = random.randint(0, len(original_texts)-1)
        
        test_original = original_texts[test_idx]
        detector_examples = convert_to_detector_examples(self.case_manager.get_examples_dict())
        
        if self.use_confidence:
            from utils.confidence_calculators import get_confidence_voting
            confidence = get_confidence_voting(test_original, detector_examples, rounds=3)
            predicted_original = confidence < 0.5
        else:
            from utils.api_service import call_model_api
            prompt = make_prompt_for_detection(test_original, detector_examples)
            response = call_model_api(prompt).lower()
            predicted_original = 'original' in response
        
        # 记录结果
        labels.append("0")
        preds.append("0" if predicted_original else "1")
    
    def _calculate_epoch_metrics(self, labels: list, preds: list, epoch_metrics: dict, total_samples: int) -> dict:
        """计算epoch指标"""
        human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(labels, preds)
        
        metrics = {
            "epoch": len(self.performance_log) + 1,
            "accuracy": f"{acc:.2%}",
            "f1": f"{f1:.2%}",
            "watermarked_recall": f"{machine_rec/100:.2%}",
            "original_recall": f"{human_rec/100:.2%}",
            "success_rate": f"{epoch_metrics['success']/total_samples:.2%}",
            "high_quality_cases": len(self.case_manager.get_high_quality_cases()),
        }
        
        self.performance_log.append(metrics)
        
        print(f"\n本轮指标:")
        print(f"准确率: {metrics['accuracy']}")
        print(f"F1分数: {metrics['f1']}")
        print(f"水印召回率: {metrics['watermarked_recall']}")
        print(f"原始召回率: {metrics['original_recall']}")
        print(f"高质量案例数: {metrics['high_quality_cases']}")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_path = f"../results/checkpoint_{epoch}.pkl"
        save_pkl(self.case_manager.get_examples_dict(), checkpoint_path)
    
    def train(self, original_texts: list):
        """训练主循环"""
        for epoch in range(self.max_epochs):
            self.train_epoch(original_texts, epoch)
        
        return self._generate_final_report()
    
    def _generate_final_report(self) -> dict:
        """生成最终报告"""
        quality_scores = [case.quality_score for case in self.case_manager.success_cases]
        confidence_levels = [case.confidence for case in self.case_manager.success_cases]
        
        report = {
            "performance_history": self.performance_log,
            "final_stats": {
                "success_cases": len(self.case_manager.success_cases),
                "failure_cases": len(self.case_manager.failure_cases),
                "high_quality_ratio": len(self.case_manager.get_high_quality_cases()) / max(1, len(self.case_manager.success_cases)),
                "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "avg_confidence": sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0,
            },
            "config_used": {
                "use_confidence": self.use_confidence,
                "max_epochs": self.max_epochs,
                "quality_threshold": QUALITY_CONFIG['threshold']
            }
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description='简化的水印检测系统')
    parser.add_argument('--data_dir', type=str, default='../../data/',
                       help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='../results/',
                       help='结果输出目录')
    parser.add_argument('--use_confidence', action='store_true',
                       help='是否使用置信度系统')
    parser.add_argument('--max_epochs', type=int, default=2,
                       help='最大训练轮数')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 根据调试模式调整参数
    if args.debug:
        max_examples = 20
        print("🐛 调试模式已启用")
    else:
        max_examples = 200
    
    # 加载数据
    dataset_path = os.path.join(args.data_dir, "common", "train", "train_humans.pkl")
    dataset_path = os.path.normpath(dataset_path)
    
    if not os.path.exists(dataset_path):
        print(f"⚠️ 数据文件不存在: {dataset_path}")
        print("🎯 生成模拟数据进行测试...")
        original_texts = [
            f"This is a sample human-written text {i}. It contains natural language patterns."
            for i in range(10 if args.debug else 100)
        ]
    else:
        original_texts = load_pkl(dataset_path)
        limit = 20 if args.debug else 1000
        original_texts = original_texts[:limit]
    
    print(f"加载了 {len(original_texts)} 篇原始文章")
    
    # 创建并运行系统
    system = WatermarkDetectionSystem(
        use_confidence=args.use_confidence,
        max_epochs=args.max_epochs,
        max_examples=max_examples
    )
    
    final_report = system.train(original_texts)
    
    # 保存报告
    with open(os.path.join(args.output_dir, "final_report_simple.json"), "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n=== 最终统计 ===")
    stats = final_report["final_stats"]
    print(f"成功案例数: {stats['success_cases']}")
    print(f"失败案例数: {stats['failure_cases']}")
    print(f"高质量案例比例: {stats['high_quality_ratio']:.2%}")
    print(f"平均质量分数: {stats['avg_quality_score']:.2%}")
    print(f"使用置信度: {'是' if args.use_confidence else '否'}")


if __name__ == '__main__':
    main()
