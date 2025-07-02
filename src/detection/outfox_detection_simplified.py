"""
OUTFOX检测系统 - 简化重构版
移除复杂的置信度系统和冗余逻辑，保持核心功能
"""

import random
random.seed(42)
import os
import sys
import argparse
from tqdm import tqdm
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.utils import (
    load_pkl, save_pkl, call_model_api, compute_metrics,
    get_comprehensive_similarity, string2token_nums
)

# 重要：保持这两个函数不变
from utils.utils import (
    make_prompt_for_watermark_generation,
    generation_by_qwen,
    make_prompt_for_detection,
    convert_to_detector_examples
)

# 简化配置
CONFIG = {
    'max_epochs': 2,
    'max_examples': 100,
    'max_prompt_examples': 10,
    'similarity_threshold': 0.7,
    'use_simple_evaluation': True
}

print("🎯 简化模式启用 - 移除复杂的置信度和质量评分系统")


class SimpleCase:
    """简化的案例类 - 移除复杂逻辑"""
    
    def __init__(self, original, watermarked):
        self.original = original
        self.watermarked = watermarked
        self.label = 'Unknown'
        self.is_good = False
    
    def evaluate(self, detector_examples):
        """简化的评估逻辑"""
        try:
            # 简单检测水印文本
            watermark_prompt = make_prompt_for_detection(self.watermarked, detector_examples[:5])
            watermark_response = call_model_api(watermark_prompt).lower()
            
            # 简单检测原始文本
            original_prompt = make_prompt_for_detection(self.original, detector_examples[:5])
            original_response = call_model_api(original_prompt).lower()
            
            # 简单判断逻辑
            watermark_detected = 'watermarked' in watermark_response
            original_correct = 'original' in original_response
            
            self.is_good = watermark_detected and original_correct
            self.label = 'Good' if self.is_good else 'Bad'
            
        except Exception as e:
            print(f"评估失败: {e}")
            self.is_good = False
            self.label = 'Bad'
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'original': self.original,
            'watermarked': self.watermarked,
            'label': self.label
        }


class SimplifiedTrainer:
    """简化的训练器 - 移除复杂的置信度和质量评分"""
    
    def __init__(self):
        self.examples = {"success": [], "failure": []}
        self.performance_log = []
    
    def process_text(self, original_text):
        """处理单个文本 - 简化逻辑"""
        # 生成水印文本
        prompt = make_prompt_for_watermark_generation(
            original_text, 
            self.examples, 
            max_examples=CONFIG['max_prompt_examples']
        )
        
        orig_tokens = string2token_nums(original_text)
        watermarked_text = generation_by_qwen(prompt, orig_tokens)
        
        # 创建案例
        case = SimpleCase(original_text, watermarked_text)
        
        # 评估案例
        detector_examples = convert_to_detector_examples(self.examples)
        case.evaluate(detector_examples)
        
        # 添加到适当的列表
        if case.is_good:
            self.examples["success"].append(case.to_dict())
            # 保持数量限制
            if len(self.examples["success"]) > CONFIG['max_examples']:
                self.examples["success"] = self.examples["success"][-CONFIG['max_examples']:]
        else:
            self.examples["failure"].append(case.to_dict())
        
        return case
    
    def run_epoch(self, texts, epoch):
        """运行单个epoch - 简化逻辑"""
        labels, predictions = [], []
        
        pbar = tqdm(enumerate(texts), total=len(texts), desc=f"第{epoch+1}轮")
        
        for idx, original_text in pbar:
            try:
                # 处理文本
                case = self.process_text(original_text)
                
                # 记录结果
                labels.append("1")  # 水印文本
                predictions.append("1" if case.is_good else "0")
                
                # 每5个样本测试一个原始文本
                if idx % 5 == 0 and len(texts) > 1:
                    test_idx = random.randint(0, len(texts)-1)
                    while test_idx == idx:
                        test_idx = random.randint(0, len(texts)-1)
                    
                    test_original = texts[test_idx]
                    test_case = SimpleCase(test_original, test_original)
                    detector_examples = convert_to_detector_examples(self.examples)
                    test_case.evaluate(detector_examples)
                    
                    labels.append("0")  # 原始文本
                    predictions.append("0" if 'original' in call_model_api(
                        make_prompt_for_detection(test_original, detector_examples[:5])
                    ).lower() else "1")
                
                # 更新进度条
                if len(labels) > 0:
                    acc = sum(1 for l, p in zip(labels, predictions) if l == p) / len(labels)
                    pbar.set_postfix({
                        'acc': f"{acc:.2%}",
                        'good': len(self.examples["success"]),
                        'bad': len(self.examples["failure"])
                    })
                
            except Exception as e:
                print(f"处理文本 {idx} 时出错: {e}")
                labels.append("1")
                predictions.append("0")
        
        pbar.close()
        
        # 计算指标
        if labels and predictions:
            human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(labels, predictions)
            
            metrics = {
                "epoch": epoch + 1,
                "accuracy": f"{acc:.2%}",
                "f1": f"{f1:.2%}", 
                "watermarked_recall": f"{machine_rec:.1f}%",
                "original_recall": f"{human_rec:.1f}%",
                "success_cases": len(self.examples["success"]),
                "failure_cases": len(self.examples["failure"])
            }
            
            self.performance_log.append(metrics)
            return metrics
        
        return {}


def main():
    """主函数 - 简化逻辑"""
    parser = argparse.ArgumentParser(description='简化的OUTFOX检测系统')
    parser.add_argument('--data_dir', type=str, default='../../data/',
                       help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='../results/',
                       help='结果输出目录')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    # 调试模式配置
    if args.debug:
        CONFIG['max_epochs'] = 1
        CONFIG['max_examples'] = 20
        print("🐛 调试模式启用")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化训练器
    trainer = SimplifiedTrainer()
    
    # 加载数据
    dataset_path = os.path.join(args.data_dir, "common", "train", "train_humans.pkl")
    dataset_path = os.path.normpath(dataset_path)
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据文件不存在: {dataset_path}")
        print("🎯 生成模拟数据...")
        texts = [f"Sample text {i} for testing watermark detection." for i in range(50)]
    else:
        texts = load_pkl(dataset_path)[:500]  # 限制数量
    
    print(f"📚 加载了 {len(texts)} 篇文章")
    
    # 运行训练
    for epoch in range(CONFIG['max_epochs']):
        print(f"\n=== 第 {epoch+1}/{CONFIG['max_epochs']} 轮训练 ===")
        
        metrics = trainer.run_epoch(texts, epoch)
        
        if metrics:
            print(f"准确率: {metrics['accuracy']}")
            print(f"F1分数: {metrics['f1']}")
            print(f"水印召回率: {metrics['watermarked_recall']}")
            print(f"原始召回率: {metrics['original_recall']}")
            print(f"成功案例: {metrics['success_cases']}")
        
        # 保存检查点
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_simplified_{epoch}.pkl")
        save_pkl(trainer.examples, checkpoint_path)
    
    # 保存最终结果
    final_report = {
        "performance_history": trainer.performance_log,
        "final_examples": trainer.examples,
        "config_used": CONFIG
    }
    
    report_path = os.path.join(args.output_dir, "final_report_simplified.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 最终统计 ===")
    print(f"成功案例数: {len(trainer.examples['success'])}")
    print(f"失败案例数: {len(trainer.examples['failure'])}")
    print(f"📄 报告已保存: {report_path}")
    print("\n🎉 训练完成！")


if __name__ == '__main__':
    main()
