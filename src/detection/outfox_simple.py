"""
简化的OUTFOX检测系统 - SOLID原则重构版
去除冗余逻辑，保持核心功能
"""

import os
import sys
import argparse
import json
from typing import List

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.utils import load_pkl, save_pkl
from src.detection.config import DetectionConfig
from src.detection.case_manager import CaseManager
from src.detection.detection_engine import DetectionEngine


class OutfoxDetectionApp:
    """主应用程序 - 单一职责原则"""
    
    def __init__(self, args):
        self.args = args
        self.case_manager = CaseManager()
        self.detection_engine = DetectionEngine(self.case_manager)
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
    
    def load_data(self) -> List[str]:
        """加载数据"""
        dataset_path = os.path.join(self.args.data_dir, "common", "train", "train_humans.pkl")
        dataset_path = os.path.normpath(dataset_path)
        
        if not os.path.exists(dataset_path):
            print(f"⚠️  数据文件不存在: {dataset_path}")
            print("🎯 生成模拟数据进行测试...")
            return [
                f"This is a sample human-written text {i}. It contains natural language patterns."
                for i in range(50)
            ]
        
        texts = load_pkl(dataset_path)
        return texts[:1000]  # 限制数量
    
    def setup_mode(self, mode: str = None):
        """设置运行模式"""
        if mode:
            DetectionConfig.set_mode(mode)
        
        status = DetectionConfig.get_status()
        print(f"🚀 运行模式: {status['mode']}")
        print(f"📊 配置: 轮数={status['max_epochs']}, 样本={status['max_examples']}")
        print(f"⚙️  置信度: {'启用' if status['use_confidence'] else '禁用'}")
        print("-" * 40)
    
    def run_training(self, texts: List[str]):
        """运行训练过程"""
        print(f"📚 加载了 {len(texts)} 篇文章")
        
        for epoch in range(DetectionConfig.MAX_EPOCHS):
            print(f"\n=== 第 {epoch+1}/{DetectionConfig.MAX_EPOCHS} 轮训练 ===")
            
            epoch_result = self.detection_engine.run_epoch(texts, epoch)
            
            # 显示结果
            print(f"准确率: {epoch_result['accuracy']}")
            print(f"F1分数: {epoch_result['f1']}")
            print(f"水印召回率: {epoch_result['watermarked_recall']}")
            print(f"成功案例: {epoch_result['success_count']}")
            print(f"平均质量: {epoch_result['average_quality']:.2%}")
            
            # 保存检查点
            self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_data = {
            'examples': self.case_manager.get_examples_dict(),
            'performance': self.detection_engine.get_performance_summary(),
            'config': DetectionConfig.get_status()
        }
        
        checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint_simple_{epoch}.pkl")
        save_pkl(checkpoint_data, checkpoint_path)
    
    def save_final_report(self):
        """保存最终报告"""
        summary = self.detection_engine.get_performance_summary()
        
        # 保存详细报告
        report_path = os.path.join(self.args.output_dir, "final_report_simple.json")
        with open(report_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存最终示例
        examples_path = os.path.join(self.args.output_dir, "final_examples_simple.pkl")
        save_pkl(self.case_manager.get_examples_dict(), examples_path)
        
        print(f"\n📄 报告已保存: {report_path}")
        print(f"💾 示例已保存: {examples_path}")
    
    def print_summary(self):
        """打印最终摘要"""
        summary = self.detection_engine.get_performance_summary()
        
        if 'case_statistics' in summary:
            stats = summary['case_statistics']
            print(f"\n=== 最终统计 ===")
            print(f"成功案例数: {stats['success_count']}")
            print(f"失败案例数: {stats['failure_count']}")
            print(f"高质量案例数: {stats['high_quality_count']}")
            print(f"平均质量分数: {stats['average_quality']:.2%}")
            print(f"高质量比例: {stats['high_quality_ratio']:.2%}")
        
        if 'config_used' in summary:
            config = summary['config_used']
            print(f"\n=== 使用配置 ===")
            print(f"模式: {config['mode']}")
            print(f"置信度: {'启用' if config['use_confidence'] else '禁用'}")
            print(f"训练轮数: {config['max_epochs']}")
    
    def run(self):
        """主运行方法"""
        print("=" * 50)
        print("🔥 OUTFOX-v2 简化检测系统")
        print("=" * 50)
        
        # 设置模式
        mode = "debug" if self.args.debug else "fast" if self.args.fast else "standard"
        self.setup_mode(mode)
        
        # 加载数据
        texts = self.load_data()
        
        # 运行训练
        self.run_training(texts)
        
        # 保存结果
        self.save_final_report()
        
        # 打印摘要
        self.print_summary()
        
        print("\n🎉 检测完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OUTFOX-v2 简化检测系统')
    
    parser.add_argument('--data_dir', type=str, default='../../data/',
                       help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='../results/',
                       help='结果输出目录')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式（少量数据）')
    parser.add_argument('--fast', action='store_true',
                       help='启用快速模式')
    parser.add_argument('--confidence', action='store_true',
                       help='启用置信度系统')
    
    args = parser.parse_args()
    
    # 根据参数设置置信度
    if args.confidence:
        DetectionConfig.USE_CONFIDENCE = True
    
    # 创建并运行应用
    app = OutfoxDetectionApp(args)
    app.run()


if __name__ == '__main__':
    main()

