import random
import os
import argparse
from tqdm import tqdm
from typing import Dict, List

# 导入重构后的模块
from config.settings import system_config, watermark_config, update_config_from_args
from utils.file_utils import FileManager
from utils.text_utils import TextProcessor
from utils.metrics_utils import MetricsCalculator
from utils.similarity_utils import HybridSimilarityManager
from services.api_service import APIServiceFactory
from services.watermark_service import WatermarkGenerator, WatermarkDetector
from models.case_model import EnhancedCase

class WatermarkTrainingSystem:
    """水印训练系统主类 - 集成智能相似度系统"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化服务组件
        self.file_manager = FileManager()
        self.text_processor = TextProcessor()
        self.metrics_calculator = MetricsCalculator()
        
        # 初始化相似度管理器（新增）
        print("正在初始化相似度计算系统...")
        self.similarity_manager = HybridSimilarityManager()
        
        # 创建API服务
        api_service = APIServiceFactory.create_service()
        
        # 传入相似度管理器到水印服务
        self.watermark_generator = WatermarkGenerator(
            api_service, self.text_processor, self.similarity_manager
        )
        # 创建水印检测器时传入相似度管理器
        self.watermark_detector = WatermarkDetector(
            api_service, self.text_processor, self.similarity_manager
        )
        
        # 初始化数据
        self.enhanced_examples = {"success": [], "failure": []}
        self.performance_log = []
        
        # 设置随机种子
        random.seed(system_config.random_seed)
        
        print("水印训练系统初始化完成（包含BERT+TF-IDF相似度系统）")
    
    def load_dataset(self) -> List[str]:
        """加载数据集"""
        dataset_path = os.path.join(system_config.data_dir, "common", "train", "train_humans.pkl")
        original_texts = self.file_manager.load_pkl(dataset_path)
        # 使用配置文件中的样本数量限制
        original_texts = original_texts[:system_config.max_train_samples]
        print(f"从 {dataset_path} 加载了 {len(original_texts)} 篇原始文章")
        return original_texts
    
    def enhanced_to_basic_examples(self, enhanced_examples: Dict) -> Dict:
        """将增强型案例转换为基本字典格式"""
        basic_examples = {"success": [], "failure": []}
        
        for case in enhanced_examples["success"]:
            if isinstance(case, EnhancedCase):
                basic_examples["success"].append(case.to_dict())
            else:
                basic_examples["success"].append(case)
        
        for case in enhanced_examples["failure"]:
            if isinstance(case, EnhancedCase):
                basic_examples["failure"].append(case.to_dict())
            else:
                basic_examples["failure"].append(case)
        
        return basic_examples
    
    def train_epoch(self, epoch: int, original_texts: List[str]) -> Dict:
        """训练一个epoch - 增强的质量评估"""
        print(f"\n=== 第 {epoch+1}/{self.args.max_epochs} 轮训练 ===")
        
        epoch_metrics = {"success": 0, "failure": 0}
        labels, preds = [], []
        
        total_samples = len(original_texts)
        display_interval = max(1, int(total_samples * system_config.display_interval_percent))
        
        pbar = tqdm(range(total_samples), total=total_samples, desc=f"第{epoch+1}轮")
        
        # 获取当前的生成器案例
        current_gen_examples = self.enhanced_to_basic_examples(self.enhanced_examples)
        
        for idx, original in enumerate(tqdm(original_texts, desc=f"Epoch {epoch}")):
            pbar.update(1)
            
            # 生成水印文本（现在使用智能案例检索）
            basic_examples = self.enhanced_to_basic_examples(self.enhanced_examples)
            prompt_gen, orig_tokens = self.watermark_generator.make_prompt_for_watermark_generation(
                original, basic_examples, max_examples=watermark_config.max_prompt_examples
            )
            
            watermarked_text = self.watermark_generator.generate_watermarked_text(prompt_gen, orig_tokens)
            
            # 创建案例时传入生成器案例
            case = EnhancedCase(
                original, watermarked_text, 
                text_processor=self.text_processor,
                detector=self.watermark_detector,
                similarity_manager=self.similarity_manager
            )
            
            # 更新标签时传入生成器案例
            case.update_label(current_gen_examples)
            
            # 记录结果
            self._record_case_results(case, labels, preds)
            
            # 显示详细过程（增强版）
            if idx % display_interval == 0:
                self._display_enhanced_progress(epoch, idx, total_samples, original, watermarked_text, case)
            
            # 分类案例
            self._classify_case_by_detection_result(case, epoch_metrics)
            
            # 测试原始文本
            if idx % system_config.test_original_interval == 0:
                self._test_original_text(idx, original_texts, labels, preds, display_interval, current_gen_examples)
            
            # 更新进度条（增强版）
            self._update_enhanced_progress_bar(pbar, labels, preds)
            
            # 保存检查点
            if (idx + 1) % system_config.checkpoint_interval == 0 or idx == len(original_texts) - 1:
                self._save_checkpoint(epoch)
        
        pbar.close()
        return self._calculate_enhanced_epoch_metrics(epoch, labels, preds, epoch_metrics, len(original_texts))
    
    def _record_case_results(self, case: EnhancedCase, labels: List, preds: List):
        """记录案例结果 - 包含水印和原文检测"""
        # 记录水印文本的检测结果
        labels.append("1")  # 期望：水印文本应该被检测为Watermarked
        preds.append("1" if case.watermark_detected else "0")
        
        # 记录原文的检测结果  
        labels.append("0")  # 期望：原文应该被检测为Original
        preds.append("0" if case.original_correctly_identified else "1")
    
    def _display_enhanced_progress(self, epoch: int, idx: int, total: int, original: str, watermarked: str, case: EnhancedCase):
        """显示增强的详细进度"""
        current_percent = (idx / total) * 100
        print(f"\n--- 第{epoch+1}轮 {current_percent:.1f}% 进度详细过程 ---")
        print(f"样本索引: {idx}/{total}")
        print(f"原始文本: {original[:500]}...")
        print(f"水印文本: {watermarked[:500]}...")
        
        # 显示详细的检测结果
        print(f"\n=== 检测结果详情 ===")
        print(case.get_detection_summary())
        
        # 如果是失败案例，显示失败分析（新增）
        if case.label == 'Bad':
            print(f"\n=== 失败分析 ===")
            print(case.get_failure_analysis())
        
        print(f"\n=== 质量评估 ===")
        print(f"总体质量分数: {case.quality_score:.3f}")
        
        # 显示详细的语义质量指标
        if case.quality_metrics:
            print(f"语义相似度: {case.quality_metrics.get('semantic_similarity', 0):.3f}")
            print(f"长度比例: {case.quality_metrics.get('length_ratio', 0):.3f}")
            print(f"词汇重叠: {case.quality_metrics.get('word_overlap', 0):.3f}")
            print(f"高语义质量: {'是' if case.is_high_semantic_quality() else '否'}")
        print("----------------------------\n")
    
    def _classify_case_by_detection_result(self, case: EnhancedCase, epoch_metrics: Dict):
        """基于检测结果分类案例（完全抛弃质量阈值方法）"""
        # 完全基于检测器分类结果决定案例分类
        if case.label == 'Good':
            self.enhanced_examples["success"].append(case)
            epoch_metrics["success"] += 1
            
            # 控制成功案例数量
            if len(self.enhanced_examples["success"]) > watermark_config.max_examples_size:
                self.enhanced_examples["success"].sort(key=lambda x: x.quality_score, reverse=True)
                self.enhanced_examples["success"] = self.enhanced_examples["success"][:watermark_config.max_examples_size]
        else:
            self.enhanced_examples["failure"].append(case)
            epoch_metrics["failure"] += 1
            
            # 控制失败案例数量
            if len(self.enhanced_examples["failure"]) > watermark_config.max_examples_size:
                self.enhanced_examples["failure"].sort(key=lambda x: x.quality_score, reverse=True)
                self.enhanced_examples["failure"] = self.enhanced_examples["failure"][:watermark_config.max_examples_size]
    
    def _test_original_text(self, idx: int, original_texts: List, 
                       labels: List, preds: List, display_interval: int,
                       current_gen_examples: Dict):
        """测试原文检测 - 修复标签逻辑"""
        if idx % display_interval == 0:
            original_text = original_texts[idx]
            original_pred = self.watermark_detector.detect_text(original_text, gen_examples=current_gen_examples)
            
            labels.append('0')  # 原文期望标签：Original
            pred_label = '0' if 'original' in original_pred else '1'
            preds.append(pred_label)
            
            print(f"原文检测: {original_pred} -> {pred_label}")
    
    def _update_enhanced_progress_bar(self, pbar: tqdm, labels: List, preds: List):
        """更新增强的进度条"""
        if len(labels) > 0:
            acc = sum([1 for l, p in zip(labels, preds) if l == p]) / len(labels)
            wm_rec = sum([1 for l, p in zip(labels, preds) if l == p == "1"]) / labels.count("1") if labels.count("1") > 0 else 0
            
            # 计算高语义质量案例数
            high_semantic_quality = len([c for c in self.enhanced_examples["success"] if c.is_high_semantic_quality()])
            
            pbar.set_postfix({
                'acc': f"{acc:.2%}",
                'w_rec': f"{wm_rec:.2%}",
                'h_sem': high_semantic_quality  # 高语义质量案例数
            })
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_examples = self.enhanced_to_basic_examples(self.enhanced_examples)
        checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint_{epoch}.pkl")
        self.file_manager.save_pkl(checkpoint_examples, checkpoint_path)
    
    def _calculate_enhanced_epoch_metrics(self, epoch: int, labels: List, preds: List, 
                                        epoch_metrics: Dict, total_samples: int) -> Dict:
        """计算增强的epoch指标"""
        human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = self.metrics_calculator.compute_metrics(labels, preds)
        
        final_metrics = {
            "accuracy": acc,
            "f1": f1,
            "watermarked_recall": machine_rec / 100,
            "original_recall": human_rec / 100,
            "average_recall": avg_rec / 100
        }
        
        # 计算语义质量统计
        success_cases = self.enhanced_examples["success"]
        failure_cases = self.enhanced_examples["failure"]  # 新增失败案例统计
        
        if success_cases:
            avg_semantic_similarity = sum(c.get_semantic_similarity() for c in success_cases) / len(success_cases)
            high_semantic_cases = len([c for c in success_cases if c.is_high_semantic_quality()])
            avg_overall_quality = sum(c.quality_score for c in success_cases) / len(success_cases)
        else:
            avg_semantic_similarity = 0.0
            high_semantic_cases = 0
            avg_overall_quality = 0.0
        
        # 失败案例分析（新增）
        failure_stats = {"detection_failures": 0, "semantic_failures": 0, "mixed_failures": 0}
        for case in failure_cases:
            if not case.watermark_detected and not case.original_correctly_identified:
                failure_stats["mixed_failures"] += 1
            elif not case.watermark_detected:
                failure_stats["detection_failures"] += 1
            else:
                failure_stats["semantic_failures"] += 1
        
        # 添加到性能日志
        epoch_log = {
            "epoch": epoch+1,
            "accuracy": f"{final_metrics['accuracy']:.2%}",
            "f1": f"{final_metrics['f1']:.2%}",
            "watermarked_recall": f"{final_metrics['watermarked_recall']:.2%}",
            "original_recall": f"{final_metrics['original_recall']:.2%}",
            "average_recall": f"{final_metrics['average_recall']:.2%}",
            "success_rate": f"{epoch_metrics['success']/total_samples:.2%}",
            "avg_semantic_similarity": f"{avg_semantic_similarity:.3f}",
            "high_semantic_cases": high_semantic_cases,
            "avg_overall_quality": f"{avg_overall_quality:.3f}",
            "failure_count": len(failure_cases),
            "failure_breakdown": failure_stats
        }
        
        self.performance_log.append(epoch_log)
        
        print(f"\n本轮评估指标:")
        print(f"准确率: {final_metrics['accuracy']:.2%}")
        print(f"F1分数: {final_metrics['f1']:.2%}")
        print(f"水印召回率: {final_metrics['watermarked_recall']:.2%}")
        print(f"原始文本召回率: {final_metrics['original_recall']:.2%}")
        print(f"平均召回率: {final_metrics['average_recall']:.2%}")
        print(f"平均语义相似度: {avg_semantic_similarity:.3f}")
        print(f"高语义质量案例数: {high_semantic_cases}")
        print(f"平均总体质量: {avg_overall_quality:.3f}")
        print(f"失败案例数: {len(failure_cases)} (检测失败:{failure_stats['detection_failures']}, 语义失败:{failure_stats['semantic_failures']}, 混合失败:{failure_stats['mixed_failures']})")
        
        return final_metrics
    
    def save_final_report(self):
        """保存最终报告"""
        quality_scores = [case.quality_score for case in self.enhanced_examples["success"]]
        
        final_report = {
            "performance_history": self.performance_log,
            "quality_distribution": {
                "min": min(quality_scores) if quality_scores else 0,
                "max": max(quality_scores) if quality_scores else 0,
                "mean": sum(quality_scores)/len(quality_scores) if quality_scores else 0,
                "median": sorted(quality_scores)[len(quality_scores)//2] if quality_scores else 0,
                "high_quality_ratio": len([q for q in quality_scores if q > 0.8])/len(quality_scores) if quality_scores else 0
            },
            "final_examples_count": {
                "success": len(self.enhanced_examples["success"]),
                "failure": len(self.enhanced_examples["failure"])
            }
        }
        
        report_path = os.path.join(self.args.output_dir, "final_report_0723")
        self.file_manager.save_json(final_report, report_path)
        
        print("\n=== 最终统计 ===")
        print(f"成功案例数: {len(self.enhanced_examples['success'])}")
        print(f"失败案例数: {len(self.enhanced_examples['failure'])}")
        print(f"高质量案例比例: {final_report['quality_distribution']['high_quality_ratio']:.2%}")
        print(f"平均质量分数: {final_report['quality_distribution']['mean']:.2%}")

def main():
    parser = argparse.ArgumentParser(description='对抗式水印生成与检测系统')
    parser.add_argument('--data_dir', type=str, default=system_config.data_dir, help='数据集目录路径')
    parser.add_argument('--model', type=str, default=system_config.default_model, 
                       choices=system_config.available_models, help='使用的生成模型')
    parser.add_argument('--max_epochs', type=int, default=system_config.max_epochs, help='最大训练轮次')
    parser.add_argument('--output_dir', type=str, default=system_config.output_dir, help='结果输出目录')
    parser.add_argument('--max_examples', type=int, default=watermark_config.max_examples_size, help='保留的最大示例数量')
    parser.add_argument('--prompt_examples', type=int, default=watermark_config.max_prompt_examples, help='prompt中使用的最大示例数')
    args = parser.parse_args()
    
    # 使用统一的配置更新函数
    update_config_from_args(args)
    
    # 创建输出目录
    os.makedirs(system_config.output_dir, exist_ok=True)
    
    # 创建训练系统
    training_system = WatermarkTrainingSystem(args)
    
    # 加载数据
    original_texts = training_system.load_dataset()
    
    # 训练循环 - 使用配置文件中的参数
    for epoch in range(system_config.max_epochs):
        training_system.train_epoch(epoch, original_texts)
    
    # 保存最终报告
    training_system.save_final_report()

if __name__ == '__main__':
    main()