import random
random.seed(42)
import os
import sys
import argparse
from tqdm import tqdm
import time
import json
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.utils import (
    load_pkl, save_pkl, call_model_api, compute_metrics,
    get_confidence_voting, get_cached_embedding, find_similar_cases,
    get_comprehensive_similarity, string2token_nums
)

# 水印系统核心组件
from utils.utils import (
    make_prompt_for_watermark_generation,
    generation_by_qwen,
    make_prompt_for_detection,
    convert_to_detector_examples,
    identify_watermark_effectiveness
)

# ==================== 快速配置开关 ====================
# 修改这些变量来控制系统行为
USE_CONFIDENCE = False       # 改为 False 禁用置信度
FAST_MODE = False           # 改为 True 启用快速模式
DEBUG_MODE = False          # 改为 True 启用调试模式

# 根据模式自动调整设置
if DEBUG_MODE:
    USE_CONFIDENCE = False
    MAX_EPOCHS = 2
    MAX_EXAMPLES_SIZE = 20
    print("🐛 调试模式已启用")
elif FAST_MODE:
    USE_CONFIDENCE = False
    MAX_EPOCHS = 5
    MAX_EXAMPLES_SIZE = 100
    print("⚡ 快速模式已启用")
else:
    MAX_EPOCHS = 2
    MAX_EXAMPLES_SIZE = 200
    print("🎯 标准模式已启用")

# 系统配置
MAX_PROMPT_EXAMPLES = 10  # prompt中使用的最大示例数

# 案例质量评估权重
LABEL_WEIGHT = 0.8  # 标签权重
CONFIDENCE_WEIGHT = 0.2  # 置信度权重
QUALITY_THRESHOLD = 0.6  # 案例质量阈值
SIMILARITY_THRESHOLD = 0.7  # 最低相似度要求

# 显示当前设置
print(f"置信度系统: {'✅ 启用' if USE_CONFIDENCE else '❌ 禁用'}")
print(f"训练轮数: {MAX_EPOCHS}")
print("-" * 40)

def calculate_case_quality_score(case):
    """
    改进的质量评分，使用更准确的文本相似度计算
    :param case: EnhancedCase实例
    :return: 0.0-1.0之间的质量分数
    """
    # 水印强度得分 - 使用置信度直接作为水印强度的度量
    watermark_strength = case.confidence
    
    # 原文正确分类得分
    original_detection_score = 1.0 if getattr(case, 'original_correctly_identified', False) else 0.0
    
    # 使用改进的相似度计算方法
    try:
        similarity_score = get_comprehensive_similarity(case.original, case.watermarked)
    except Exception as e:
        # 如果新函数不可用，使用简单的相似度计算
        print(f"相似度计算失败，使用简单方法: {e}")
        len_ratio = min(len(case.watermarked), len(case.original)) / max(len(case.watermarked), len(case.original))
        similarity_score = len_ratio
    
    # 调整权重 - 增加相似度的重要性
    w1, w2, w3 = 0.4, 0.2, 0.4  # 水印强度权重, 原文识别权重, 相似度权重
    quality_score = (w1 * watermark_strength) + (w2 * original_detection_score) + (w3 * similarity_score)
    
    return quality_score

def check_original_classification(original_text, detector_examples, use_confidence=True):
    """
    检查原文是否被正确分类为Original
    :param original_text: 原始文本
    :param detector_examples: 检测器示例
    :param use_confidence: 是否使用置信度
    :return: 是否正确分类
    """
    try:
        if use_confidence:
            # 使用置信度投票
            original_confidence = get_confidence_voting(
                original_text,
                detector_examples,
                rounds=3
            )
            # 低置信度表示被识别为原始文本
            predicted_original = original_confidence < 0.5
        else:
            # 不使用置信度时，直接进行单次检测
            prompt = make_prompt_for_detection(original_text, detector_examples)
            response = call_model_api(prompt).lower()
            predicted_original = 'original' in response
        
        return predicted_original
    except Exception as e:
        print(f"检查原文分类时出错: {e}")
        return False

def filter_by_similarity(case, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    根据相似度筛选案例
    :param case: EnhancedCase实例
    :param similarity_threshold: 相似度阈值
    :return: 是否通过筛选
    """
    try:
        similarity = get_comprehensive_similarity(case.original, case.watermarked)
        return similarity >= similarity_threshold
    except Exception as e:
        # 如果计算失败，使用原来的长度比方法
        print(f"相似度计算失败，使用长度比方法: {e}")
        len_ratio = min(len(case.watermarked), len(case.original)) / max(len(case.watermarked), len(case.original))
        return len_ratio >= 0.8  # 保守的长度相似度要求

class EnhancedCase:
    """增强型案例类，包含置信度和嵌入向量"""
    def __init__(self, original, watermarked, label=None, confidence=0.0):
        self.original = original
        self.watermarked = watermarked
        self.label = label  # 'Good' 或 'Bad'
        self.confidence = confidence
        self.embedding = None
        self.quality_score = 0.0  # 新增：案例质量分数
        # 计算嵌入向量
        self.update_embedding()
    
    def update_embedding(self):
        """更新嵌入向量"""
        self.embedding = get_cached_embedding(self.original)
    
    def update_confidence_and_label(self, detector_examples, use_confidence=True):
        """分离水印文本检测评估和原文评估"""
        if use_confidence:
            # 单独评估水印文本的检测效果
            watermark_confidence = get_confidence_voting(
                self.watermarked, 
                detector_examples,
                rounds=5  # 默认5轮投票
           )
        
            # 单独评估原文是否被正确识别为非水印
            original_confidence = get_confidence_voting(
                self.original,
                detector_examples,
                rounds=3  # 原文检测可以使用较少轮数
            )
        
            # 记录原文是否被正确识别为"Original"
            self.original_correctly_identified = original_confidence < 0.3  # 低置信度表示被识别为原始文本
        
            # 更新水印文本的标签和置信度
            self.watermark_detected = watermark_confidence > 0.6  # 水印检测阈值
            self.confidence = watermark_confidence
            
            # 综合判断标签
            if self.watermark_detected:
                self.label = 'Good'
            else:
                self.label = 'Bad'
                # 如果水印未被检测到，降低置信度但不为零
                self.confidence = watermark_confidence * 0.3
            
            # 存储额外信息以供分析
            self.watermark_confidence = watermark_confidence
            self.original_confidence = original_confidence
        else:
            # 不使用置信度时，使用简单的二分类方法
            # 只进行一次检测，不用多轮投票
            prompt = make_prompt_for_detection(self.watermarked, detector_examples)
            watermark_response = call_model_api(prompt).lower()
            
            prompt_orig = make_prompt_for_detection(self.original, detector_examples)
            original_response = call_model_api(prompt_orig).lower()
            
            # 直接根据检测结果设置标签
            self.watermark_detected = 'watermarked' in watermark_response
            self.original_correctly_identified = 'original' in original_response
            
            if self.watermark_detected and self.original_correctly_identified:
                self.label = 'Good'
                self.confidence = 1.0  # 不使用置信度时设为固定值
            else:
                self.label = 'Bad'
                self.confidence = 0.0  # 不使用置信度时设为固定值
            
            # 存储额外信息以供分析
            self.watermark_confidence = 1.0 if self.watermark_detected else 0.0
            self.original_confidence = 0.0 if self.original_correctly_identified else 1.0
        
        # 更新质量分数
        self.quality_score = calculate_case_quality_score(self)
    
    def to_dict(self):
        """转换为字典，方便保存"""
        return {
            'original': self.original,
            'watermarked': self.watermarked,
            'label': self.label,
            'confidence': self.confidence,
            'quality_score': self.quality_score
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建案例对象"""
        case = cls(
            data['original'], 
            data['watermarked'], 
            data.get('label', None),
            data.get('confidence', 0.0)
        )
        case.quality_score = data.get('quality_score', 0.0)
        return case

def enhanced_to_basic_examples(enhanced_examples):
    """将增强型案例转换为基本字典格式"""
    basic_examples = {"success": [], "failure": []}
    
    # 处理成功案例
    for case in enhanced_examples["success"]:
        if isinstance(case, EnhancedCase):
            basic_examples["success"].append(case.to_dict())
        else:
            basic_examples["success"].append(case)
    
    # 处理失败案例
    for case in enhanced_examples["failure"]:
        if isinstance(case, EnhancedCase):
            basic_examples["failure"].append(case.to_dict())
        else:
            basic_examples["failure"].append(case)
    
    return basic_examples

def show_simple_text_diff(original, watermarked, case_id=None):
    """最简单的差异显示"""
    orig_words = original.split()
    water_words = watermarked.split()
    
    # 找出不同的词
    orig_set = set(orig_words)
    water_set = set(water_words)
    
    removed = orig_set - water_set
    added = water_set - orig_set
    
    prefix = f"案例{case_id}: " if case_id else ""
    
    # 显示原文（标记删除的词）
    orig_display = []
    for word in orig_words:
        if word in removed:
            orig_display.append(f"[删除:{word}]")
        else:
            orig_display.append(word)
    
    # 显示水印文本（标记添加的词）
    water_display = []
    for word in water_words:
        if word in added:
            water_display.append(f"[添加:{word}]")
        else:
            water_display.append(word)
    
    print(f"{prefix}原始: {' '.join(orig_display)}")
    print(f"{prefix}水印: {' '.join(water_display)}")
    
    # 显示变化统计
    if removed or added:
        changes = []
        if removed:
            changes.append(f"删除{len(removed)}词")
        if added:
            changes.append(f"添加{len(added)}词")
        print(f"{prefix}变化: {', '.join(changes)}")
    else:
        print(f"{prefix}变化: 无明显词汇变化")
    print()

def main():
    parser = argparse.ArgumentParser(description='对抗式水印生成与检测系统')
    parser.add_argument('--data_dir', type=str, default='../../data/',
                       help='数据集目录路径')
    parser.add_argument('--model', type=str, default='qwen-turbo',
                       choices=['qwen-turbo', 'gpt-3.5-turbo'],
                       help='使用的生成模型')
    parser.add_argument('--output_dir', type=str, default='../results/',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 检查输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")

    # 初始化系统
    enhanced_examples = {"success": [], "failure": []}
    performance_log = []

    # 加载原始数据集
    dataset_path = os.path.join(args.data_dir, "common", "train", "train_humans.pkl")
    dataset_path = os.path.normpath(dataset_path)  # 修复路径分隔符
    
    # 检查数据文件是否存在
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据文件不存在: {dataset_path}")
        print("🎯 生成模拟数据进行测试...")
        # 生成一些模拟数据用于测试
        original_texts = [
            f"This is a sample human-written text {i}. It contains natural language patterns and typical human writing characteristics."
            for i in range(10)
        ]
    else:
        original_texts = load_pkl(dataset_path)
        original_texts = original_texts[:1000]  # 决定使用文章范围
    
    print(f"加载了 {len(original_texts)} 篇原始文章")

    # 对抗训练主循环
    for epoch in range(MAX_EPOCHS):
        print(f"\n=== 第 {epoch+1}/{MAX_EPOCHS} 轮训练 ===")
        epoch_metrics = {"success": 0, "failure": 0}
        
        # 初始化本轮的标签和预测列表
        labels, preds = [], []
        
        # 记录当前处理的样本位置
        current_position = 0
        total_samples = len(original_texts)
        
        # 计算5%的样本数量，确保至少为1
        display_interval = max(1, int(total_samples * 0.05))
        
        # 创建第一个进度条
        pbar = tqdm(range(total_samples), total=total_samples, desc=f"第{epoch+1}轮")
        
        for idx, original_text in enumerate(original_texts):
            # 更新进度条
            pbar.update(1)
            current_position += 1
            
            # 转换为基本字典格式用于现有函数
            basic_examples = enhanced_to_basic_examples(enhanced_examples)
            
            # 生成水印文本 - 直接使用make_prompt_for_watermark_generation，它已经实现了相似度选择
            prompt_gen = make_prompt_for_watermark_generation(
                original_text, basic_examples, max_examples=MAX_PROMPT_EXAMPLES
            )
            orig_tokens = string2token_nums(original_text)  # 单独计算原文token数
            
            watermarked_text = generation_by_qwen(prompt_gen, orig_tokens)

            # 转换检测器示例
            detector_examples = convert_to_detector_examples(basic_examples)
            
            # 创建增强型案例
            new_case = EnhancedCase(original_text, watermarked_text)
            
            # 同时更新标签和置信度，确保一致性
            try:
                new_case.update_confidence_and_label(detector_examples, use_confidence=USE_CONFIDENCE)
            except Exception as e:
                print(f"标签和置信度更新失败: {e}")
                new_case.label = 'Bad'
                new_case.confidence = 0.3  # 失败时使用较低的默认置信度
                new_case.quality_score = calculate_case_quality_score(new_case)
            
            # 记录检测结果 - 用于真实评估
            # 注意：这里使用二分类标签用于评估，而不是"Good"/"Bad"
            if new_case.label == 'Good':
                labels.append("1")  # 水印成功 (预期是Watermarked)
                preds.append("1")   # 预测为Watermarked
            else:
                labels.append("1")  # 水印失败 (预期是Watermarked但未被正确检测)
                preds.append("0")   # 预测为Original

            # 每处理5%的样本显示一次详细过程
            if idx % display_interval == 0:
                current_percent = (idx / total_samples) * 100
                print(f"\n--- 第{epoch+1}轮 {current_percent:.1f}% 进度详细过程 ---")
                print(f"样本索引: {idx}/{total_samples}")
                show_simple_text_diff(original_text, watermarked_text, case_id=idx+1)
                print(f"标签分类: {new_case.label} (Good=成功水印，Bad=失败水印)")
                print(f"置信度: {new_case.confidence:.2%}")
                print(f"质量分数: {new_case.quality_score:.2%} (标签权重={LABEL_WEIGHT}, 置信度权重={CONFIDENCE_WEIGHT})")
                print("----------------------------\n")

            # 使用加权质量分数决定案例分类
            if new_case.quality_score >= QUALITY_THRESHOLD:
                enhanced_examples["success"].append(new_case)
                epoch_metrics["success"] += 1
            else:
                enhanced_examples["failure"].append(new_case)
                epoch_metrics["failure"] += 1

            # 对success列表按质量分数排序
            enhanced_examples["success"].sort(key=lambda x: x.quality_score, reverse=True)
            
            # 保留质量分数最高的MAX_EXAMPLES_SIZE个案例
            if len(enhanced_examples["success"]) > MAX_EXAMPLES_SIZE:
                enhanced_examples["success"] = enhanced_examples["success"][:MAX_EXAMPLES_SIZE]

            # 保存检查点 - 转换为基本格式后保存
            if (idx + 1) % 50 == 0 or idx == len(original_texts) - 1:
                checkpoint_examples = enhanced_to_basic_examples(enhanced_examples)
                save_pkl(checkpoint_examples, os.path.join(args.output_dir, f"checkpoint_{epoch}.pkl"))
            
            # 每隔N个样本，添加一个"原始"样本进行测试
            if idx % 5 == 0:  # 每5个样本添加一个原始文本检测案例
                # 随机选择一篇不同的文章作为原始样本
                random_idx = random.randint(0, len(original_texts)-1)
                while random_idx == idx:  # 确保不是当前处理的文章
                    random_idx = random.randint(0, len(original_texts)-1)
                
                test_original = original_texts[random_idx]
                
                # 评估检测器对原始文本的识别能力
                if USE_CONFIDENCE:
                    original_confidence = get_confidence_voting(
                        test_original,
                        detector_examples,
                        rounds=3
                    )
                    predicted_original = original_confidence < 0.5
                else:
                    # 不使用置信度时，直接进行单次检测
                    prompt = make_prompt_for_detection(test_original, detector_examples)
                    response = call_model_api(prompt).lower()
                    predicted_original = 'original' in response
                    original_confidence = 0.0 if predicted_original else 1.0  # 模拟置信度用于显示
                
                # 记录评估结果
                labels.append("0")  # 真实标签为原始文本
                preds.append("0" if predicted_original else "1")  # 预测
                
                # 打印详细信息（如果在显示间隔内）
                if idx % display_interval == 0:
                    print(f"\n--- 原始文本检测测试 ---")
                    print(f"原始文本: {test_original[:500]}...")
                    print(f"检测置信度: {original_confidence:.2%}")
                    print(f"预测结果: {'Original' if predicted_original else 'Watermarked'}")
                    print(f"使用置信度: {'是' if USE_CONFIDENCE else '否'}")
                    print("----------------------------\n")
                
            # 计算当前指标
            if len(labels) > 0:
                acc = sum([1 for l, p in zip(labels, preds) if l == p]) / len(labels)
                wm_rec = sum([1 for l, p in zip(labels, preds) if l == p == "1"]) / labels.count("1") if labels.count("1") > 0 else 0
                high_quality = len([c for c in enhanced_examples["success"] if c.quality_score > 0.8])
                
                # 更新进度条信息
                pbar.set_postfix({
                    'acc': f"{acc:.2%}",
                    'w_rec': f"{wm_rec:.2%}",
                    'high_q': high_quality
                })
            
            # 每5个样本后保留当前进度条并创建新的进度条
            if (idx + 1) % 5 == 0 and idx < len(original_texts) - 1:
                # 设置当前进度条为保留状态
                pbar.leave = True
                pbar.close()
                
                # 创建新的进度条，但仍然覆盖全部样本
                remaining = total_samples - current_position
                pbar = tqdm(range(remaining), total=total_samples, initial=current_position, 
                           desc=f"第{epoch+1}轮 ({current_position}/{total_samples})")
                
                # 立即设置指标信息以避免空白进度条
                if len(labels) > 0:
                    pbar.set_postfix({
                        'acc': f"{acc:.2%}",
                        'w_rec': f"{wm_rec:.2%}",
                        'high_q': high_quality
                    })

        # 确保最后的进度条正确关闭
        pbar.close()
        
        # 计算本轮最终指标
        human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(labels, preds)
        
        # 转换为字典格式以匹配原有输出
        final_metrics = {
            "accuracy": acc,
            "f1": f1,
            "watermarked_recall": machine_rec / 100,  # 原函数返回百分比，转换为小数
            "original_recall": human_rec / 100,
            "average_recall": avg_rec / 100
        }
        
        # 添加到性能日志
        performance_log.append({
            "epoch": epoch+1,
            "accuracy": f"{final_metrics['accuracy']:.2%}",
            "f1": f"{final_metrics['f1']:.2%}",
            "watermarked_recall": f"{final_metrics['watermarked_recall']:.2%}",
            "original_recall": f"{final_metrics['original_recall']:.2%}",
            "average_recall": f"{final_metrics['average_recall']:.2%}",
            "success_rate": f"{epoch_metrics['success']/len(original_texts):.2%}",
            "high_quality_cases": len([c for c in enhanced_examples["success"] if c.quality_score > 0.8]),
            "avg_quality_score": f"{sum(c.quality_score for c in enhanced_examples['success'])/len(enhanced_examples['success']):.2%}" if enhanced_examples['success'] else "0.00%"
        })

        print(f"\n本轮评估指标:")
        print(f"准确率: {final_metrics['accuracy']:.2%}")
        print(f"F1分数: {final_metrics['f1']:.2%}")
        print(f"水印召回率: {final_metrics['watermarked_recall']:.2%}")
        print(f"原始文本召回率: {final_metrics['original_recall']:.2%}")
        print(f"平均召回率: {final_metrics['average_recall']:.2%}")
        print(f"平均质量分数: {sum(c.quality_score for c in enhanced_examples['success'])/len(enhanced_examples['success']):.2%}" if enhanced_examples['success'] else "0.00%")
        print(f"高质量案例数(>0.8): {len([c for c in enhanced_examples['success'] if c.quality_score > 0.8])}")

    # 最终评估
    print("\n=== 最终评估 ===")
    basic_examples = enhanced_to_basic_examples(enhanced_examples)
    detector_examples = convert_to_detector_examples(basic_examples)
    
    # 获取所有案例的质量分数分布
    quality_scores = [case.quality_score for case in enhanced_examples["success"]]
    confidence_levels = [case.confidence for case in enhanced_examples["success"]]
    
    # 保存最终报告
    final_report = {
        "performance_history": performance_log,
        "quality_distribution": {
            "min": min(quality_scores) if quality_scores else 0,
            "max": max(quality_scores) if quality_scores else 0,
            "mean": sum(quality_scores)/len(quality_scores) if quality_scores else 0,
            "median": sorted(quality_scores)[len(quality_scores)//2] if quality_scores else 0,
            "high_quality_ratio": len([q for q in quality_scores if q > 0.8])/len(quality_scores) if quality_scores else 0
        },
        "confidence_distribution": {
            "min": min(confidence_levels) if confidence_levels else 0,
            "max": max(confidence_levels) if confidence_levels else 0,
            "mean": sum(confidence_levels)/len(confidence_levels) if confidence_levels else 0,
            "median": sorted(confidence_levels)[len(confidence_levels)//2] if confidence_levels else 0,
            "high_conf_ratio": len([c for c in confidence_levels if c > 0.8])/len(confidence_levels) if confidence_levels else 0
        },
        "final_examples_count": {
            "success": len(enhanced_examples["success"]),
            "failure": len(enhanced_examples["failure"])
        },
        "weights_used": {
            "label_weight": LABEL_WEIGHT,
            "confidence_weight": CONFIDENCE_WEIGHT,
            "quality_threshold": QUALITY_THRESHOLD,
            "use_confidence": USE_CONFIDENCE
        }
    }
    
    with open(os.path.join(args.output_dir, "final_report.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n=== 最终统计 ===")
    print(f"成功案例数: {len(enhanced_examples['success'])}")
    print(f"失败案例数: {len(enhanced_examples['failure'])}")
    print(f"高质量案例比例: {final_report['quality_distribution']['high_quality_ratio']:.2%}")
    print(f"平均质量分数: {final_report['quality_distribution']['mean']:.2%}")
    print(f"使用置信度: {'是' if USE_CONFIDENCE else '否'}")
    print(f"使用的权重设置: 标签权重={LABEL_WEIGHT}, 置信度权重={CONFIDENCE_WEIGHT}, 质量阈值={QUALITY_THRESHOLD}")

if __name__ == '__main__':
    main()