"""置信度计算实现模块"""
import time
import random
import numpy as np
from collections import Counter
from typing import List, Dict, Any
from .confidence_interface import ConfidenceCalculator
from .prompt_generator import make_prompt_for_detection
from .api_service import call_model_api, remove_common_prefixes


class VotingConfidenceCalculator(ConfidenceCalculator):
    """投票置信度计算器"""
    
    def __init__(self, rounds: int = 5):
        self.rounds = rounds
    
    def calculate_confidence(self, text: str, examples: List[Dict], **kwargs) -> Dict[str, Any]:
        """通过多次检测投票获取置信度"""
        votes = 0
        processed_text = remove_common_prefixes(text)

        if len(examples) == 0:
            return {'prediction': 'original', 'confidence': 0.5, 'method_used': 'voting_fallback'}

        for _ in range(self.rounds):
            sample_size = min(len(examples), 5)
            random_examples = random.sample(examples, sample_size)
            prompt = make_prompt_for_detection(processed_text, random_examples)
            response = call_model_api(prompt).lower()
            if 'watermarked' in response:
                votes += 1
            time.sleep(0.5)

        confidence = votes / self.rounds
        prediction = 'watermarked' if confidence > 0.5 else 'original'
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'method_used': 'voting',
            'rounds_used': self.rounds
        }


class EntropyConfidenceCalculator(ConfidenceCalculator):
    """基于信息熵的置信度计算器"""
    
    def __init__(self, min_rounds: int = 3, max_rounds: int = 7, confidence_threshold: float = 0.8):
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
    
    def calculate_confidence(self, text: str, examples: List[Dict], **kwargs) -> Dict[str, Any]:
        """自适应置信度投票"""
        predictions = []
        confidences = []

        for round_num in range(self.max_rounds):
            prediction, individual_conf = self._single_detection(text, examples, round_num)
            predictions.append(prediction)
            confidences.append(individual_conf)

            # 在最小轮数后检查是否可以提前停止
            if round_num >= self.min_rounds - 1:
                entropy = self._calculate_entropy(predictions)
                consistency = self._calculate_consistency(predictions)

                if consistency >= self.confidence_threshold or entropy < 0.5:
                    break

        final_confidence = self._calculate_weighted_confidence(predictions, confidences)
        final_prediction = self._get_weighted_prediction(predictions, confidences)

        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'rounds_used': len(predictions),
            'entropy': self._calculate_entropy(predictions),
            'method_used': 'entropy'
        }
    
    def _single_detection(self, text: str, examples: List[Dict], round_num: int) -> tuple:
        """单次检测"""
        sample_size = min(len(examples), 5)
        random_examples = random.sample(examples, sample_size)
        prompt = make_prompt_for_detection(text, random_examples)
        response = call_model_api(prompt).lower()
        
        prediction = 'watermarked' if 'watermarked' in response else 'original'
        confidence = 0.8 if 'watermarked' in response else 0.2
        
        return prediction, confidence
    
    def _calculate_entropy(self, predictions: List[str]) -> float:
        """计算预测分布的熵"""
        if not predictions:
            return float('inf')

        counter = Counter(predictions)
        total = len(predictions)
        entropy = 0

        for count in counter.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)

        return entropy
    
    def _calculate_consistency(self, predictions: List[str]) -> float:
        """计算预测一致性"""
        if not predictions:
            return 0.0

        counter = Counter(predictions)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(predictions)
    
    def _calculate_weighted_confidence(self, predictions: List[str], confidences: List[float]) -> float:
        """计算加权置信度"""
        if not predictions:
            return 0.0

        base_confidence = sum(confidences) / len(confidences)
        consistency = self._calculate_consistency(predictions)
        consistency_bonus = consistency * 0.2

        return min(1.0, base_confidence + consistency_bonus)
    
    def _get_weighted_prediction(self, predictions: List[str], confidences: List[float]) -> str:
        """基于置信度加权的最终预测"""
        prediction_scores = {}

        for pred, conf in zip(predictions, confidences):
            if pred not in prediction_scores:
                prediction_scores[pred] = 0.0
            prediction_scores[pred] += conf

        return max(prediction_scores.items(), key=lambda x: x[1])[0]


# 兼容性函数
def get_confidence_voting(text: str, detector_examples: List[Dict], rounds: int = 5) -> float:
    """兼容性函数"""
    calculator = VotingConfidenceCalculator(rounds)
    result = calculator.calculate_confidence(text, detector_examples)
    return result['confidence']
