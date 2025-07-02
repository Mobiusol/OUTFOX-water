"""指标计算模块"""
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_three_recalls(labels: List[str], preds: List[str]) -> Tuple[float, float, float]:
    """计算三种召回率"""
    all_n, all_p, tn, tp = 0, 0, 0, 0
    
    for label, pred in zip(labels, preds):
        if label == '0':
            all_p += 1
        if label == '1':
            all_n += 1
        if label == pred == '0':
            tp += 1
        if label == pred == '1':
            tn += 1
    
    human_rec = tp * 100 / all_p if all_p > 0 else 0
    machine_rec = tn * 100 / all_n if all_n > 0 else 0
    avg_rec = (human_rec + machine_rec) / 2
    
    return (human_rec, machine_rec, avg_rec)


def compute_metrics(labels: List[str], preds: List[str]) -> Tuple[float, float, float, float, float, float, float]:
    """计算各种指标"""
    human_rec, machine_rec, avg_rec = compute_three_recalls(labels, preds)
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label='1')
    recall = recall_score(labels, preds, pos_label='1')
    f1 = f1_score(labels, preds, pos_label='1')
    
    return (human_rec, machine_rec, avg_rec, acc, precision, recall, f1)
