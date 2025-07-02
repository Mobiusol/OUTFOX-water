#!/usr/bin/env python3
"""
测试脚本：验证所有导入和基本功能是否正常工作
"""

print("开始测试导入...")

try:
    # 测试基础导入
    import sys
    import os
    
    # 添加项目路径
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    
    print("✓ 基础导入成功")
    
    # 测试utils模块
    from utils.utils import (
        load_pkl, save_pkl, call_model_api, compute_metrics,
        get_confidence_voting, get_cached_embedding, find_similar_cases,
        get_comprehensive_similarity, make_prompt_for_detection,
        make_prompt_for_watermark_generation, generation_by_qwen,
        convert_to_detector_examples, identify_watermark_effectiveness
    )
    print("✓ utils模块导入成功")
    
    # 测试基本功能
    test_text1 = "这是一个测试文本。"
    test_text2 = "这是另一个测试文本。"
    
    # 测试嵌入功能
    embedding = get_cached_embedding(test_text1)
    print(f"✓ 文本嵌入功能正常，嵌入维度: {embedding.shape}")
    
    # 测试相似度计算
    similarity = get_comprehensive_similarity(test_text1, test_text2)
    print(f"✓ 相似度计算功能正常，相似度: {similarity:.3f}")
    
    # 测试案例查找
    test_cases = [
        {'original': test_text1, 'watermarked': test_text1 + " 修改版本"},
        {'original': test_text2, 'watermarked': test_text2 + " 另一个修改版本"}
    ]
    similar_cases = find_similar_cases(test_text1, test_cases, top_k=1)
    print(f"✓ 案例查找功能正常，找到 {len(similar_cases)} 个相似案例")
    
    # 测试主模块导入
    from src.detection.outfox_detection_with_considering_attack import (
        EnhancedCase, calculate_case_quality_score, filter_by_similarity
    )
    print("✓ 主模块导入成功")
    
    # 测试EnhancedCase
    test_case = EnhancedCase(test_text1, test_text2)
    print(f"✓ EnhancedCase创建成功，质量分数: {test_case.quality_score:.3f}")
    
    print("\n🎉 所有测试通过！代码可以正常运行。")
    print("\n建议安装以下依赖以获得最佳性能：")
    print("pip install sentence-transformers>=2.2.0")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖都已安装：pip install -r requirements.txt")
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()
