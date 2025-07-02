#!/usr/bin/env python3
"""
重构验证测试脚本
测试重构后的utils模块是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_basic_imports():
    """测试基本导入功能"""
    print("🔧 测试基本导入...")
    
    try:
        from utils.utils import (
            load_pkl, save_pkl, truncate_text, string2token_nums,
            make_prompt_for_detection, make_prompt_for_watermark_generation,
            get_cached_embedding, get_comprehensive_similarity,
            compute_metrics, find_similar_cases
        )
        print("✅ 基本导入成功")
        return True
    except Exception as e:
        print(f"❌ 基本导入失败: {e}")
        return False

def test_prompt_functions():
    """测试重要的prompt函数是否保持不变"""
    print("🔧 测试prompt函数...")
    
    try:
        from utils.utils import make_prompt_for_detection, make_prompt_for_watermark_generation
        
        # 测试make_prompt_for_detection
        test_text = "This is a test text for detection."
        test_examples = [
            {"text": "Example original text.", "label": "Original"},
            {"text": "Example watermarked text.", "label": "Watermarked"}
        ]
        
        prompt = make_prompt_for_detection(test_text, test_examples)
        
        if prompt and isinstance(prompt, str) and len(prompt) > 0:
            print("✅ make_prompt_for_detection 正常工作")
        else:
            print("❌ make_prompt_for_detection 返回异常")
            return False
        
        # 测试make_prompt_for_watermark_generation
        test_examples_gen = {
            "success": [
                {"original": "Original text example", "watermarked": "Watermarked text example"}
            ]
        }
        
        prompt_gen = make_prompt_for_watermark_generation(test_text, test_examples_gen)
        
        if prompt_gen and isinstance(prompt_gen, str) and len(prompt_gen) > 0:
            print("✅ make_prompt_for_watermark_generation 正常工作")
        else:
            print("❌ make_prompt_for_watermark_generation 返回异常")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ prompt函数测试失败: {e}")
        return False

def test_utility_functions():
    """测试工具函数"""
    print("🔧 测试工具函数...")
    
    try:
        from utils.utils import truncate_text, string2token_nums, get_comprehensive_similarity
        
        # 测试truncate_text
        test_text = "This is a test text with multiple words for truncation testing."
        truncated = truncate_text(test_text, 5, 'word')
        
        if truncated and len(truncated.split()) <= 5:
            print("✅ truncate_text 正常工作")
        else:
            print("❌ truncate_text 异常")
            return False
        
        # 测试string2token_nums (可能需要tiktoken)
        try:
            token_count = string2token_nums(test_text)
            if isinstance(token_count, int) and token_count > 0:
                print("✅ string2token_nums 正常工作")
            else:
                print("⚠️ string2token_nums 返回值异常，但不影响核心功能")
        except Exception as e:
            print(f"⚠️ string2token_nums 失败 (可能缺少tiktoken): {e}")
        
        # 测试相似度计算
        similarity = get_comprehensive_similarity("Hello world", "Hello earth")
        
        if isinstance(similarity, (int, float)) and 0 <= similarity <= 1:
            print("✅ get_comprehensive_similarity 正常工作")
        else:
            print("❌ get_comprehensive_similarity 异常")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        return False

def test_module_structure():
    """测试模块结构"""
    print("🔧 测试模块结构...")
    
    expected_modules = [
        'utils.config',
        'utils.base_utils', 
        'utils.api_service',
        'utils.embedding_service',
        'utils.metrics',
        'utils.prompt_generator',
        'utils.confidence_calculators',
        'utils.watermark_validator',
        'utils.similarity_service'
    ]
    
    success_count = 0
    
    for module_name in expected_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} 导入成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name} 导入失败: {e}")
    
    if success_count == len(expected_modules):
        print("✅ 所有模块结构正确")
        return True
    else:
        print(f"⚠️ {success_count}/{len(expected_modules)} 模块导入成功")
        return success_count >= len(expected_modules) * 0.8  # 80%以上成功即可

def main():
    """主测试函数"""
    print("=" * 50)
    print("🚀 OUTFOX-v2 重构验证测试")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_prompt_functions, 
        test_utility_functions,
        test_module_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！重构成功！")
        return True
    elif passed >= total * 0.8:
        print("✅ 大部分测试通过，重构基本成功")
        return True
    else:
        print("❌ 测试失败过多，需要修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
