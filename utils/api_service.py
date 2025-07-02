"""API服务模块 - 模型调用和文本处理"""
import time
import requests
import backoff
import re
from typing import List, Optional

from .config import DASHSCOPE_API_KEY, DASHSCOPE_ENDPOINT, COMMON_PREFIXES


@backoff.on_exception(backoff.expo, Exception, max_time=60, max_tries=5)
def completions_with_backoff(**kwargs):
    """带重试机制的API调用"""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-turbo",
        "input": {
            "prompt": kwargs.get("prompt"),
            "messages": kwargs.get("messages")
        },
        "parameters": {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
    }
    response = requests.post(DASHSCOPE_ENDPOINT, headers=headers, json=payload)
    result = response.json()
    return result["output"]["text"]


def remove_common_prefixes(text: str) -> str:
    """移除常见前缀"""
    processed_text = text.strip()
    for prefix in COMMON_PREFIXES:
        if processed_text.startswith(prefix):
            processed_text = processed_text.replace(prefix, "", 1).strip()
            break
    return processed_text


def generation_by_qwen(prompt: str, original_text_tokens: int) -> str:
    """使用Qwen生成文本"""
    while True:
        try:
            lm_essay = completions_with_backoff(
                prompt=prompt,
                temperature=0.8,
                top_p=1,
                max_tokens=original_text_tokens + 50
            )
            break
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            time.sleep(3)

    return remove_common_prefixes(lm_essay)


def call_model_api(prompt: str) -> str:
    """调用模型API进行检测"""
    res = completions_with_backoff(
        prompt=prompt,
        temperature=0.1,
        top_p=0.9,
        max_tokens=10
    )
    return res.strip().lower()


def process_reply_from_qwen(res: str) -> Optional[str]:
    """处理Qwen的回复"""
    if not res:
        return None
    res = res.lower()
    if 'original' in res and 'watermarked' not in res:
        return '0'
    elif 'watermarked' in res and 'original' not in res:
        return '1'
    else:
        return None
