from abc import ABC, abstractmethod
import requests
import backoff
import time
from config.settings import api_config

class APIServiceInterface(ABC):
    """API服务接口"""
    
    @abstractmethod
    def call_api(self, prompt: str, **kwargs) -> str:
        """调用API"""
        pass

class DashScopeAPIService(APIServiceInterface):
    """百炼API服务实现"""
    
    def __init__(self):
        self.api_key = api_config.dashscope_api_key
        self.endpoint = api_config.dashscope_endpoint

    @backoff.on_exception(backoff.expo, Exception, max_time=60, max_tries=api_config.max_retries, base=api_config.retry_delay)
    def call_api(self, prompt: str, **kwargs) -> str:
        """调用百炼API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": api_config.model_name,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "temperature": kwargs.get("temperature", api_config.default_temperature),
                "top_p": kwargs.get("top_p", api_config.default_top_p),
                "max_tokens": kwargs.get("max_tokens", api_config.default_max_tokens)
            }
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()  # 添加状态检查
            result = response.json()
            
            # 添加详细的响应检查
            if "output" not in result:
                print(f"API响应格式错误: {result}")
                if "error" in result:
                    raise Exception(f"API错误: {result['error']}")
                else:
                    raise Exception(f"响应中缺少output字段: {result}")
            
            return result["output"]["text"]
        except requests.exceptions.RequestException as e:
            print(f"HTTP请求失败: {e}")
            raise
        except KeyError as e:
            print(f"响应解析失败: {e}, 完整响应: {result}")
            raise
        except Exception as e:
            print(f"API调用失败: {e}")
            raise

class APIServiceFactory:
    """API服务工厂"""
    
    @staticmethod
    def create_service(service_type: str = "dashscope") -> APIServiceInterface:
        """创建API服务实例"""
        if service_type == "dashscope":
            return DashScopeAPIService()
        else:
            raise ValueError(f"不支持的服务类型: {service_type}")
