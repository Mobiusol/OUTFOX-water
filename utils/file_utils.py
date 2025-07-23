import pickle
import json
from typing import Any, Dict

class FileManager:
    """文件操作管理器"""
    
    @staticmethod
    def load_pkl(path: str) -> Any:
        """加载pickle文件"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"错误：加载pickle文件失败 {path}: {e}")
            raise
    
    @staticmethod
    def save_pkl(obj: Any, path: str) -> None:
        """保存pickle文件"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            print(f"错误：保存pickle文件失败 {path}: {e}")
            raise
    
    @staticmethod
    def load_json(path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(path, mode="rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"错误：加载JSON文件失败 {path}: {e}")
            raise
    
    @staticmethod
    def save_json(obj: Dict, path: str) -> None:
        """保存JSON文件"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"错误：保存JSON文件失败 {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"保存JSON文件失败 {path}: {e}")
            raise
