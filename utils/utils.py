"""
遗留utils模块 - 仅用于向后兼容
新代码请使用模块化的导入方式
"""
import warnings

# 发出弃用警告
warnings.warn(
    "utils.utils模块已弃用，请使用新的模块化导入方式",
    DeprecationWarning,
    stacklevel=2
)

# 为向后兼容保留的导入
from utils.file_utils import FileManager
from utils.text_utils import TextProcessor
from utils.metrics_utils import MetricsCalculator
from services.api_service import APIServiceFactory

# 创建全局实例以保持向后兼容
_file_manager = FileManager()
_text_processor = TextProcessor()
_metrics_calculator = MetricsCalculator()

# 向后兼容的函数别名
load_pkl = _file_manager.load_pkl
save_pkl = _file_manager.save_pkl
get_cached_embedding = _text_processor.get_cached_embedding
find_similar_cases = _text_processor.find_similar_cases
compute_metrics = _metrics_calculator.compute_metrics