"""
系统配置模块 - 单一职责原则
"""

class DetectionConfig:
    """检测系统配置"""
    
    # 模式配置
    USE_CONFIDENCE = False
    FAST_MODE = False
    DEBUG_MODE = False
    
    # 训练配置
    MAX_EPOCHS = 2
    MAX_EXAMPLES_SIZE = 200
    MAX_PROMPT_EXAMPLES = 10
    
    # 质量评估配置
    SIMILARITY_THRESHOLD = 0.7
    QUALITY_THRESHOLD = 0.6
    
    # 权重配置
    LABEL_WEIGHT = 0.8
    CONFIDENCE_WEIGHT = 0.2
    
    @classmethod
    def set_mode(cls, mode: str):
        """设置运行模式"""
        if mode == "debug":
            cls.DEBUG_MODE = True
            cls.USE_CONFIDENCE = False
            cls.MAX_EPOCHS = 2
            cls.MAX_EXAMPLES_SIZE = 20
        elif mode == "fast":
            cls.FAST_MODE = True
            cls.USE_CONFIDENCE = False
            cls.MAX_EPOCHS = 5
            cls.MAX_EXAMPLES_SIZE = 100
        else:  # standard
            cls.MAX_EPOCHS = 2
            cls.MAX_EXAMPLES_SIZE = 200
    
    @classmethod
    def get_status(cls):
        """获取当前配置状态"""
        return {
            'mode': 'debug' if cls.DEBUG_MODE else 'fast' if cls.FAST_MODE else 'standard',
            'use_confidence': cls.USE_CONFIDENCE,
            'max_epochs': cls.MAX_EPOCHS,
            'max_examples': cls.MAX_EXAMPLES_SIZE
        }
