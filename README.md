# OUTFOX-v2 Water Mark System

基于智能相似度检索的对抗式水印生成与检测系统

## 项目特性

- **智能案例检索**: 使用BERT+TF-IDF混合相似度计算
- **对抗式训练**: 生成器和检测器协同进化
- **质量评估**: 多维度案例质量评分
- **结构化提示**: 成功案例和失败案例分析

## 项目结构

```
OUTFOX-v2/
├── config/           # 配置文件
├── services/         # 核心服务
├── utils/           # 工具类
├── models/          # 数据模型
├── src/             # 主要源码
└── data/            # 数据文件
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行训练
python src/detection/outfox_detection_with_considering_attack.py
```

## 配置说明

主要配置文件在 `config/settings.py` 中，包含：
- API配置
- 水印参数
- 相似度阈值
- 系统参数

## 更新日志

- v2.0: 集成智能相似度检索系统
- v2.1: 优化案例分类逻辑，移除文本差异性对成功/失败判断的影响
