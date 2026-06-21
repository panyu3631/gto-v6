"""
GTO-GameFlow v5.11 — 多模型集成模块

路径4方案: 多个独立模型分别预测，只在多模型同时指向同一方向时才投注。

模型列表:
1. EloPredictor — 基于 Elo 评分差的概率预测
2. DixonColesPredictor — 基于泊松 + Dixon-Coles 的比分概率预测
3. XGBoostPredictor — 基于梯度提升的特征工程预测
4. MarketOddsPredictor — 基于市场赔率的隐含概率预测

集成方式:
- 每个模型独立输出 {home, draw, away} 概率
- 只有当 ≥N 个模型对同一方向（主胜/平/客胜）的置信度 > 阈值时才投注
- 目标: 大幅减少投注频率，提高每次投注的置信度
"""

from .elo_predictor import EloPredictor
from .dixon_coles_predictor import DixonColesPredictor
from .xgboost_predictor import XGBoostPredictor
from .market_odds_predictor import MarketOddsPredictor
from .ensemble_gate import EnsembleDecisionGate, ModelPrediction, EnsembleResult

__all__ = [
    "EloPredictor",
    "DixonColesPredictor",
    "XGBoostPredictor",
    "MarketOddsPredictor",
    "EnsembleDecisionGate",
    "ModelPrediction",
    "EnsembleResult",
]
