"""
GTO-GameFlow v5.11 — XGBoost 梯度提升预测器

基于 XGBoost 梯度提升算法的特征工程预测器。
使用现有的因子计算引擎作为特征来源。

模型特点:
- 使用 60+ 个因子作为特征
- 自动特征选择 (LASSO)
- 支持增量训练
- 可解释性强 (特征重要性)

使用方式:
    predictor = XGBoostPredictor(league_id="premier_league")
    predictor.train(training_data)
    probs = predictor.predict(match_context)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    XGBoost 梯度提升预测器。
    
    使用因子计算引擎的输出作为特征，训练 XGBoost 模型预测 1X2 结果。
    优势: 能捕捉非线性关系，自动特征选择。
    劣势: 需要大量训练数据，可能过拟合。
    """
    
    # 特征列表 (与 FactorComputationEngine 对齐)
    FEATURE_NAMES = [
        # 基础因子 F1-F18
        "F1_elo_diff", "F2_recent_form", "F3_h2h", "F4_fatigue",
        "F5_rank", "F6_goal_diff", "F7_xg_diff", "F8_market_move",
        "F9_weather", "F12_ref_style", "F13_coach_change",
        "F14_rotation", "F15_derby", "F16_style_matchup",
        "F17_streak", "F18_player_form",
        # 增强因子 F19-F32
        "F19_market_sentiment", "F20_odds_std", "F21_nlp_sentiment",
        "F22_time_decay", "F23_league_strength", "F24_poisson_correction",
        "F25_handicap_depth", "F26_totals_trend", "F27_value_signal",
        "F28_contrarian", "F29_market_efficiency", "F30_motivation",
        "F31_financial", "F32_winter_break",
        # 联赛特定因子 F33-F41
        "F33_christmas", "F34_complacency", "F35_streak_league",
        "F36_position", "F37_promoted", "F38_schedule",
        "F39_derby_intensity", "F40_ht_momentum", "F41_shot_eff",
        # v5.10.8 比赛统计衍生因子
        "F42_territorial", "F43_discipline", "F44_odds_drift",
        "F45_market_disagree", "F46_ref_bias", "F47_comeback",
        "F48_streak_enriched", "F49_goal_vol", "F50_corner_dom",
        "F51_sot_rate", "F52_ah_drift", "F53_totals_drift",
        # v5.11 平局专属因子
        "F56_draw_tactical", "F57_draw_goal_exp", "F58_draw_tendency",
    ]
    
    def __init__(
        self,
        league_id: str = "",
        model_path: Optional[str] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        参数:
            league_id: 联赛ID
            model_path: 预训练模型路径 (None=需要训练)
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率
            subsample: 行采样比例
            colsample_bytree: 列采样比例
            random_state: 随机种子
        """
        self.league_id = league_id
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        
        # 如果提供了模型路径，尝试加载
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        训练 XGBoost 模型。
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,) — 0=主胜, 1=平, 2=客胜
            feature_names: 特征名称列表
            eval_set: 验证集 (X_val, y_val)
        
        返回:
            训练统计信息
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost 未安装，无法训练模型")
            return {"error": "xgboost not installed"}
        
        if feature_names is None:
            feature_names = self.FEATURE_NAMES[:X.shape[1]]
        
        # 创建 DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
        
        # 参数
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.random_state,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        
        # 训练
        evals = [(dtrain, "train")]
        if eval_set:
            dval = xgb.DMatrix(eval_set[0], label=eval_set[1], feature_names=feature_names)
            evals.append((dval, "val"))
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=10 if eval_set else None,
            verbose_eval=False,
        )
        
        self.is_trained = True
        
        # 计算特征重要性
        self.feature_importance = self.model.get_score(importance_type="gain")
        
        stats = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "best_iteration": getattr(self.model, "best_iteration", self.n_estimators),
            "feature_importance_top10": sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
        
        logger.info(f"XGBoost 训练完成: {stats}")
        return stats
    
    def predict(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        预测 1X2 概率。
        
        参数:
            features: 特征向量 (n_features,)
            feature_names: 特征名称列表
            **kwargs: 其他参数 (忽略)
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        if not self.is_trained or self.model is None:
            logger.warning("XGBoost 模型未训练，返回均匀分布")
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        try:
            import xgboost as xgb
        except ImportError:
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        if feature_names is None:
            feature_names = self.FEATURE_NAMES[:features.shape[0]]
        
        # 转换为 DMatrix
        if features.ndim == 1:
            features = features.reshape(1, -1)
        dtest = xgb.DMatrix(features, feature_names=feature_names)
        
        # 预测
        probs = self.model.predict(dtest)[0]  # [P_home, P_draw, P_away]
        
        return {
            "home": round(float(probs[0]), 6),
            "draw": round(float(probs[1]), 6),
            "away": round(float(probs[2]), 6),
        }
    
    def predict_from_context(self, match_context: dict) -> Dict[str, float]:
        """
        从比赛上下文预测。
        
        参数:
            match_context: 包含因子值的字典
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        if not self.is_trained:
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        # 从上下文提取特征
        features = self._extract_features(match_context)
        return self.predict(features)
    
    def _extract_features(self, match_context: dict) -> np.ndarray:
        """从比赛上下文提取特征向量"""
        features = []
        for fname in self.FEATURE_NAMES:
            # 尝试多种键名格式
            value = (
                match_context.get(fname, 0.0) or
                match_context.get(fname.lower(), 0.0) or
                match_context.get(fname.replace("F", "factor_"), 0.0) or
                0.0
            )
            features.append(float(value))
        return np.array(features)
    
    def save_model(self, path: str) -> None:
        """保存模型到文件"""
        if self.model is None:
            logger.warning("没有可保存的模型")
            return
        
        model_data = {
            "model": self.model,
            "feature_importance": self.feature_importance,
            "league_id": self.league_id,
            "is_trained": self.is_trained,
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str) -> bool:
        """从文件加载模型"""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.feature_importance = model_data.get("feature_importance")
            self.league_id = model_data.get("league_id", self.league_id)
            self.is_trained = model_data.get("is_trained", True)
            
            logger.info(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.feature_importance:
            return {}
        return dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        ))
