"""
GTO-GameFlow v5.11 — 多模型集成编排器

路径4核心编排器: 将所有预测器组合成一个统一的预测流水线。

数据流:
    MatchContext → EloPredictor → elo_probs
    MatchContext → DixonColesPredictor → dc_probs
    MatchContext → XGBoostPredictor → xgb_probs
    MatchContext → MarketOddsPredictor → market_probs
    
    [elo_probs, dc_probs, xgb_probs, market_probs] → EnsembleDecisionGate → 最终决策
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .elo_predictor import EloPredictor
from .dixon_coles_predictor import DixonColesPredictor
from .xgboost_predictor import XGBoostPredictor
from .market_odds_predictor import MarketOddsPredictor
from .ensemble_gate import EnsembleDecisionGate, ModelPrediction, EnsembleResult

logger = logging.getLogger(__name__)


class EnsembleOrchestrator:
    """
    多模型集成编排器。
    
    管理所有预测器，收集预测结果，通过集成决策门做出最终决策。
    
    使用方式:
        orchestrator = EnsembleOrchestrator(league_id="premier_league")
        result = orchestrator.predict(match_context)
        if result.is_approved:
            # 批准投注
    """
    
    def __init__(
        self,
        league_id: str = "",
        min_agreement: int = 3,
        confidence_threshold: float = 0.55,
        value_threshold: float = 0.03,
        enable_elo: bool = True,
        enable_dc: bool = True,
        enable_xgb: bool = True,
        enable_market: bool = True,
        xgb_model_path: Optional[str] = None,
    ):
        """
        参数:
            league_id: 联赛ID
            min_agreement: 最少同意模型数
            confidence_threshold: 平均置信度阈值
            value_threshold: 价值阈值
            enable_elo: 启用 Elo 预测器
            enable_dc: 启用 Dixon-Coles 预测器
            enable_xgb: 启用 XGBoost 预测器
            enable_market: 启用市场赔率预测器
            xgb_model_path: XGBoost 模型路径
        """
        self.league_id = league_id
        
        # 初始化预测器
        self.predictors = []
        
        if enable_elo:
            self.predictors.append(("elo", EloPredictor(league_id=league_id)))
        
        if enable_dc:
            self.predictors.append(("dixon_coles", DixonColesPredictor(league_id=league_id)))
        
        if enable_xgb:
            self.predictors.append(("xgboost", XGBoostPredictor(
                league_id=league_id,
                model_path=xgb_model_path,
            )))
        
        if enable_market:
            self.predictors.append(("market_odds", MarketOddsPredictor(league_id=league_id)))
        
        # 初始化集成决策门
        self.gate = EnsembleDecisionGate(
            min_agreement=min_agreement,
            confidence_threshold=confidence_threshold,
            value_threshold=value_threshold,
        )
        
        logger.info(f"集成编排器初始化完成: {len(self.predictors)} 个预测器")
    
    def predict(
        self,
        match_context: Dict[str, Any],
        market_probs: Optional[Dict[str, float]] = None,
    ) -> EnsembleResult:
        """
        执行多模型集成预测。
        
        参数:
            match_context: 比赛上下文 (包含所有必要信息)
            market_probs: 市场隐含概率
        
        返回:
            EnsembleResult
        """
        # 收集各模型预测
        predictions = []
        
        for model_name, predictor in self.predictors:
            try:
                probs = predictor.predict_from_context(match_context)
                
                # 计算模型置信度
                confidence = self._calculate_confidence(probs, model_name)
                
                pred = ModelPrediction(
                    model_name=model_name,
                    probs=probs,
                    confidence=confidence,
                    metadata={"model_type": type(predictor).__name__},
                )
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 预测失败: {e}")
                continue
        
        # 通过集成决策门
        result = self.gate.evaluate(
            predictions=predictions,
            market_probs=market_probs,
            strategy="1x2",
        )
        
        return result
    
    def predict_for_totals(
        self,
        match_context: Dict[str, Any],
        market_probs: Optional[Dict[str, float]] = None,
        totals_line: float = 2.5,
    ) -> EnsembleResult:
        """
        执行大小球策略的多模型集成预测。
        
        参数:
            match_context: 比赛上下文
            market_probs: 市场隐含概率
            totals_line: 大小球线
        
        返回:
            EnsembleResult
        """
        predictions = []
        
        for model_name, predictor in self.predictors:
            try:
                probs = predictor.predict_from_context(match_context)
                
                # 计算大小球概率
                avg_goals = match_context.get("avg_goals", 2.65)
                over_prob = self._calculate_over_prob(avg_goals, totals_line)
                under_prob = 1.0 - over_prob
                
                confidence = self._calculate_confidence(probs, model_name)
                
                pred = ModelPrediction(
                    model_name=model_name,
                    probs=probs,
                    confidence=confidence,
                    metadata={
                        "over_prob": over_prob,
                        "under_prob": under_prob,
                        "totals_line": totals_line,
                    },
                )
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 预测失败: {e}")
                continue
        
        result = self.gate.evaluate_for_totals(
            predictions=predictions,
            market_probs=market_probs,
            totals_line=totals_line,
        )
        
        return result
    
    def get_all_predictions(
        self,
        match_context: Dict[str, Any],
    ) -> List[ModelPrediction]:
        """
        获取所有模型的预测结果 (不做集成决策)。
        
        参数:
            match_context: 比赛上下文
        
        返回:
            各模型的预测结果列表
        """
        predictions = []
        
        for model_name, predictor in self.predictors:
            try:
                probs = predictor.predict_from_context(match_context)
                confidence = self._calculate_confidence(probs, model_name)
                
                pred = ModelPrediction(
                    model_name=model_name,
                    probs=probs,
                    confidence=confidence,
                    metadata={"model_type": type(predictor).__name__},
                )
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 预测失败: {e}")
                continue
        
        return predictions
    
    def _calculate_confidence(
        self,
        probs: Dict[str, float],
        model_name: str,
    ) -> float:
        """
        计算模型置信度。
        
        基于概率分布的熵: 熵越低，置信度越高。
        """
        import math
        
        # 计算熵
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log2(p)
        
        # 最大熵 (均匀分布)
        max_entropy = math.log2(len(probs))
        
        # 归一化到 0~1
        if max_entropy > 0:
            confidence = 1.0 - (entropy / max_entropy)
        else:
            confidence = 0.5
        
        return round(confidence, 4)
    
    def _calculate_over_prob(
        self,
        avg_goals: float,
        totals_line: float,
    ) -> float:
        """
        计算大小球 over 概率。
        
        使用泊松分布近似。
        """
        import math
        
        # 泊松分布 P(goals > line)
        over_prob = 0.0
        for k in range(int(totals_line) + 1, 20):
            pmf = math.exp(-avg_goals) * (avg_goals ** k) / math.factorial(k)
            over_prob += pmf
        
        return round(over_prob, 4)
    
    def get_model_names(self) -> List[str]:
        """获取所有模型名称"""
        return [name for name, _ in self.predictors]
    
    def get_model_count(self) -> int:
        """获取模型数量"""
        return len(self.predictors)


def create_ensemble_orchestrator(
    league_id: str = "",
    min_agreement: int = 3,
    confidence_threshold: float = 0.55,
    value_threshold: float = 0.03,
) -> EnsembleOrchestrator:
    """创建集成编排器的便捷函数"""
    return EnsembleOrchestrator(
        league_id=league_id,
        min_agreement=min_agreement,
        confidence_threshold=confidence_threshold,
        value_threshold=value_threshold,
    )
