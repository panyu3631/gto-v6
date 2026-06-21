"""
GTO-GameFlow v5.11 — 多模型集成决策门 (Ensemble Decision Gate)

路径4核心组件: 多个独立模型分别预测，只在多模型同时指向同一方向时才投注。

集成策略:
1. 每个模型独立输出 {home, draw, away} 概率
2. 计算每个方向的"投票数"和"平均置信度"
3. 只有当 ≥N 个模型对同一方向的置信度 > 阈值时才投注
4. 目标: 大幅减少投注频率，提高每次投注的置信度

投票规则:
- 方向一致性: ≥min_agreement 个模型指向同一方向
- 置信度门槛: 平均置信度 > confidence_threshold
- 价值门槛: 最大概率 - 市场隐含概率 > value_threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """单个模型的预测结果"""
    model_name: str
    probs: Dict[str, float]  # {"home": P, "draw": P, "away": P}
    confidence: float = 0.0  # 模型置信度 (0~1)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """集成决策结果"""
    # 最终决策
    selected_direction: Optional[str] = None  # "home" / "draw" / "away" / None
    ensemble_prob: float = 0.0  # 集成后的概率
    market_prob: float = 0.0  # 市场隐含概率
    value: float = 0.0  # 价值 = ensemble_prob - market_prob
    
    # 投票统计
    agreement_count: int = 0  # 同意该方向的模型数
    total_models: int = 0  # 总模型数
    avg_confidence: float = 0.0  # 平均置信度
    
    # 各模型详情
    predictions: List[ModelPrediction] = field(default_factory=list)
    
    # 决策原因
    reason: str = ""
    
    @property
    def is_approved(self) -> bool:
        """是否通过集成决策门"""
        return self.selected_direction is not None
    
    def summary(self) -> dict:
        return {
            "approved": self.is_approved,
            "direction": self.selected_direction,
            "ensemble_prob": self.ensemble_prob,
            "market_prob": self.market_prob,
            "value": self.value,
            "agreement": f"{self.agreement_count}/{self.total_models}",
            "avg_confidence": self.avg_confidence,
            "reason": self.reason,
        }


class EnsembleDecisionGate:
    """
    多模型集成决策门。
    
    只有当多个模型同时指向同一方向时才批准投注。
    
    使用方式:
        gate = EnsembleDecisionGate(min_agreement=3, confidence_threshold=0.55)
        result = gate.evaluate(predictions, market_probs)
        if result.is_approved:
            # 批准投注
    """
    
    def __init__(
        self,
        min_agreement: int = 3,
        confidence_threshold: float = 0.55,
        value_threshold: float = 0.03,
        min_models: int = 4,
        max_models: int = 5,
        direction_weights: Optional[Dict[str, float]] = None,
    ):
        """
        参数:
            min_agreement: 最少同意模型数 (≥此值才投注)
            confidence_threshold: 平均置信度阈值
            value_threshold: 价值阈值 (ensemble_prob - market_prob)
            min_models: 最少模型数 (不足则不投注)
            max_models: 最大模型数
            direction_weights: 方向权重 (可选)
        """
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self.value_threshold = value_threshold
        self.min_models = min_models
        self.max_models = max_models
        self.direction_weights = direction_weights or {
            "home": 1.0,
            "draw": 1.0,
            "away": 1.0,
        }
    
    def evaluate(
        self,
        predictions: List[ModelPrediction],
        market_probs: Optional[Dict[str, float]] = None,
        strategy: str = "1x2",
    ) -> EnsembleResult:
        """
        评估多模型预测，返回集成决策。
        
        参数:
            predictions: 各模型的预测结果列表
            market_probs: 市场隐含概率 {"home": P, "draw": P, "away": P}
            strategy: 策略类型 ("1x2" / "asian_handicap" / "over_under")
        
        返回:
            EnsembleResult
        """
        result = EnsembleResult(
            predictions=predictions,
            total_models=len(predictions),
        )
        
        # 检查模型数量
        if len(predictions) < self.min_models:
            result.reason = f"模型数量不足 ({len(predictions)} < {self.min_models})"
            return result
        
        # 设置市场概率
        if market_probs:
            result.market_prob = market_probs.get("home", 0.33)
        
        # 计算每个方向的投票
        direction_votes = self._count_direction_votes(predictions)
        
        # 找到票数最多的方向
        best_direction = max(direction_votes.items(), key=lambda x: x[1]["count"])
        direction_name, direction_data = best_direction
        
        result.agreement_count = direction_data["count"]
        result.avg_confidence = direction_data["avg_confidence"]
        result.ensemble_prob = direction_data["avg_prob"]
        
        # 检查是否满足所有条件
        reasons = []
        
        # 条件1: 方向一致性
        if direction_data["count"] < self.min_agreement:
            reasons.append(f"同意数不足 ({direction_data['count']} < {self.min_agreement})")
        
        # 条件2: 置信度门槛
        if direction_data["avg_confidence"] < self.confidence_threshold:
            reasons.append(f"置信度不足 ({direction_data['avg_confidence']:.3f} < {self.confidence_threshold})")
        
        # 条件3: 价值门槛
        if market_probs:
            market_prob = market_probs.get(direction_name, 0.33)
            value = direction_data["avg_prob"] - market_prob
            result.value = value
            result.market_prob = market_prob
            
            if value < self.value_threshold:
                reasons.append(f"价值不足 ({value:.4f} < {self.value_threshold})")
        
        # 所有条件满足
        if not reasons:
            result.selected_direction = direction_name
            result.reason = f"通过: {direction_name} 方向, {direction_data['count']}/{len(predictions)} 模型同意"
        else:
            result.reason = "; ".join(reasons)
        
        return result
    
    def evaluate_for_totals(
        self,
        predictions: List[ModelPrediction],
        market_probs: Optional[Dict[str, float]] = None,
        totals_line: float = 2.5,
    ) -> EnsembleResult:
        """
        评估大小球策略的多模型预测。
        
        参数:
            predictions: 各模型的预测结果列表
            market_probs: 市场隐含概率 {"over": P, "under": P}
            totals_line: 大小球线
        
        返回:
            EnsembleResult
        """
        result = EnsembleResult(
            predictions=predictions,
            total_models=len(predictions),
        )
        
        if len(predictions) < self.min_models:
            result.reason = f"模型数量不足 ({len(predictions)} < {self.min_models})"
            return result
        
        # 计算 over/under 投票
        over_votes = 0
        under_votes = 0
        over_probs = []
        under_probs = []
        over_confidences = []
        under_confidences = []
        
        for pred in predictions:
            # 从比分概率推导大小球概率
            over_prob = pred.metadata.get("over_prob", 0.0)
            under_prob = pred.metadata.get("under_prob", 0.0)
            
            if over_prob > under_prob:
                over_votes += 1
                over_probs.append(over_prob)
                over_confidences.append(pred.confidence)
            else:
                under_votes += 1
                under_probs.append(under_prob)
                under_confidences.append(pred.confidence)
        
        # 选择票数最多的方向
        if over_votes >= under_votes:
            direction = "over"
            count = over_votes
            avg_prob = sum(over_probs) / len(over_probs) if over_probs else 0.0
            avg_conf = sum(over_confidences) / len(over_confidences) if over_confidences else 0.0
        else:
            direction = "under"
            count = under_votes
            avg_prob = sum(under_probs) / len(under_probs) if under_probs else 0.0
            avg_conf = sum(under_confidences) / len(under_confidences) if under_confidences else 0.0
        
        result.selected_direction = direction if count >= self.min_agreement else None
        result.agreement_count = count
        result.ensemble_prob = avg_prob
        result.avg_confidence = avg_conf
        
        if market_probs:
            result.market_prob = market_probs.get(direction, 0.5)
            result.value = avg_prob - result.market_prob
        
        if result.selected_direction:
            result.reason = f"通过: {direction} 方向, {count}/{len(predictions)} 模型同意"
        else:
            result.reason = f"同意数不足 ({count} < {self.min_agreement})"
        
        return result
    
    def _count_direction_votes(
        self,
        predictions: List[ModelPrediction],
    ) -> Dict[str, Dict[str, Any]]:
        """
        统计每个方向的投票数。
        
        返回:
            {
                "home": {"count": n, "probs": [...], "confidences": [...]},
                "draw": {"count": n, "probs": [...], "confidences": [...]},
                "away": {"count": n, "probs": [...], "confidences": [...]},
            }
        """
        votes = {
            "home": {"count": 0, "probs": [], "confidences": []},
            "draw": {"count": 0, "probs": [], "confidences": []},
            "away": {"count": 0, "probs": [], "confidences": []},
        }
        
        for pred in predictions:
            probs = pred.probs
            
            # 找到概率最大的方向
            best_dir = max(probs.items(), key=lambda x: x[1])
            direction = best_dir[0]
            prob = best_dir[1]
            
            if direction in votes:
                votes[direction]["count"] += 1
                votes[direction]["probs"].append(prob)
                votes[direction]["confidences"].append(pred.confidence)
        
        # 计算平均值
        for direction in votes:
            if votes[direction]["probs"]:
                votes[direction]["avg_prob"] = sum(votes[direction]["probs"]) / len(votes[direction]["probs"])
                votes[direction]["avg_confidence"] = sum(votes[direction]["confidences"]) / len(votes[direction]["confidences"])
            else:
                votes[direction]["avg_prob"] = 0.0
                votes[direction]["avg_confidence"] = 0.0
        
        return votes
    
    def get_ensemble_probs(
        self,
        predictions: List[ModelPrediction],
        method: str = "average",
    ) -> Dict[str, float]:
        """
        计算集成后的概率。
        
        参数:
            predictions: 各模型的预测结果列表
            method: 集成方法 ("average" / "weighted" / "best")
        
        返回:
            {"home": P, "draw": P, "away": P}
        """
        if not predictions:
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        if method == "average":
            # 简单平均
            home_avg = sum(p.probs.get("home", 0) for p in predictions) / len(predictions)
            draw_avg = sum(p.probs.get("draw", 0) for p in predictions) / len(predictions)
            away_avg = sum(p.probs.get("away", 0) for p in predictions) / len(predictions)
            return {"home": home_avg, "draw": draw_avg, "away": away_avg}
        
        elif method == "weighted":
            # 置信度加权
            total_weight = sum(p.confidence for p in predictions)
            if total_weight <= 0:
                return self.get_ensemble_probs(predictions, method="average")
            
            home_w = sum(p.probs.get("home", 0) * p.confidence for p in predictions) / total_weight
            draw_w = sum(p.probs.get("draw", 0) * p.confidence for p in predictions) / total_weight
            away_w = sum(p.probs.get("away", 0) * p.confidence for p in predictions) / total_weight
            return {"home": home_w, "draw": draw_w, "away": away_w}
        
        elif method == "best":
            # 选择置信度最高的模型
            best = max(predictions, key=lambda p: p.confidence)
            return best.probs
        
        return {"home": 0.33, "draw": 0.33, "away": 0.34}


def create_ensemble_gate(
    min_agreement: int = 3,
    confidence_threshold: float = 0.55,
    value_threshold: float = 0.03,
) -> EnsembleDecisionGate:
    """创建集成决策门的便捷函数"""
    return EnsembleDecisionGate(
        min_agreement=min_agreement,
        confidence_threshold=confidence_threshold,
        value_threshold=value_threshold,
    )
