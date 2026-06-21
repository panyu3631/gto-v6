"""
GTO-GameFlow v5.11 — Dixon-Coles 泊松预测器

基于 Dixon-Coles 泊松模型预测比分概率。
这是对现有 UnifiedProbabilityEngine 的封装，提供统一的预测接口。

模型特点:
- 使用泊松分布建模进球数
- Dixon-Coles tau 校正低比分的负相关性
- 联赛特化的 ρ 参数
- 考虑主场优势和近期状态

公式:
    P(home_goals = h, away_goals = a) = τ(h, a) × Poisson(h, λ_h) × Poisson(a, λ_a)
    
    其中:
    λ_h = base_lambda × (1 + home_attack) × (1 - away_defense) × home_advantage
    λ_a = base_lambda × (1 + away_attack) × (1 - home_defense)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DixonColesPredictor:
    """
    Dixon-Coles 泊松预测器。
    
    使用泊松分布 + Dixon-Coles 校正预测比分概率。
    优势: 理论基础扎实，能建模比分相关性。
    劣势: 对参数敏感，需要校准。
    """
    
    # 联赛特化 ρ 参数 (Dixon-Coles 相关系数)
    LEAGUE_RHO = {
        "premier_league": -0.10,
        "la_liga": -0.12,
        "bundesliga": -0.08,
        "serie_a": -0.11,
        "ligue_1": -0.09,
    }
    
    # 联赛场均进球数
    LEAGUE_AVG_GOALS = {
        "premier_league": 2.65,
        "la_liga": 2.55,
        "bundesliga": 2.85,
        "serie_a": 2.50,
        "ligue_1": 2.60,
    }
    
    DEFAULT_RHO = -0.10
    DEFAULT_AVG_GOALS = 2.65
    
    def __init__(
        self,
        league_id: str = "",
        rho: Optional[float] = None,
        avg_goals: Optional[float] = None,
        max_goals: int = 8,
    ):
        """
        参数:
            league_id: 联赛ID
            rho: Dixon-Coles 相关系数 (None=使用联赛默认值)
            avg_goals: 场均进球数 (None=使用联赛默认值)
            max_goals: 最大进球数
        """
        self.league_id = league_id
        self.rho = rho or self.LEAGUE_RHO.get(league_id, self.DEFAULT_RHO)
        self.avg_goals = avg_goals or self.LEAGUE_AVG_GOALS.get(league_id, self.DEFAULT_AVG_GOALS)
        self.max_goals = max_goals
    
    def predict(
        self,
        home_attack: float = 0.0,
        home_defense: float = 0.0,
        away_attack: float = 0.0,
        away_defense: float = 0.0,
        home_advantage: float = 0.35,
        **kwargs,
    ) -> Dict[str, float]:
        """
        预测 1X2 概率。
        
        参数:
            home_attack: 主队攻击因子 (0=平均水平)
            home_defense: 主队防守因子 (0=平均水平)
            away_attack: 客队攻击因子 (0=平均水平)
            away_defense: 客队防守因子 (0=平均水平)
            home_advantage: 主场优势乘数
            **kwargs: 其他参数 (忽略)
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        # 计算预期进球数
        base = self.avg_goals / 2.0
        home_lambda = base * (1.0 + home_attack) * (1.0 - away_defense) * (1.0 + home_advantage)
        away_lambda = base * (1.0 + away_attack) * (1.0 - home_defense)
        
        # 限制在合理范围
        home_lambda = max(0.3, min(4.0, home_lambda))
        away_lambda = max(0.3, min(4.0, away_lambda))
        
        # 计算比分概率矩阵
        score_probs = self._build_score_matrix(home_lambda, away_lambda)
        
        # 汇总 1X2 概率
        home_prob = sum(p for (h, a), p in score_probs.items() if h > a)
        draw_prob = sum(p for (h, a), p in score_probs.items() if h == a)
        away_prob = sum(p for (h, a), p in score_probs.items() if h < a)
        
        # 归一化
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        
        return {
            "home": round(home_prob, 6),
            "draw": round(draw_prob, 6),
            "away": round(away_prob, 6),
        }
    
    def predict_with_lambdas(
        self,
        home_lambda: float,
        away_lambda: float,
    ) -> Dict[str, float]:
        """
        直接使用 lambda 参数预测。
        
        参数:
            home_lambda: 主队预期进球
            away_lambda: 客队预期进球
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        score_probs = self._build_score_matrix(home_lambda, away_lambda)
        
        home_prob = sum(p for (h, a), p in score_probs.items() if h > a)
        draw_prob = sum(p for (h, a), p in score_probs.items() if h == a)
        away_prob = sum(p for (h, a), p in score_probs.items() if h < a)
        
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        
        return {
            "home": round(home_prob, 6),
            "draw": round(draw_prob, 6),
            "away": round(away_prob, 6),
        }
    
    def predict_from_context(self, match_context: dict) -> Dict[str, float]:
        """
        从比赛上下文预测。
        
        参数:
            match_context: 包含 attack/defense 因子的字典
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        return self.predict(
            home_attack=match_context.get("home_attack", 0.0),
            home_defense=match_context.get("home_defense", 0.0),
            away_attack=match_context.get("away_attack", 0.0),
            away_defense=match_context.get("away_defense", 0.0),
            home_advantage=match_context.get("home_advantage", 0.35),
        )
    
    def _build_score_matrix(
        self,
        home_lambda: float,
        away_lambda: float,
    ) -> Dict[Tuple[int, int], float]:
        """构建比分概率矩阵 (含 Dixon-Coles 校正)"""
        probs = {}
        
        for h in range(self.max_goals + 1):
            h_p = self._poisson_pmf(h, home_lambda)
            for a in range(self.max_goals + 1):
                a_p = self._poisson_pmf(a, away_lambda)
                tau = self._dixon_coles_tau(h, a, home_lambda, away_lambda)
                probs[(h, a)] = h_p * a_p * tau
        
        # 归一化
        total = sum(probs.values())
        if total > 0:
            for key in probs:
                probs[key] /= total
        
        return probs
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        """泊松分布概率质量函数"""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    
    def _dixon_coles_tau(
        self,
        x: int,
        y: int,
        lam_h: float,
        lam_a: float,
    ) -> float:
        """Dixon-Coles tau 校正 (仅作用于低比分 x,y ∈ {0,1})"""
        rho = self.rho
        
        if x >= 2 or y >= 2:
            return 1.0
        if x == 0 and y == 0:
            return max(0.0, 1.0 - lam_h * lam_a * rho)
        elif x == 0 and y == 1:
            return max(0.0, 1.0 + lam_h * rho)
        elif x == 1 and y == 0:
            return max(0.0, 1.0 + lam_a * rho)
        elif x == 1 and y == 1:
            return max(0.0, 1.0 - rho)
        return 1.0
