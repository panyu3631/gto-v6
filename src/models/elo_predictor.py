"""
GTO-GameFlow v5.11 — Elo 评分预测器

基于 Elo 评分差预测比赛结果概率。
使用标准 Elo 公式 + 联赛特化的主场优势修正。

公式:
    expected_home = 1.0 / (1.0 + 10^(-(elo_diff + home_advantage) / 400))
    
    P(home) = expected_home × (1 - draw_factor)
    P(draw) = draw_factor × (1 - |expected_home - 0.5| × 2)
    P(away) = (1 - expected_home) × (1 - draw_factor)

其中 draw_factor 根据联赛历史平局率校准。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EloPredictor:
    """
    Elo 评分预测器。
    
    使用 Elo 评分差预测 1X2 概率。
    优势: 简单、稳定、无需大量特征。
    劣势: 不考虑近期状态、伤病等动态因素。
    """
    
    # 联赛历史平局率 (用于校准 draw_factor)
    LEAGUE_DRAW_RATES = {
        "premier_league": 0.24,
        "la_liga": 0.23,
        "bundesliga": 0.25,
        "serie_a": 0.26,
        "ligue_1": 0.25,
    }
    
    # 联赛主场优势 (Elo 点数)
    LEAGUE_HOME_ADVANTAGE = {
        "premier_league": 65,
        "la_liga": 70,
        "bundesliga": 80,
        "serie_a": 75,
        "ligue_1": 70,
    }
    
    DEFAULT_DRAW_RATE = 0.25
    DEFAULT_HOME_ADVANTAGE = 70
    
    def __init__(
        self,
        league_id: str = "",
        home_advantage: Optional[float] = None,
        draw_factor: Optional[float] = None,
    ):
        """
        参数:
            league_id: 联赛ID
            home_advantage: 主场优势 Elo 点数 (None=使用联赛默认值)
            draw_factor: 平局因子 (None=根据联赛平局率自动计算)
        """
        self.league_id = league_id
        self.home_advantage = (
            home_advantage 
            or self.LEAGUE_HOME_ADVANTAGE.get(league_id, self.DEFAULT_HOME_ADVANTAGE)
        )
        
        draw_rate = (
            draw_factor
            or self.LEAGUE_DRAW_RATES.get(league_id, self.DEFAULT_DRAW_RATE)
        )
        # 将平局率转换为 draw_factor (经验公式)
        self.draw_factor = draw_rate * 1.2
    
    def predict(
        self,
        home_elo: float,
        away_elo: float,
        **kwargs,
    ) -> Dict[str, float]:
        """
        预测 1X2 概率。
        
        参数:
            home_elo: 主队 Elo 评分
            away_elo: 客队 Elo 评分
            **kwargs: 其他参数 (忽略，保持接口一致)
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        elo_diff = home_elo - away_elo + self.home_advantage
        
        # 标准 Elo 期望值
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        # 平局概率: 当双方实力接近时平局概率最高
        closeness = 1.0 - abs(expected_home - 0.5) * 2.0  # 0~1, 越接近0.5越高
        draw_prob = self.draw_factor * closeness
        draw_prob = max(0.10, min(0.35, draw_prob))  # 限制在合理范围
        
        # 主胜和客胜概率
        remaining = 1.0 - draw_prob
        home_prob = expected_home * remaining
        away_prob = (1.0 - expected_home) * remaining
        
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
    
    def predict_from_context(self, match_context: dict) -> Dict[str, float]:
        """
        从比赛上下文预测。
        
        参数:
            match_context: 包含 home_elo, away_elo 的字典
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        home_elo = match_context.get("home_elo", 1500.0)
        away_elo = match_context.get("away_elo", 1500.0)
        return self.predict(home_elo, away_elo)
