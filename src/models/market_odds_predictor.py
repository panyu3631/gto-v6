"""
GTO-GameFlow v5.11 — 市场赔率预测器

基于市场赔率的隐含概率预测器。
直接使用博彩公司的赔率作为市场共识的预测。

模型特点:
- 使用市场赔率的隐含概率
- 去除边际 (overround) 以获得真实概率
- 支持多来源赔率聚合
- 反映市场参与者的集体智慧

公式:
    implied_prob = 1 / odds
    overround = Σ implied_prob
    true_prob = implied_prob / overround
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketOddsPredictor:
    """
    市场赔率预测器。
    
    使用市场赔率的隐含概率作为预测。
    优势: 反映市场共识，信息量大。
    劣势: 受博彩公司边际影响，可能有偏差。
    """
    
    # 默认边际率 (overround - 1.0)
    DEFAULT_OVERROUND = 0.05  # 5% 边际
    
    def __init__(
        self,
        league_id: str = "",
        overround_method: str = "proportional",
    ):
        """
        参数:
            league_id: 联赛ID
            overround_method: 去边际方法 ("proportional" / "shin" / "basic")
        """
        self.league_id = league_id
        self.overround_method = overround_method
    
    def predict(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        **kwargs,
    ) -> Dict[str, float]:
        """
        预测 1X2 概率。
        
        参数:
            home_odds: 主胜赔率
            draw_odds: 平局赔率
            away_odds: 客胜赔率
            **kwargs: 其他参数 (忽略)
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        if home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
            logger.warning("赔率无效，返回均匀分布")
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        # 计算隐含概率
        implied_home = 1.0 / home_odds
        implied_draw = 1.0 / draw_odds
        implied_away = 1.0 / away_odds
        
        # 去边际
        if self.overround_method == "proportional":
            return self._proportional_method(implied_home, implied_draw, implied_away)
        elif self.overround_method == "shin":
            return self._shin_method(implied_home, implied_draw, implied_away)
        else:
            return self._basic_method(implied_home, implied_draw, implied_away)
    
    def predict_from_odds_dict(
        self,
        odds: Dict[str, float],
    ) -> Dict[str, float]:
        """
        从赔率字典预测。
        
        参数:
            odds: {"home": home_odds, "draw": draw_odds, "away": away_odds}
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        return self.predict(
            home_odds=odds.get("home", 1.0),
            draw_odds=odds.get("draw", 1.0),
            away_odds=odds.get("away", 1.0),
        )
    
    def predict_from_context(self, match_context: dict) -> Dict[str, float]:
        """
        从比赛上下文预测。
        
        参数:
            match_context: 包含 market_probs 或 odds 的字典
        
        返回:
            {"home": P_home, "draw": P_draw, "away": P_away}
        """
        # 优先使用已计算好的市场概率
        if "market_probs" in match_context:
            probs = match_context["market_probs"]
            return {
                "home": probs.get("home", 0.33),
                "draw": probs.get("draw", 0.33),
                "away": probs.get("away", 0.34),
            }
        
        # 否则从赔率计算
        if "odds" in match_context:
            odds = match_context["odds"]
            return self.predict_from_odds_dict(odds)
        
        # 尝试分别获取赔率
        home_odds = match_context.get("home_odds", 0.0)
        draw_odds = match_context.get("draw_odds", 0.0)
        away_odds = match_context.get("away_odds", 0.0)
        
        if home_odds > 0 and draw_odds > 0 and away_odds > 0:
            return self.predict(home_odds, draw_odds, away_odds)
        
        logger.warning("无法从上下文获取赔率信息，返回均匀分布")
        return {"home": 0.33, "draw": 0.33, "away": 0.34}
    
    def _proportional_method(
        self,
        implied_home: float,
        implied_draw: float,
        implied_away: float,
    ) -> Dict[str, float]:
        """
        比例去边际法。
        
        假设边际均匀分布在各结果上。
        true_prob = implied_prob / overround
        """
        overround = implied_home + implied_draw + implied_away
        
        if overround <= 0:
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        return {
            "home": round(implied_home / overround, 6),
            "draw": round(implied_draw / overround, 6),
            "away": round(implied_away / overround, 6),
        }
    
    def _shin_method(
        self,
        implied_home: float,
        implied_draw: float,
        implied_away: float,
    ) -> Dict[str, float]:
        """
        Shin 去边际法。
        
        考虑到博彩公司可能对某些结果有信息优势。
        使用迭代方法求解。
        """
        # 简化版 Shin 方法
        overround = implied_home + implied_draw + implied_away
        
        if overround <= 0:
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        # 迭代求解 (简化版，最多10次)
        z = 0.0  # 信息份额参数
        for _ in range(10):
            # 计算调整后的概率
            adj_home = implied_home - z * (implied_home ** 2)
            adj_draw = implied_draw - z * (implied_draw ** 2)
            adj_away = implied_away - z * (implied_away ** 2)
            
            total = adj_home + adj_draw + adj_away
            if total <= 0:
                break
            
            # 更新 z
            z_new = (overround - 1.0) / (implied_home ** 2 + implied_draw ** 2 + implied_away ** 2)
            
            if abs(z_new - z) < 1e-6:
                break
            z = z_new
        
        # 归一化
        total = adj_home + adj_draw + adj_away
        if total > 0:
            return {
                "home": round(adj_home / total, 6),
                "draw": round(adj_draw / total, 6),
                "away": round(adj_away / total, 6),
            }
        
        return self._proportional_method(implied_home, implied_draw, implied_away)
    
    def _basic_method(
        self,
        implied_home: float,
        implied_draw: float,
        implied_away: float,
    ) -> Dict[str, float]:
        """
        基础去边际法。
        
        简单地从每个隐含概率中减去等量的边际。
        """
        overround = implied_home + implied_draw + implied_away
        margin = overround - 1.0
        
        if margin <= 0:
            return {
                "home": round(implied_home, 6),
                "draw": round(implied_draw, 6),
                "away": round(implied_away, 6),
            }
        
        # 从每个概率中减去 margin/3
        adj_margin = margin / 3.0
        
        home = max(0.01, implied_home - adj_margin)
        draw = max(0.01, implied_draw - adj_margin)
        away = max(0.01, implied_away - adj_margin)
        
        # 归一化
        total = home + draw + away
        return {
            "home": round(home / total, 6),
            "draw": round(draw / total, 6),
            "away": round(away / total, 6),
        }
    
    def aggregate_odds(
        self,
        odds_list: List[Dict[str, float]],
        method: str = "average",
    ) -> Dict[str, float]:
        """
        聚合多个来源的赔率。
        
        参数:
            odds_list: 赔率字典列表
            method: 聚合方法 ("average" / "best" / "median")
        
        返回:
            聚合后的概率
        """
        if not odds_list:
            return {"home": 0.33, "draw": 0.33, "away": 0.34}
        
        if len(odds_list) == 1:
            return self.predict_from_odds_dict(odds_list[0])
        
        # 将每个赔率转换为概率
        all_probs = []
        for odds in odds_list:
            probs = self.predict_from_odds_dict(odds)
            all_probs.append(probs)
        
        if method == "average":
            # 平均概率
            avg_home = sum(p["home"] for p in all_probs) / len(all_probs)
            avg_draw = sum(p["draw"] for p in all_probs) / len(all_probs)
            avg_away = sum(p["away"] for p in all_probs) / len(all_probs)
            return {
                "home": round(avg_home, 6),
                "draw": round(avg_draw, 6),
                "away": round(avg_away, 6),
            }
        elif method == "median":
            # 中位数概率
            homes = sorted(p["home"] for p in all_probs)
            draws = sorted(p["draw"] for p in all_probs)
            aways = sorted(p["away"] for p in all_probs)
            mid = len(all_probs) // 2
            return {
                "home": round(homes[mid], 6),
                "draw": round(draws[mid], 6),
                "away": round(aways[mid], 6),
            }
        else:
            # 最佳赔率 (最高概率)
            best_home = max(p["home"] for p in all_probs)
            best_draw = max(p["draw"] for p in all_probs)
            best_away = max(p["away"] for p in all_probs)
            return {
                "home": round(best_home, 6),
                "draw": round(best_draw, 6),
                "away": round(best_away, 6),
            }
