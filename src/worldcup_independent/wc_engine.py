"""
GTO v6.0 — 世界杯独立计算模块

完全独立于五大联赛模块，可单独移除不影响主系统。

模块结构:
├── wc_engine.py      — 世界杯预测引擎
├── wc_backtest.py    — 世界杯回测
├── wc_data.py        — 世界杯数据获取
└── wc_config.py      — 世界杯配置

使用方式:
    from src.worldcup独立.wc_engine import WorldCupEngine
    engine = WorldCupEngine()
    prediction = engine.predict_match("Argentina", "France")
"""

from __future__ import annotations

import math
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# 世界杯独立数据目录
WC_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'worldcup')
os.makedirs(WC_DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 世界杯配置
# ═══════════════════════════════════════════════════════════════

WC_CONFIG = {
    "avg_goals": 2.5,           # 世界杯场均进球
    "home_advantage": 0.0,      # 中立场，无主场优势
    "draw_rate_group": 0.28,    # 小组赛平局率
    "draw_rate_knockout": 0.15, # 淘汰赛平局率
    "elo_k": 20,                # Elo更新系数
    "min_edge": 0.03,           # 最小价值阈值
    "kelly_fraction": 0.20,     # Kelly系数
    "max_stake_pct": 0.02,      # 最大仓位
}


# ═══════════════════════════════════════════════════════════════
# 世界杯数据
# ═══════════════════════════════════════════════════════════════

WC_2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USMNT", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

WC_TEAM_CN = {
    "Mexico": "墨西哥", "South Africa": "南非", "South Korea": "韩国", "Czechia": "捷克",
    "Canada": "加拿大", "Bosnia Herzegovina": "波黑", "Qatar": "卡塔尔", "Switzerland": "瑞士",
    "Brazil": "巴西", "Morocco": "摩洛哥", "Haiti": "海地", "Scotland": "苏格兰",
    "USMNT": "美国", "Paraguay": "巴拉圭", "Australia": "澳大利亚", "Turkey": "土耳其",
    "Germany": "德国", "Curaçao": "库拉索", "Ivory Coast": "科特迪瓦", "Ecuador": "厄瓜多尔",
    "Netherlands": "荷兰", "Japan": "日本", "Sweden": "瑞典", "Tunisia": "突尼斯",
    "Belgium": "比利时", "Egypt": "埃及", "Iran": "伊朗", "New Zealand": "新西兰",
    "Spain": "西班牙", "Cape Verde": "佛得角", "Saudi Arabia": "沙特阿拉伯", "Uruguay": "乌拉圭",
    "France": "法国", "Senegal": "塞内加尔", "Iraq": "伊拉克", "Norway": "挪威",
    "Argentina": "阿根廷", "Algeria": "阿尔及利亚", "Austria": "奥地利", "Jordan": "约旦",
    "Portugal": "葡萄牙", "DR Congo": "刚果", "Uzbekistan": "乌兹别克斯坦", "Colombia": "哥伦比亚",
    "England": "英格兰", "Croatia": "克罗地亚", "Ghana": "加纳", "Panama": "巴拿马",
}

WC_ELO = {
    "Argentina": 1860, "France": 1850, "Brazil": 1835, "England": 1810,
    "Belgium": 1790, "Portugal": 1780, "Spain": 1775, "Netherlands": 1770,
    "Germany": 1760, "Croatia": 1750, "Uruguay": 1740, "Colombia": 1730,
    "Mexico": 1720, "USMNT": 1710, "Senegal": 1700, "Japan": 1690,
    "Morocco": 1680, "Switzerland": 1670, "Norway": 1620, "Australia": 1650,
    "South Korea": 1640, "Iran": 1630, "Saudi Arabia": 1620, "Ecuador": 1610,
    "Turkey": 1600, "Scotland": 1580, "Egypt": 1550, "Ivory Coast": 1570,
    "Paraguay": 1550, "Cape Verde": 1450, "Bosnia Herzegovina": 1520,
    "Qatar": 1500, "Haiti": 1400, "Curaçao": 1350, "Iraq": 1480,
    "Jordan": 1460, "Algeria": 1520, "Austria": 1580, "DR Congo": 1480,
    "Uzbekistan": 1450, "New Zealand": 1420, "Panama": 1500, "Ghana": 1530,
    "Sweden": 1600, "Tunisia": 1540,
}

# 世界杯历史数据
WC_HISTORY = {
    "Argentina": {"titles": 3, "finals": 6, "semis": 5},
    "Brazil": {"titles": 5, "finals": 7, "semis": 11},
    "France": {"titles": 2, "finals": 4, "semis": 6},
    "Germany": {"titles": 4, "finals": 8, "semis": 13},
    "England": {"titles": 1, "finals": 1, "semis": 3},
    "Spain": {"titles": 1, "finals": 1, "semis": 2},
    "Netherlands": {"titles": 0, "finals": 3, "semis": 5},
    "Uruguay": {"titles": 2, "finals": 2, "semis": 5},
    "Italy": {"titles": 4, "finals": 6, "semis": 8},
    "Portugal": {"titles": 0, "finals": 0, "semis": 2},
    "Belgium": {"titles": 0, "finals": 0, "semis": 1},
    "Croatia": {"titles": 0, "finals": 1, "semis": 3},
}


# ═══════════════════════════════════════════════════════════════
# 世界杯数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class WCPrediction:
    """世界杯比赛预测"""
    match_id: str
    home_team: str
    away_team: str
    home_team_cn: str
    away_team_cn: str
    stage: str  # group / r32 / r16 / qf / sf / final
    group: str = ""
    
    # 概率
    home_prob: float = 0.0
    draw_prob: float = 0.0
    away_prob: float = 0.0
    
    # 因子
    factors: Dict[str, float] = field(default_factory=dict)
    
    # 推荐
    recommended: str = ""
    recommended_odds: float = 0.0
    value: float = 0.0
    confidence: float = 0.0
    
    # 比分矩阵
    score_matrix: Dict[str, float] = field(default_factory=dict)
    
    # 元数据
    home_lambda: float = 0.0
    away_lambda: float = 0.0


@dataclass
class WCMatchResult:
    """世界杯比赛结果"""
    match_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    stage: str
    group: str = ""


# ═══════════════════════════════════════════════════════════════
# 世界杯预测引擎
# ═══════════════════════════════════════════════════════════════

class WorldCupEngine:
    """
    世界杯独立预测引擎。
    
    完全独立于五大联赛引擎，可单独移除。
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or WC_CONFIG
        self.elo = dict(WC_ELO)
        self.history = WC_HISTORY
        self.groups = WC_2026_GROUPS
        self.team_cn = WC_TEAM_CN
        self.match_history: Dict[str, List] = {}
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        stage: str = "group",
        group: str = "",
        odds_home: float = 0.0,
        odds_draw: float = 0.0,
        odds_away: float = 0.0,
    ) -> WCPrediction:
        """
        预测世界杯比赛。
        
        参数:
            home_team: 主队
            away_team: 客队
            stage: 比赛阶段
            group: 小组 (小组赛时)
            odds_home/draw/away: 市场赔率 (可选)
        
        返回:
            WCPrediction
        """
        match_id = f"wc_{home_team}_vs_{away_team}_{stage}"
        
        # 获取Elo
        home_elo = self.elo.get(home_team, 1500)
        away_elo = self.elo.get(away_team, 1500)
        
        # 计算因子
        factors = self._compute_factors(home_team, away_team, stage)
        
        # 调整Elo差值
        elo_diff = home_elo - away_elo
        adjusted_diff = elo_diff + factors.get("experience_bonus", 0) + factors.get("form_bonus", 0)
        
        # 计算期望进球
        base_lambda = self.config["avg_goals"] / 2.0
        ef = adjusted_diff / 400.0
        
        home_lambda = base_lambda * (1.0 + ef * 0.5)
        away_lambda = base_lambda * (1.0 - ef * 0.5)
        
        # 泊松分布
        score_matrix = {}
        for h in range(8):
            for a in range(8):
                p = self._poisson_pmf(h, home_lambda) * self._poisson_pmf(a, away_lambda)
                score_matrix[f"{h}-{a}"] = p
        
        # 归一化
        total = sum(score_matrix.values())
        if total > 0:
            for k in score_matrix:
                score_matrix[k] /= total
        
        # 计算1X2概率
        home_prob = sum(v for k, v in score_matrix.items() if int(k.split("-")[0]) > int(k.split("-")[1]))
        draw_prob = sum(v for k, v in score_matrix.items() if int(k.split("-")[0]) == int(k.split("-")[1]))
        away_prob = 1.0 - home_prob - draw_prob
        
        # 淘汰赛调整
        if stage in ("r32", "r16", "qf", "sf", "final"):
            draw_prob *= 0.6
            remaining = 1.0 - draw_prob
            home_prob = home_prob / (home_prob + away_prob) * remaining
            away_prob = 1.0 - home_prob - draw_prob
        
        # 推荐
        recommended = ""
        value = 0.0
        confidence = 0.0
        
        if odds_home > 1 and odds_draw > 1 and odds_away > 1:
            probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
            odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}
            
            best_value = 0
            best_dir = None
            for direction in ["home", "draw", "away"]:
                v = probs[direction] - (1.0 / odds[direction])
                if v > best_value:
                    best_value = v
                    best_dir = direction
            
            if best_dir and best_value > self.config["min_edge"]:
                recommended = best_dir
                value = best_value
                confidence = min(1.0, best_value / self.config["min_edge"])
        
        return WCPrediction(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_team_cn=self.team_cn.get(home_team, home_team),
            away_team_cn=self.team_cn.get(away_team, away_team),
            stage=stage,
            group=group,
            home_prob=round(home_prob, 4),
            draw_prob=round(draw_prob, 4),
            away_prob=round(away_prob, 4),
            factors=factors,
            recommended=recommended,
            value=round(value, 4),
            confidence=round(confidence, 4),
            score_matrix=score_matrix,
            home_lambda=round(home_lambda, 4),
            away_lambda=round(away_lambda, 4),
        )
    
    def update_result(self, result: WCMatchResult):
        """更新比赛结果（用于动态Elo更新）"""
        # 更新Elo
        home_elo = self.elo.get(result.home_team, 1500)
        away_elo = self.elo.get(result.away_team, 1500)
        
        expected = 1.0 / (1.0 + 10 ** (-(home_elo - away_elo) / 400.0))
        
        if result.home_score > result.away_score:
            actual = 1.0
        elif result.home_score == result.away_score:
            actual = 0.5
        else:
            actual = 0.0
        
        gd = abs(result.home_score - result.away_score)
        margin = 1.0 + min(gd, 3) * 0.33
        delta = self.config["elo_k"] * margin * (actual - expected)
        
        self.elo[result.home_team] = home_elo + delta
        self.elo[result.away_team] = away_elo - delta
        
        # 记录历史
        self.match_history.setdefault(result.home_team, []).append({
            "gf": result.home_score, "ga": result.away_score,
            "opponent": result.away_team, "is_home": True,
        })
        self.match_history.setdefault(result.away_team, []).append({
            "gf": result.away_score, "ga": result.home_score,
            "opponent": result.home_team, "is_home": False,
        })
    
    def _compute_factors(self, home: str, away: str, stage: str) -> Dict[str, float]:
        """计算世界杯特殊因子"""
        factors = {}
        
        # 世界杯经验因子
        home_exp = self.history.get(home, {}).get("semis", 0)
        away_exp = self.history.get(away, {}).get("semis", 0)
        factors["experience_bonus"] = (home_exp - away_exp) * 5
        
        # 大赛基因因子
        home_titles = self.history.get(home, {}).get("titles", 0)
        away_titles = self.history.get(away, {}).get("titles", 0)
        factors["big_game_bonus"] = (home_titles - away_titles) * 3
        
        # 近期状态因子
        home_form = self._get_form(home)
        away_form = self._get_form(away)
        factors["form_bonus"] = (home_form - away_form) * 10
        
        # 阶段因子
        stage_mult = {"group": 1.0, "r32": 1.1, "r16": 1.1, "qf": 1.2, "sf": 1.3, "final": 1.5}
        factors["stage_factor"] = stage_mult.get(stage, 1.0)
        
        return factors
    
    def _get_form(self, team: str) -> float:
        """获取球队近期状态"""
        history = self.match_history.get(team, [])
        if not history:
            return 0.5
        
        recent = history[-5:]
        points = sum(3 if m["gf"] > m["ga"] else (1 if m["gf"] == m["ga"] else 0) for m in recent)
        return points / (len(recent) * 3)
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        """泊松分布PMF"""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ═══════════════════════════════════════════════════════════════
# 世界杯回测模块
# ═══════════════════════════════════════════════════════════════

class WorldCupBacktest:
    """世界杯独立回测"""
    
    def __init__(self, engine: WorldCupEngine = None):
        self.engine = engine or WorldCupEngine()
        self.bets: List[Dict] = []
        self.bankroll = 10000.0
        self.peak = 10000.0
        self.mdd = 0.0
    
    def run(self, matches: List[WCMatchResult], odds_data: Dict = None):
        """
        运行回测。
        
        参数:
            matches: 比赛结果列表
            odds_data: 赔率数据 {match_id: {home, draw, away}}
        """
        for match in matches:
            # 预测
            pred = self.engine.predict_match(
                match.home_team, match.away_team,
                stage=match.stage, group=match.group,
            )
            
            # 获取赔率
            odds = odds_data.get(match.match_id, {}) if odds_data else {}
            
            # 下注逻辑
            if pred.recommended and odds:
                self._place_bet(pred, match, odds)
            
            # 更新引擎
            self.engine.update_result(match)
    
    def _place_bet(self, pred: WCPrediction, match: WCMatchResult, odds: Dict):
        """下注"""
        direction = pred.recommended
        bet_odds = odds.get(direction, 0)
        
        if bet_odds <= 1:
            return
        
        # Kelly公式
        b = bet_odds - 1
        fk = (b * pred.value - (1 - pred.value)) / b if b > 0 else 0
        stake = min(
            self.bankroll * fk * self.engine.config["kelly_fraction"],
            self.bankroll * self.engine.config["max_stake_pct"]
        )
        
        if stake < 10:
            return
        
        # 结算
        rmap = {"home": match.home_team, "draw": "draw", "away": match.away_team}
        actual = "home" if match.home_score > match.away_score else ("away" if match.home_score < match.away_score else "draw")
        won = direction == actual
        
        profit = stake * (bet_odds - 1) if won else -stake
        self.bankroll += profit
        
        if self.bankroll > self.peak:
            self.peak = self.bankroll
        dd = (self.peak - self.bankroll) / self.peak if self.peak > 0 else 0
        if dd > self.mdd:
            self.mdd = dd
        
        self.bets.append({
            "match_id": match.match_id,
            "home": match.home_team,
            "away": match.away_team,
            "direction": direction,
            "odds": bet_odds,
            "stake": stake,
            "won": won,
            "profit": profit,
            "model_prob": pred.value + (1.0 / bet_odds),
            "market_prob": 1.0 / bet_odds,
        })
    
    def get_results(self) -> Dict:
        """获取回测结果"""
        total = len(self.bets)
        wins = sum(1 for b in self.bets if b["won"])
        staked = sum(b["stake"] for b in self.bets)
        returned = sum(b["stake"] + b["profit"] for b in self.bets if b["won"])
        
        return {
            "total_bets": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0,
            "total_staked": staked,
            "total_returned": returned,
            "profit": self.bankroll - 10000,
            "roi": (self.bankroll - 10000) / staked if staked > 0 else 0,
            "max_drawdown": self.mdd,
            "final_bankroll": self.bankroll,
        }


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def get_wc_engine(config: Dict = None) -> WorldCupEngine:
    """获取世界杯引擎"""
    return WorldCupEngine(config)


def get_wc_backtest() -> WorldCupBacktest:
    """获取世界杯回测"""
    return WorldCupBacktest()
