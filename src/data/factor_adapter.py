"""
GTO v6.0 — 因子适配器

将采集的数据转换为58因子格式，供预测引擎使用。

核心逻辑:
1. 从数据库读取历史数据
2. 从采集器获取实时数据
3. 计算所有可用因子
4. 跳过不可用因子（返回0）
"""

from __future__ import annotations
import json
import os
import logging
import math
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# 联赛参数
LEAGUE_PARAMS = {
    "premier_league": {"avg_goals": 2.98, "home_adv": 0.38, "draw_rate": 0.225, "rho": -0.08},
    "la_liga": {"avg_goals": 2.55, "home_adv": 0.42, "draw_rate": 0.26, "rho": -0.10},
    "bundesliga": {"avg_goals": 3.17, "home_adv": 0.45, "draw_rate": 0.249, "rho": -0.06},
    "serie_a": {"avg_goals": 2.65, "home_adv": 0.40, "draw_rate": 0.28, "rho": -0.13},
    "ligue_1": {"avg_goals": 2.78, "home_adv": 0.42, "draw_rate": 0.258, "rho": -0.12},
    "worldcup": {"avg_goals": 2.50, "home_adv": 0.00, "draw_rate": 0.25, "rho": -0.10},
}

# 球队历史数据（从比赛结果累积）
TEAM_HISTORY: Dict[str, Dict] = defaultdict(lambda: {
    "results": [], "gf": [], "ga": [], "shots": [], "sot": [],
    "corners": [], "yellows": [], "ht_goals": [],
})


class FactorAdapter:
    """因子适配器 — 将采集数据转换为58因子"""
    
    def __init__(self, league_id: str):
        self.league_id = league_id
        self.params = LEAGUE_PARAMS.get(league_id, LEAGUE_PARAMS["worldcup"])
        self.elo: Dict[str, float] = defaultdict(lambda: 1500.0)
    
    def compute_factors(
        self,
        home_team: str,
        away_team: str,
        match_data: Dict = None,
        odds_data: Dict = None,
        weather_data: Dict = None,
        standings_data: Dict = None,
    ) -> Dict[str, float]:
        """
        计算所有可用因子。
        
        参数:
            home_team: 主队
            away_team: 客队
            match_data: 比赛数据 (时间、场地等)
            odds_data: 赔率数据
            weather_data: 天气数据
            standings_data: 积分榜数据
        
        返回:
            {factor_id: delta_value}
        """
        factors = {}
        match_data = match_data or {}
        odds_data = odds_data or {}
        weather_data = weather_data or {}
        standings_data = standings_data or {}
        
        # 获取历史数据
        home_hist = TEAM_HISTORY[home_team]
        away_hist = TEAM_HISTORY[away_team]
        
        # ── F1: Elo ──
        elo_diff = self.elo[home_team] - self.elo[away_team]
        factors["F1"] = 0.5 * (elo_diff / 400.0)
        
        # ── F3: 近期状态 ──
        if home_hist["results"]:
            home_form = self._ewma(home_hist["results"][-5:])
            away_form = self._ewma(away_hist["results"][-5:]) if away_hist["results"] else 1.5
            factors["F3"] = (home_form - away_form) * 0.15
        
        # ── F4: 主场优势 ──
        if self.league_id != "worldcup":
            factors["F4"] = self.params["home_adv"] * 0.1
        
        # ── F5: 历史交锋 ──
        # 需要历史交锋数据，暂跳过
        
        # ── F6: 赛程密度 ──
        factors["F6"] = -0.03  # 默认值
        
        # ── F7: 排名差 ──
        if standings_data:
            home_pos = standings_data.get(home_team, {}).get("position", 10)
            away_pos = standings_data.get(away_team, {}).get("position", 10)
            factors["F7"] = ((away_pos - home_pos) / 20.0) * 0.1
        
        # ── F8: 净胜球差 ──
        if home_hist["gf"] and away_hist["gf"]:
            home_gd = sum(home_hist["gf"][-10:]) - sum(home_hist["ga"][-10:])
            away_gd = sum(away_hist["gf"][-10:]) - sum(away_hist["ga"][-10:])
            factors["F8"] = ((home_gd - away_gd) / 10.0) * 0.06
        
        # ── F9: xG差 ──
        # 无数据源，跳过
        
        # ── F10: 赔率隐含概率 ──
        if odds_data and odds_data.get("home", 0) > 1:
            h, d, a = odds_data["home"], odds_data["draw"], odds_data["away"]
            mg = 1.0/h + 1.0/d + 1.0/a
            ip_home = (1.0/h) / mg
            ip_draw = (1.0/d) / mg
            ip_away = (1.0/a) / mg
            factors["F10"] = self._logit(ip_home) - self._logit(0.33)
        
        # ── F12: 天气 ──
        if weather_data:
            impact = weather_data.get("impact", 0)
            factors["F12"] = impact * 0.08
        
        # ── F13: 裁判 ──
        # 暂跳过
        
        # ── F15: 教练更替 ──
        # 暂跳过
        
        # ── F16: 欧战影响 ──
        # 暂跳过
        
        # ── F18: 德比战 ──
        # 暂跳过
        
        # ── F19: 风格匹配 ──
        if home_hist["shots"] and away_hist["shots"]:
            home_shot_avg = sum(home_hist["shots"][-5:]) / max(len(home_hist["shots"][-5:]), 1)
            away_shot_avg = sum(away_hist["shots"][-5:]) / max(len(away_hist["shots"][-5:]), 1)
            style = 0.5 + (home_shot_avg - away_shot_avg) / 20.0
            factors["F19"] = (style - 0.5) * 0.10
        
        # ── F20: 连胜动量 ──
        if home_hist["results"]:
            streak = self._calc_streak(home_hist["results"][-5:])
            factors["F20"] = streak * 0.08
        
        # ── F25: 时间衰减 ──
        factors["F25"] = 1.0 * 0.05
        
        # ── F26: 联赛强度 ──
        league_strength = {"premier_league": 1.0, "la_liga": 0.95, "bundesliga": 0.88,
                          "serie_a": 0.85, "ligue_1": 0.78, "worldcup": 0.90}.get(self.league_id, 0.85)
        factors["F26"] = (league_strength - 0.9) * 0.05
        
        # ── F27: 泊松修正 ──
        if home_hist["gf"]:
            goals = home_hist["gf"][-10:]
            avg_g = sum(goals) / len(goals)
            var_g = sum((g - avg_g)**2 for g in goals) / len(goals)
            factors["F27"] = ((var_g - avg_g) / max(avg_g, 1)) * 0.20
        
        # ── F29: 大小球趋势 ──
        if home_hist["gf"] and home_hist["ga"]:
            avg_total = (sum(home_hist["gf"][-5:]) + sum(home_hist["ga"][-5:])) / max(len(home_hist["gf"][-5:]), 1)
            factors["F29"] = (avg_total - self.params["avg_goals"]) * 0.5
        
        # ── F33: 保级/争冠动力 ──
        if standings_data:
            home_pos = standings_data.get(home_team, {}).get("position", 10)
            away_pos = standings_data.get(away_team, {}).get("position", 10)
            mb = 0.0
            if home_pos <= 3: mb += 0.05
            if away_pos <= 3: mb -= 0.03
            if home_pos >= 18: mb += 0.08
            if away_pos >= 18: mb -= 0.05
            factors["F33"] = mb
        
        # ── F37: 中游无欲 ──
        if standings_data:
            home_pos = standings_data.get(home_team, {}).get("position", 10)
            if 8 <= home_pos <= 14:
                factors["F37"] = 0.03
        
        # ── F39: 积分榜位置 ──
        if standings_data:
            home_pts = standings_data.get(home_team, {}).get("points", 0)
            away_pts = standings_data.get(away_team, {}).get("points", 0)
            factors["F39"] = ((away_pts - home_pts) / 20.0) * 0.12
        
        # ── F42-F53: 比赛统计（需要赛后数据）──
        # 暂跳过，赛后累积
        
        # ── F46: 赔率漂移 ──
        # 需要开盘vs收盘赔率，暂跳过
        
        # ── F56-F58: 平局因子 ──
        if home_hist["gf"]:
            avg_total = (sum(home_hist["gf"][-5:]) + sum(home_hist["ga"][-5:])) / max(len(home_hist["gf"][-5:]), 1)
            if 2.0 < avg_total < 2.8:
                factors["F57"] = 0.2 * 0.10
        
        draws = sum(1 for r in home_hist["results"][-10:] if r == 1)
        if len(home_hist["results"]) >= 5:
            factors["F58"] = ((draws / len(home_hist["results"][-10:]) - 0.25) * 2.0) * 0.06
        
        return factors
    
    def update_history(
        self,
        team: str,
        gf: int,
        ga: int,
        shots: int = 0,
        sot: int = 0,
        corners: int = 0,
        yellows: int = 0,
        ht_goals: int = 0,
    ):
        """更新球队历史数据"""
        hist = TEAM_HISTORY[team]
        
        # 结果 (胜=3, 平=1, 负=0)
        if gf > ga:
            result = 3
        elif gf == ga:
            result = 1
        else:
            result = 0
        
        hist["results"].append(result)
        hist["gf"].append(gf)
        hist["ga"].append(ga)
        if shots > 0:
            hist["shots"].append(shots)
        if sot > 0:
            hist["sot"].append(sot)
        if corners > 0:
            hist["corners"].append(corners)
        if yellows > 0:
            hist["yellows"].append(yellows)
        if ht_goals > 0:
            hist["ht_goals"].append(ht_goals)
        
        # 限制历史长度
        for key in hist:
            if isinstance(hist[key], list):
                hist[key] = hist[key][-30:]
    
    def update_elo(self, home_team: str, away_team: str, home_score: int, away_score: int):
        """更新Elo评分"""
        eh = self.elo[home_team]
        ea = self.elo[away_team]
        
        expected = 1.0 / (1.0 + 10 ** (-(eh - ea) / 400.0))
        
        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0
        
        gd = abs(home_score - away_score)
        margin = 1.0 + min(gd, 3) * 0.33
        delta = 20 * margin * (actual - expected)
        
        self.elo[home_team] = eh + delta
        self.elo[away_team] = ea - delta
    
    def _ewma(self, values: List[float], alpha: float = 0.3) -> float:
        """指数加权移动平均"""
        if not values:
            return 1.5
        ewma = 0.0
        weight_sum = 0.0
        for i, v in enumerate(values):
            w = (1 - alpha) ** i
            ewma += v * w
            weight_sum += w
        return ewma / weight_sum if weight_sum > 0 else 1.5
    
    def _calc_streak(self, results: List[int]) -> float:
        """计算连胜/连败动量"""
        if not results:
            return 0.0
        streak = 0
        for r in reversed(results):
            if r == 3:
                streak += 1
            elif r == 0:
                streak -= 1
            else:
                break
        return max(-3, min(3, streak)) / 3.0
    
    def _logit(self, p: float) -> float:
        """logit函数"""
        p = max(0.001, min(0.999, p))
        return math.log(p / (1.0 - p))


# 全局实例
_adapters: Dict[str, FactorAdapter] = {}


def get_factor_adapter(league_id: str) -> FactorAdapter:
    """获取因子适配器"""
    if league_id not in _adapters:
        _adapters[league_id] = FactorAdapter(league_id)
    return _adapters[league_id]
