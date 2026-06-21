"""
GTO v6.0 — 世界杯模块

世界杯预测支持：
1. 小组赛 — 32支球队，8组，每组4队
2. 淘汰赛 — 16强→8强→4强→决赛
3. 特殊因子 — 国家队经验、赛程密度、主客场（中立场）
4. 历史数据 — 世界杯历史战绩

使用方式:
    wc = WorldCupModule()
    prediction = wc.predict_match("Argentina", "France", stage="final")
"""

from __future__ import annotations
import json
import os
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# 2026世界杯参赛队伍（示例）
WC_2026_TEAMS = {
    "pot1": ["Argentina", "France", "Brazil", "England", "Belgium", "Portugal", "Spain", "Netherlands"],
    "pot2": ["Germany", "Croatia", "Uruguay", "Colombia", "Mexico", "USA", "Senegal", "Japan"],
    "pot3": ["Morocco", "Switzerland", "Denmark", "Australia", "South Korea", "Iran", "Saudi Arabia", "Ecuador"],
    "pot4": ["Nigeria", "Serbia", "Poland", "Cameroon", "Canada", "Wales", "Tunisia", "Ghana"],
}

# 球队中文名映射
TEAM_NAMES_CN = {
    "Argentina": "阿根廷", "France": "法国", "Brazil": "巴西", "England": "英格兰",
    "Belgium": "比利时", "Portugal": "葡萄牙", "Spain": "西班牙", "Netherlands": "荷兰",
    "Germany": "德国", "Croatia": "克罗地亚", "Uruguay": "乌拉圭", "Colombia": "哥伦比亚",
    "Mexico": "墨西哥", "USA": "美国", "Senegal": "塞内加尔", "Japan": "日本",
    "Morocco": "摩洛哥", "Switzerland": "瑞士", "Denmark": "丹麦", "Australia": "澳大利亚",
    "South Korea": "韩国", "Iran": "伊朗", "Saudi Arabia": "沙特阿拉伯", "Ecuador": "厄瓜多尔",
    "Nigeria": "尼日利亚", "Serbia": "塞尔维亚", "Poland": "波兰", "Cameroon": "喀麦隆",
    "Canada": "加拿大", "Wales": "威尔士", "Tunisia": "突尼斯", "Ghana": "加纳",
    "Italy": "意大利", "Scotland": "苏格兰", "Norway": "挪威", "Sweden": "瑞典",
    "Austria": "奥地利", "Czech Republic": "捷克", "Romania": "罗马尼亚", "Hungary": "匈牙利",
    "Turkey": "土耳其", "Greece": "希腊", "Russia": "俄罗斯", "Ukraine": "乌克兰",
    "Peru": "秘鲁", "Chile": "智利", "Paraguay": "巴拉圭", "Bolivia": "玻利维亚",
    "Egypt": "埃及", "Algeria": "阿尔及利亚", "Tunisia": "突尼斯", "Morocco": "摩洛哥",
    "Ivory Coast": "科特迪瓦", "Cameroon": "喀麦隆", "Ghana": "加纳", "Nigeria": "尼日利亚",
    "South Africa": "南非", "DR Congo": "刚果民主共和国", "Mali": "马里", "Burkina Faso": "布基纳法索",
    "China": "中国", "Iraq": "伊拉克", "Qatar": "卡塔尔", "UAE": "阿联酋",
    "New Zealand": "新西兰", "Jamaica": "牙买加", "Honduras": "洪都拉斯", "Costa Rica": "哥斯达黎加",
    "Panama": "巴拿马", "Trinidad and Tobago": "特立尼达和多巴哥",
}

def get_cn_name(team: str) -> str:
    """获取球队中文名"""
    return TEAM_NAMES_CN.get(team, team)

# 国家队Elo评分（2024年数据）
NATIONAL_ELO = {
    "Argentina": 1860, "France": 1850, "Brazil": 1835, "England": 1810,
    "Belgium": 1790, "Portugal": 1780, "Spain": 1775, "Netherlands": 1770,
    "Germany": 1760, "Croatia": 1750, "Uruguay": 1740, "Colombia": 1730,
    "Mexico": 1720, "USA": 1710, "Senegal": 1700, "Japan": 1690,
    "Morocco": 1680, "Switzerland": 1670, "Denmark": 1660, "Australia": 1650,
    "South Korea": 1640, "Iran": 1630, "Saudi Arabia": 1620, "Ecuador": 1610,
    "Nigeria": 1600, "Serbia": 1590, "Poland": 1580, "Cameroon": 1570,
    "Canada": 1560, "Wales": 1550, "Tunisia": 1540, "Ghana": 1530,
}

# 世界杯历史数据
WC_HISTORY = {
    "Argentina": {"titles": 3, "finals": 6, "semis": 5, "top_scorer": "Messi"},
    "Brazil": {"titles": 5, "finals": 7, "semis": 11, "top_scorer": "Ronaldo"},
    "France": {"titles": 2, "finals": 4, "semis": 6, "top_scorer": "Mbappé"},
    "Germany": {"titles": 4, "finals": 8, "semis": 13, "top_scorer": "Klose"},
    "England": {"titles": 1, "finals": 1, "semis": 3, "top_scorer": "Kane"},
    "Spain": {"titles": 1, "finals": 1, "semis": 2, "top_scorer": "Villa"},
    "Netherlands": {"titles": 0, "finals": 3, "semis": 5, "top_scorer": "van Persie"},
    "Uruguay": {"titles": 2, "finals": 2, "semis": 5, "top_scorer": "Suárez"},
    "Italy": {"titles": 4, "finals": 6, "semis": 8, "top_scorer": "Riva"},
    "Portugal": {"titles": 0, "finals": 0, "semis": 2, "top_scorer": "Ronaldo"},
    "Belgium": {"titles": 0, "finals": 0, "semis": 1, "top_scorer": "Lukaku"},
    "Croatia": {"titles": 0, "finals": 1, "semis": 3, "top_scorer": "Modrić"},
}


@dataclass
class WorldCupMatch:
    """世界杯比赛"""
    match_id: str
    stage: str  # group, r16, qf, sf, final
    home_team: str
    away_team: str
    group: str = ""
    venue: str = ""
    date: str = ""


@dataclass
class WorldCupPrediction:
    """世界杯预测"""
    match_id: str
    home_team: str
    away_team: str
    home_prob: float
    draw_prob: float
    away_prob: float
    recommended_bet: Optional[Dict] = None
    factors: Dict[str, float] = field(default_factory=dict)


class WorldCupModule:
    """世界杯预测模块"""
    
    def __init__(self):
        self.elo = dict(NATIONAL_ELO)
        self.history = WC_HISTORY
        self.teams = WC_2026_TEAMS
        
        # 世界杯特殊参数
        self.home_advantage = 0.0  # 中立场，无主场优势
        self.draw_rate_group = 0.28  # 小组赛平局率较高
        self.draw_rate_knockout = 0.15  # 淘汰赛平局率较低（含加时）
        self.avg_goals = 2.5  # 世界杯场均进球
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        stage: str = "group",
        odds_home: float = 0.0,
        odds_draw: float = 0.0,
        odds_away: float = 0.0,
    ) -> WorldCupPrediction:
        """
        预测世界杯比赛。
        
        参数:
            home_team: 主队
            away_team: 客队
            stage: 比赛阶段 (group/r16/qf/sf/final)
            odds_home/draw/away: 市场赔率（可选）
        
        返回:
            WorldCupPrediction
        """
        # 获取Elo
        home_elo = self.elo.get(home_team, 1500)
        away_elo = self.elo.get(away_team, 1500)
        
        # 计算基础概率
        elo_diff = home_elo - away_elo
        
        # 世界杯特殊因子
        factors = self._compute_factors(home_team, away_team, stage)
        
        # 调整Elo差值
        adjusted_diff = elo_diff + factors.get("experience_bonus", 0) + factors.get("form_bonus", 0)
        
        # 计算期望进球
        base_lambda = self.avg_goals / 2.0
        ef = adjusted_diff / 400.0
        
        home_lambda = base_lambda * (1.0 + ef * 0.5)
        away_lambda = base_lambda * (1.0 - ef * 0.5)
        
        # 泊松分布
        score_matrix = {}
        for h in range(8):
            for a in range(8):
                p = self._poisson_pmf(h, home_lambda) * self._poisson_pmf(a, away_lambda)
                score_matrix[(h, a)] = p
        
        # 归一化
        total = sum(score_matrix.values())
        if total > 0:
            for k in score_matrix:
                score_matrix[k] /= total
        
        # 计算1X2概率
        home_prob = sum(v for (h, a), v in score_matrix.items() if h > a)
        draw_prob = sum(v for (h, a), v in score_matrix.items() if h == a)
        away_prob = sum(v for (h, a), v in score_matrix.items() if h < a)
        
        # 淘汰赛调整（平局后有点球）
        if stage in ("r16", "qf", "sf", "final"):
            # 淘汰赛平局概率降低
            draw_prob *= 0.6
            remaining = 1.0 - draw_prob
            home_prob = home_prob / (home_prob + away_prob) * remaining
            away_prob = 1.0 - home_prob - draw_prob
        
        # 推荐投注
        recommended = None
        if odds_home > 1 and odds_draw > 1 and odds_away > 1:
            probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
            odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}
            
            best_value = 0
            best_dir = None
            for direction in ["home", "draw", "away"]:
                value = probs[direction] - (1.0 / odds[direction])
                if value > best_value:
                    best_value = value
                    best_dir = direction
            
            if best_dir and best_value > 0.03:
                recommended = {
                    "direction": best_dir,
                    "odds": odds[best_dir],
                    "value": best_value,
                    "model_prob": probs[best_dir],
                }
        
        return WorldCupPrediction(
            match_id=f"wc_{home_team}_vs_{away_team}",
            home_team=home_team,
            away_team=away_team,
            home_prob=round(home_prob, 4),
            draw_prob=round(draw_prob, 4),
            away_prob=round(away_prob, 4),
            recommended_bet=recommended,
            factors=factors,
        )
    
    def _compute_factors(self, home_team: str, away_team: str, stage: str) -> Dict[str, float]:
        """计算世界杯特殊因子"""
        factors = {}
        
        # 世界杯经验因子
        home_exp = self.history.get(home_team, {}).get("semis", 0)
        away_exp = self.history.get(away_team, {}).get("semis", 0)
        factors["experience_bonus"] = (home_exp - away_exp) * 5  # 每次进四强+5 Elo
        
        # 近期状态因子（简化）
        factors["form_bonus"] = 0.0
        
        # 大赛基因因子
        home_titles = self.history.get(home_team, {}).get("titles", 0)
        away_titles = self.history.get(away_team, {}).get("titles", 0)
        factors["big_game_bonus"] = (home_titles - away_titles) * 3
        
        # 阶段因子
        if stage == "group":
            factors["stage_factor"] = 1.0
        elif stage == "r16":
            factors["stage_factor"] = 1.1
        elif stage == "qf":
            factors["stage_factor"] = 1.2
        elif stage == "sf":
            factors["stage_factor"] = 1.3
        elif stage == "final":
            factors["stage_factor"] = 1.5
        
        return factors
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    
    def predict_group(self, group_teams: List[str]) -> Dict[str, Dict]:
        """
        预测小组出线。
        
        参数:
            group_teams: 小组4支球队
        
        返回:
            {team: {points, gd, gf, position}}
        """
        results = {team: {"points": 0, "gd": 0, "gf": 0, "ga": 0} for team in group_teams}
        
        # 模拟所有对阵
        for i, home in enumerate(group_teams):
            for j, away in enumerate(group_teams):
                if i >= j:
                    continue
                
                pred = self.predict_match(home, away, stage="group")
                
                # 模拟结果
                if pred.home_prob > pred.away_prob and pred.home_prob > pred.draw_prob:
                    results[home]["points"] += 3
                    results[home]["gf"] += 2
                    results[away]["ga"] += 2
                elif pred.away_prob > pred.home_prob and pred.away_prob > pred.draw_prob:
                    results[away]["points"] += 3
                    results[away]["gf"] += 2
                    results[home]["ga"] += 2
                else:
                    results[home]["points"] += 1
                    results[away]["points"] += 1
                    results[home]["gf"] += 1
                    results[away]["gf"] += 1
                    results[home]["ga"] += 1
                    results[away]["ga"] += 1
        
        # 计算净胜球
        for team in results:
            results[team]["gd"] = results[team]["gf"] - results[team]["ga"]
        
        # 排名
        sorted_teams = sorted(results.items(), key=lambda x: (-x[1]["points"], -x[1]["gd"], -x[1]["gf"]))
        for i, (team, _) in enumerate(sorted_teams):
            results[team]["position"] = i + 1
        
        return results
    
    def get_team_info(self, team: str) -> Dict:
        """获取球队信息"""
        return {
            "team": team,
            "elo": self.elo.get(team, 1500),
            "history": self.history.get(team, {}),
            "pot": self._get_pot(team),
        }
    
    def _get_pot(self, team: str) -> int:
        """获取球队档次"""
        for pot, teams in self.teams.items():
            if team in teams:
                return int(pot[-1])
        return 4
    
    def list_teams(self) -> List[str]:
        """列出所有参赛队伍"""
        all_teams = []
        for pot_teams in self.teams.values():
            all_teams.extend(pot_teams)
        return all_teams


# 全局世界杯模块
_wc_module = None


def get_world_cup_module() -> WorldCupModule:
    """获取全局世界杯模块"""
    global _wc_module
    if _wc_module is None:
        _wc_module = WorldCupModule()
    return _wc_module
