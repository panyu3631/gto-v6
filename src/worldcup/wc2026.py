"""
GTO v6.0 — 世界杯2026实时数据

48支球队，12组，104场比赛
"""

from __future__ import annotations
import json
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# 2026世界杯分组 (实际分组)
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

# 球队中文名
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

# 球队Elo评分
WC_ELO = {
    "Argentina": 1860, "France": 1850, "Brazil": 1835, "England": 1810,
    "Belgium": 1790, "Portugal": 1780, "Spain": 1775, "Netherlands": 1770,
    "Germany": 1760, "Croatia": 1750, "Uruguay": 1740, "Colombia": 1730,
    "Mexico": 1720, "USMNT": 1710, "Senegal": 1700, "Japan": 1690,
    "Morocco": 1680, "Switzerland": 1670, "Denmark": 1660, "Australia": 1650,
    "South Korea": 1640, "Iran": 1630, "Saudi Arabia": 1620, "Ecuador": 1610,
    "Nigeria": 1600, "Serbia": 1590, "Poland": 1580, "Cameroon": 1570,
    "Canada": 1560, "Wales": 1550, "Tunisia": 1540, "Ghana": 1530,
    "Turkey": 1600, "Scotland": 1580, "Norway": 1620, "Egypt": 1550,
    "Ivory Coast": 1570, "Paraguay": 1550, "Cape Verde": 1450,
    "Bosnia Herzegovina": 1520, "Qatar": 1500, "Haiti": 1400,
    "Curaçao": 1350, "Iraq": 1480, "Jordan": 1460, "Algeria": 1520,
    "Austria": 1580, "DR Congo": 1480, "Uzbekistan": 1450, "New Zealand": 1420,
    "Panama": 1500,
}

# 比赛结果 (已踢完的比赛)
WC_RESULTS = {
    "2026-06-11": [
        {"home": "Mexico", "away": "South Africa", "home_score": 2, "away_score": 0, "group": "A"},
        {"home": "South Korea", "away": "Czechia", "home_score": 2, "away_score": 1, "group": "A"},
    ],
    "2026-06-12": [
        {"home": "Canada", "away": "Bosnia Herzegovina", "home_score": 1, "away_score": 1, "group": "B"},
        {"home": "USMNT", "away": "Paraguay", "home_score": 4, "away_score": 1, "group": "D"},
    ],
    "2026-06-13": [
        {"home": "Qatar", "away": "Switzerland", "home_score": 1, "away_score": 1, "group": "B"},
        {"home": "Brazil", "away": "Morocco", "home_score": 1, "away_score": 1, "group": "C"},
        {"home": "Haiti", "away": "Scotland", "home_score": 0, "away_score": 1, "group": "C"},
        {"home": "Australia", "away": "Turkey", "home_score": 2, "away_score": 0, "group": "D"},
    ],
    "2026-06-14": [
        {"home": "Germany", "away": "Curaçao", "home_score": 7, "away_score": 1, "group": "E"},
        {"home": "Netherlands", "away": "Japan", "home_score": 2, "away_score": 2, "group": "F"},
        {"home": "Ivory Coast", "away": "Ecuador", "home_score": 1, "away_score": 0, "group": "E"},
        {"home": "Sweden", "away": "Tunisia", "home_score": 5, "away_score": 1, "group": "F"},
    ],
    "2026-06-15": [
        {"home": "Spain", "away": "Cape Verde", "home_score": 0, "away_score": 0, "group": "H"},
        {"home": "Belgium", "away": "Egypt", "home_score": 1, "away_score": 1, "group": "G"},
        {"home": "Saudi Arabia", "away": "Uruguay", "home_score": 1, "away_score": 1, "group": "H"},
        {"home": "Iran", "away": "New Zealand", "home_score": 2, "away_score": 2, "group": "G"},
    ],
    "2026-06-16": [
        {"home": "France", "away": "Senegal", "home_score": 3, "away_score": 1, "group": "I"},
        {"home": "Iraq", "away": "Norway", "home_score": 1, "away_score": 4, "group": "I"},
        {"home": "Argentina", "away": "Algeria", "home_score": 3, "away_score": 0, "group": "J"},
        {"home": "Austria", "away": "Jordan", "home_score": 3, "away_score": 1, "group": "J"},
    ],
    "2026-06-17": [
        {"home": "Portugal", "away": "DR Congo", "home_score": 1, "away_score": 1, "group": "K"},
        {"home": "England", "away": "Croatia", "home_score": 4, "away_score": 2, "group": "L"},
        {"home": "Ghana", "away": "Panama", "home_score": 1, "away_score": 0, "group": "L"},
        {"home": "Uzbekistan", "away": "Colombia", "home_score": 1, "away_score": 3, "group": "K"},
    ],
    "2026-06-18": [
        {"home": "South Africa", "away": "Czechia", "home_score": None, "away_score": None, "group": "A"},
        {"home": "Switzerland", "away": "Bosnia Herzegovina", "home_score": None, "away_score": None, "group": "B"},
        {"home": "Canada", "away": "Qatar", "home_score": None, "away_score": None, "group": "B"},
        {"home": "Mexico", "away": "South Korea", "home_score": None, "away_score": None, "group": "A"},
    ],
    "2026-06-19": [
        {"home": "USMNT", "away": "Australia", "home_score": None, "away_score": None, "group": "D"},
        {"home": "Scotland", "away": "Morocco", "home_score": None, "away_score": None, "group": "C"},
        {"home": "Brazil", "away": "Haiti", "home_score": None, "away_score": None, "group": "C"},
        {"home": "Paraguay", "away": "Turkey", "home_score": None, "away_score": None, "group": "D"},
    ],
    "2026-06-20": [
        {"home": "Netherlands", "away": "Sweden", "home_score": None, "away_score": None, "group": "F"},
        {"home": "Germany", "away": "Ivory Coast", "home_score": None, "away_score": None, "group": "E"},
        {"home": "Ecuador", "away": "Curaçao", "home_score": None, "away_score": None, "group": "E"},
        {"home": "Tunisia", "away": "Japan", "home_score": None, "away_score": None, "group": "F"},
    ],
    "2026-06-21": [
        {"home": "Spain", "away": "Saudi Arabia", "home_score": None, "away_score": None, "group": "H"},
        {"home": "Belgium", "away": "Iran", "home_score": None, "away_score": None, "group": "G"},
        {"home": "Uruguay", "away": "Cape Verde", "home_score": None, "away_score": None, "group": "H"},
        {"home": "New Zealand", "away": "Egypt", "home_score": None, "away_score": None, "group": "G"},
    ],
    "2026-06-22": [
        {"home": "Argentina", "away": "Austria", "home_score": None, "away_score": None, "group": "J"},
        {"home": "France", "away": "Iraq", "home_score": None, "away_score": None, "group": "I"},
        {"home": "Norway", "away": "Senegal", "home_score": None, "away_score": None, "group": "I"},
        {"home": "Jordan", "away": "Algeria", "home_score": None, "away_score": None, "group": "J"},
    ],
    "2026-06-23": [
        {"home": "Portugal", "away": "Uzbekistan", "home_score": None, "away_score": None, "group": "K"},
        {"home": "Colombia", "away": "DR Congo", "home_score": None, "away_score": None, "group": "K"},
        {"home": "England", "away": "Ghana", "home_score": None, "away_score": None, "group": "L"},
        {"home": "Croatia", "away": "Panama", "home_score": None, "away_score": None, "group": "L"},
    ],
    "2026-06-24": [
        {"home": "Mexico", "away": "Czechia", "home_score": None, "away_score": None, "group": "A"},
        {"home": "South Korea", "away": "South Africa", "home_score": None, "away_score": None, "group": "A"},
        {"home": "Canada", "away": "Switzerland", "home_score": None, "away_score": None, "group": "B"},
        {"home": "Qatar", "away": "Bosnia Herzegovina", "home_score": None, "away_score": None, "group": "B"},
    ],
    "2026-06-25": [
        {"home": "Brazil", "away": "Scotland", "home_score": None, "away_score": None, "group": "C"},
        {"home": "Morocco", "away": "Haiti", "home_score": None, "away_score": None, "group": "C"},
        {"home": "USMNT", "away": "Turkey", "home_score": None, "away_score": None, "group": "D"},
        {"home": "Paraguay", "away": "Australia", "home_score": None, "away_score": None, "group": "D"},
    ],
    "2026-06-26": [
        {"home": "Germany", "away": "Ecuador", "home_score": None, "away_score": None, "group": "E"},
        {"home": "Curaçao", "away": "Ivory Coast", "home_score": None, "away_score": None, "group": "E"},
        {"home": "Netherlands", "away": "Tunisia", "home_score": None, "away_score": None, "group": "F"},
        {"home": "Japan", "away": "Sweden", "home_score": None, "away_score": None, "group": "F"},
    ],
    "2026-06-27": [
        {"home": "Spain", "away": "Uruguay", "home_score": None, "away_score": None, "group": "H"},
        {"home": "Cape Verde", "away": "Saudi Arabia", "home_score": None, "away_score": None, "group": "H"},
        {"home": "Belgium", "away": "New Zealand", "home_score": None, "away_score": None, "group": "G"},
        {"home": "Egypt", "away": "Iran", "home_score": None, "away_score": None, "group": "G"},
    ],
    "2026-06-28": [
        {"home": "France", "away": "Norway", "home_score": None, "away_score": None, "group": "I"},
        {"home": "Senegal", "away": "Iraq", "home_score": None, "away_score": None, "group": "I"},
        {"home": "Argentina", "away": "Jordan", "home_score": None, "away_score": None, "group": "J"},
        {"home": "Algeria", "away": "Austria", "home_score": None, "away_score": None, "group": "J"},
        {"home": "Portugal", "away": "Colombia", "home_score": None, "away_score": None, "group": "K"},
        {"home": "DR Congo", "away": "Uzbekistan", "home_score": None, "away_score": None, "group": "K"},
        {"home": "England", "away": "Panama", "home_score": None, "away_score": None, "group": "L"},
        {"home": "Ghana", "away": "Croatia", "home_score": None, "away_score": None, "group": "L"},
    ],
}


@dataclass
class WCMatch:
    """世界杯比赛"""
    date: str
    home_team: str
    away_team: str
    home_team_cn: str
    away_team_cn: str
    group: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "upcoming"  # upcoming, live, finished
    home_prob: float = 0.0
    draw_prob: float = 0.0
    away_prob: float = 0.0
    recommended: str = ""
    value: float = 0.0


class WorldCup2026:
    """2026世界杯模块"""
    
    def __init__(self):
        self.groups = WC_2026_GROUPS
        self.team_cn = WC_TEAM_CN
        self.elo = WC_ELO
        self.results = WC_RESULTS
    
    def get_cn_name(self, team: str) -> str:
        """获取球队中文名"""
        return self.team_cn.get(team, team)
    
    def get_all_matches(self) -> List[WCMatch]:
        """获取所有比赛"""
        matches = []
        for date, games in self.results.items():
            for game in games:
                home = game["home"]
                away = game["away"]
                home_score = game.get("home_score")
                away_score = game.get("away_score")
                
                # 判断状态
                if home_score is not None:
                    status = "finished"
                elif date < "2026-06-21":
                    status = "finished"  # 假设之前的比赛已结束
                elif date == "2026-06-21":
                    status = "today"
                else:
                    status = "upcoming"
                
                # 计算概率
                probs = self._predict(home, away)
                
                # 推荐
                if probs["home"] > 45:
                    rec = self.get_cn_name(home)
                elif probs["away"] > 45:
                    rec = self.get_cn_name(away)
                else:
                    rec = "观望"
                
                match = WCMatch(
                    date=date,
                    home_team=home,
                    away_team=away,
                    home_team_cn=self.get_cn_name(home),
                    away_team_cn=self.get_cn_name(away),
                    group=game["group"],
                    home_score=home_score,
                    away_score=away_score,
                    status=status,
                    home_prob=probs["home"],
                    draw_prob=probs["draw"],
                    away_prob=probs["away"],
                    recommended=rec,
                    value=probs["home"] - 45 if probs["home"] > probs["away"] else probs["away"] - 45,
                )
                matches.append(match)
        
        return matches
    
    def get_today_matches(self) -> List[WCMatch]:
        """获取今日比赛"""
        all_matches = self.get_all_matches()
        return [m for m in all_matches if m.status == "today"]
    
    def get_upcoming_matches(self) -> List[WCMatch]:
        """获取未来比赛"""
        all_matches = self.get_all_matches()
        return [m for m in all_matches if m.status == "upcoming"]
    
    def get_finished_matches(self) -> List[WCMatch]:
        """获取已结束比赛"""
        all_matches = self.get_all_matches()
        return [m for m in all_matches if m.status == "finished"]
    
    def get_group_standings(self) -> Dict[str, List[Dict]]:
        """获取小组积分榜"""
        standings = {}
        
        for group, teams in self.groups.items():
            team_stats = {}
            for team in teams:
                team_stats[team] = {
                    "team": team,
                    "team_cn": self.get_cn_name(team),
                    "played": 0, "wins": 0, "draws": 0, "losses": 0,
                    "gf": 0, "ga": 0, "gd": 0, "points": 0,
                }
            
            # 计算积分
            for date, games in self.results.items():
                for game in games:
                    if game["group"] != group:
                        continue
                    if game.get("home_score") is None:
                        continue
                    
                    home = game["home"]
                    away = game["away"]
                    hs = game["home_score"]
                    as_ = game["away_score"]
                    
                    if home in team_stats:
                        team_stats[home]["played"] += 1
                        team_stats[home]["gf"] += hs
                        team_stats[home]["ga"] += as_
                        team_stats[home]["gd"] = team_stats[home]["gf"] - team_stats[home]["ga"]
                    
                    if away in team_stats:
                        team_stats[away]["played"] += 1
                        team_stats[away]["gf"] += as_
                        team_stats[away]["ga"] += hs
                        team_stats[away]["gd"] = team_stats[away]["gf"] - team_stats[away]["ga"]
                    
                    if hs > as_:
                        if home in team_stats:
                            team_stats[home]["wins"] += 1
                            team_stats[home]["points"] += 3
                        if away in team_stats:
                            team_stats[away]["losses"] += 1
                    elif hs < as_:
                        if away in team_stats:
                            team_stats[away]["wins"] += 1
                            team_stats[away]["points"] += 3
                        if home in team_stats:
                            team_stats[home]["losses"] += 1
                    else:
                        if home in team_stats:
                            team_stats[home]["draws"] += 1
                            team_stats[home]["points"] += 1
                        if away in team_stats:
                            team_stats[away]["draws"] += 1
                            team_stats[away]["points"] += 1
            
            # 排名
            sorted_teams = sorted(
                team_stats.values(),
                key=lambda x: (-x["points"], -x["gd"], -x["gf"])
            )
            for i, t in enumerate(sorted_teams):
                t["position"] = i + 1
            
            standings[group] = sorted_teams
        
        return standings
    
    def _predict(self, home: str, away: str) -> Dict[str, float]:
        """预测比赛概率"""
        h_elo = self.elo.get(home, 1500)
        a_elo = self.elo.get(away, 1500)
        diff = h_elo - a_elo
        
        expected = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        draw = 0.25 * (1.0 - abs(expected - 0.5) * 2)
        home_prob = expected * (1.0 - draw)
        away_prob = (1.0 - expected) * (1.0 - draw)
        
        return {
            "home": round(home_prob * 100),
            "draw": round(draw * 100),
            "away": round(away_prob * 100),
        }


# 全局实例
_wc = None


def get_world_cup() -> WorldCup2026:
    """获取世界杯模块"""
    global _wc
    if _wc is None:
        _wc = WorldCup2026()
    return _wc
