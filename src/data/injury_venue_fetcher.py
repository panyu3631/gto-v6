"""
GTO v6.0 — 伤停数据获取

从网页搜索获取伤停信息。
"""

from __future__ import annotations
import re
import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 球队名英文到中文映射
TEAM_CN = {
    "Manchester City": "曼城", "Arsenal": "阿森纳", "Liverpool": "利物浦",
    "Aston Villa": "阿斯顿维拉", "Tottenham": "热刺", "Chelsea": "切尔西",
    "Newcastle": "纽卡斯尔", "Man United": "曼联", "West Ham": "西汉姆",
    "Crystal Palace": "水晶宫", "Brighton": "布莱顿", "Bournemouth": "伯恩茅斯",
    "Fulham": "富勒姆", "Wolves": "狼队", "Everton": "埃弗顿",
    "Brentford": "布伦特福德", "Nott'm Forest": "诺丁汉森林", "Luton": "卢顿",
    "Burnley": "伯恩利", "Sheffield United": "谢菲尔德联",
    "Real Madrid": "皇家马德里", "Barcelona": "巴塞罗那",
    "Atletico Madrid": "马德里竞技", "Sevilla": "塞维利亚",
    "Bayern Munich": "拜仁慕尼黑", "Borussia Dortmund": "多特蒙德",
    "Inter Milan": "国际米兰", "AC Milan": "AC米兰", "Juventus": "尤文图斯",
    "Paris Saint-Germain": "巴黎圣日耳曼", "Marseille": "马赛", "Lyon": "里昂",
}


class InjuryFetcher:
    """伤停数据获取器"""
    
    def __init__(self):
        self._cache = {}
    
    def get_injuries(self, team: str, league: str = "") -> List[Dict]:
        """
        获取球队伤停信息。
        
        参数:
            team: 球队名
            league: 联赛名
        
        返回:
            [{"player_name": str, "injury_type": str, "status": str, "expected_return": str}]
        """
        cache_key = f"{team}_{league}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从网页搜索获取
        injuries = self._search_injuries(team)
        
        self._cache[cache_key] = injuries
        return injuries
    
    def _search_injuries(self, team: str) -> List[Dict]:
        """从网页搜索伤停信息"""
        team_cn = TEAM_CN.get(team, team)
        query = f"{team_cn} {team} 伤停 伤病 injured"
        
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
            
            return self._parse_injuries(html, team)
        
        except Exception as e:
            logger.debug(f"伤停搜索失败: {e}")
            return []
    
    def _parse_injuries(self, html: str, team: str) -> List[Dict]:
        """从HTML解析伤停信息"""
        injuries = []
        
        # 尝试匹配伤停列表格式
        # 常见格式: "球员名 - 伤停类型 - 预计回归日期"
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–]\s*(knee|ankle|hamstring|muscle|groin|calf|thigh|shoulder|back|hip)\s*(?:injury|受伤|伤停)?',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s*[:：]\s*(knee|ankle|hamstring|muscle|groin|calf|thigh)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for m in matches:
                player = m[0].strip()
                injury = m[1].strip()
                if len(player) > 3 and len(player) < 30:
                    injuries.append({
                        "player_name": player,
                        "injury_type": injury,
                        "status": "out",
                        "expected_return": "",
                    })
        
        # 去重
        seen = set()
        unique = []
        for inj in injuries:
            key = inj["player_name"]
            if key not in seen:
                seen.add(key)
                unique.append(inj)
        
        return unique[:10]  # 最多返回10条


# 球队场地信息 (静态数据)
TEAM_VENUES = {
    # 英超
    "Manchester City": {"venue_name": "伊蒂哈德球场", "city": "Manchester", "capacity": 53400},
    "Arsenal": {"venue_name": "酋长球场", "city": "London", "capacity": 60704},
    "Liverpool": {"venue_name": "安菲尔德球场", "city": "Liverpool", "capacity": 53394},
    "Aston Villa": {"venue_name": "维拉公园球场", "city": "Birmingham", "capacity": 42682},
    "Tottenham": {"venue_name": "托特纳姆热刺球场", "city": "London", "capacity": 62850},
    "Chelsea": {"venue_name": "斯坦福桥球场", "city": "London", "capacity": 40343},
    "Newcastle": {"venue_name": "圣詹姆斯公园球场", "city": "Newcastle", "capacity": 52305},
    "Man United": {"venue_name": "老特拉福德球场", "city": "Manchester", "capacity": 74140},
    "West Ham": {"venue_name": "伦敦碗", "city": "London", "capacity": 62500},
    "Crystal Palace": {"venue_name": "塞尔赫斯特公园球场", "city": "London", "capacity": 25486},
    "Brighton": {"venue_name": "法尔马球场", "city": "Brighton", "capacity": 31876},
    "Bournemouth": {"venue_name": "活力球场", "city": "Bournemouth", "capacity": 11364},
    "Fulham": {"venue_name": "克拉文农场球场", "city": "London", "capacity": 25700},
    "Wolves": {"venue_name": "莫利纽球场", "city": "Wolverhampton", "capacity": 31750},
    "Everton": {"venue_name": "古迪逊公园球场", "city": "Liverpool", "capacity": 39414},
    "Brentford": {"venue_name": "布伦特福德社区球场", "city": "London", "capacity": 17250},
    "Nott'm Forest": {"venue_name": "城市球场", "city": "Nottingham", "capacity": 30445},
    "Luton": {"venue_name": "肯尼尔沃思路球场", "city": "Luton", "capacity": 10356},
    "Burnley": {"venue_name": "特夫摩尔球场", "city": "Burnley", "capacity": 21944},
    "Sheffield United": {"venue_name": "布拉莫巷球场", "city": "Sheffield", "capacity": 32050},
    
    # 西甲
    "Real Madrid": {"venue_name": "伯纳乌球场", "city": "Madrid", "capacity": 81044},
    "Barcelona": {"venue_name": "诺坎普球场", "city": "Barcelona", "capacity": 99354},
    "Atletico Madrid": {"venue_name": "万达大都会球场", "city": "Madrid", "capacity": 68456},
    
    # 德甲
    "Bayern Munich": {"venue_name": "安联球场", "city": "Munich", "capacity": 75024},
    "Borussia Dortmund": {"venue_name": "威斯特法伦球场", "city": "Dortmund", "capacity": 81365},
    
    # 意甲
    "Inter Milan": {"venue_name": "梅阿查球场", "city": "Milan", "capacity": 75923},
    "AC Milan": {"venue_name": "圣西罗球场", "city": "Milan", "capacity": 75923},
    "Juventus": {"venue_name": "安联球场", "city": "Turin", "capacity": 41507},
    
    # 法甲
    "Paris Saint-Germain": {"venue_name": "王子公园球场", "city": "Paris", "capacity": 47929},
    "Marseille": {"venue_name": "韦洛德罗姆球场", "city": "Marseille", "capacity": 67394},
    "Lyon": {"venue_name": "奥林匹克公园球场", "city": "Lyon", "capacity": 59186},
}


class VenueFetcher:
    """场地信息获取器"""
    
    def get_venue(self, team: str) -> Optional[Dict]:
        """
        获取球队主场信息。
        
        参数:
            team: 球队名
        
        返回:
            {"venue_name": str, "city": str, "capacity": int}
        """
        return TEAM_VENUES.get(team)


# 全局实例
_injury_fetcher = None
_venue_fetcher = None


def get_injury_fetcher() -> InjuryFetcher:
    """获取全局伤停获取器"""
    global _injury_fetcher
    if _injury_fetcher is None:
        _injury_fetcher = InjuryFetcher()
    return _injury_fetcher


def get_venue_fetcher() -> VenueFetcher:
    """获取全局场地获取器"""
    global _venue_fetcher
    if _venue_fetcher is None:
        _venue_fetcher = VenueFetcher()
    return _venue_fetcher
