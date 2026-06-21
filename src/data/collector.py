"""
GTO v6.0 — 统一数据采集模块

永久性数据获取能力，支持所有赛事。

数据源:
1. Football-Data.org API — 赛程、积分榜、比赛结果
2. 网页搜索 — 赔率、伤停、比赛统计
3. Open-Meteo API — 天气

使用方式:
    collector = DataCollector(league_id="premier_league")
    collector.collect_upcoming_matches()
    collector.collect_match_odds(match_id)
    collector.collect_match_stats(match_id)
"""

from __future__ import annotations
import json
import os
import re
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# API配置
API_KEY_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.json')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'collected')
os.makedirs(DATA_DIR, exist_ok=True)

# 联赛ID映射
LEAGUE_API_IDS = {
    "premier_league": "PL",
    "la_liga": "PD",
    "bundesliga": "BL1",
    "serie_a": "SA",
    "ligue_1": "FL1",
    "worldcup": "WC",
}

# 城市坐标（天气用）
CITY_COORDS = {
    "London": (51.51, -0.13), "Manchester": (53.48, -2.24),
    "Liverpool": (53.41, -2.98), "Madrid": (40.42, -3.70),
    "Barcelona": (41.39, 2.17), "Munich": (48.14, 11.58),
    "Berlin": (52.52, 13.41), "Milan": (45.46, 9.19),
    "Rome": (41.90, 12.50), "Paris": (48.86, 2.35),
    "Lyon": (45.76, 4.84), "Dortmund": (51.51, 7.47),
    "Turin": (45.07, 7.69), "Naples": (40.85, 14.27),
    "Marseille": (43.30, 5.37), "Atlanta": (33.75, -84.39),
    "Houston": (29.76, -95.37), "Miami": (25.76, -80.19),
    "Seattle": (47.61, -122.33), "Dallas": (32.78, -96.80),
    "New York": (40.71, -74.01), "Los Angeles": (34.05, -118.24),
    "Vancouver": (49.28, -123.12), "Toronto": (43.65, -79.38),
    "Mexico City": (19.43, -99.13), "Guadalajara": (20.67, -103.35),
    "Monterrey": (25.69, -100.32),
}


@dataclass
class CollectedMatch:
    """采集的比赛数据"""
    match_id: str
    league_id: str
    home_team: str
    away_team: str
    kickoff_time: str
    stage: str = "regular"
    group: str = ""
    venue: str = ""
    city: str = ""
    
    # 结果
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "scheduled"
    
    # 赔率
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    odds_over25: float = 0.0
    odds_under25: float = 0.0
    ah_line: float = 0.0
    ah_home: float = 0.0
    ah_away: float = 0.0
    
    # 比赛统计
    shots_home: int = 0
    shots_away: int = 0
    sot_home: int = 0
    sot_away: int = 0
    corners_home: int = 0
    corners_away: int = 0
    yellows_home: int = 0
    yellows_away: int = 0
    ht_home: int = 0
    ht_away: int = 0
    
    # 天气
    temperature: float = 0.0
    wind_speed: float = 0.0
    precipitation: float = 0.0
    weather_desc: str = ""
    
    # 元数据
    source: str = ""
    collected_at: str = ""


class DataCollector:
    """统一数据采集器"""
    
    def __init__(self, league_id: str):
        self.league_id = league_id
        self.api_id = LEAGUE_API_IDS.get(league_id, "")
        self.api_key = self._load_api_key()
        self._cache: Dict[str, Any] = {}
        self._cache_dir = os.path.join(DATA_DIR, league_id)
        os.makedirs(self._cache_dir, exist_ok=True)
    
    def _load_api_key(self) -> str:
        """加载API Key"""
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, 'r') as f:
                config = json.load(f)
                return config.get("football_data", {}).get("api_key", "")
        return ""
    
    # ═══════════════════════════════════════════════════════════════
    # 赛程数据
    # ═══════════════════════════════════════════════════════════════
    
    def collect_upcoming_matches(self, days: int = 7) -> List[CollectedMatch]:
        """采集未来N天的比赛"""
        matches = []
        
        # 从Football-Data.org获取
        if self.api_key:
            api_matches = self._fetch_from_api()
            matches.extend(api_matches)
        
        # 从ESPN获取（备用）
        if not matches:
            espn_matches = self._fetch_from_espn()
            matches.extend(espn_matches)
        
        # 保存到文件
        if matches:
            self._save_matches(matches)
        
        return matches
    
    def _fetch_from_api(self) -> List[CollectedMatch]:
        """从Football-Data.org获取赛程"""
        if not self.api_id:
            return []
        
        try:
            url = f"https://api.football-data.org/v4/competitions/{self.api_id}/matches"
            req = urllib.request.Request(url)
            req.add_header('X-Auth-Token', self.api_key)
            
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            
            matches = []
            for m in data.get("matches", []):
                home = m.get("homeTeam", {}).get("name", "")
                away = m.get("awayTeam", {}).get("name", "")
                
                match = CollectedMatch(
                    match_id=str(m.get("id", "")),
                    league_id=self.league_id,
                    home_team=home,
                    away_team=away,
                    kickoff_time=m.get("utcDate", ""),
                    stage=m.get("stage", "REGULAR"),
                    group=m.get("group", ""),
                    status=m.get("status", "SCHEDULED").lower(),
                    home_score=m.get("score", {}).get("fullTime", {}).get("home"),
                    away_score=m.get("score", {}).get("fullTime", {}).get("away"),
                    source="football-data.org",
                    collected_at=datetime.now().isoformat(),
                )
                matches.append(match)
            
            logger.info(f"从API获取 {len(matches)} 场比赛")
            return matches
        
        except Exception as e:
            logger.warning(f"API获取失败: {e}")
            return []
    
    def _fetch_from_espn(self) -> List[CollectedMatch]:
        """从ESPN获取赛程"""
        espn_ids = {
            "premier_league": "eng.1",
            "la_liga": "esp.1",
            "bundesliga": "ger.1",
            "serie_a": "ita.1",
            "ligue_1": "fra.1",
            "worldcup": "fifa.world",
        }
        
        espn_id = espn_ids.get(self.league_id)
        if not espn_id:
            return []
        
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/scoreboard"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            matches = []
            for event in data.get("events", []):
                comp = event.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                
                if len(competitors) < 2:
                    continue
                
                home = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away = next((c for c in competitors if c.get("homeAway") == "away"), {})
                
                match = CollectedMatch(
                    match_id=event.get("id", ""),
                    league_id=self.league_id,
                    home_team=home.get("team", {}).get("shortDisplayName", ""),
                    away_team=away.get("team", {}).get("shortDisplayName", ""),
                    kickoff_time=event.get("date", ""),
                    status="scheduled",
                    source="espn",
                    collected_at=datetime.now().isoformat(),
                )
                matches.append(match)
            
            logger.info(f"从ESPN获取 {len(matches)} 场比赛")
            return matches
        
        except Exception as e:
            logger.warning(f"ESPN获取失败: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════
    # 赔率数据
    # ═══════════════════════════════════════════════════════════════
    
    def collect_match_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """采集比赛赔率（网页搜索）"""
        cache_key = f"odds_{home_team}_{away_team}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 构建搜索查询
        home_cn = self._get_team_cn(home_team)
        away_cn = self._get_team_cn(away_team)
        query = f"{home_cn} {away_cn} 赔率 胜平负 大小球"
        
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
            
            odds = self._parse_odds_from_html(html)
            if odds:
                self._cache[cache_key] = odds
                self._save_odds(home_team, away_team, odds)
            
            return odds
        
        except Exception as e:
            logger.warning(f"赔率获取失败: {e}")
            return None
    
    def _parse_odds_from_html(self, html: str) -> Optional[Dict]:
        """从HTML解析赔率"""
        patterns = [
            r'(\d+\.?\d*)\s*[/／]\s*(\d+\.?\d*)\s*[/／]\s*(\d+\.?\d*)',
            r'(?:主胜|1)\s*[:：]?\s*(\d+\.?\d*)\s*(?:平|X)\s*[:：]?\s*(\d+\.?\d*)\s*(?:客胜|2)\s*[:：]?\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                o1, o2, o3 = float(match.group(1)), float(match.group(2)), float(match.group(3))
                if self._validate_odds(o1, o2, o3):
                    # 确定哪个是主胜（通常最小）
                    if o1 <= o2 and o1 <= o3:
                        return {"home": o1, "draw": o2, "away": o3}
                    elif o3 <= o1 and o3 <= o2:
                        return {"home": o3, "draw": o1, "away": o2}
                    else:
                        return {"home": o1, "draw": o2, "away": o3}
        
        return None
    
    def _validate_odds(self, o1: float, o2: float, o3: float) -> bool:
        """验证赔率是否合理"""
        if o1 <= 1 or o2 <= 1 or o3 <= 1:
            return False
        if o1 > 50 or o2 > 50 or o3 > 50:
            return False
        margin = 1.0/o1 + 1.0/o2 + 1.0/o3
        return 0.8 < margin < 1.3
    
    # ═══════════════════════════════════════════════════════════════
    # 天气数据
    # ═══════════════════════════════════════════════════════════════
    
    def collect_weather(self, city: str) -> Optional[Dict]:
        """采集天气数据"""
        cache_key = f"weather_{city}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        coords = CITY_COORDS.get(city)
        if not coords:
            # 模糊匹配
            for c, coord in CITY_COORDS.items():
                if c.lower() in city.lower() or city.lower() in c.lower():
                    coords = coord
                    break
        
        if not coords:
            return None
        
        try:
            lat, lon = coords
            url = (f"https://api.open-meteo.com/v1/forecast?"
                   f"latitude={lat}&longitude={lon}"
                   f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code"
                   f"&timezone=auto")
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            current = data.get("current", {})
            result = {
                "temperature": current.get("temperature_2m", 15),
                "humidity": current.get("relative_humidity_2m", 50),
                "wind_speed": current.get("wind_speed_10m", 0),
                "precipitation": current.get("precipitation", 0),
                "weather_code": current.get("weather_code", 0),
                "weather_desc": self._get_weather_desc(current.get("weather_code", 0)),
                "impact": self._calc_weather_impact(
                    current.get("temperature_2m", 15),
                    current.get("wind_speed_10m", 0),
                    current.get("precipitation", 0),
                ),
            }
            
            self._cache[cache_key] = result
            return result
        
        except Exception as e:
            logger.warning(f"天气获取失败: {e}")
            return None
    
    def _get_weather_desc(self, code: int) -> str:
        desc = {0: "晴天", 1: "多云", 2: "阴天", 3: "阴天",
                45: "雾", 51: "毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
                71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨", 95: "雷暴"}
        return desc.get(code, "未知")
    
    def _calc_weather_impact(self, temp: float, wind: float, precip: float) -> float:
        score = 0.0
        if 15 <= temp <= 25: score += 0.3
        elif temp < 5 or temp > 35: score -= 0.3
        if precip == 0: score += 0.3
        elif precip > 10: score -= 0.3
        if wind < 10: score += 0.2
        elif wind > 25: score -= 0.2
        return round(score, 2)
    
    # ═══════════════════════════════════════════════════════════════
    # 比赛统计（赛后）
    # ═══════════════════════════════════════════════════════════════
    
    def collect_match_stats(self, home_team: str, away_team: str) -> Optional[Dict]:
        """采集比赛统计（赛后）"""
        cache_key = f"stats_{home_team}_{away_team}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从Football-Data.org API获取
        if self.api_key:
            stats = self._fetch_match_stats_from_api(home_team, away_team)
            if stats:
                self._cache[cache_key] = stats
                return stats
        
        return None
    
    def _fetch_match_stats_from_api(self, home_team: str, away_team: str) -> Optional[Dict]:
        """从API获取比赛统计"""
        # Football-Data.org免费版不提供详细统计
        # 需要从其他来源获取
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # 积分榜
    # ═══════════════════════════════════════════════════════════════
    
    def collect_standings(self) -> Optional[Dict]:
        """采集积分榜"""
        if not self.api_key or not self.api_id:
            return None
        
        try:
            url = f"https://api.football-data.org/v4/competitions/{self.api_id}/standings"
            req = urllib.request.Request(url)
            req.add_header('X-Auth-Token', self.api_key)
            
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            
            standings = {}
            for standing in data.get("standings", []):
                if standing.get("type") == "TOTAL":
                    for entry in standing.get("table", []):
                        team = entry.get("team", {}).get("name", "")
                        standings[team] = {
                            "position": entry.get("position", 0),
                            "points": entry.get("points", 0),
                            "played": entry.get("playedGames", 0),
                            "won": entry.get("won", 0),
                            "draw": entry.get("draw", 0),
                            "lost": entry.get("lost", 0),
                            "gf": entry.get("goalsFor", 0),
                            "ga": entry.get("goalsAgainst", 0),
                            "gd": entry.get("goalDifference", 0),
                        }
            
            if standings:
                self._save_standings(standings)
            
            return standings
        
        except Exception as e:
            logger.warning(f"积分榜获取失败: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════
    # 工具函数
    # ═══════════════════════════════════════════════════════════════
    
    def _get_team_cn(self, team: str) -> str:
        """获取球队中文名（简化版）"""
        cn_map = {
            "Manchester City": "曼城", "Arsenal": "阿森纳", "Liverpool": "利物浦",
            "Real Madrid": "皇家马德里", "Barcelona": "巴塞罗那",
            "Bayern Munich": "拜仁慕尼黑", "Borussia Dortmund": "多特蒙德",
            "Inter Milan": "国际米兰", "AC Milan": "AC米兰", "Juventus": "尤文图斯",
            "Paris Saint-Germain": "巴黎圣日耳曼",
            "Argentina": "阿根廷", "France": "法国", "Brazil": "巴西",
            "England": "英格兰", "Spain": "西班牙", "Germany": "德国",
            "Netherlands": "荷兰", "Portugal": "葡萄牙", "Belgium": "比利时",
        }
        return cn_map.get(team, team)
    
    def _save_matches(self, matches: List[CollectedMatch]):
        """保存比赛数据"""
        data = []
        for m in matches:
            data.append({
                "match_id": m.match_id,
                "league_id": m.league_id,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "kickoff_time": m.kickoff_time,
                "stage": m.stage,
                "group": m.group,
                "status": m.status,
                "home_score": m.home_score,
                "away_score": m.away_score,
                "source": m.source,
            })
        
        path = os.path.join(self._cache_dir, f"matches_{datetime.now().strftime('%Y%m%d')}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_odds(self, home_team: str, away_team: str, odds: Dict):
        """保存赔率数据"""
        path = os.path.join(self._cache_dir, f"odds_{home_team}_{away_team}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(odds, f, indent=2)
    
    def _save_standings(self, standings: Dict):
        """保存积分榜"""
        path = os.path.join(self._cache_dir, f"standings_{datetime.now().strftime('%Y%m%d')}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(standings, f, ensure_ascii=False, indent=2)


# 全局实例
_collectors: Dict[str, DataCollector] = {}


def get_collector(league_id: str) -> DataCollector:
    """获取数据采集器"""
    if league_id not in _collectors:
        _collectors[league_id] = DataCollector(league_id)
    return _collectors[league_id]
