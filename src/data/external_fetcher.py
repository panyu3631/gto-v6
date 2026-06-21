"""
GTO v6.0 — 外部数据源接入模块

可接入的数据源:
1. football-data.org — 免费，无需API Key (积分榜/赛程)
2. OpenWeatherMap — 需要API Key (天气)
3. API-Football — 需要API Key (伤病/赔率/xG)

使用方式:
    fetcher = ExternalDataFetcher(api_keys={"weather": "xxx"})
    standings = fetcher.get_standings("premier_league", "2023-24")
    weather = fetcher.get_weather("London")
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# football-data.org 免费API (无需Key)
FOOTBALL_DATA_IDS = {
    "premier_league": "PL",
    "la_liga": "PD",
    "bundesliga": "BL1",
    "serie_a": "SA",
    "ligue_1": "FL1",
}

# Open-Meteo 免费天气API (无需Key)
WEATHER_CITIES = {
    "premier_league": {"lat": 51.5, "lon": -0.12, "name": "London"},
    "la_liga": {"lat": 40.4, "lon": -3.7, "name": "Madrid"},
    "bundesliga": {"lat": 52.52, "lon": 13.4, "name": "Berlin"},
    "serie_a": {"lat": 41.9, "lon": 12.5, "name": "Rome"},
    "ligue_1": {"lat": 48.85, "lon": 2.35, "name": "Paris"},
}


class ExternalDataFetcher:
    """外部数据源获取器"""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self._cache: Dict[str, Any] = {}

    def get_standings(self, league_id: str, season: str) -> Optional[Dict]:
        """
        从 football-data.org 获取积分榜 (免费，无需Key)

        返回:
            {
                "team_name": {
                    "position": int,
                    "points": int,
                    "played": int,
                    "won": int,
                    "draw": int,
                    "lost": int,
                    "goals_for": int,
                    "goals_against": int,
                    "goal_difference": int
                }
            }
        """
        cache_key = f"standings_{league_id}_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        comp_id = FOOTBALL_DATA_IDS.get(league_id)
        if not comp_id:
            return None

        try:
            import urllib.request
            url = f"https://api.football-data.org/v4/competitions/{comp_id}/standings?season={season[:4]}"
            req = urllib.request.Request(url)
            req.add_header("X-Auth-Token", self.api_keys.get("football_data", ""))
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            standings = {}
            for entry in data.get("standings", [{}])[0].get("table", []):
                team = entry.get("team", {}).get("name", "")
                if team:
                    standings[team] = {
                        "position": entry.get("position", 0),
                        "points": entry.get("points", 0),
                        "played": entry.get("playedGames", 0),
                        "won": entry.get("won", 0),
                        "draw": entry.get("draw", 0),
                        "lost": entry.get("lost", 0),
                        "goals_for": entry.get("goalsFor", 0),
                        "goals_against": entry.get("goalsAgainst", 0),
                        "goal_difference": entry.get("goalDifference", 0),
                    }

            self._cache[cache_key] = standings
            return standings

        except Exception as e:
            logger.warning(f"获取积分榜失败: {e}")
            return None

    def get_weather(self, league_id: str) -> Optional[Dict]:
        """
        从 Open-Meteo 获取天气 (免费，无需Key)

        返回:
            {"temp": float, "rain": float, "wind": float, "impact": float}
        """
        cache_key = f"weather_{league_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        city = WEATHER_CITIES.get(league_id)
        if not city:
            return None

        try:
            import urllib.request
            url = f"https://api.open-meteo.com/v1/forecast?latitude={city['lat']}&longitude={city['lon']}&current_weather=true&hourly=precipitation"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            current = data.get("current_weather", {})
            temp = current.get("temperature", 15)
            wind = current.get("windspeed", 0)
            # 降水量从hourly获取
            hourly = data.get("hourly", {})
            precip = hourly.get("precipitation", [0])
            rain = precip[0] if precip else 0

            # 计算天气影响因子 (-1到1)
            temp_score = max(0, min(1, 1 - abs(temp - 18) / 20))
            rain_score = max(0, 1 - rain / 10)
            wind_score = max(0, 1 - wind / 30)
            impact = (temp_score * 0.3 + rain_score * 0.4 + wind_score * 0.3) * 2 - 1

            result = {"temp": temp, "rain": rain, "wind": wind, "impact": impact}
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"获取天气失败: {e}")
            return None

    def get_injuries(self, team: str) -> Optional[List]:
        """
        从 API-Football 获取伤病数据 (需要API Key)

        返回:
            [{"player": str, "injury": str, "status": str}]
        """
        if "api_football" not in self.api_keys:
            return None

        # API-Football 需要付费Key，暂不实现
        return None

    def get_xg(self, team: str, league_id: str) -> Optional[Dict]:
        """
        从 understat 获取 xG 数据 (需要爬虫)

        返回:
            {"xg_for": float, "xg_against": float, "xg_diff": float}
        """
        # understat 没有官方API，需要爬虫
        return None

    def get_odds(self, league_id: str, match_id: str) -> Optional[Dict]:
        """
        从 API-Football 获取实时赔率 (需要API Key)

        返回:
            {"home": float, "draw": float, "away": float}
        """
        if "api_football" not in self.api_keys:
            return None

        # API-Football 需要付费Key，暂不实现
        return None

    def is_available(self, source: str) -> bool:
        """检查数据源是否可用"""
        if source == "football_data":
            return True  # 免费，总是可用
        elif source == "weather":
            return True  # Open-Meteo免费
        elif source == "api_football":
            return "api_football" in self.api_keys
        elif source == "understat":
            return False  # 需要爬虫
        elif source == "whoscored":
            return False  # 需要爬虫
        elif source == "transfermarkt":
            return False  # 需要爬虫
        return False

    def get_available_sources(self) -> List[str]:
        """获取所有可用的数据源"""
        available = []
        for source in ["football_data", "weather", "api_football", "understat", "whoscored", "transfermarkt"]:
            if self.is_available(source):
                available.append(source)
        return available
