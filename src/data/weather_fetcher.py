"""
GTO v6.0 — 天气数据获取

从 Open-Meteo API 获取比赛日天气。
免费，无需API Key。
"""

from __future__ import annotations
import json
import logging
import math
import urllib.request
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 主要城市坐标
CITY_COORDS = {
    # 英超
    "Manchester": {"lat": 53.48, "lon": -2.24},
    "London": {"lat": 51.51, "lon": -0.13},
    "Liverpool": {"lat": 53.41, "lon": -2.98},
    "Birmingham": {"lat": 52.49, "lon": -1.89},
    "Newcastle": {"lat": 54.98, "lon": -1.62},
    "Leeds": {"lat": 53.80, "lon": -1.55},
    "Leicester": {"lat": 52.64, "lon": -1.13},
    "Brighton": {"lat": 50.83, "lon": -0.14},
    "Southampton": {"lat": 50.90, "lon": -1.40},
    "Wolverhampton": {"lat": 52.59, "lon": -2.13},
    "Burnley": {"lat": 53.79, "lon": -2.23},
    "Sheffield": {"lat": 53.38, "lon": -1.47},
    "Nottingham": {"lat": 52.95, "lon": -1.15},
    "Bournemouth": {"lat": 50.72, "lon": -1.88},
    "Luton": {"lat": 51.88, "lon": -0.42},
    "Fulham": {"lat": 51.47, "lon": -0.22},
    "Crystal Palace": {"lat": 51.40, "lon": -0.08},
    "West Ham": {"lat": 51.54, "lon": 0.02},
    "Tottenham": {"lat": 51.60, "lon": -0.07},
    "Arsenal": {"lat": 51.55, "lon": -0.11},
    "Chelsea": {"lat": 51.48, "lon": -0.19},
    "Brentford": {"lat": 51.49, "lon": -0.29},
    "Aston Villa": {"lat": 52.51, "lon": -1.88},
    "Everton": {"lat": 53.44, "lon": -2.97},
    
    # 西甲
    "Madrid": {"lat": 40.42, "lon": -3.70},
    "Barcelona": {"lat": 41.39, "lon": 2.17},
    "Seville": {"lat": 37.39, "lon": -5.98},
    "Valencia": {"lat": 39.47, "lon": -0.38},
    "Bilbao": {"lat": 43.26, "lon": -2.93},
    "San Sebastian": {"lat": 43.32, "lon": -1.98},
    "Villarreal": {"lat": 39.94, "lon": -0.10},
    "Vigo": {"lat": 42.23, "lon": -8.71},
    "Granada": {"lat": 37.18, "lon": -3.60},
    "Malaga": {"lat": 36.72, "lon": -4.42},
    
    # 德甲
    "Munich": {"lat": 48.14, "lon": 11.58},
    "Dortmund": {"lat": 51.51, "lon": 7.47},
    "Berlin": {"lat": 52.52, "lon": 13.41},
    "Hamburg": {"lat": 53.55, "lon": 9.99},
    "Cologne": {"lat": 50.94, "lon": 6.96},
    "Frankfurt": {"lat": 50.11, "lon": 8.68},
    "Stuttgart": {"lat": 48.78, "lon": 9.18},
    "Leipzig": {"lat": 51.34, "lon": 12.37},
    "Leverkusen": {"lat": 51.05, "lon": 6.98},
    "Gladbach": {"lat": 51.19, "lon": 6.44},
    "Freiburg": {"lat": 47.99, "lon": 7.85},
    "Wolfsburg": {"lat": 52.42, "lon": 10.79},
    "Bremen": {"lat": 53.08, "lon": 8.81},
    "Augsburg": {"lat": 48.37, "lon": 10.90},
    "Hoffenheim": {"lat": 49.28, "lon": 8.89},
    "Mainz": {"lat": 50.00, "lon": 8.27},
    "Heidenheim": {"lat": 48.68, "lon": 10.15},
    "Darmstadt": {"lat": 49.87, "lon": 8.65},
    "Bochum": {"lat": 51.48, "lon": 7.22},
    "Berlin Union": {"lat": 52.49, "lon": 13.45},
    
    # 意甲
    "Milan": {"lat": 45.46, "lon": 9.19},
    "Rome": {"lat": 41.90, "lon": 12.50},
    "Turin": {"lat": 45.07, "lon": 7.69},
    "Naples": {"lat": 40.85, "lon": 14.27},
    "Florence": {"lat": 43.77, "lon": 11.25},
    "Bologna": {"lat": 44.49, "lon": 11.34},
    "Genoa": {"lat": 44.41, "lon": 8.93},
    "Verona": {"lat": 45.44, "lon": 10.99},
    "Bergamo": {"lat": 45.70, "lon": 9.67},
    "Cagliari": {"lat": 39.22, "lon": 9.12},
    "Lecce": {"lat": 40.35, "lon": 18.17},
    "Udine": {"lat": 46.07, "lon": 13.24},
    "Empoli": {"lat": 43.72, "lon": 10.95},
    "Sassuolo": {"lat": 44.54, "lon": 10.79},
    "Frosinone": {"lat": 41.64, "lon": 13.35},
    "Salerno": {"lat": 40.68, "lon": 14.77},
    "Monza": {"lat": 45.58, "lon": 9.27},
    
    # 法甲
    "Paris": {"lat": 48.86, "lon": 2.35},
    "Marseille": {"lat": 43.30, "lon": 5.37},
    "Lyon": {"lat": 45.76, "lon": 4.84},
    "Monaco": {"lat": 43.73, "lon": 7.42},
    "Nice": {"lat": 43.71, "lon": 7.26},
    "Lille": {"lat": 50.63, "lon": 3.06},
    "Rennes": {"lat": 48.11, "lon": -1.68},
    "Strasbourg": {"lat": 48.57, "lon": 7.75},
    "Bordeaux": {"lat": 44.84, "lon": -0.58},
    "Toulouse": {"lat": 43.60, "lon": 1.44},
    "Montpellier": {"lat": 43.61, "lon": 3.88},
    "Lens": {"lat": 50.43, "lon": 2.83},
    "Reims": {"lat": 49.25, "lon": 3.52},
    "Nantes": {"lat": 47.22, "lon": -1.55},
    "Brest": {"lat": 48.39, "lon": -4.49},
    "Le Havre": {"lat": 49.49, "lon": 0.11},
    "Clermont": {"lat": 45.78, "lon": 3.08},
    "Lorient": {"lat": 47.75, "lon": -3.37},
    "Metz": {"lat": 49.12, "lon": 6.18},
    "Angers": {"lat": 47.47, "lon": -0.55},
}


class WeatherFetcher:
    """天气数据获取器"""
    
    def __init__(self):
        self._cache = {}
    
    def get_weather(self, city: str, date: str = None) -> Optional[Dict]:
        """
        获取天气数据。
        
        参数:
            city: 城市名
            date: 日期 (YYYY-MM-DD)，None=今天
        
        返回:
            {"temperature": float, "humidity": float, "wind_speed": float, 
             "precipitation": float, "weather_desc": str, "weather_impact": float}
        """
        cache_key = f"{city}_{date or 'today'}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        coords = CITY_COORDS.get(city)
        if not coords:
            # 尝试模糊匹配
            for c, coord in CITY_COORDS.items():
                if c.lower() in city.lower() or city.lower() in c.lower():
                    coords = coord
                    break
        
        if not coords:
            logger.warning(f"未找到城市坐标: {city}")
            return None
        
        try:
            url = (f"https://api.open-meteo.com/v1/forecast?"
                   f"latitude={coords['lat']}&longitude={coords['lon']}"
                   f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code"
                   f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
                   f"&timezone=auto")
            
            if date:
                url += f"&start_date={date}&end_date={date}"
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            # 解析数据
            current = data.get("current", {})
            daily = data.get("daily", {})
            
            temp = current.get("temperature_2m", 15)
            humidity = current.get("relative_humidity_2m", 50)
            wind = current.get("wind_speed_10m", 0)
            precip = current.get("precipitation", 0)
            weather_code = current.get("weather_code", 0)
            
            # 天气描述
            weather_desc = self._get_weather_desc(weather_code)
            
            # 计算天气影响因子 (-1 到 1)
            weather_impact = self._calculate_impact(temp, humidity, wind, precip)
            
            result = {
                "temperature": temp,
                "humidity": humidity,
                "wind_speed": wind,
                "precipitation": precip,
                "weather_code": weather_code,
                "weather_desc": weather_desc,
                "weather_impact": weather_impact,
            }
            
            self._cache[cache_key] = result
            return result
        
        except Exception as e:
            logger.warning(f"获取天气失败 ({city}): {e}")
            return None
    
    def _get_weather_desc(self, code: int) -> str:
        """天气代码转描述"""
        desc_map = {
            0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
            45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "中毛毛雨",
            55: "大毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
            80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
            85: "小阵雪", 86: "大阵雪", 95: "雷暴",
            96: "雷暴+小冰雹", 99: "雷暴+大冰雹",
        }
        return desc_map.get(code, "未知")
    
    def _calculate_impact(self, temp: float, humidity: float, wind: float, precip: float) -> float:
        """计算天气影响因子 (-1 到 1)"""
        # 温度影响 (最佳: 15-25°C)
        if 15 <= temp <= 25:
            temp_score = 1.0
        elif 5 <= temp < 15 or 25 < temp <= 35:
            temp_score = 0.5
        else:
            temp_score = -0.5
        
        # 降水影响
        if precip == 0:
            precip_score = 1.0
        elif precip < 2:
            precip_score = 0.5
        elif precip < 10:
            precip_score = -0.5
        else:
            precip_score = -1.0
        
        # 风速影响
        if wind < 10:
            wind_score = 1.0
        elif wind < 20:
            wind_score = 0.5
        elif wind < 30:
            wind_score = -0.5
        else:
            wind_score = -1.0
        
        # 综合评分
        impact = (temp_score * 0.3 + precip_score * 0.4 + wind_score * 0.3)
        return round(impact, 2)


# 全局实例
_fetcher = None


def get_weather_fetcher() -> WeatherFetcher:
    """获取全局天气获取器"""
    global _fetcher
    if _fetcher is None:
        _fetcher = WeatherFetcher()
    return _fetcher
