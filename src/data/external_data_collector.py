"""
GTO-GameFlow v5.10.7 — 外部真实数据采集器

从 FBref、Understat 等真实数据源获取高维度数据。
直接通过 HTTP 下载 CSV 文件，无需浏览器/Selenium。

数据源:
  - FBref: xG, xGA, 射门位置, 传球数据 — 免费 CSV 导出
  - 球队身价: 预收集的静态数据
"""

import csv
import io
import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

# ============================================================
# FBref 联赛 → URL 路径
# ============================================================

# comp ID → league name
FBREF_COMP_IDS = {
    "premier_league": 9,
    "la_liga": 12,
    "bundesliga": 20,
    "serie_a": 11,
    "ligue_1": 13,
}

# CSV 文件名格式
FBREF_SEASON_FORMAT = {
    "premier_league": "Premier-League",
    "la_liga": "La-Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie-A",
    "ligue_1": "Ligue-1",
}

# CSV 队名 → 代码内队名映射
TEAM_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "Tottenham": "Tottenham",
    "Leeds United": "Leeds",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    "Brighton": "Brighton",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Luton Town": "Luton",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Cardiff City": "Cardiff",
    "Huddersfield": "Huddersfield",
    "West Brom": "West Brom",
    "Burnley": "Burnley",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Watford": "Watford",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Liverpool": "Liverpool",
    "Southampton": "Southampton",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Wigan Athletic": "Wigan",
    "Middlesbrough": "Middlesbrough",
    "Hull City": "Hull",
    "Reading": "Reading",
    "Blackpool": "Blackpool",
    "Blackburn": "Blackburn",
    "Bolton": "Bolton",
    "Portsmouth": "Portsmouth",
    "Birmingham City": "Birmingham",
    "Derby County": "Derby",
    "Ipswich Town": "Ipswich",
    "Charlton Ath": "Charlton",
    "Coventry City": "Coventry",
    "Oxford United": "Oxford",
    "Bradford City": "Bradford",
    "Barnsley": "Barnsley",
    "Oldham Athletic": "Oldham",
    "Sheffield Weds": "Sheffield Weds",
    "Swindon Town": "Swindon",
    "Wimbledon": "Wimbledon",
    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "Getafe": "Getafe",
    "Osasuna": "Osasuna",
    "Mallorca": "Mallorca",
    "Alaves": "Alaves",
    "Rayo Vallecano": "Vallecano",
    "Girona": "Girona",
    "Valladolid": "Valladolid",
    "Granada": "Granada",
    "Cadiz": "Cadiz",
    "Elche": "Elche",
    "Levante": "Levante",
    "Almeria": "Almeria",
    "Huesca": "Huesca",
    "Leganes": "Leganes",
    "Eibar": "Eibar",
    "La Coruna": "La Coruna",
    "Malaga": "Malaga",
    "Las Palmas": "Las Palmas",
    "Tenerife": "Tenerife",
    "Numancia": "Numancia",
    "Gijon": "Gijon",
    "Zaragoza": "Zaragoza",
    "Racing Sant": "Racing Sant",
    "Recreativo": "Recreativo",
    "Murcia": "Murcia",
    "Cordoba": "Cordoba",
    "Xerez": "Xerez",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Eint Frankfurt": "Ein Frankfurt",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "B. Leverkusen",
    "Wolfsburg": "Wolfsburg",
    "Hoffenheim": "Hoffenheim",
    "M'Gladbach": "M'gladbach",
    "Freiburg": "Freiburg",
    "Union Berlin": "Union Berlin",
    "Mainz 05": "Mainz",
    "FC Koln": "FC Koln",
    "Werder Bremen": "Werder Bremen",
    "Augsburg": "Augsburg",
    "Stuttgart": "Stuttgart",
    "Bochum": "Bochum",
    "Schalke 04": "Schalke 04",
    "Hertha BSC": "Hertha",
    "Darmstadt 98": "Darmstadt",
    "Heidenheim": "Heidenheim",
    "Arminia": "Arminia",
    "Greuther Furth": "Greuther Furth",
    "Paderborn 07": "Paderborn",
    "Hannover 96": "Hannover",
    "Nurnberg": "Nurnberg",
    "Dusseldorf": "Dusseldorf",
    "Hamburger SV": "Hamburg",
    "Ingolstadt 04": "Ingolstadt",
    "Braunschweig": "Braunschweig",
    "Kaiserslautern": "Kaiserslautern",
    # Serie A
    "Juventus": "Juventus",
    "Inter": "Inter",
    "AC Milan": "AC Milan",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Bologna": "Bologna",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Sampdoria": "Sampdoria",
    "Genoa": "Genoa",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Lecce": "Lecce",
    "Monza": "Monza",
    "Hellas Verona": "Verona",
    "Salernitana": "Salernitana",
    "Spezia": "Spezia",
    "Cremonese": "Cremonese",
    "Venezia": "Venezia",
    "Benevento": "Benevento",
    "Frosinone": "Frosinone",
    "Brescia": "Brescia",
    "Parma": "Parma",
    "SPAL": "SPAL",
    "Chievo": "Chievo",
    "Palermo": "Palermo",
    "Pescara": "Pescara",
    "Carpi": "Carpi",
    "Cesena": "Cesena",
    "Catania": "Catania",
    "Livorno": "Livorno",
    "Novara": "Novara",
    "Siena": "Siena",
    "Bari": "Bari",
    # Ligue 1
    "Paris S-G": "Paris SG",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Lens": "Lens",
    "Strasbourg": "Strasbourg",
    "Montpellier": "Montpellier",
    "Reims": "Reims",
    "Nantes": "Nantes",
    "Brest": "Brest",
    "Lorient": "Lorient",
    "Toulouse": "Toulouse",
    "Auxerre": "Auxerre",
    "Ajaccio": "Ajaccio",
    "Angers": "Angers",
    "Clermont Foot": "Clermont",
    "Troyes": "Troyes",
    "Saint-Etienne": "St Etienne",
    "Metz": "Metz",
    "Bordeaux": "Bordeaux",
    "Le Havre": "Le Havre",
    "Amiens": "Amiens",
    "Dijon": "Dijon",
    "Guingamp": "Guingamp",
    "Caen": "Caen",
    "Bastia": "Bastia",
    "Nancy": "Nancy",
    "Evian": "Evian",
    "GFC Ajaccio": "GFC Ajaccio",
    "Sochaux": "Sochaux",
    "Valenciennes": "Valenciennes",
    "Arles-Avignon": "Arles-Avignon",
}


class ExternalDataCollector:
    """
    统一的外部数据采集器。

    从 FBref 直接下载 CSV 获取 xG 数据，无需浏览器。
    缓存策略: 首次下载后缓存到本地 pickle 文件。
    """

    def __init__(self, cache_dir: str = "src/data/external_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._xg_data: Dict[str, Dict] = {}  # key → {team → [match_data]}
        self._loaded = False

    # ================================================================
    # 主入口: 获取 xG 数据
    # ================================================================

    def fetch_xg_data(
        self,
        leagues: List[str],
        seasons: List[str],
        force_refresh: bool = False,
    ) -> None:
        """
        从 FBref 下载 CSV 获取 xG 数据。

        seasons: ["2022-2023", "2023-2024", ...]
        """
        cache_file = self.cache_dir / "xg_data_fbref.pkl"

        if cache_file.exists() and not force_refresh:
            print(f"    加载 xG 缓存: {cache_file}")
            with open(cache_file, "rb") as f:
                self._xg_data = pickle.load(f)
            self._loaded = True
            return

        print("    从 FBref 下载 xG 数据 (CSV 模式)...")
        for league in leagues:
            comp_id = FBREF_COMP_IDS.get(league)
            if not comp_id:
                continue

            league_name = FBREF_SEASON_FORMAT.get(league, league)
            for season in seasons:
                key = f"{league}_{season}"
                if key not in self._xg_data:
                    self._xg_data[key] = {}

                year = season.split("-")[0]
                url = (f"https://fbref.com/en/comps/{comp_id}/{year}-{int(year)+1}/"
                       f"schedule/{year}-{int(year)+1}-{league_name}-Scores-and-Fixtures")

                try:
                    self._download_fbref_csv(url, key, league, season)
                    print(f"    {league} {season}: OK")
                except Exception as e:
                    print(f"    {league} {season}: 失败 - {e}")

                time.sleep(1.0)  # 速率限制

        # 保存缓存
        with open(cache_file, "wb") as f:
            pickle.dump(self._xg_data, f)
        print(f"    已保存 xG 缓存: {cache_file}")
        self._loaded = True

    def _download_fbref_csv(self, url: str, key: str, league: str, season: str) -> None:
        """下载 FBref 赛程页面并解析 xG 数据"""
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GTO-Bot/1.0)"})
        with urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")

        # 方案1: 查找嵌入的 match data JSON
        # FBref 赛程表在 HTML 中嵌入了一个 JSON 数据块
        xg_matches = self._parse_fbref_html(html, key, league, season)

        if not xg_matches:
            # 方案2: 尝试 CSV 导出
            csv_url = url.replace("Scores-and-Fixtures", "Scores-and-Fixtures.csv")
            try:
                req2 = Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req2, timeout=30) as resp2:
                    csv_text = resp2.read().decode("utf-8")
                xg_matches = self._parse_fbref_csv(csv_text, key)
            except Exception:
                pass

        if xg_matches:
            self._xg_data[key] = xg_matches

    def _parse_fbref_html(
        self, html: str, key: str, league: str, season: str,
    ) -> Dict:
        """从 FBref HTML 中提取 xG 数据"""
        result = {}

        # 查找表格中的 xG 列
        # FBref 的赛程表使用 <table id="sched_2022-2023_9_1">
        table_pattern = re.compile(r'<table[^>]*id="sched_[^"]*"[^>]*>(.*?)</table>', re.DOTALL)
        matches = table_pattern.findall(html)

        if not matches:
            return result

        for table_html in matches:
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL)
            for row in rows:
                # 检测是否包含 xG 数据
                if 'xG' not in row and 'xg' not in row:
                    continue

                # 提取球队名称
                team_cells = re.findall(
                    r'<td[^>]*data-stat="(home_team|away_team)"[^>]*>.*?<a[^>]*>(.*?)</a>',
                    row, re.DOTALL,
                )

                # 提取 xG 值
                xg_cells = re.findall(
                    r'<td[^>]*data-stat="(home_xg|away_xg)"[^>]*>\s*([\d.]+)\s*</td>',
                    row, re.DOTALL,
                )

                # 提取日期
                date_match = re.search(
                    r'<td[^>]*data-stat="date"[^>]*>.*?(\d{4}-\d{2}-\d{2}).*?</td>',
                    row, re.DOTALL,
                )
                match_date = date_match.group(1) if date_match else ""

                # 提取比分
                score_match = re.search(
                    r'<td[^>]*data-stat="score"[^>]*>.*?(\d+)&ndash;(\d+).*?</td>',
                    row, re.DOTALL,
                )

                home_team = ""
                away_team = ""
                for stat, name in team_cells:
                    if stat == "home_team":
                        home_team = name.strip()
                    elif stat == "away_team":
                        away_team = name.strip()

                home_xg = None
                away_xg = None
                for stat, val in xg_cells:
                    try:
                        if stat == "home_xg":
                            home_xg = float(val)
                        elif stat == "away_xg":
                            away_xg = float(val)
                    except ValueError:
                        pass

                if home_team and away_team and home_xg is not None and away_xg is not None:
                    # 转换为代码内队名
                    home_csv = TEAM_NAME_MAP.get(home_team, home_team)
                    away_csv = TEAM_NAME_MAP.get(away_team, away_team)

                    match_data = {
                        "home_team": home_csv,
                        "away_team": away_csv,
                        "home_xg": home_xg,
                        "away_xg": away_xg,
                        "date": match_date,
                    }

                    if home_csv not in result:
                        result[home_csv] = []
                    result[home_csv].append(match_data)

                    if away_csv not in result:
                        result[away_csv] = []
                    # 为客队也添加 (交换 xG)
                    result[away_csv].append({
                        "home_team": away_csv,
                        "away_team": home_csv,
                        "home_xg": away_xg,
                        "away_xg": home_xg,
                        "date": match_date,
                    })

        return result

    def _parse_fbref_csv(self, csv_text: str, key: str) -> Dict:
        """从 FBref CSV 导出中解析 xG 数据"""
        result = {}
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            home = row.get("Home", "").strip()
            away = row.get("Away", "").strip()
            home_xg_str = row.get("xG", "") or row.get("xG.1", "")
            away_xg_str = row.get("xG.2", "") or row.get("xG.1", "")
            date_str = row.get("Date", "")

            if not home or not away:
                continue

            try:
                home_xg = float(home_xg_str) if home_xg_str else None
                away_xg = float(away_xg_str) if away_xg_str else None
            except ValueError:
                continue

            if home_xg is not None and away_xg is not None:
                home_csv = TEAM_NAME_MAP.get(home, home)
                away_csv = TEAM_NAME_MAP.get(away, away)

                match_data = {
                    "home_team": home_csv,
                    "away_team": away_csv,
                    "home_xg": home_xg,
                    "away_xg": away_xg,
                    "date": date_str,
                }

                if home_csv not in result:
                    result[home_csv] = []
                result[home_csv].append(match_data)

                if away_csv not in result:
                    result[away_csv] = []
                result[away_csv].append({
                    "home_team": away_csv,
                    "away_team": home_csv,
                    "home_xg": away_xg,
                    "away_xg": home_xg,
                    "date": date_str,
                })

        return result

    # ================================================================
    # 查询接口
    # ================================================================

    def get_team_xg(
        self,
        league: str,
        season: str,
        csv_team_name: str,
        before_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """获取球队 xG 数据"""
        key = f"{league}_{season}"
        if key not in self._xg_data:
            return self._empty_xg()

        team_matches = self._xg_data[key].get(csv_team_name, [])
        if not team_matches:
            return self._empty_xg()

        xg_for = 0.0
        xg_against = 0.0
        count = 0

        for m in team_matches:
            if before_date and m.get("date"):
                try:
                    m_date = datetime.strptime(str(m["date"])[:10], "%Y-%m-%d")
                    if m_date >= before_date:
                        continue
                except ValueError:
                    pass

            if m["home_team"] == csv_team_name:
                xg_for += m["home_xg"]
                xg_against += m["away_xg"]
            else:
                xg_for += m["away_xg"]
                xg_against += m["home_xg"]
            count += 1

        if count > 0:
            return {
                "xg_for": xg_for,
                "xg_against": xg_against,
                "avg_xg_for": xg_for / count,
                "avg_xg_against": xg_against / count,
                "matches": count,
            }
        return self._empty_xg()

    def get_xg_diff(
        self,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        before_date: Optional[datetime] = None,
    ) -> float:
        """获取主客队 xG 差值"""
        home_xg = self.get_team_xg(league, season, home_team, before_date)
        away_xg = self.get_team_xg(league, season, away_team, before_date)
        home_avg = home_xg.get("avg_xg_for", 0.0) - home_xg.get("avg_xg_against", 0.0)
        away_avg = away_xg.get("avg_xg_for", 0.0) - away_xg.get("avg_xg_against", 0.0)
        return home_avg - away_avg

    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _empty_xg() -> Dict[str, float]:
        return {"xg_for": 0.0, "xg_against": 0.0, "avg_xg_for": 0.0, "avg_xg_against": 0.0, "matches": 0}


# ============================================================
# 便捷函数
# ============================================================

def create_external_collector(
    cache_dir: str = "src/data/external_cache",
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> ExternalDataCollector:
    """创建并初始化外部数据采集器"""
    if leagues is None:
        leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    if seasons is None:
        seasons = [
            "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019",
            "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024",
        ]

    collector = ExternalDataCollector(cache_dir=cache_dir)
    collector.fetch_xg_data(leagues=leagues, seasons=seasons, force_refresh=force_refresh)
    return collector


if __name__ == "__main__":
    collector = create_external_collector(
        leagues=["premier_league"],
        seasons=["2022-2023"],
        force_refresh=True,
    )
    diff = collector.get_xg_diff(
        "premier_league", "2022-2023",
        "Arsenal", "Tottenham",
        before_date=datetime(2022, 10, 1),
    )
    print(f"Arsenal vs Tottenham xG diff: {diff:.4f}")