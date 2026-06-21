"""
GTO-GameFlow v5.10.5 — xG 数据获取器 [已废弃]

⚠️ DEPRECATED: Understat 已改为动态加载 (JavaScript), HTTP 解析失效。
实际 xG 代理已移至 enhanced_data_provider._compute_shot_quality_proxy(),
基于 CSV 射门数据 (HS/AS/HST/AST) 计算。

本文件保留用于未来可能的 API 集成参考。
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

# ============================================================
# 联赛 → Understat 代码映射
# ============================================================

LEAGUE_TO_UNDERSTAT = {
    "premier_league": "EPL",
    "la_liga": "La_liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie_A",
    "ligue_1": "Ligue_1",
}

# 赛季 → Understat 年份
def season_to_year(season: str) -> str:
    """'2022-23' → '2022'"""
    return season[:4]


# ============================================================
# xG 数据获取
# ============================================================

class XGFetcher:
    """
    从 Understat 获取 xG 数据。

    用法:
        fetcher = XGFetcher(cache_dir="src/data/xg_cache")
        fetcher.download_all_leagues(["premier_league"], ["2014-15", ..., "2023-24"])
        xg_data = fetcher.get_xg_for_match(league, home, away, date)
    """

    def __init__(self, cache_dir: str = "src/data/xg_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._team_data: Dict[str, Dict] = {}  # league_season → {team: [matches]}
        self._loaded = False

    def download_all_leagues(
        self,
        leagues: List[str],
        seasons: List[str],
        delay: float = 0.5,
    ) -> None:
        """下载所有联赛-赛季的 xG 数据"""
        for league in leagues:
            for season in seasons:
                self._download_league_season(league, season, delay)
        self._load_cache()

    def _download_league_season(self, league: str, season: str, delay: float) -> None:
        """下载单个联赛-赛季的 xG 数据"""
        league_code = LEAGUE_TO_UNDERSTAT.get(league)
        if not league_code:
            print(f"    跳过 {league} (无 Understat 映射)")
            return

        year = season_to_year(season)
        cache_file = os.path.join(self.cache_dir, f"{league}_{season}.json")

        if os.path.exists(cache_file):
            print(f"    xG 缓存已存在: {league} {season}")
            return

        url = f"https://understat.com/league/{league_code}/{year}"
        print(f"    下载 xG: {url}")

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")

            # 提取 JSON 数据 (Understat 在 HTML 中嵌入 JSON)
            # teamsData 包含所有球队的比赛数据
            pattern = r"var teamsData\s*=\s*JSON\.parse\('(.+?)'\)"
            match = re.search(pattern, html)
            if not match:
                print(f"    未找到 teamsData: {league} {season}")
                return

            json_str = match.group(1)
            # 解码 HTML 实体
            json_str = json_str.replace("\\'", "'")
            json_str = json_str.replace('\\"', '"')
            json_str = json_str.replace("\\\\", "\\")
            json_str = json_str.encode().decode("unicode_escape")
            data = json.loads(json_str)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"    已保存: {len(data)} 支球队, {cache_file}")

        except Exception as e:
            print(f"    下载失败 ({league} {season}): {e}")

        time.sleep(delay)

    def _load_cache(self) -> None:
        """加载缓存到内存"""
        if self._loaded:
            return

        for filename in os.listdir(self.cache_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.cache_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # filename: premier_league_2022-23.json
                key = filename.replace(".json", "")
                self._team_data[key] = data
            except Exception as e:
                print(f"    加载缓存失败: {filename}: {e}")

        self._loaded = True
        print(f"    xG 缓存已加载: {len(self._team_data)} 个联赛-赛季")

    def get_team_xg(
        self,
        league: str,
        season: str,
        team: str,
        before_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        获取球队在指定日期之前的 xG 数据。

        返回: {xg_for, xg_against, avg_xg_for, avg_xg_against, npxG_for, npxG_against}
        """
        key = f"{league}_{season}"
        if key not in self._team_data:
            return {"xg_for": 0.0, "xg_against": 0.0, "avg_xg_for": 0.0, "avg_xg_against": 0.0}

        data = self._team_data[key]
        xg_for = 0.0
        xg_against = 0.0
        count = 0

        for team_id, team_data in data.items():
            if not isinstance(team_data, dict):
                continue
            history = team_data.get("history", [])
            if not history:
                continue

            for match in history:
                # 检查球队名称匹配
                h_team = match.get("h_title", "")
                a_team = match.get("a_title", "")
                if team.lower() not in (h_team.lower(), a_team.lower()):
                    continue

                # 检查日期
                match_date_str = match.get("date", "")
                if match_date_str and before_date:
                    try:
                        match_date = datetime.strptime(match_date_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            match_date = datetime.strptime(match_date_str, "%Y-%m-%d")
                        except ValueError:
                            match_date = None
                    if match_date and match_date >= before_date:
                        continue

                if team.lower() == h_team.lower():
                    xg_for += float(match.get("xG", 0) or 0)
                    xg_against += float(match.get("xGA", 0) or 0)
                else:
                    xg_for += float(match.get("xGA", 0) or 0)
                    xg_against += float(match.get("xG", 0) or 0)
                count += 1

        if count > 0:
            return {
                "xg_for": xg_for,
                "xg_against": xg_against,
                "avg_xg_for": xg_for / count,
                "avg_xg_against": xg_against / count,
                "matches": count,
            }
        return {"xg_for": 0.0, "xg_against": 0.0, "avg_xg_for": 0.0, "avg_xg_against": 0.0}

    def get_xg_diff(
        self,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        before_date: Optional[datetime] = None,
    ) -> float:
        """
        获取主客队 xG 差值。

        返回: home_avg_xg_diff - away_avg_xg_diff
        (正值 = 主队 xG 优势)
        """
        home_xg = self.get_team_xg(league, season, home_team, before_date)
        away_xg = self.get_team_xg(league, season, away_team, before_date)

        home_avg = home_xg.get("avg_xg_for", 0.0) - home_xg.get("avg_xg_against", 0.0)
        away_avg = away_xg.get("avg_xg_for", 0.0) - away_xg.get("avg_xg_against", 0.0)

        return home_avg - away_avg

    def get_match_xg(
        self,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        match_date: datetime,
    ) -> Optional[Dict[str, float]]:
        """
        获取特定比赛的实际 xG 数据 (用于回测验证)。

        返回: {home_xg, away_xg, home_npxg, away_npxg, home_deep, away_deep}
        如果找不到则返回 None。
        """
        key = f"{league}_{season}"
        if key not in self._team_data:
            return None

        data = self._team_data[key]
        for team_id, team_data in data.items():
            if not isinstance(team_data, dict):
                continue
            history = team_data.get("history", [])
            for match in history:
                h_team = match.get("h_title", "")
                a_team = match.get("a_title", "")
                if home_team.lower() in (h_team.lower(), "") and away_team.lower() in (a_team.lower(), ""):
                    match_date_str = match.get("date", "")
                    if match_date_str:
                        try:
                            md = datetime.strptime(match_date_str[:10], "%Y-%m-%d")
                            if md.date() == match_date.date():
                                return {
                                    "home_xg": float(match.get("xG", 0) or 0),
                                    "away_xg": float(match.get("xGA", 0) or 0),
                                    "home_deep": float(match.get("deep", 0) or 0),
                                    "away_deep": float(match.get("deep_allowed", 0) or 0),
                                }
                        except ValueError:
                            pass
        return None


# ============================================================
# 便捷函数
# ============================================================

def create_xg_fetcher(
    cache_dir: str = "src/data/xg_cache",
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> XGFetcher:
    """
    创建并初始化 xG 数据获取器。

    如果缓存不存在，自动下载数据。
    """
    if leagues is None:
        leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    if seasons is None:
        seasons = [
            "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
            "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
        ]

    fetcher = XGFetcher(cache_dir=cache_dir)

    # 检查是否需要下载
    need_download = False
    for league in leagues:
        for season in seasons:
            if not os.path.exists(os.path.join(cache_dir, f"{league}_{season}.json")):
                need_download = True
                break

    if need_download:
        print("    开始下载 xG 数据 (首次运行可能需要几分钟)...")
        fetcher.download_all_leagues(leagues, seasons, delay=0.3)
    else:
        print("    xG 缓存完整，无需下载")

    fetcher._load_cache()
    return fetcher


if __name__ == "__main__":
    # 测试
    fetcher = create_xg_fetcher()
    diff = fetcher.get_xg_diff(
        "premier_league", "2022-23",
        "Arsenal", "Tottenham",
        before_date=datetime(2022, 10, 1),
    )
    print(f"Arsenal vs Tottenham xG diff (before Oct 1 2022): {diff:.4f}")