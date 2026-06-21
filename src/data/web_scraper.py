"""
GTO v6.0 — 网页数据抓取器

使用 web_fetch 从公开网页抓取足球数据，无需API Key。

数据源:
1. 积分榜 — ESPN/BBC Sport
2. 伤病 — Transfermarkt/Physio Room
3. xG — FBref
4. 教练更替 — 维基百科/新闻
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'scraped')


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_data(filename: str, data: Any):
    ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_data(filename: str) -> Optional[Any]:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def parse_standings_from_html(html: str, league: str) -> Dict[str, Dict]:
    """
    从HTML解析积分榜。

    支持ESPN/BBC Sport等网站格式。
    返回: {team_name: {position, points, played, won, draw, lost, gf, ga, gd}}
    """
    standings = {}

    # 尝试解析表格格式
    # 匹配: 排位 球队 场次 胜 平 负 进球 失球 净胜球 积分
    pattern = r'(\d+)\s+([A-Za-z\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d]+)\s+(\d+)'
    matches = re.findall(pattern, html)

    for m in matches:
        pos, team, played, won, draw, lost, gf, ga, gd, pts = m
        team = team.strip()
        if team:
            standings[team] = {
                "position": int(pos),
                "points": int(pts),
                "played": int(played),
                "won": int(won),
                "draw": int(draw),
                "lost": int(lost),
                "goals_for": int(gf),
                "goals_against": int(ga),
                "goal_difference": int(gd),
            }

    return standings


def scrape_standings(league: str, season: str = "2023-24") -> Optional[Dict]:
    """
    抓取积分榜数据。

    使用 web_fetch 从公开网站获取。
    """
    cache_file = f"standings_{league}_{season}.json"
    cached = load_data(cache_file)
    if cached:
        return cached

    # ESPN积分榜URL
    urls = {
        "premier_league": "https://www.espn.com/soccer/standings/_/league/eng.1",
        "la_liga": "https://www.espn.com/soccer/standings/_/league/esp.1",
        "bundesliga": "https://www.espn.com/soccer/standings/_/league/ger.1",
        "serie_a": "https://www.espn.com/soccer/standings/_/league/ita.1",
        "ligue_1": "https://www.espn.com/soccer/standings/_/league/fra.1",
    }

    url = urls.get(league)
    if not url:
        return None

    # 注意: web_fetch 是 OpenClaw 工具，需要通过 exec 调用
    # 这里返回URL，由调用方使用 web_fetch 获取
    return {"url": url, "needs_fetch": True}


def parse_injuries_from_text(text: str, team: str) -> List[Dict]:
    """
    从文本解析伤病信息。

    返回: [{player, injury, status, expected_return}]
    """
    injuries = []

    # 匹配常见伤病格式
    # "Player Name - Knee Injury - Out for 3 weeks"
    pattern = r'([A-Za-z\s]+?)\s*[-–]\s*([A-Za-z\s]+?)\s*[-–]\s*(Out|Doubtful|Day-to-Day)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    for m in matches:
        player, injury, status = m
        injuries.append({
            "player": player.strip(),
            "injury": injury.strip(),
            "status": status.strip(),
        })

    return injuries


def parse_xg_from_text(text: str) -> Optional[Dict]:
    """
    从文本解析xG数据。

    返回: {xg_for, xg_against, xg_diff}
    """
    # 匹配xG数据
    pattern = r'xG[:\s]+([0-9.]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        xg = float(match.group(1))
        return {"xg_for": xg, "xg_against": 0, "xg_diff": xg}
    return None


def scrape_team_data(team: str, league: str) -> Dict:
    """
    抓取球队综合数据。

    返回包含积分榜、伤病、xG等信息的字典。
    """
    result = {
        "team": team,
        "league": league,
        "standings": None,
        "injuries": [],
        "xg": None,
        "coach_change": False,
        "last_updated": datetime.now().isoformat(),
    }

    # 积分榜
    standings = scrape_standings(league)
    if standings and not standings.get("needs_fetch"):
        result["standings"] = standings

    return result


def get_scraped_data(league: str, team: str) -> Dict:
    """
    获取已抓取的数据。

    如果没有缓存数据，返回默认值。
    """
    cache_file = f"team_data_{league}_{team.replace(' ', '_')}.json"
    cached = load_data(cache_file)
    if cached:
        return cached

    return {
        "team": team,
        "league": league,
        "standings": None,
        "injuries": [],
        "xg": None,
        "coach_change": False,
        "last_updated": None,
    }
