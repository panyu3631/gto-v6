"""
GTO-GameFlow v5.9.2 — 真实历史赔率数据加载器

数据源: football-data.co.uk (免费, 覆盖 5 大联赛 + 多赛季)
提供 Bet365 / Pinnacle / 市场平均 赔率，含 1X2 / 亚盘 / 大小球
"""
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "historical_odds"

# 联赛 → football-data.co.uk 代码
LEAGUE_CODES = {
    "premier_league": "E0",
    "la_liga": "SP1",
    "bundesliga": "D1",
    "serie_a": "I1",
    "ligue_1": "F1",
}

# 赛季代码映射: "2023/24" → "2324"
def season_code(season: str) -> str:
    y1, y2 = season.split("/")
    return y1[-2:] + y2[-2:]

# 数据 URL 模板
# 2023/24 英超: https://www.football-data.co.uk/mmz4281/2324/E0.csv
# 2014/15 英超: https://www.football-data.co.uk/mmz4281/1415/E0.csv
DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"

# 英文队名 → 中文队名映射
TEAM_NAME_MAP: Dict[str, Dict[str, str]] = {
    "premier_league": {
        "Man City": "曼城", "Manchester City": "曼城",
        "Arsenal": "阿森纳",
        "Liverpool": "利物浦",
        "Chelsea": "切尔西",
        "Tottenham": "热刺", "Spurs": "热刺",
        "Man United": "曼联", "Manchester Utd": "曼联",
        "Newcastle": "纽卡斯尔联", "Newcastle Utd": "纽卡斯尔联",
        "Aston Villa": "阿斯顿维拉",
        "West Ham": "西汉姆联", "West Ham Utd": "西汉姆联",
        "Brighton": "布莱顿",
        "Everton": "埃弗顿",
        "Wolves": "狼队", "Wolverhampton": "狼队",
        "Crystal Palace": "水晶宫",
        "Fulham": "富勒姆",
        "Bournemouth": "伯恩茅斯",
        "Brentford": "布伦特福德",
        "Nott'm Forest": "诺丁汉森林", "Nottingham Forest": "诺丁汉森林",
        "Leicester": "莱斯特城",
        "Southampton": "南安普顿",
        "Leeds": "利兹联", "Leeds Utd": "利兹联",
        "Burnley": "伯恩利",
        "Watford": "沃特福德",
        "Norwich": "诺维奇",
        "West Brom": "西布朗", "West Bromwich": "西布朗",
        "Sheffield United": "谢菲尔德联", "Sheffield Utd": "谢菲尔德联",
        "Middlesbrough": "米德尔斯堡",
        "Swansea": "斯旺西",
        "Stoke": "斯托克城",
        "Huddersfield": "哈德斯菲尔德",
        "Cardiff": "加的夫城",
        "Luton": "卢顿",
        "Ipswich": "伊普斯维奇",
        "Sunderland": "桑德兰",
        "Hull": "赫尔城",
        "QPR": "女王公园巡游者",
        "Reading": "雷丁",
        "Wigan": "维冈竞技",
        "Blackburn": "布莱克本",
        "Bolton": "博尔顿",
        "Blackpool": "布莱克浦",
        "Portsmouth": "朴茨茅斯",
        "Birmingham": "伯明翰",
    },
    "la_liga": {
        "Real Madrid": "皇家马德里",
        "Barcelona": "巴塞罗那",
        "Atletico Madrid": "马德里竞技", "Ath Madrid": "马德里竞技",
        "Sevilla": "塞维利亚",
        "Real Sociedad": "皇家社会",
        "Villarreal": "比利亚雷亚尔",
        "Ath Bilbao": "毕尔巴鄂竞技", "Athletic Bilbao": "毕尔巴鄂竞技",
        "Betis": "皇家贝蒂斯", "Real Betis": "皇家贝蒂斯",
        "Valencia": "瓦伦西亚",
        "Girona": "赫罗纳",
        "Osasuna": "奥萨苏纳",
        "Celta": "塞尔塔", "Celta Vigo": "塞尔塔",
        "Getafe": "赫塔费",
        "Mallorca": "马略卡",
        "Vallecano": "巴列卡诺", "Rayo Vallecano": "巴列卡诺",
        "Alaves": "阿拉维斯",
        "Espanol": "西班牙人", "Espanyol": "西班牙人",
        "Leganes": "莱加内斯",
        "Valladolid": "巴拉多利德",
        "Granada": "格拉纳达",
        "Eibar": "埃瓦尔",
        "Levante": "莱万特",
        "La Coruna": "拉科鲁尼亚", "Dep La Coruna": "拉科鲁尼亚",
        "Malaga": "马拉加",
        "Las Palmas": "拉斯帕尔马斯",
        "Cadiz": "加的斯",
        "Almeria": "阿尔梅里亚",
        "Elche": "埃尔切",
        "Huesca": "韦斯卡",
        "Gijon": "希洪竞技", "Sp Gijon": "希洪竞技",
        "Cordoba": "科尔多瓦",
        "Tenerife": "特内里费",
    },
    "bundesliga": {
        "Bayern Munich": "拜仁慕尼黑", "FC Bayern": "拜仁慕尼黑",
        "Dortmund": "多特蒙德", "Borussia Dortmund": "多特蒙德",
        "RB Leipzig": "莱比锡红牛",
        "Leverkusen": "勒沃库森", "Bayer Leverkusen": "勒沃库森",
        "Eintracht Frankfurt": "法兰克福",
        "Stuttgart": "斯图加特", "VfB Stuttgart": "斯图加特",
        "Wolfsburg": "沃尔夫斯堡",
        "Freiburg": "弗赖堡",
        "Hoffenheim": "霍芬海姆",
        "M'gladbach": "门兴格拉德巴赫", "Borussia M.Gladbach": "门兴格拉德巴赫",
        "Union Berlin": "柏林联合",
        "Werder Bremen": "云达不莱梅",
        "Augsburg": "奥格斯堡",
        "Mainz": "美因茨", "Mainz 05": "美因茨",
        "Bochum": "波鸿",
        "Schalke 04": "沙尔克04",
        "Hamburg": "汉堡",
        "FC Koln": "科隆", "Cologne": "科隆",
        "Hertha Berlin": "柏林赫塔", "Hertha": "柏林赫塔",
        "Dusseldorf": "杜塞尔多夫", "Fortuna Dusseldorf": "杜塞尔多夫",
        "Paderborn": "帕德博恩",
        "Nurnberg": "纽伦堡",
        "Ingolstadt": "因戈尔施塔特",
        "Darmstadt": "达姆施塔特",
        "Heidenheim": "海登海姆",
        "Greuther Furth": "菲尔特",
        "Hannover": "汉诺威96",
        "Bielefeld": "比勒费尔德",
        "St Pauli": "圣保利",
        "Karlsruhe": "卡尔斯鲁厄",
    },
    "serie_a": {
        "Juventus": "尤文图斯",
        "Inter": "国际米兰", "Inter Milan": "国际米兰",
        "AC Milan": "AC米兰", "Milan": "AC米兰",
        "Napoli": "那不勒斯",
        "Roma": "罗马",
        "Atalanta": "亚特兰大",
        "Lazio": "拉齐奥",
        "Fiorentina": "佛罗伦萨",
        "Bologna": "博洛尼亚",
        "Torino": "都灵",
        "Udinese": "乌迪内斯",
        "Genoa": "热那亚",
        "Sampdoria": "桑普多利亚",
        "Sassuolo": "萨索洛",
        "Cagliari": "卡利亚里",
        "Verona": "维罗纳", "Hellas Verona": "维罗纳",
        "Empoli": "恩波利",
        "Lecce": "莱切",
        "Monza": "蒙扎",
        "Frosinone": "弗罗西诺内",
        "Benevento": "贝内文托",
        "Spezia": "斯佩齐亚",
        "Venezia": "威尼斯",
        "Parma": "帕尔马",
        "Crotone": "克罗托内",
        "Brescia": "布雷西亚",
        "Salernitana": "萨勒尼塔纳",
        "Chievo": "切沃",
        "Palermo": "巴勒莫",
        "Pescara": "佩斯卡拉",
        "Spal": "斯帕尔",
        "Como": "科莫",
    },
    "ligue_1": {
        "Paris SG": "巴黎圣日耳曼", "PSG": "巴黎圣日耳曼",
        "Marseille": "马赛",
        "Monaco": "摩纳哥",
        "Lyon": "里昂",
        "Lille": "里尔",
        "Nice": "尼斯",
        "Lens": "朗斯",
        "Rennes": "雷恩",
        "Reims": "兰斯",
        "Strasbourg": "斯特拉斯堡",
        "Brest": "布雷斯特",
        "Toulouse": "图卢兹",
        "Montpellier": "蒙彼利埃",
        "Nantes": "南特",
        "Le Havre": "勒阿弗尔",
        "Metz": "梅斯",
        "Lorient": "洛里昂",
        "Clermont": "克莱蒙",
        "Angers": "昂热",
        "St Etienne": "圣埃蒂安",
        "Bordeaux": "波尔多",
        "Guingamp": "甘冈",
        "Caen": "卡昂",
        "Dijon": "第戎",
        "Amiens": "亚眠",
        "Troyes": "特鲁瓦",
        "Nimes": "尼姆",
        "Auxerre": "欧塞尔",
        "Ajaccio": "阿雅克肖",
        "Bastia": "巴斯蒂亚",
        "GFC Ajaccio": "阿雅克肖GFCO",
        "Nancy": "南锡",
        "Evian": "埃维昂",
        "Sochaux": "索肖",
        "Valenciennes": "瓦朗谢讷",
    },
}


@dataclass
class HistoricalMatchOdds:
    """一场比赛的真实历史赔率"""
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str  # 'H', 'D', 'A'

    # 1X2 赔率 (Bet365)
    b365_h: Optional[float] = None
    b365_d: Optional[float] = None
    b365_a: Optional[float] = None

    # 1X2 赔率 (Pinnacle, 接近真实市场)
    ps_h: Optional[float] = None
    ps_d: Optional[float] = None
    ps_a: Optional[float] = None

    # 市场平均赔率
    avg_h: Optional[float] = None
    avg_d: Optional[float] = None
    avg_a: Optional[float] = None

    # 亚盘 (Bet365)
    asian_handicap: Optional[float] = None  # 让球线
    asian_home_odds: Optional[float] = None
    asian_away_odds: Optional[float] = None

    # 大小球 2.5 (Bet365)
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None


def parse_odds_csv(filepath: str, league_id: str) -> Dict[str, HistoricalMatchOdds]:
    """
    解析 football-data.co.uk CSV 文件。
    返回: {match_key: HistoricalMatchOdds}
    match_key = "Man City_Arsenal" (主队_客队，按字母排序)
    """
    odds_data = {}
    team_map = TEAM_NAME_MAP.get(league_id, {})

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                home_en = row.get('HomeTeam', '').strip()
                away_en = row.get('AwayTeam', '').strip()
                if not home_en or not away_en:
                    continue

                home_cn = team_map.get(home_en, home_en)
                away_cn = team_map.get(away_en, away_en)

                # 解析数值
                def _f(val, default=None):
                    try:
                        return float(val) if val and val.strip() else default
                    except (ValueError, TypeError):
                        return default

                match = HistoricalMatchOdds(
                    date=row.get('Date', ''),
                    home_team=home_cn,
                    away_team=away_cn,
                    home_goals=int(row.get('FTHG', '0') or '0'),
                    away_goals=int(row.get('FTAG', '0') or '0'),
                    result=row.get('FTR', '').strip(),
                    b365_h=_f(row.get('B365H')),
                    b365_d=_f(row.get('B365D')),
                    b365_a=_f(row.get('B365A')),
                    ps_h=_f(row.get('PSH')),
                    ps_d=_f(row.get('PSD')),
                    ps_a=_f(row.get('PSA')),
                    avg_h=_f(row.get('BbAvH')),
                    avg_d=_f(row.get('BbAvD')),
                    avg_a=_f(row.get('BbAvA')),
                    asian_handicap=_f(row.get('BbAHh')),
                    asian_home_odds=_f(row.get('BbAvAHH')),
                    asian_away_odds=_f(row.get('BbAvAHA')),
                    over_odds=_f(row.get('BbAv>2.5')),
                    under_odds=_f(row.get('BbAv<2.5')),
                )

                # 仅保存有完整 1X2 赔率的比赛
                if match.b365_h and match.b365_d and match.b365_a:
                    key = f"{home_cn}_{away_cn}"
                    odds_data[key] = match

            except Exception:
                continue

    return odds_data


def download_and_cache(league_id: str, season: str) -> Optional[str]:
    """
    下载并缓存一个赛季的数据。
    返回本地文件路径，失败返回 None。
    """
    league_code = LEAGUE_CODES.get(league_id)
    if not league_code:
        return None

    code = season_code(season)
    url = DATA_URL.format(code=code, league=league_code)

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = DATA_DIR / f"{league_id}_{season.replace('/', '-')}.csv"

    if dest.exists():
        return str(dest)

    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest, 'wb') as f:
            f.write(data)
        return str(dest)
    except Exception:
        return None


def load_odds_for_season(league_id: str, season: str) -> Dict[str, HistoricalMatchOdds]:
    """
    加载指定联赛+赛季的真实赔率数据。
    先尝试本地缓存，失败则尝试下载。
    """
    # 先检查本地
    local_path = DATA_DIR / f"{league_id}_{season.replace('/', '-')}.csv"
    if local_path.exists():
        return parse_odds_csv(str(local_path), league_id)

    # 尝试下载
    saved = download_and_cache(league_id, season)
    if saved:
        return parse_odds_csv(saved, league_id)

    return {}


def get_real_odds(
    odds_data: Dict[str, HistoricalMatchOdds],
    home_team: str,
    away_team: str,
) -> Optional[HistoricalMatchOdds]:
    """
    查找一场比赛的真实赔率。
    odds_data: load_odds_for_season 的返回值
    """
    key = f"{home_team}_{away_team}"
    return odds_data.get(key)


# 数据可用性查询
def available_seasons(league_id: str) -> list:
    """检查本地缓存中有哪些赛季数据"""
    if not DATA_DIR.exists():
        return []
    seasons = []
    for f in DATA_DIR.glob(f"{league_id}_*.csv"):
        name = f.stem.replace(f"{league_id}_", "").replace("-", "/")
        seasons.append(name)
    return sorted(seasons)