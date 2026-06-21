"""
GTO v6.0 — 全局中文名称映射

所有显示内容使用中文。
"""

# 球队中文名映射（五大联赛 + 世界杯）
TEAM_NAMES_CN = {
    # 英超
    "Man City": "曼城", "Arsenal": "阿森纳", "Liverpool": "利物浦",
    "Aston Villa": "阿斯顿维拉", "Tottenham": "热刺", "Chelsea": "切尔西",
    "Newcastle": "纽卡斯尔", "Man United": "曼联", "West Ham": "西汉姆",
    "Crystal Palace": "水晶宫", "Brighton": "布莱顿", "Bournemouth": "伯恩茅斯",
    "Fulham": "富勒姆", "Wolves": "狼队", "Everton": "埃弗顿",
    "Brentford": "布伦特福德", "Nott'm Forest": "诺丁汉森林", "Luton": "卢顿",
    "Burnley": "伯恩利", "Sheffield United": "谢菲尔德联",
    
    # 西甲
    "Real Madrid": "皇家马德里", "Barcelona": "巴塞罗那", "Girona": "赫罗纳",
    "Atletico Madrid": "马德里竞技", "Athletic Club": "毕尔巴鄂竞技",
    "Real Sociedad": "皇家社会", "Real Betis": "皇家贝蒂斯", "Villarreal": "比利亚雷亚尔",
    "Valencia": "巴伦西亚", "Alaves": "阿拉维斯", "Osasuna": "奥萨苏纳",
    "Getafe": "赫塔费", "Sevilla": "塞维利亚", "Celta Vigo": "塞尔塔",
    "Las Palmas": "拉斯帕尔马斯", "Mallorca": "马略卡", "Rayo Vallecano": "巴列卡诺",
    "Cadiz": "加的斯", "Granada": "格拉纳达", "Almeria": "阿尔梅里亚",
    
    # 德甲
    "Bayer Leverkusen": "勒沃库森", "VfB Stuttgart": "斯图加特", "Bayern Munich": "拜仁慕尼黑",
    "RB Leipzig": "莱比锡", "Borussia Dortmund": "多特蒙德", "Eintracht Frankfurt": "法兰克福",
    "Hoffenheim": "霍芬海姆", "Werder Bremen": "不莱梅", "Freiburg": "弗赖堡",
    "Augsburg": "奥格斯堡", "Heidenheim": "海登海姆", "Union Berlin": "柏林联合",
    "Mainz 05": "美因茨", "Borussia Monchengladbach": "门兴格拉德巴赫",
    "Wolfsburg": "沃尔夫斯堡", "Bochum": "波鸿", "Koln": "科隆", "Darmstadt": "达姆施塔特",
    
    # 意甲
    "Inter Milan": "国际米兰", "AC Milan": "AC米兰", "Juventus": "尤文图斯",
    "Atalanta": "亚特兰大", "Bologna": "博洛尼亚", "Roma": "罗马", "Lazio": "拉齐奥",
    "Fiorentina": "佛罗伦萨", "Napoli": "那不勒斯", "Torino": "都灵",
    "Monza": "蒙扎", "Genoa": "热那亚", "Lecce": "莱切", "Cagliari": "卡利亚里",
    "Verona": "维罗纳", "Empoli": "恩波利", "Udinese": "乌迪内斯",
    "Frosinone": "弗罗西诺内", "Sassuolo": "萨索洛", "Salernitana": "萨勒尼塔纳",
    
    # 法甲
    "Paris Saint-Germain": "巴黎圣日耳曼", "Monaco": "摩纳哥", "Brest": "布雷斯特",
    "Lille": "里尔", "Nice": "尼斯", "Lyon": "里昂", "Lens": "朗斯",
    "Marseille": "马赛", "Rennes": "雷恩", "Toulouse": "图卢兹",
    "Reims": "兰斯", "Montpellier": "蒙彼利埃", "Strasbourg": "斯特拉斯堡",
    "Nantes": "南特", "Le Havre": "勒阿弗尔", "Clermont": "克莱蒙",
    "Lorient": "洛里昂", "Metz": "梅斯",
    
    # 国家队
    "Argentina": "阿根廷", "France": "法国", "Brazil": "巴西", "England": "英格兰",
    "Belgium": "比利时", "Portugal": "葡萄牙", "Spain": "西班牙", "Netherlands": "荷兰",
    "Germany": "德国", "Croatia": "克罗地亚", "Uruguay": "乌拉圭", "Colombia": "哥伦比亚",
    "Mexico": "墨西哥", "USA": "美国", "Senegal": "塞内加尔", "Japan": "日本",
    "Morocco": "摩洛哥", "Switzerland": "瑞士", "Denmark": "丹麦", "Australia": "澳大利亚",
    "South Korea": "韩国", "Iran": "伊朗", "Saudi Arabia": "沙特阿拉伯", "Ecuador": "厄瓜多尔",
    "Nigeria": "尼日利亚", "Serbia": "塞尔维亚", "Poland": "波兰", "Cameroon": "喀麦隆",
    "Canada": "加拿大", "Wales": "威尔士", "Tunisia": "突尼斯", "Ghana": "加纳",
    "Italy": "意大利",
}

# 联赛中文名
LEAGUE_NAMES_CN = {
    "premier_league": "英超",
    "la_liga": "西甲",
    "bundesliga": "德甲",
    "serie_a": "意甲",
    "ligue_1": "法甲",
    "worldcup": "世界杯",
}

# 策略中文名
STRATEGY_NAMES_CN = {
    "1x2": "胜平负",
    "over_under": "大小球",
    "asian_handicap": "亚盘",
    "parlay": "串关",
}

# 方向中文名
DIRECTION_NAMES_CN = {
    "home": "主胜",
    "draw": "平局",
    "away": "客胜",
    "over": "大球",
    "under": "小球",
}


def get_cn_name(team: str) -> str:
    """获取球队中文名"""
    return TEAM_NAMES_CN.get(team, team)


def get_league_cn(league_id: str) -> str:
    """获取联赛中文名"""
    return LEAGUE_NAMES_CN.get(league_id, league_id)


def get_strategy_cn(strategy: str) -> str:
    """获取策略中文名"""
    return STRATEGY_NAMES_CN.get(strategy, strategy)


def get_direction_cn(direction: str) -> str:
    """获取方向中文名"""
    return DIRECTION_NAMES_CN.get(direction, direction)
