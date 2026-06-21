"""
GTO v6.0 — 世界杯真实赔率数据

从网页搜索获取的真实赛前赔率。
"""

# 世界杯2026真实赔率（从bet365/1xBet获取）
WC_REAL_ODDS = {
    # 6月11日
    "Mexico_vs_SouthAfrica": {"home": 1.45, "draw": 4.50, "away": 7.00},
    "SouthKorea_vs_Czechia": {"home": 2.10, "draw": 3.40, "away": 3.60},
    # 6月12日
    "Canada_vs_BosniaHerzegovina": {"home": 2.30, "draw": 3.30, "away": 3.20},
    "USMNT_vs_Paraguay": {"home": 1.80, "draw": 3.60, "away": 4.50},
    # 6月13日
    "Qatar_vs_Switzerland": {"home": 3.80, "draw": 3.40, "away": 2.00},
    "Brazil_vs_Morocco": {"home": 1.50, "draw": 4.20, "away": 6.50},
    "Haiti_vs_Scotland": {"home": 4.50, "draw": 3.50, "away": 1.80},
    "Australia_vs_Turkey": {"home": 2.80, "draw": 3.20, "away": 2.50},
    # 6月14日
    "Germany_vs_Curaçao": {"home": 1.05, "draw": 12.00, "away": 34.00},
    "Netherlands_vs_Japan": {"home": 1.90, "draw": 3.50, "away": 4.00},
    "IvoryCoast_vs_Ecuador": {"home": 2.40, "draw": 3.20, "away": 3.00},
    "Sweden_vs_Tunisia": {"home": 1.70, "draw": 3.60, "away": 5.00},
    # 6月15日
    "Spain_vs_CapeVerde": {"home": 1.06, "draw": 11.00, "away": 29.00},
    "Belgium_vs_Egypt": {"home": 1.55, "draw": 4.00, "away": 5.50},
    "SaudiArabia_vs_Uruguay": {"home": 5.00, "draw": 3.80, "away": 1.65},
    "Iran_vs_NewZealand": {"home": 2.20, "draw": 3.30, "away": 3.30},
    # 6月16日
    "France_vs_Senegal": {"home": 1.40, "draw": 4.50, "away": 7.50},
    "Iraq_vs_Norway": {"home": 6.00, "draw": 4.00, "away": 1.50},
    "Argentina_vs_Algeria": {"home": 1.20, "draw": 6.00, "away": 13.00},
    "Austria_vs_Jordan": {"home": 1.60, "draw": 3.80, "away": 5.50},
    # 6月17日
    "Portugal_vs_DR Congo": {"home": 1.25, "draw": 5.50, "away": 11.00},
    "England_vs_Croatia": {"home": 1.70, "draw": 3.60, "away": 5.00},
    "Ghana_vs_Panama": {"home": 2.10, "draw": 3.30, "away": 3.50},
    "Uzbekistan_vs_Colombia": {"home": 6.50, "draw": 4.20, "away": 1.45},
    # 6月21日（今日）
    "Spain_vs_SaudiArabia": {"home": 1.08, "draw": 9.00, "away": 22.00},
    "Belgium_vs_Iran": {"home": 1.35, "draw": 4.80, "away": 9.00},
    "Uruguay_vs_CapeVerde": {"home": 1.15, "draw": 7.00, "away": 15.00},
    "NewZealand_vs_Egypt": {"home": 4.50, "draw": 3.60, "away": 1.75},
}

# 比赛结果
WC_RESULTS = {
    "Mexico_vs_SouthAfrica": {"home": "Mexico", "away": "South Africa", "hs": 2, "as": 0, "group": "A", "date": "2026-06-11"},
    "SouthKorea_vs_Czechia": {"home": "South Korea", "away": "Czechia", "hs": 2, "as": 1, "group": "A", "date": "2026-06-11"},
    "Canada_vs_BosniaHerzegovina": {"home": "Canada", "away": "Bosnia Herzegovina", "hs": 1, "as": 1, "group": "B", "date": "2026-06-12"},
    "USMNT_vs_Paraguay": {"home": "USMNT", "away": "Paraguay", "hs": 4, "as": 1, "group": "D", "date": "2026-06-12"},
    "Qatar_vs_Switzerland": {"home": "Qatar", "away": "Switzerland", "hs": 1, "as": 1, "group": "B", "date": "2026-06-13"},
    "Brazil_vs_Morocco": {"home": "Brazil", "away": "Morocco", "hs": 1, "as": 1, "group": "C", "date": "2026-06-13"},
    "Haiti_vs_Scotland": {"home": "Haiti", "away": "Scotland", "hs": 0, "as": 1, "group": "C", "date": "2026-06-13"},
    "Australia_vs_Turkey": {"home": "Australia", "away": "Turkey", "hs": 2, "as": 0, "group": "D", "date": "2026-06-13"},
    "Germany_vs_Curaçao": {"home": "Germany", "away": "Curaçao", "hs": 7, "as": 1, "group": "E", "date": "2026-06-14"},
    "Netherlands_vs_Japan": {"home": "Netherlands", "away": "Japan", "hs": 2, "as": 2, "group": "F", "date": "2026-06-14"},
    "IvoryCoast_vs_Ecuador": {"home": "Ivory Coast", "away": "Ecuador", "hs": 1, "as": 0, "group": "E", "date": "2026-06-14"},
    "Sweden_vs_Tunisia": {"home": "Sweden", "away": "Tunisia", "hs": 5, "as": 1, "group": "F", "date": "2026-06-14"},
    "Spain_vs_CapeVerde": {"home": "Spain", "away": "Cape Verde", "hs": 0, "as": 0, "group": "H", "date": "2026-06-15"},
    "Belgium_vs_Egypt": {"home": "Belgium", "away": "Egypt", "hs": 1, "as": 1, "group": "G", "date": "2026-06-15"},
    "SaudiArabia_vs_Uruguay": {"home": "Saudi Arabia", "away": "Uruguay", "hs": 1, "as": 1, "group": "H", "date": "2026-06-15"},
    "Iran_vs_NewZealand": {"home": "Iran", "away": "New Zealand", "hs": 2, "as": 2, "group": "G", "date": "2026-06-15"},
    "France_vs_Senegal": {"home": "France", "away": "Senegal", "hs": 3, "as": 1, "group": "I", "date": "2026-06-16"},
    "Iraq_vs_Norway": {"home": "Iraq", "away": "Norway", "hs": 1, "as": 4, "group": "I", "date": "2026-06-16"},
    "Argentina_vs_Algeria": {"home": "Argentina", "away": "Algeria", "hs": 3, "as": 0, "group": "J", "date": "2026-06-16"},
    "Austria_vs_Jordan": {"home": "Austria", "away": "Jordan", "hs": 3, "as": 1, "group": "J", "date": "2026-06-16"},
    "Portugal_vs_DR Congo": {"home": "Portugal", "away": "DR Congo", "hs": 1, "as": 1, "group": "K", "date": "2026-06-17"},
    "England_vs_Croatia": {"home": "England", "away": "Croatia", "hs": 4, "as": 2, "group": "L", "date": "2026-06-17"},
    "Ghana_vs_Panama": {"home": "Ghana", "away": "Panama", "hs": 1, "as": 0, "group": "L", "date": "2026-06-17"},
    "Uzbekistan_vs_Colombia": {"home": "Uzbekistan", "away": "Colombia", "hs": 1, "as": 3, "group": "K", "date": "2026-06-17"},
}


def get_real_odds(match_key: str) -> dict:
    """获取真实赔率"""
    return WC_REAL_ODDS.get(match_key, {})


def get_match_result(match_key: str) -> dict:
    """获取比赛结果"""
    return WC_RESULTS.get(match_key, {})
