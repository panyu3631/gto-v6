"""
GTO-GameFlow v5.9.3 — 中文翻译/显示模块

提供统一的映射表，将内部英文字段名转换为中文显示名。
所有翻译集中在此模块，不依赖任何计算逻辑，避免代码混乱。

使用方式:
    from src.utils.i18n import cn_league, cn_strategy, cn_selection, cn_result

    print(cn_league("premier_league"))  # → "英超"
    print(cn_strategy("1x2"))           # → "胜平负"
    print(cn_selection("home_win"))     # → "主胜"
    print(cn_result("win"))             # → "赢"

设计原则:
    - 纯函数，无副作用，不依赖任何计算模块
    - 未知键返回原始值（静默降级）
    - 可作为字典直接使用，也可作为函数调用
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any

# ═══════════════════════════════════════════════════════════════
# 联赛名称映射
# ═══════════════════════════════════════════════════════════════

LEAGUE_NAME_CN: Dict[str, str] = {
    "premier_league": "英超",
    "la_liga": "西甲",
    "bundesliga": "德甲",
    "serie_a": "意甲",
    "ligue_1": "法甲",
    # 别名
    "epl": "英超",
    "laliga": "西甲",
    "bundesliga_1": "德甲",
    "seria_a": "意甲",
    "league_1": "法甲",
}


def cn_league(key: str) -> str:
    """将联赛内部 ID 转换为中文显示名。未知键返回原始值。"""
    return LEAGUE_NAME_CN.get(key, key)


# ═══════════════════════════════════════════════════════════════
# 策略名称映射
# ═══════════════════════════════════════════════════════════════

STRATEGY_NAME_CN: Dict[str, str] = {
    "1x2": "胜平负",
    "1X2": "胜平负",
    "asian": "亚盘",
    "asian_handicap": "亚盘",
    "over_under": "大小球",
    "parlay": "串关",
    # 英文别名
    "moneyline": "胜平负",
    "totals": "大小球",
}


def cn_strategy(key: str) -> str:
    """将策略内部 ID 转换为中文显示名。未知键返回原始值。"""
    return STRATEGY_NAME_CN.get(key, key)


# ═══════════════════════════════════════════════════════════════
# 投注方向映射 (BetSelection.value → 中文)
# ═══════════════════════════════════════════════════════════════

SELECTION_NAME_CN: Dict[str, str] = {
    "home_win": "主胜",
    "draw": "平局",
    "away_win": "客胜",
    "over": "大球",
    "under": "小球",
    "HOME_WIN": "主胜",
    "DRAW": "平局",
    "AWAY_WIN": "客胜",
    "OVER": "大球",
    "UNDER": "小球",
}


def cn_selection(key: str) -> str:
    """将投注方向枚举值转换为中文显示名。"""
    return SELECTION_NAME_CN.get(key, key)


# ═══════════════════════════════════════════════════════════════
# 结算结果映射
# ═══════════════════════════════════════════════════════════════

RESULT_NAME_CN: Dict[str, str] = {
    "win": "赢",
    "loss": "输",
    "push": "走水",
    "void": "无效",
    "pending": "待定",
    "full_win": "全赢",
    "half_win": "赢半",
    "half_loss": "输半",
    "full_loss": "全输",
    "WIN": "赢",
    "LOSS": "输",
    "PUSH": "走水",
    "VOID": "无效",
    "PENDING": "待定",
}


def cn_result(key: str) -> str:
    """将结算结果枚举值转换为中文显示名。"""
    return RESULT_NAME_CN.get(key, key)


# ═══════════════════════════════════════════════════════════════
# 赛场结果 (FTR) 映射
# ═══════════════════════════════════════════════════════════════

FTR_NAME_CN: Dict[str, str] = {
    "H": "主胜",
    "D": "平局",
    "A": "客胜",
    "home_win": "主胜",
    "draw": "平局",
    "away_win": "客胜",
}


def cn_ftr(key: str) -> str:
    """将赛场结果代码转换为中文显示名。"""
    return FTR_NAME_CN.get(key, key)


# ═══════════════════════════════════════════════════════════════
# 投注建议标签映射 (BetSelection 枚举值 → 短标签)
# ═══════════════════════════════════════════════════════════════

SELECTION_LABEL_CN: Dict[str, str] = {
    "home_win": "主胜",
    "draw": "平局",
    "away_win": "客胜",
    "over": "大球",
    "under": "小球",
}


def cn_label(key: str) -> str:
    """获取投注方向的短中文标签。"""
    return SELECTION_LABEL_CN.get(key, key)


# ═══════════════════════════════════════════════════════════════
# 通用便捷函数
# ═══════════════════════════════════════════════════════════════

def cn(key: str, category: str = "auto") -> str:
    """
    自动识别类型并翻译为中文。

    参数:
        key: 需要翻译的键
        category: 类型提示 ("league" | "strategy" | "selection" | "result" | "auto")

    返回:
        中文翻译后的字符串
    """
    if category == "league":
        return cn_league(key)
    if category == "strategy":
        return cn_strategy(key)
    if category == "selection":
        return cn_selection(key)
    if category == "result":
        return cn_result(key)

    # 自动识别
    for mapping in [LEAGUE_NAME_CN, STRATEGY_NAME_CN, SELECTION_NAME_CN,
                     RESULT_NAME_CN, FTR_NAME_CN]:
        if key in mapping:
            return mapping[key]
    return key


def cn_all() -> Dict[str, Dict[str, str]]:
    """返回所有翻译映射表 (用于前端或配置文件导出)。"""
    return {
        "leagues": LEAGUE_NAME_CN,
        "strategies": STRATEGY_NAME_CN,
        "selections": SELECTION_NAME_CN,
        "results": RESULT_NAME_CN,
        "ftr": FTR_NAME_CN,
    }


# ═══════════════════════════════════════════════════════════════
# 默认格式化模板
# ═══════════════════════════════════════════════════════════════

# 投注记录格式化模板
BET_LOG_FORMAT_CN = "【{strategy}】{league} {match_desc} | 投注: {selection} @{odds} | 结果: {result} | 盈亏: {pnl:+.0f}"

# 回测摘要格式化模板
BACKTEST_SUMMARY_CN = """\
╔══════════════════════════════════════════╗
║  GTO-GameFlow 回测摘要
╠══════════════════════════════════════════╣
║  总投注: {total_bets:>5} 注
║  胜率:   {win_rate:>7.1%}
║  ROI:    {roi:>+8.1%}
║  利润:   {profit:>+10,.0f}
╚══════════════════════════════════════════╝"""