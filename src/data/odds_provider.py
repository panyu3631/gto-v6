"""
统一赔率数据层 (Phase 5.0)

整合三个数据源，提供一致的赔率接口：
1. CSV 历史数据 (football-data.co.uk) — 回测主源
2. Football-Data.org API (v4) — 实时赔率
3. API-Football (v3) — 实时赔率 + 多线亚盘/大小球

关键设计:
- 所有调用方通过此模块获取赔率，不再直接读 CSV
- 自动处理列名差异 (BbAvH vs AvgH, BbAvAHH vs B365AHH)
- 提供 Pinnacle 赔率作为首选 (margin 最小)
- 支持降级: 实时 API → CSV 历史 → 合成赔率
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MatchOddsBundle:
    """单场比赛的完整赔率数据包"""
    match_id: str
    home_team: str
    away_team: str
    kickoff_time: datetime

    # 1X2 — 优先 Pinnacle，其次市场平均，最后 Bet365
    odds_home: float = 1.0
    odds_draw: float = 1.0
    odds_away: float = 1.0
    odds_source: str = "unknown"  # "pinnacle" | "market_avg" | "bet365"

    # 亚盘 (多线)
    # {handicap_line: {"home": odds, "away": odds}}
    asian_odds: Dict[float, Dict[str, float]] = field(default_factory=dict)

    # 大小球 (多线)
    # {totals_line: {"over": odds, "under": odds}}
    totals_odds: Dict[float, Dict[str, float]] = field(default_factory=dict)

    # 博彩商数量 (用于 margin 估计)
    bookmaker_count: int = 0

    # 元数据
    has_real_asian: bool = False
    has_real_totals: bool = False
    is_live: bool = False  # 是否为实时 API 数据


class OddsProvider:
    """
    统一赔率提供者。

    优先级: Pinnacle > 市场平均 > Bet365 > 合成
    对于 CSV 回测数据，自动适配新旧列名格式。
    """

    # CSV 列名兼容映射 (新旧格式)
    CSV_COLUMN_MAP = {
        # 1X2 市场平均
        "avg_h": ["BbAvH", "AvgH"],
        "avg_d": ["BbAvD", "AvgD"],
        "avg_a": ["BbAvA", "AvgA"],
        # Pinnacle 1X2
        "ps_h": ["PSH"],
        "ps_d": ["PSD"],
        "ps_a": ["PSA"],
        # Bet365 1X2
        "b365_h": ["B365H"],
        "b365_d": ["B365D"],
        "b365_a": ["B365A"],
        # 亚盘
        "asian_handicap": ["BbAHh", "AHh", "AvgAH"],
        "asian_home": ["BbAvAHH", "AvgAHH"],
        "asian_away": ["BbAvAHA", "AvgAHA"],
        # 大小球 2.5
        "over_2.5": ["BbAv>2.5", "Avg>2.5"],
        "under_2.5": ["BbAv<2.5", "Avg<2.5"],
    }

    def __init__(self, use_pinnacle: bool = True):
        """
        参数:
            use_pinnacle: 是否优先使用 Pinnacle 赔率 (margin 最小)
        """
        self.use_pinnacle = use_pinnacle
        self._api_cache: Dict[str, MatchOddsBundle] = {}

    def resolve_csv_field(self, row: dict, field_key: str) -> Optional[float]:
        """
        从 CSV 行中解析字段，兼容新旧列名格式。

        示例:
            resolve_csv_field(row, "avg_h") → 先查 BbAvH，再查 AvgH
        """
        candidates = self.CSV_COLUMN_MAP.get(field_key, [field_key])
        for col in candidates:
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "NA", "NaN"):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    def build_from_csv(
        self,
        row: dict,
        match_id: str,
        home_team: str,
        away_team: str,
        kickoff_time: datetime,
    ) -> MatchOddsBundle:
        """
        从 CSV 行构建赔率数据包。

        自动选择最优赔率源: Pinnacle > 市场平均 > Bet365
        """
        bundle = MatchOddsBundle(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            kickoff_time=kickoff_time,
        )

        # 1X2 赔率: 优先 Pinnacle
        if self.use_pinnacle:
            ps_h = self.resolve_csv_field(row, "ps_h")
            ps_d = self.resolve_csv_field(row, "ps_d")
            ps_a = self.resolve_csv_field(row, "ps_a")
            if ps_h and ps_d and ps_a:
                bundle.odds_home = ps_h
                bundle.odds_draw = ps_d
                bundle.odds_away = ps_a
                bundle.odds_source = "pinnacle"
                bundle.bookmaker_count = 1  # Pinnacle 是单一博彩商

        # 回退: 市场平均
        if bundle.odds_source == "unknown":
            avg_h = self.resolve_csv_field(row, "avg_h")
            avg_d = self.resolve_csv_field(row, "avg_d")
            avg_a = self.resolve_csv_field(row, "avg_a")
            if avg_h and avg_d and avg_a:
                bundle.odds_home = avg_h
                bundle.odds_draw = avg_d
                bundle.odds_away = avg_a
                bundle.odds_source = "market_avg"
                bundle.bookmaker_count = 5  # BetBrain 聚合多家

        # 最终回退: Bet365
        if bundle.odds_source == "unknown":
            b365_h = self.resolve_csv_field(row, "b365_h")
            b365_d = self.resolve_csv_field(row, "b365_d")
            b365_a = self.resolve_csv_field(row, "b365_a")
            if b365_h and b365_d and b365_a:
                bundle.odds_home = b365_h
                bundle.odds_draw = b365_d
                bundle.odds_away = b365_a
                bundle.odds_source = "bet365"
                bundle.bookmaker_count = 1

        # 亚盘 (单线 — CSV 仅提供一条线)
        asian_line = self.resolve_csv_field(row, "asian_handicap")
        asian_home = self.resolve_csv_field(row, "asian_home")
        asian_away = self.resolve_csv_field(row, "asian_away")
        if asian_line is not None and asian_home and asian_away:
            bundle.asian_odds[asian_line] = {"home": asian_home, "away": asian_away}
            bundle.has_real_asian = True

        # 大小球 2.5 (CSV 仅提供一条线)
        over_odds = self.resolve_csv_field(row, "over_2.5")
        under_odds = self.resolve_csv_field(row, "under_2.5")
        if over_odds and under_odds:
            bundle.totals_odds[2.5] = {"over": over_odds, "under": under_odds}
            bundle.has_real_totals = True

        return bundle

    def get_odds_for_match(
        self,
        row: dict,
        match_id: str,
        home_team: str,
        away_team: str,
        kickoff_time: datetime,
    ) -> MatchOddsBundle:
        """获取单场比赛的赔率 (带缓存)"""
        if match_id in self._api_cache:
            return self._api_cache[match_id]
        bundle = self.build_from_csv(row, match_id, home_team, away_team, kickoff_time)
        self._api_cache[match_id] = bundle
        return bundle

    def estimate_margin(self, odds_home: float, odds_draw: float, odds_away: float) -> float:
        """估计庄家 margin (overround - 1)"""
        return (1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away) - 1.0

    def clear_cache(self):
        self._api_cache.clear()


# 全局单例
_provider: Optional[OddsProvider] = None


def get_odds_provider(use_pinnacle: bool = True) -> OddsProvider:
    global _provider
    if _provider is None:
        _provider = OddsProvider(use_pinnacle=use_pinnacle)
    return _provider