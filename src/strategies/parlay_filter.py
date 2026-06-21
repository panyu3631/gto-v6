"""
GTO v6.0 — 串关专用筛选器

核心原则: 串关有独立的筛选逻辑，不复用单场筛选器。

与单场筛选器的区别:
| 参数 | 单场 | 串关 |
|------|------|------|
| 价值阈值 | 5% | 2% |
| 概率下限 | 无 | 20% |
| 赔率范围 | 无限制 | 1.20+ |
| 最大长度 | 1 | 2-3 |
| 单注占比 | 0.5% | 0.3% |

串关类型:
- 2串1: 主力，命中率高
- 3串1: 高赔率，少量配置

腿类型:
- 单选腿: 一个市场一个选项
- 双选腿: 一个市场两个选项 (胜平/平负/胜负)
- 跨市场腿: 同场不同市场各一

使用方式:
    filter = ParlayFilter(league_id="premier_league")
    legs = filter.filter_legs(single_bets)
    parlays = filter.generate_parlays(legs, bankroll=10000)
"""

from __future__ import annotations

import math
import logging
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParlayLeg:
    """串关的一条腿"""
    match_id: str
    league_id: str
    strategy: str          # "1x2" / "over_under" / "asian_handicap"
    direction: str         # "home" / "draw" / "away" / "over" / "under"
    odds: float
    model_prob: float
    market_prob: float
    value: float
    confidence: float = 0.0
    leg_type: str = "single"  # "single" / "double_chance" / "cross_market"
    handicap_line: float = 0.0
    totals_line: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.odds > 1.0 and self.model_prob > 0 and self.value > 0


@dataclass
class ParlayCombo:
    """串关组合"""
    parlay_id: str
    legs: List[ParlayLeg]
    combined_odds: float
    combined_prob: float
    combined_value: float
    kelly_stake: float
    adjusted_stake: float
    priority_score: float
    cross_league: bool
    cross_strategy: bool
    correlation_penalty: float

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def match_ids(self) -> Set[str]:
        return {leg.match_id for leg in self.legs}

    def summary(self) -> dict:
        return {
            "parlay_id": self.parlay_id,
            "n_legs": self.n_legs,
            "matches": list(self.match_ids),
            "combined_odds": round(self.combined_odds, 2),
            "combined_prob": round(self.combined_prob, 4),
            "combined_value": round(self.combined_value, 4),
            "stake": round(self.adjusted_stake, 2),
            "cross_league": self.cross_league,
        }


# ═══════════════════════════════════════════════════════════════
# 串关筛选器参数
# ═══════════════════════════════════════════════════════════════

# 联赛特化参数
LEAGUE_PARLAY_PARAMS = {
    "premier_league": {
        "min_single_value": 0.02,
        "min_single_prob": 0.20,
        "min_odds": 1.20,
        "max_odds": 10.0,
        "min_combined_odds": 2.0,
        "min_combined_prob": 0.05,
        "max_legs": 3,
        "kelly_discount": 0.15,
        "corr_penalty_same_league": 0.85,
        "corr_penalty_same_strategy": 0.90,
        "cross_league_bonus": 1.15,
        "max_single_exposure": 0.003,
        "max_total_exposure": 0.01,
    },
    "la_liga": {
        "min_single_value": 0.02,
        "min_single_prob": 0.20,
        "min_odds": 1.20,
        "max_odds": 10.0,
        "min_combined_odds": 2.0,
        "min_combined_prob": 0.05,
        "max_legs": 3,
        "kelly_discount": 0.15,
        "corr_penalty_same_league": 0.85,
        "corr_penalty_same_strategy": 0.90,
        "cross_league_bonus": 1.15,
        "max_single_exposure": 0.003,
        "max_total_exposure": 0.01,
    },
    "bundesliga": {
        "min_single_value": 0.02,
        "min_single_prob": 0.20,
        "min_odds": 1.20,
        "max_odds": 10.0,
        "min_combined_odds": 2.0,
        "min_combined_prob": 0.05,
        "max_legs": 3,
        "kelly_discount": 0.15,
        "corr_penalty_same_league": 0.85,
        "corr_penalty_same_strategy": 0.90,
        "cross_league_bonus": 1.15,
        "max_single_exposure": 0.003,
        "max_total_exposure": 0.01,
    },
    "serie_a": {
        "min_single_value": 0.02,
        "min_single_prob": 0.20,
        "min_odds": 1.20,
        "max_odds": 10.0,
        "min_combined_odds": 2.0,
        "min_combined_prob": 0.05,
        "max_legs": 3,
        "kelly_discount": 0.15,
        "corr_penalty_same_league": 0.85,
        "corr_penalty_same_strategy": 0.90,
        "cross_league_bonus": 1.15,
        "max_single_exposure": 0.003,
        "max_total_exposure": 0.01,
    },
    "ligue_1": {
        "min_single_value": 0.02,
        "min_single_prob": 0.20,
        "min_odds": 1.20,
        "max_odds": 10.0,
        "min_combined_odds": 2.0,
        "min_combined_prob": 0.05,
        "max_legs": 3,
        "kelly_discount": 0.15,
        "corr_penalty_same_league": 0.85,
        "corr_penalty_same_strategy": 0.90,
        "cross_league_bonus": 1.15,
        "max_single_exposure": 0.003,
        "max_total_exposure": 0.01,
    },
}

DEFAULT_PARLAY_PARAMS = {
    "min_single_value": 0.02,
    "min_single_prob": 0.20,
    "min_odds": 1.20,
    "max_odds": 10.0,
    "min_combined_odds": 2.0,
    "min_combined_prob": 0.05,
    "max_legs": 3,
    "kelly_discount": 0.15,
    "corr_penalty_same_league": 0.85,
    "corr_penalty_same_strategy": 0.90,
    "cross_league_bonus": 1.15,
    "max_single_exposure": 0.003,
    "max_total_exposure": 0.01,
}

# 仓位控制 (按赔率区间)
STAKE_TIERS = [
    {"max_odds": 5.0, "max_exposure": 0.003, "kelly_discount": 0.20},
    {"max_odds": 10.0, "max_exposure": 0.002, "kelly_discount": 0.15},
    {"max_odds": 20.0, "max_exposure": 0.001, "kelly_discount": 0.12},
    {"max_odds": float('inf'), "max_exposure": 0.0005, "kelly_discount": 0.10},
]


# ═══════════════════════════════════════════════════════════════
# 串关筛选器
# ═══════════════════════════════════════════════════════════════

class ParlayFilter:
    """
    串关专用筛选器。

    与单场筛选器完全独立，使用更宽松的阈值。

    使用方式:
        filter = ParlayFilter(league_id="premier_league")
        legs = filter.filter_legs(single_bets)
        parlays = filter.generate_parlays(legs, bankroll=10000)
    """

    def __init__(
        self,
        league_id: str = "",
        min_single_value: Optional[float] = None,
        min_single_prob: Optional[float] = None,
        min_odds: Optional[float] = None,
        max_odds: Optional[float] = None,
        min_combined_odds: Optional[float] = None,
        max_legs: Optional[int] = None,
        kelly_discount: Optional[float] = None,
    ):
        params = LEAGUE_PARLAY_PARAMS.get(league_id, DEFAULT_PARLAY_PARAMS)

        self.league_id = league_id
        self.min_single_value = min_single_value or params["min_single_value"]
        self.min_single_prob = min_single_prob or params["min_single_prob"]
        self.min_odds = min_odds or params["min_odds"]
        self.max_odds = max_odds or params["max_odds"]
        self.min_combined_odds = min_combined_odds or params["min_combined_odds"]
        self.max_legs = max_legs or params["max_legs"]
        self.kelly_discount = kelly_discount or params["kelly_discount"]
        self.corr_penalty_same_league = params["corr_penalty_same_league"]
        self.corr_penalty_same_strategy = params["corr_penalty_same_strategy"]
        self.cross_league_bonus = params["cross_league_bonus"]
        self.max_single_exposure = params["max_single_exposure"]
        self.max_total_exposure = params["max_total_exposure"]

    def filter_legs(
        self,
        candidates: List[Dict],
    ) -> List[ParlayLeg]:
        """
        从候选池筛选合格的串关腿。

        参数:
            candidates: 候选列表，每个元素是字典:
                {
                    "match_id": str,
                    "league_id": str,
                    "strategy": str,
                    "direction": str,
                    "odds": float,
                    "model_prob": float,
                    "market_prob": float,
                    "value": float,
                    "confidence": float,
                }

        返回:
            合格的 ParlayLeg 列表
        """
        legs = []

        for c in candidates:
            # 基础筛选
            if c.get("value", 0) < self.min_single_value:
                continue
            if c.get("model_prob", 0) < self.min_single_prob:
                continue
            if c.get("odds", 0) < self.min_odds:
                continue
            if c.get("odds", 0) > self.max_odds:
                continue

            leg = ParlayLeg(
                match_id=c["match_id"],
                league_id=c.get("league_id", ""),
                strategy=c.get("strategy", "1x2"),
                direction=c.get("direction", ""),
                odds=c["odds"],
                model_prob=c["model_prob"],
                market_prob=c.get("market_prob", 0),
                value=c["value"],
                confidence=c.get("confidence", 0),
                handicap_line=c.get("handicap_line", 0),
                totals_line=c.get("totals_line", 0),
            )

            if leg.is_valid:
                legs.append(leg)

        # 每场比赛保留最优的2条腿
        legs = self._deduplicate(legs, max_per_match=2)

        # 按价值排序
        legs.sort(key=lambda x: x.value, reverse=True)

        return legs

    def generate_parlays(
        self,
        legs: List[ParlayLeg],
        bankroll: float,
        max_parlays: int = 10,
    ) -> List[ParlayCombo]:
        """
        从合格腿中生成串关组合。

        参数:
            legs: 合格的 ParlayLeg 列表
            bankroll: 当前资金
            max_parlays: 最大输出数量

        返回:
            ParlayCombo 列表 (按优先级排序)
        """
        if len(legs) < 2:
            return []

        # 限制候选池大小
        legs = legs[:30]

        all_parlays = []

        # 生成2串1
        for combo in itertools.combinations(legs, 2):
            parlay = self._build_parlay(list(combo), bankroll, n_legs=2)
            if parlay:
                all_parlays.append(parlay)

        # 生成3串1 (如果允许)
        if self.max_legs >= 3:
            for combo in itertools.combinations(legs, 3):
                parlay = self._build_parlay(list(combo), bankroll, n_legs=3)
                if parlay:
                    all_parlays.append(parlay)

        # 按优先级排序
        all_parlays.sort(key=lambda x: x.priority_score, reverse=True)

        # 去重
        seen = set()
        deduped = []
        for p in all_parlays:
            key = frozenset(p.match_ids)
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        return deduped[:max_parlays]

    def _build_parlay(
        self,
        legs: List[ParlayLeg],
        bankroll: float,
        n_legs: int,
    ) -> Optional[ParlayCombo]:
        """构建单个串关组合"""
        # 检查: 同一比赛不能出现两次
        match_ids = [leg.match_id for leg in legs]
        if len(set(match_ids)) < n_legs:
            return None

        # 计算组合赔率和概率
        combined_odds = 1.0
        raw_combined_prob = 1.0
        for leg in legs:
            combined_odds *= leg.odds
            raw_combined_prob *= leg.model_prob

        # 最低组合赔率
        if combined_odds < self.min_combined_odds:
            return None

        # 相关性惩罚
        corr_penalty, cross_league, cross_strategy = self._compute_correlation(legs)
        combined_prob = raw_combined_prob * corr_penalty

        # 组合价值
        combined_value = combined_prob - (1.0 / combined_odds)

        if combined_value < 0.01:  # 串关最低价值1%
            return None

        # 仓位计算
        stake = self._compute_stake(combined_odds, combined_prob, bankroll)
        if stake < 1.0:
            return None

        # 优先级
        cross_bonus = self.cross_league_bonus if cross_league else 1.0
        priority = combined_value * combined_prob * cross_bonus

        parlay_id = f"parlay_{n_legs}x1_{hash(frozenset(match_ids)) % 100000:05d}"

        return ParlayCombo(
            parlay_id=parlay_id,
            legs=legs,
            combined_odds=round(combined_odds, 4),
            combined_prob=round(combined_prob, 6),
            combined_value=round(combined_value, 6),
            kelly_stake=round(stake, 2),
            adjusted_stake=round(stake, 2),
            priority_score=round(priority, 6),
            cross_league=cross_league,
            cross_strategy=cross_strategy,
            correlation_penalty=round(corr_penalty, 4),
        )

    def _compute_correlation(
        self,
        legs: List[ParlayLeg],
    ) -> Tuple[float, bool, bool]:
        """计算相关性惩罚"""
        penalty = 1.0
        leagues = {leg.league_id for leg in legs}
        strategies = {leg.strategy for leg in legs}

        cross_league = len(leagues) > 1
        cross_strategy = len(strategies) > 1

        if len(leagues) == 1:
            penalty *= self.corr_penalty_same_league
        if len(strategies) == 1:
            penalty *= self.corr_penalty_same_strategy

        return penalty, cross_league, cross_strategy

    def _compute_stake(
        self,
        combined_odds: float,
        combined_prob: float,
        bankroll: float,
    ) -> float:
        """计算串关仓位 (按赔率区间分层)"""
        # Kelly 公式
        b = combined_odds - 1.0
        if b <= 0:
            return 0.0

        f_kelly = max(0.0, (b * combined_prob - (1.0 - combined_prob)) / b)

        # 按赔率区间选择折扣
        kelly_discount = self.kelly_discount
        max_exposure = self.max_single_exposure
        for tier in STAKE_TIERS:
            if combined_odds <= tier["max_odds"]:
                kelly_discount = tier["kelly_discount"]
                max_exposure = tier["max_exposure"]
                break

        # 串关长度折扣
        n_legs = len(legs) if 'legs' in dir(self) else 2
        leg_discount = 0.65 if n_legs == 2 else 0.50

        stake = f_kelly * kelly_discount * leg_discount * bankroll

        # 单注上限
        max_stake = bankroll * max_exposure
        return min(stake, max_stake)

    def _deduplicate(
        self,
        legs: List[ParlayLeg],
        max_per_match: int = 2,
    ) -> List[ParlayLeg]:
        """每场比赛保留最优的N条腿"""
        by_match: Dict[str, List[ParlayLeg]] = defaultdict(list)
        for leg in legs:
            by_match[leg.match_id].append(leg)

        result = []
        for match_id, match_legs in by_match.items():
            match_legs.sort(key=lambda x: x.value, reverse=True)
            result.extend(match_legs[:max_per_match])

        return result


# ═══════════════════════════════════════════════════════════════
# 双选腿生成器
# ═══════════════════════════════════════════════════════════════

class DoubleChanceGenerator:
    """
    双选腿生成器。

    从单选腿生成双选腿:
    - 胜平(1X): P(胜) + P(平)
    - 平负(X2): P(平) + P(负)
    - 胜负(12): P(胜) + P(负)
    """

    @staticmethod
    def generate(
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        match_id: str,
        league_id: str,
    ) -> List[Dict]:
        """
        生成双选腿。

        参数:
            home_odds/draw_odds/away_odds: 单选赔率
            home_prob/draw_prob/away_prob: 模型概率
            match_id: 比赛ID
            league_id: 联赛ID

        返回:
            双选腿列表
        """
        legs = []

        # 胜平(1X)
        if home_odds > 1 and draw_odds > 1:
            dc_odds = 1.0 / (1.0/home_odds + 1.0/draw_odds)
            dc_prob = home_prob + draw_prob
            dc_value = dc_prob - (1.0 / dc_odds)
            if dc_value > 0:
                legs.append({
                    "match_id": match_id,
                    "league_id": league_id,
                    "strategy": "1x2",
                    "direction": "1x",
                    "odds": dc_odds,
                    "model_prob": dc_prob,
                    "market_prob": 1.0 / dc_odds,
                    "value": dc_value,
                    "leg_type": "double_chance",
                })

        # 平负(X2)
        if draw_odds > 1 and away_odds > 1:
            dc_odds = 1.0 / (1.0/draw_odds + 1.0/away_odds)
            dc_prob = draw_prob + away_prob
            dc_value = dc_prob - (1.0 / dc_odds)
            if dc_value > 0:
                legs.append({
                    "match_id": match_id,
                    "league_id": league_id,
                    "strategy": "1x2",
                    "direction": "x2",
                    "odds": dc_odds,
                    "model_prob": dc_prob,
                    "market_prob": 1.0 / dc_odds,
                    "value": dc_value,
                    "leg_type": "double_chance",
                })

        # 胜负(12)
        if home_odds > 1 and away_odds > 1:
            dc_odds = 1.0 / (1.0/home_odds + 1.0/away_odds)
            dc_prob = home_prob + away_prob
            dc_value = dc_prob - (1.0 / dc_odds)
            if dc_value > 0:
                legs.append({
                    "match_id": match_id,
                    "league_id": league_id,
                    "strategy": "1x2",
                    "direction": "12",
                    "odds": dc_odds,
                    "model_prob": dc_prob,
                    "market_prob": 1.0 / dc_odds,
                    "value": dc_value,
                    "leg_type": "double_chance",
                })

        return legs


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def create_parlay_filter(league_id: str = "") -> ParlayFilter:
    """创建串关筛选器"""
    return ParlayFilter(league_id=league_id)


def generate_parlays_from_bets(
    bets: List[Dict],
    bankroll: float,
    league_id: str = "",
    max_parlays: int = 10,
) -> List[ParlayCombo]:
    """
    便捷函数: 从单场投注池生成串关。

    参数:
        bets: 单场投注列表
        bankroll: 资金
        league_id: 联赛ID
        max_parlays: 最大输出数量

    返回:
        ParlayCombo 列表
    """
    filter = ParlayFilter(league_id=league_id)
    legs = filter.filter_legs(bets)
    return filter.generate_parlays(legs, bankroll, max_parlays)
