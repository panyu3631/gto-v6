"""
GTO-GameFlow v5.10 — 统一资金管理器 (UnifiedBankrollManager)

将所有策略的 Kelly 计算、资金分配、风险控制统一到单一执行路径，
消除 v5.9 的 "资金碎片化" 问题。

v5.9 问题:
- 每个策略独立计算 Kelly stake，资金分配互相独立
- 同场比赛可能同时有 1X2、亚盘、大小球投注，总暴露不受控
- MPT 权重在策略层面分配，但单场层面没有统一限制

v5.10 统一方案:
- 单一 Kelly 引擎: 所有策略使用相同的 Kelly 分数和折扣
- 单场暴露上限: 同场比赛总投注 ≤ 资金池的 5%
- 日暴露上限: 单日总投注 ≤ 资金池的 20%
- 策略内权重: 在统一 Kelly 框架内按策略分配
- 动态 Kelly 分数: 根据回测表现自动调整

数据流:
    approved_proposals → unified_kelly → single_match_cap → daily_cap → final_stakes

使用方式:
    mgr = UnifiedBankrollManager(bankroll=100000.0, league_id="premier_league")
    staked = mgr.allocate(approved_proposals, daily_staked=5000.0)
    # staked → 带最终投注额的提案列表
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 联赛校准的 Kelly 参数
# ═══════════════════════════════════════════════════════════════

LEAGUE_KELLY_PARAMS = {
    "premier_league": {
        "kelly_fraction": 0.25,      # Kelly 折扣系数
        "max_single_match_pct": 0.05,  # 单场最大暴露 (资金池%)
        "max_daily_pct": 0.20,         # 单日最大暴露 (资金池%)
        "min_stake": 10.0,             # 最小投注额
        "max_stake": 5000.0,           # 单注最大投注额
        "max_concurrent_bets": 12,     # 最大并发投注数
    },
    "la_liga": {
        "kelly_fraction": 0.22,
        "max_single_match_pct": 0.04,
        "max_daily_pct": 0.18,
        "min_stake": 10.0,
        "max_stake": 4000.0,
        "max_concurrent_bets": 10,
    },
    "bundesliga": {
        "kelly_fraction": 0.25,
        "max_single_match_pct": 0.05,
        "max_daily_pct": 0.20,
        "min_stake": 10.0,
        "max_stake": 5000.0,
        "max_concurrent_bets": 12,
    },
    "serie_a": {
        "kelly_fraction": 0.20,
        "max_single_match_pct": 0.04,
        "max_daily_pct": 0.15,
        "min_stake": 10.0,
        "max_stake": 3000.0,
        "max_concurrent_bets": 8,
    },
    "ligue_1": {
        "kelly_fraction": 0.22,
        "max_single_match_pct": 0.04,
        "max_daily_pct": 0.18,
        "min_stake": 10.0,
        "max_stake": 4000.0,
        "max_concurrent_bets": 10,
    },
}

DEFAULT_KELLY_PARAMS = {
    "kelly_fraction": 0.25,
    "max_single_match_pct": 0.05,
    "max_daily_pct": 0.20,
    "min_stake": 10.0,
    "max_stake": 5000.0,
    "max_concurrent_bets": 10,
}


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class StakedProposal:
    """带最终投注额的提案"""
    match_id: str
    strategy: str
    selection: str
    odds: float
    model_prob: float
    value: float
    kelly_stake_raw: float      # 原始 Kelly 计算
    final_stake: float           # 经过所有限制后的最终投注额
    priority_score: float
    league_id: str = ""
    handicap_line: float = 0.0
    totals_line: float = 0.0
    original: Any = None

    # 风控标记
    capped_by_match: bool = False
    capped_by_daily: bool = False
    capped_by_max: bool = False


@dataclass 
class AllocationResult:
    """资金分配结果"""
    proposals: List[StakedProposal] = field(default_factory=list)
    total_staked: float = 0.0
    match_exposures: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "total_proposals": len(self.proposals),
            "total_staked": round(self.total_staked, 2),
            "by_strategy": self.stats.get("by_strategy", {}),
            "match_exposures": {k: round(v, 2) for k, v in self.match_exposures.items()},
        }


# ═══════════════════════════════════════════════════════════════
# 统一资金管理器
# ═══════════════════════════════════════════════════════════════

class UnifiedBankrollManager:
    """
    统一资金管理器 (v5.10)。

    所有策略通过同一 Kelly 引擎分配资金。

    核心逻辑:
    1. 统一 Kelly: 所有策略使用相同的 Kelly 分数
    2. 单场上限: 同场比赛总投注 ≤ max_single_match_pct × bankroll
    3. 日上限: 单日总投注 ≤ max_daily_pct × bankroll
    4. 单注上限: 单注 ≤ max_stake
    5. 单注下限: 单注 ≥ min_stake

    使用方式:
        mgr = UnifiedBankrollManager(bankroll=100000.0, league_id="premier_league")
        result = mgr.allocate(approved_proposals, daily_staked=5000.0)
        for p in result.proposals:
            print(f"{p.strategy} {p.selection}: stake={p.final_stake}")
    """

    def __init__(
        self,
        bankroll: float = 100000.0,
        league_id: str = "",
        kelly_fraction: Optional[float] = None,
        max_single_match_pct: Optional[float] = None,
        max_daily_pct: Optional[float] = None,
        min_stake: Optional[float] = None,
        max_stake: Optional[float] = None,
        max_concurrent_bets: Optional[int] = None,
    ):
        """
        参数:
            bankroll: 总资金池
            league_id: 联赛ID
            kelly_fraction: Kelly 折扣系数 (None=使用联赛默认值)
            max_single_match_pct: 单场最大暴露比例
            max_daily_pct: 单日最大暴露比例
            min_stake: 最小投注额
            max_stake: 单注最大投注额
            max_concurrent_bets: 最大并发投注数
        """
        self.bankroll = bankroll
        self.league_id = league_id
        params = LEAGUE_KELLY_PARAMS.get(league_id, DEFAULT_KELLY_PARAMS)

        self.kelly_fraction = kelly_fraction or params["kelly_fraction"]
        self.max_single_match_pct = max_single_match_pct or params["max_single_match_pct"]
        self.max_daily_pct = max_daily_pct or params["max_daily_pct"]
        self.min_stake = min_stake or params["min_stake"]
        self.max_stake = max_stake or params["max_stake"]
        self.max_concurrent_bets = max_concurrent_bets or params["max_concurrent_bets"]

        # 单注基准 (用于 Kelly 标准化)
        self.base_stake = self.bankroll * 0.01  # 1% of bankroll

    def allocate(
        self,
        proposals: List[Any],
        daily_staked: float = 0.0,
        concurrent_bets: int = 0,
    ) -> AllocationResult:
        """
        为所有提案分配资金。

        参数:
            proposals: 统一提案列表 (UnifiedProposal 或兼容)
            daily_staked: 当日已投注额
            concurrent_bets: 当前并发投注数

        返回:
            AllocationResult
        """
        result = AllocationResult()
        by_strategy: Dict[str, Dict[str, float]] = {}

        if not proposals:
            return result

        daily_remaining = self.bankroll * self.max_daily_pct - daily_staked
        if daily_remaining <= 0:
            logger.warning("当日投注额已满，跳过所有提案")
            result.stats = {"reason": "daily_limit_reached", "daily_staked": daily_staked}
            return result

        # ── 阶段 1: 原始 Kelly 计算 ──
        kelly_proposals: List[tuple] = []  # (StakedProposal, raw_stake)

        for p in proposals:
            # 提取通用字段
            odds = getattr(p, 'odds', 1.0)
            model_prob = getattr(p, 'model_prob', 0.0)
            value = getattr(p, 'value', 0.0)
            priority = getattr(p, 'priority_score', 0.0)
            match_id = getattr(p, 'match_id', '')
            strategy = getattr(p, 'strategy', '1x2')
            selection = getattr(p, 'selection', '')
            league_id = getattr(p, 'league_id', self.league_id)
            handicap = getattr(p, 'handicap_line', 0.0)
            totals = getattr(p, 'totals_line', 0.0)
            original = getattr(p, 'original', p)

            # 统一 Kelly 公式: f = (b × p - q) / b
            b = max(odds - 1.0, 0.01)
            q = 1.0 - model_prob
            f_kelly = max(0.0, (b * model_prob - q) / b)
            f_kelly *= self.kelly_fraction
            raw_stake = f_kelly * self.base_stake

            sp = StakedProposal(
                match_id=match_id,
                strategy=strategy,
                selection=selection,
                odds=odds,
                model_prob=model_prob,
                value=value,
                kelly_stake_raw=raw_stake,
                final_stake=raw_stake,
                priority_score=priority,
                league_id=league_id,
                handicap_line=handicap,
                totals_line=totals,
                original=original,
            )
            kelly_proposals.append((sp, raw_stake))

        # 按优先级排序
        kelly_proposals.sort(key=lambda x: x[0].priority_score, reverse=True)

        # ── 阶段 2: 单注上限 ──
        for sp, _ in kelly_proposals:
            if sp.final_stake > self.max_stake:
                sp.final_stake = self.max_stake
                sp.capped_by_max = True
            if sp.final_stake < self.min_stake:
                sp.final_stake = 0.0  # 不足最小投注额，跳过

        # ── 阶段 3: 并发限制 ──
        remaining_slots = max(0, self.max_concurrent_bets - concurrent_bets)
        if remaining_slots < len(kelly_proposals):
            # 取优先级最高的 remaining_slots 个
            kelly_proposals = kelly_proposals[:remaining_slots]

        # ── 阶段 4: 单场暴露上限 ──
        match_exposure: Dict[str, float] = {}
        for sp, _ in kelly_proposals:
            if sp.final_stake <= 0:
                continue
            current = match_exposure.get(sp.match_id, 0.0)
            max_match = self.bankroll * self.max_single_match_pct
            available = max(0.0, max_match - current)

            if sp.final_stake > available:
                if available >= self.min_stake:
                    sp.final_stake = available
                    sp.capped_by_match = True
                else:
                    sp.final_stake = 0.0

            match_exposure[sp.match_id] = current + sp.final_stake

        # ── 阶段 5: 日上限 ──
        total_raw = sum(sp.final_stake for sp, _ in kelly_proposals)
        if total_raw > daily_remaining:
            scale = daily_remaining / total_raw if total_raw > 0 else 0.0
            for sp, _ in kelly_proposals:
                sp.final_stake *= scale
                if sp.final_stake < self.min_stake:
                    sp.final_stake = 0.0
                sp.capped_by_daily = True

        # ── 阶段 6: 过滤零投注 ──
        final_proposals = []
        for sp, _ in kelly_proposals:
            if sp.final_stake >= self.min_stake:
                sp.final_stake = round(sp.final_stake, 2)
                final_proposals.append(sp)

                # 统计
                if sp.strategy not in by_strategy:
                    by_strategy[sp.strategy] = {"count": 0, "staked": 0.0}
                by_strategy[sp.strategy]["count"] += 1
                by_strategy[sp.strategy]["staked"] += sp.final_stake

        result.proposals = final_proposals
        result.total_staked = sum(sp.final_stake for sp in final_proposals)
        result.match_exposures = match_exposure
        result.stats = {
            "by_strategy": by_strategy,
            "total_staked": round(result.total_staked, 2),
            "bankroll": self.bankroll,
            "kelly_fraction": self.kelly_fraction,
            "daily_remaining": round(daily_remaining, 2),
            "capped_match": sum(1 for sp in final_proposals if sp.capped_by_match),
            "capped_daily": sum(1 for sp in final_proposals if sp.capped_by_daily),
            "capped_max": sum(1 for sp in final_proposals if sp.capped_by_max),
        }

        return result

    def update_bankroll(self, new_balance: float):
        """更新资金池余额"""
        self.bankroll = new_balance
        self.base_stake = self.bankroll * 0.01

    def get_kelly_params(self) -> dict:
        """获取当前 Kelly 参数"""
        return {
            "bankroll": self.bankroll,
            "kelly_fraction": self.kelly_fraction,
            "max_single_match_pct": self.max_single_match_pct,
            "max_daily_pct": self.max_daily_pct,
            "min_stake": self.min_stake,
            "max_stake": self.max_stake,
            "max_concurrent_bets": self.max_concurrent_bets,
            "base_stake": self.base_stake,
        }


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def create_bankroll_manager_for_league(
    league_id: str,
    bankroll: float = 100000.0,
) -> UnifiedBankrollManager:
    """为指定联赛创建校准后的资金管理器"""
    return UnifiedBankrollManager(bankroll=bankroll, league_id=league_id)


def get_league_kelly_params(league_id: str) -> dict:
    """获取联赛 Kelly 参数"""
    return LEAGUE_KELLY_PARAMS.get(league_id, DEFAULT_KELLY_PARAMS)