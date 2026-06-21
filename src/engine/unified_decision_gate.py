"""
GTO-GameFlow v5.10 — 统一决策门 (UnifiedDecisionGate)

将三种策略 (1X2 / 亚盘 / 大小球) 的投注决策统一到单一阈值系统，
消除 v5.9 的 "阈值碎片化" 问题。

v5.9 问题:
- 每个策略有独立的 value_threshold 和 confidence_threshold
- 亚盘 threshold=0.015 导致占比 82%，1X2 threshold=0.020 导致占比 10%
- 阈值碎片化导致策略间不可比，跷跷板效应

v5.10 统一方案:
- 单一价值阈值: 所有策略使用相同的 value_threshold_base
- 策略调整因子: 亚盘/大小球在统一阈值基础上乘以调整因子
- 联赛校准: 每个联赛独立学习最优阈值
- 统一优先级: 所有策略提案在同一尺度上排序
- 并发限制: 单日最多 N 笔投注，按优先级截断

数据流:
    各策略 proposals → 统一过滤 → 统一排序 → 截断 → 最终投注列表

使用方式:
    gate = UnifiedDecisionGate(league_id="premier_league")
    approved = gate.evaluate([
        BetProposal(value=0.025, strategy="1x2", ...),
        AsianHandicapProposal(value=0.018, strategy="asian_handicap", ...),
        TotalsProposal(value=0.022, strategy="over_under", ...),
    ])
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 联赛校准的阈值参数
# ═══════════════════════════════════════════════════════════════

# 来源: Walk-Forward 训练窗口网格搜索最优值
# v5.10.1: 降低 confidence_threshold 以免亚盘/大小球提案全部被拒
# v5.10.2: 不同策略的置信度尺度不同，使用 per-strategy 阈值
# v5.10.7: 恢复 1X2 策略 — 完整产品框架，三个策略并存
#   1X2 (三选一) 使用更高阈值 (x2_adjustment=2.0) 因随机胜率仅 33%
#   亚盘/大小球 (二选一) 使用较低阈值
LEAGUE_THRESHOLD_PARAMS = {
    "premier_league": {
        "value_threshold_base": 0.020,
        "confidence_threshold": 0.35,
        "asian_adjustment": 0.80,
        "totals_adjustment": 0.85,
        "x2_adjustment": 2.50,  # v5.10.8: 降低阈值 (2.50→2.25) 增加1x2投注量
        "min_prob_1x2": 0.30,
        "asian_confidence_threshold": 0.25,  # v5.10.8: 匹配真实 _compute_confidence 尺度 (0.1-1.0)
        "totals_confidence_threshold": 0.20,  # v5.10.8: 匹配真实 _compute_confidence 尺度
        "max_daily_bets": 12,
        "max_same_match_bets": 2,
    },
    "la_liga": {
        "value_threshold_base": 0.020,
        "confidence_threshold": 0.35,
        "asian_adjustment": 0.80,
        "totals_adjustment": 0.85,
        "x2_adjustment": 2.50,  # v5.10.8
        "min_prob_1x2": 0.30,
        "asian_confidence_threshold": 0.25,
        "totals_confidence_threshold": 0.20,
        "max_daily_bets": 10,
        "max_same_match_bets": 2,
    },
    "bundesliga": {
        "value_threshold_base": 0.020,
        "confidence_threshold": 0.35,
        "asian_adjustment": 0.80,
        "totals_adjustment": 0.85,
        "x2_adjustment": 2.50,  # v5.10.8
        "min_prob_1x2": 0.30,
        "asian_confidence_threshold": 0.25,
        "totals_confidence_threshold": 0.20,
        "max_daily_bets": 12,
        "max_same_match_bets": 2,
    },
    "serie_a": {
        "value_threshold_base": 0.018,
        "confidence_threshold": 0.30,
        "asian_adjustment": 0.75,
        "totals_adjustment": 0.80,
        "x2_adjustment": 2.50,  # v5.10.8
        "min_prob_1x2": 0.28,
        "asian_confidence_threshold": 0.22,
        "totals_confidence_threshold": 0.18,
        "max_daily_bets": 8,
        "max_same_match_bets": 2,
    },
    "ligue_1": {
        "value_threshold_base": 0.020,
        "confidence_threshold": 0.35,
        "asian_adjustment": 0.80,
        "totals_adjustment": 0.85,
        "x2_adjustment": 2.50,  # v5.10.8
        "min_prob_1x2": 0.30,
        "asian_confidence_threshold": 0.25,
        "totals_confidence_threshold": 0.20,
        "max_daily_bets": 10,
        "max_same_match_bets": 2,
    },
}

# 默认参数
DEFAULT_THRESHOLD_PARAMS = {
    "value_threshold_base": 0.020,
    "confidence_threshold": 0.35,
    "asian_adjustment": 0.80,
    "totals_adjustment": 0.85,
    "x2_adjustment": 2.50,  # v5.10.8
    "min_prob_1x2": 0.30,
    "asian_confidence_threshold": 0.25,
    "totals_confidence_threshold": 0.20,
    "max_daily_bets": 10,
    "max_same_match_bets": 2,
}


# ═══════════════════════════════════════════════════════════════
# 统一提案接口
# ═══════════════════════════════════════════════════════════════

@dataclass
class UnifiedProposal:
    """统一投注提案 — 所有策略归一化到此结构"""
    match_id: str
    strategy: str          # "1x2" / "asian_handicap" / "over_under"
    selection: str          # "home_win" / "draw" / "away_win" / "home" / "away" / "over" / "under"
    odds: float
    model_prob: float
    implied_prob: float
    value: float
    kelly_stake: float
    confidence: float
    priority_score: float
    league_id: str = ""

    # 原始提案引用 (用于后续结算)
    original: Any = None

    # 策略特定字段
    handicap_line: float = 0.0
    totals_line: float = 0.0


@dataclass
class DecisionResult:
    """统一决策门输出"""
    approved: List[UnifiedProposal] = field(default_factory=list)
    rejected: List[UnifiedProposal] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved_count(self) -> int:
        return len(self.approved)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def summary(self) -> dict:
        return {
            "approved": self.approved_count,
            "rejected": self.rejected_count,
            "by_strategy": self.stats.get("by_strategy", {}),
            "rejection_reasons": self.stats.get("rejection_reasons", {}),
        }


# ═══════════════════════════════════════════════════════════════
# 统一决策门
# ═══════════════════════════════════════════════════════════════

class UnifiedDecisionGate:
    """
    统一决策门 (v5.10)。

    所有策略的投注提案通过此门，使用统一的阈值和排序。

    核心逻辑:
    1. 策略调整: 对亚盘/大小球应用调整因子
    2. 价值过滤: 所有策略使用统一阈值
    3. 置信度过滤: 所有策略使用统一置信度阈值
    4. 统一排序: 按 priority_score 降序排列
    5. 并发限制: 单日最多 max_daily_bets 笔
    6. 同场限制: 同场最多 max_same_match_bets 笔

    使用方式:
        gate = UnifiedDecisionGate(league_id="premier_league")
        result = gate.evaluate(unified_proposals)
        # result.approved → 最终投注列表
    """

    def __init__(
        self,
        league_id: str = "",
        value_threshold_base: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        asian_adjustment: Optional[float] = None,
        totals_adjustment: Optional[float] = None,
        x2_adjustment: Optional[float] = None,
        min_prob_1x2: Optional[float] = None,
        asian_confidence_threshold: Optional[float] = None,
        totals_confidence_threshold: Optional[float] = None,
        max_daily_bets: Optional[int] = None,
        max_same_match_bets: Optional[int] = None,
    ):
        """
        参数:
            league_id: 联赛ID
            value_threshold_base: 基础价值阈值 (None=使用联赛默认值)
            confidence_threshold: 1X2置信度阈值
            asian_adjustment: 亚盘调整因子
            totals_adjustment: 大小球调整因子
            x2_adjustment: 1X2调整因子 (v5.10.5: 三选一需要更高阈值)
            min_prob_1x2: 1X2最小模型概率 (v5.10.5: 低于此值不投注)
            asian_confidence_threshold: 亚盘置信度阈值 (不同尺度)
            totals_confidence_threshold: 大小球置信度阈值 (不同尺度)
            max_daily_bets: 单日最大投注数
            max_same_match_bets: 同场最大投注数
        """
        self.league_id = league_id
        params = LEAGUE_THRESHOLD_PARAMS.get(league_id, DEFAULT_THRESHOLD_PARAMS)

        self.value_threshold_base = value_threshold_base or params["value_threshold_base"]
        self.confidence_threshold = confidence_threshold or params["confidence_threshold"]
        self.asian_adjustment = asian_adjustment or params["asian_adjustment"]
        self.totals_adjustment = totals_adjustment or params["totals_adjustment"]
        self.x2_adjustment = x2_adjustment or params.get("x2_adjustment", 2.0)
        self.min_prob_1x2 = min_prob_1x2 or params.get("min_prob_1x2", 0.30)
        self.asian_confidence_threshold = asian_confidence_threshold or params.get("asian_confidence_threshold", 0.02)
        self.totals_confidence_threshold = totals_confidence_threshold or params.get("totals_confidence_threshold", 0.002)
        self.max_daily_bets = max_daily_bets or params["max_daily_bets"]
        self.max_same_match_bets = max_same_match_bets or params["max_same_match_bets"]

    def evaluate(
        self,
        proposals: List[UnifiedProposal],
        daily_bet_count: int = 0,
    ) -> DecisionResult:
        """
        评估所有提案，返回通过/拒绝的决策。

        参数:
            proposals: 统一提案列表
            daily_bet_count: 当日已投注数

        返回:
            DecisionResult
        """
        result = DecisionResult()
        rejection_reasons: Dict[str, int] = {}
        by_strategy: Dict[str, Dict[str, int]] = {}

        # ── 阶段 1: 策略调整 + 价值过滤 ──
        for p in proposals:
            strategy_key = p.strategy
            if strategy_key not in by_strategy:
                by_strategy[strategy_key] = {"submitted": 0, "approved": 0, "rejected_value": 0,
                                              "rejected_confidence": 0, "rejected_limit": 0}

            by_strategy[strategy_key]["submitted"] += 1

            # 策略调整后的阈值
            if strategy_key == "asian_handicap":
                effective_threshold = self.value_threshold_base * self.asian_adjustment
                effective_confidence = self.asian_confidence_threshold
            elif strategy_key == "over_under":
                effective_threshold = self.value_threshold_base * self.totals_adjustment
                effective_confidence = self.totals_confidence_threshold
            else:
                # 1X2: 使用更高的价值阈值 (三选一需要更强信号)
                effective_threshold = self.value_threshold_base * self.x2_adjustment
                effective_confidence = self.confidence_threshold
                # v5.10.5: 1X2 最小概率过滤 — 模型概率低于阈值不投注
                if p.model_prob < self.min_prob_1x2:
                    result.rejected.append(p)
                    by_strategy[strategy_key]["rejected_value"] += 1
                    reason = "min_prob_1x2_below_threshold"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    continue

            # 价值过滤
            if p.value < effective_threshold:
                result.rejected.append(p)
                by_strategy[strategy_key]["rejected_value"] += 1
                reason = f"value_below_threshold_{strategy_key}"
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            # 置信度过滤 (per-strategy 阈值)
            if p.confidence < effective_confidence:
                result.rejected.append(p)
                by_strategy[strategy_key]["rejected_confidence"] += 1
                reason = "confidence_below_threshold"
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            result.approved.append(p)

        # ── 阶段 2: 统一排序 ──
        result.approved.sort(key=lambda p: p.priority_score, reverse=True)

        # ── 阶段 3: 并发限制 ──
        remaining_slots = max(0, self.max_daily_bets - daily_bet_count)
        if remaining_slots < len(result.approved):
            cut = result.approved[remaining_slots:]
            result.approved = result.approved[:remaining_slots]
            for p in cut:
                result.rejected.append(p)
                by_strategy.get(p.strategy, {}).setdefault("rejected_limit", 0)
                by_strategy[p.strategy]["rejected_limit"] += 1
            rejection_reasons["daily_limit"] = len(cut)

        # ── 阶段 4: 同场限制 ──
        match_counts: Dict[str, int] = {}
        final_approved: List[UnifiedProposal] = []
        for p in result.approved:
            cnt = match_counts.get(p.match_id, 0)
            if cnt >= self.max_same_match_bets:
                result.rejected.append(p)
                by_strategy.get(p.strategy, {}).setdefault("rejected_limit", 0)
                by_strategy[p.strategy]["rejected_limit"] += 1
                rejection_reasons["same_match_limit"] = rejection_reasons.get("same_match_limit", 0) + 1
            else:
                final_approved.append(p)
                match_counts[p.match_id] = cnt + 1

        result.approved = final_approved

        # 统计
        for key in by_strategy:
            by_strategy[key]["approved"] = len(
                [p for p in result.approved if p.strategy == key]
            )

        result.stats = {
            "by_strategy": by_strategy,
            "rejection_reasons": rejection_reasons,
            "effective_thresholds": {
                "1x2": self.value_threshold_base,
                "asian_handicap": self.value_threshold_base * self.asian_adjustment,
                "over_under": self.value_threshold_base * self.totals_adjustment,
            },
            "confidence_thresholds": {
                "1x2": self.confidence_threshold,
                "asian_handicap": self.asian_confidence_threshold,
                "over_under": self.totals_confidence_threshold,
            },
            "max_daily_bets": self.max_daily_bets,
        }

        return result

    def get_effective_threshold(self, strategy: str) -> float:
        """获取某策略的有效价值阈值"""
        if strategy == "asian_handicap":
            return self.value_threshold_base * self.asian_adjustment
        elif strategy == "over_under":
            return self.value_threshold_base * self.totals_adjustment
        return self.value_threshold_base


# ═══════════════════════════════════════════════════════════════
# 便捷函数: 从原始提案转换为统一提案
# ═══════════════════════════════════════════════════════════════

def proposals_to_unified(
    x2_proposals: List[Any] = None,
    asian_proposals: List[Any] = None,
    totals_proposals: List[Any] = None,
) -> List[UnifiedProposal]:
    """
    将各策略的原始提案转换为统一提案格式。

    参数:
        x2_proposals: BetProposal 列表
        asian_proposals: AsianHandicapProposal 列表
        totals_proposals: TotalsProposal 列表

    返回:
        UnifiedProposal 列表
    """
    unified: List[UnifiedProposal] = []

    # 1X2
    for p in (x2_proposals or []):
        confidence = getattr(p, 'priority_score', 0.0)
        unified.append(UnifiedProposal(
            match_id=p.match_id,
            strategy="1x2",
            selection=p.selection.value if hasattr(p.selection, 'value') else str(p.selection),
            odds=p.odds,
            model_prob=p.model_prob,
            implied_prob=p.implied_prob,
            value=p.value,
            kelly_stake=p.kelly_stake,
            confidence=confidence,
            priority_score=p.priority_score,
            league_id=p.league_id,
            original=p,
        ))

    # 亚盘
    for p in (asian_proposals or []):
        confidence = getattr(p, 'confidence', None)
        if confidence is None or confidence <= 0:
            confidence = getattr(p, 'priority_score', 0.0)
        unified.append(UnifiedProposal(
            match_id=p.match_id,
            strategy="asian_handicap",
            selection=p.side,
            odds=p.odds,
            model_prob=p.cover_prob if hasattr(p, 'cover_prob') else p.model_prob,
            implied_prob=p.implied_prob,
            value=p.value,
            kelly_stake=p.kelly_stake,
            confidence=confidence,
            priority_score=p.priority_score,
            league_id=p.league_id,
            handicap_line=getattr(p, 'handicap_line', 0.0),
            original=p,
        ))

    # 大小球
    for p in (totals_proposals or []):
        confidence = getattr(p, 'confidence', None)
        if confidence is None or confidence <= 0:
            confidence = getattr(p, 'priority_score', 0.0)
        model_prob = p.over_prob if hasattr(p, 'over_prob') else p.model_prob
        unified.append(UnifiedProposal(
            match_id=p.match_id,
            strategy="over_under",
            selection=p.side,
            odds=p.odds,
            model_prob=model_prob,
            implied_prob=p.implied_prob,
            value=p.value,
            kelly_stake=p.kelly_stake,
            confidence=confidence,
            priority_score=p.priority_score,
            league_id=p.league_id,
            totals_line=getattr(p, 'totals_line', 0.0),
            original=p,
        ))

    return unified


def create_decision_gate_for_league(league_id: str) -> UnifiedDecisionGate:
    """为指定联赛创建校准后的决策门"""
    return UnifiedDecisionGate(league_id=league_id)


def get_league_thresholds(league_id: str) -> dict:
    """获取联赛阈值参数"""
    return LEAGUE_THRESHOLD_PARAMS.get(league_id, DEFAULT_THRESHOLD_PARAMS)