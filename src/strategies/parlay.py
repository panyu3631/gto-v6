"""
GTO-GameFlow v5.9.3 — 串关策略 (Parlay / Accumulator) — 重新设计

核心改进:
1. 仅支持 2串1 (max_legs=2)，降低复杂度，提高命中率
2. 相关性感知定价: 不再简单 Π prob_i，而是对同联赛/同策略的腿施加相关性惩罚
3. 质量门槛: 单腿价值 ≥ 0.03，置信度 ≥ 0.55
4. 跨联赛优先: 跨联赛串关权重更高
5. 动态 Kelly: 基于实际胜率跟踪调整

风险控制:
- 单注最大占比: 资金池的 0.5% (串关风险远高于单场)
- 组合最低赔率: 2.5
- 同一比赛不能出现在同一串关中
"""
import math
import itertools
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

from ..data.models import BetProposal, BetSelection, BetResult, BetPlacement


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParlayLeg:
    """串关中的一条腿"""
    match_id: str
    selection: str              # 投注方向
    odds: float                 # 该场赔率
    model_prob: float           # 模型概率
    value: float                # 单场价值
    confidence: float = 0.0     # 模型置信度
    league_id: str = ""
    strategy_type: str = "1x2"
    totals_line: float = 0.0    # 大小球线 (仅 over_under 策略)


@dataclass
class ParlayProposal:
    """串关投注建议"""
    parlay_id: str
    legs: List[ParlayLeg] = field(default_factory=list)
    n_legs: int = 0
    combined_odds: float = 1.0          # 组合赔率 = Π odds_i
    combined_prob: float = 1.0          # 相关性调整后的模型组合概率
    raw_combined_prob: float = 1.0      # 原始 Π prob_i (未调整)
    combined_value: float = 0.0         # 组合价值
    kelly_stake: float = 0.0
    adjusted_stake: float = 0.0
    priority_score: float = 0.0
    correlation_penalty: float = 1.0    # 相关性惩罚系数
    cross_league: bool = False          # 是否跨联赛
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def match_ids(self) -> Set[str]:
        return {leg.match_id for leg in self.legs}

    @property
    def expected_return(self) -> float:
        """预期回报 = 组合概率 × 组合赔率"""
        return self.combined_prob * self.combined_odds

    def summary(self) -> dict:
        return {
            "parlay_id": self.parlay_id,
            "n_legs": self.n_legs,
            "matches": [leg.match_id for leg in self.legs],
            "combined_odds": round(self.combined_odds, 2),
            "combined_prob": round(self.combined_prob, 4),
            "combined_value": round(self.combined_value, 4),
            "kelly_stake": round(self.kelly_stake, 2),
            "adjusted_stake": round(self.adjusted_stake, 2),
            "cross_league": self.cross_league,
            "corr_penalty": round(self.correlation_penalty, 4),
        }


@dataclass
class ParlaySettlement:
    """串关结算结果"""
    parlay_id: str
    won: bool
    stake: float
    returned: float                 # 0 或 stake × combined_odds
    profit: float
    legs_won: int = 0
    legs_total: int = 0
    settled_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 串关策略引擎 (重新设计)
# ═══════════════════════════════════════════════════════════════

class ParlayStrategy:
    """
    串关策略引擎 v5.9.3。

    重新设计要点:
    - 仅支持 2串1，降低复杂度
    - 相关性感知: 同联赛/同策略施加概率惩罚
    - 跨联赛优先: 跨联赛组合权重更高
    - 质量门槛: 单腿价值 ≥ 0.03，置信度 ≥ 0.55
    """

    def __init__(
        self,
        max_legs: int = 2,                     # 仅支持 2串1
        min_single_value: float = 0.03,        # 提高单腿价值门槛
        min_combined_value: float = 0.05,       # 提高组合价值门槛
        min_combined_odds: float = 2.5,         # 最低组合赔率
        max_single_exposure: float = 0.005,     # 降至 0.5%
        max_total_exposure: float = 0.05,       # 总串关暴露降至 5%
        min_odds_per_leg: float = 1.20,         # 最低单腿赔率
        max_odds_per_leg: float = 4.00,         # 最高单腿赔率
        kelly_discount: float = 0.25,           # Kelly 折扣
        corr_penalty_same_league: float = 0.85,  # 同联赛相关性惩罚
        corr_penalty_same_strategy: float = 0.90, # 同策略相关性惩罚
        min_confidence: float = 0.55,            # 最低置信度
    ):
        self.max_legs = max_legs
        self.min_single_value = min_single_value
        self.min_combined_value = min_combined_value
        self.min_combined_odds = min_combined_odds
        self.max_single_exposure = max_single_exposure
        self.max_total_exposure = max_total_exposure
        self.min_odds_per_leg = min_odds_per_leg
        self.max_odds_per_leg = max_odds_per_leg
        self.kelly_discount = kelly_discount
        self.corr_penalty_same_league = corr_penalty_same_league
        self.corr_penalty_same_strategy = corr_penalty_same_strategy
        self.min_confidence = min_confidence

        # 胜率跟踪
        self._total_parlays = 0
        self._total_wins = 0

    def generate_parlays(
        self,
        single_bets: List[BetProposal],
        bankroll: float,
        kelly_discount: Optional[float] = None,
    ) -> List[ParlayProposal]:
        """
        从单场投注池生成串关组合。

        参数:
            single_bets: 单场投注建议列表 (来自 1X2/亚盘/大小球)
            bankroll: 当前资金池
            kelly_discount: Kelly 折扣 (None=使用默认)

        返回:
            串关投注建议列表 (按优先级降序)
        """
        if kelly_discount is None:
            kelly_discount = self.kelly_discount

        # Step 1: 严格筛选 (质量门槛)
        eligible = self._filter_eligible(single_bets)
        if len(eligible) < 2:
            return []

        # Step 2: 每场比赛只取最优方向
        eligible = self._deduplicate_per_match(eligible, max_per_match=1)
        eligible.sort(key=lambda x: x.value, reverse=True)
        eligible = eligible[:20]  # 限制池大小

        # Step 3: 生成 2串1 组合
        all_parlays = []
        for combo in itertools.combinations(eligible, 2):
            # 检查: 同一比赛不能出现两次
            match_ids = [b.match_id for b in combo]
            if len(set(match_ids)) < 2:
                continue

            parlay = self._build_parlay(combo, bankroll, kelly_discount)
            if parlay:
                all_parlays.append(parlay)

        # Step 4: 按优先级排序
        # 跨联赛组合权重更高
        all_parlays.sort(key=lambda x: x.priority_score, reverse=True)

        # Step 5: 去重
        seen = set()
        deduped = []
        for p in all_parlays:
            key = frozenset(p.match_ids)
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        # 限制输出数量
        return deduped[:10]

    def _filter_eligible(
        self,
        single_bets: List[BetProposal],
    ) -> List[BetProposal]:
        """严格筛选符合串关条件的单场投注"""
        eligible = []
        for b in single_bets:
            if b.value < self.min_single_value:
                continue
            if b.odds < self.min_odds_per_leg:
                continue
            if b.odds > self.max_odds_per_leg:
                continue
            # 置信度检查
            if hasattr(b, 'priority_score') and b.priority_score < self.min_confidence:
                continue
            eligible.append(b)
        return eligible

    def _deduplicate_per_match(
        self,
        bets: List[BetProposal],
        max_per_match: int = 1,
    ) -> List[BetProposal]:
        """每场比赛只保留最优方向"""
        best: Dict[str, BetProposal] = {}
        for b in bets:
            if b.match_id not in best or b.value > best[b.match_id].value:
                best[b.match_id] = b
        return list(best.values())

    def _compute_correlation_penalty(
        self,
        legs: List[ParlayLeg],
    ) -> Tuple[float, bool]:
        """
        计算相关性惩罚系数。

        惩罚规则:
        - 同联赛: 乘以 corr_penalty_same_league
        - 同策略: 乘以 corr_penalty_same_strategy
        - 跨联赛: 不惩罚，且返回 cross_league=True

        返回:
            (penalty, cross_league)
        """
        penalty = 1.0
        leagues = {leg.league_id for leg in legs}
        strategies = {leg.strategy_type for leg in legs}

        cross_league = len(leagues) > 1

        if len(leagues) == 1:
            # 同联赛惩罚
            penalty *= self.corr_penalty_same_league

        if len(strategies) == 1:
            # 同策略惩罚
            penalty *= self.corr_penalty_same_strategy

        return penalty, cross_league

    def _build_parlay(
        self,
        combo: Tuple[BetProposal, ...],
        bankroll: float,
        kelly_discount: float,
    ) -> Optional[ParlayProposal]:
        """构建单个 2串1 建议"""
        legs = []
        combined_odds = 1.0
        raw_combined_prob = 1.0

        for b in combo:
            confidence = getattr(b, 'priority_score', 0.0)
            leg = ParlayLeg(
                match_id=b.match_id,
                selection=b.selection.value,
                odds=b.odds,
                model_prob=b.model_prob,
                value=b.value,
                confidence=confidence,
                league_id=b.league_id,
                strategy_type=b.strategy_type,
                totals_line=b.totals_line if b.strategy_type == "over_under" else 0.0,
            )
            legs.append(leg)
            combined_odds *= b.odds
            raw_combined_prob *= b.model_prob

        # 最低组合赔率约束
        if combined_odds < self.min_combined_odds:
            return None

        # 相关性惩罚
        corr_penalty, cross_league = self._compute_correlation_penalty(legs)
        combined_prob = raw_combined_prob * corr_penalty

        # 组合价值 (使用去 margin 的公平概率)
        combined_value = combined_prob - (1.0 / combined_odds)

        if combined_value < self.min_combined_value:
            return None

        # 串关 Kelly 计算
        b_net = combined_odds - 1.0
        if b_net <= 0:
            return None

        f_kelly = max(0.0, (b_net * combined_prob - (1.0 - combined_prob)) / b_net)

        # 动态 Kelly 调整: 基于实际胜率
        if self._total_parlays >= 10:
            actual_win_rate = self._total_wins / self._total_parlays
            expected_win_rate = combined_prob
            # 如果实际胜率低于预期，额外打折
            if actual_win_rate < expected_win_rate * 0.5:
                kelly_discount *= 0.5

        # 2串1 折扣: 0.65x
        leg_discount = 0.65
        f_actual = f_kelly * kelly_discount * leg_discount

        # 跨联赛奖励: 跨联赛组合 Kelly 不打折
        if cross_league:
            f_actual = f_kelly * kelly_discount * 0.80  # 跨联赛 0.80x 而非 0.65x

        kelly_stake = f_actual * bankroll

        # 单注上限 (0.5%)
        max_stake = bankroll * self.max_single_exposure
        adjusted_stake = min(kelly_stake, max_stake)

        if adjusted_stake < 0.5:
            return None

        # 优先级: 跨联赛加权
        cross_league_bonus = 1.15 if cross_league else 1.0
        priority = combined_value * combined_prob * cross_league_bonus

        parlay_id = f"parlay_2x1_{hash(frozenset(b.match_id for b in combo)) % 100000:05d}"

        return ParlayProposal(
            parlay_id=parlay_id,
            legs=legs,
            n_legs=2,
            combined_odds=combined_odds,
            combined_prob=combined_prob,
            raw_combined_prob=raw_combined_prob,
            combined_value=combined_value,
            kelly_stake=kelly_stake,
            adjusted_stake=adjusted_stake,
            priority_score=priority,
            correlation_penalty=corr_penalty,
            cross_league=cross_league,
        )

    def settle(
        self,
        proposal: ParlayProposal,
        match_results: Dict[str, Tuple[str, str]],  # {match_id: (actual_outcome, actual_selection, goals)}
    ) -> ParlaySettlement:
        """
        结算串关。

        参数:
            proposal: 串关建议
            match_results: {match_id: (actual_outcome_str, actual_selection, actual_goals)}
        """
        all_won = True
        legs_won = 0

        for leg in proposal.legs:
            if leg.match_id not in match_results:
                all_won = False
                continue

            result_data = match_results[leg.match_id]
            actual_outcome = result_data[0]

            # 判断该腿是否赢: 大小球需要特殊处理
            if leg.selection in ("over", "under"):
                if len(result_data) >= 3 and result_data[2] is not None:
                    home_g, away_g = result_data[2]
                    total_goals = home_g + away_g
                    totals_line = getattr(leg, 'totals_line', 2.5)
                    if leg.selection == "over":
                        leg_won = total_goals > totals_line
                    else:
                        leg_won = total_goals < totals_line
                else:
                    leg_won = False
            else:
                leg_won = (actual_outcome == leg.selection)

            if leg_won:
                legs_won += 1
            else:
                all_won = False

        if all_won and legs_won == proposal.n_legs:
            returned = proposal.adjusted_stake * proposal.combined_odds
            won = True
        else:
            returned = 0.0
            won = False

        profit = returned - proposal.adjusted_stake

        # 更新胜率跟踪
        self._total_parlays += 1
        if won:
            self._total_wins += 1

        return ParlaySettlement(
            parlay_id=proposal.parlay_id,
            won=won,
            stake=proposal.adjusted_stake,
            returned=returned,
            profit=profit,
            legs_won=legs_won,
            legs_total=proposal.n_legs,
        )

    def apply_stake_cap(
        self,
        proposals: List[ParlayProposal],
        bankroll: float,
    ) -> List[ParlayProposal]:
        """应用总曝光上限。"""
        total_stake = sum(p.adjusted_stake for p in proposals)
        max_total = bankroll * self.max_total_exposure

        if total_stake > max_total:
            scale = max_total / total_stake
            for p in proposals:
                p.adjusted_stake *= scale

        return proposals

    @property
    def win_rate(self) -> float:
        """实际串关胜率"""
        if self._total_parlays == 0:
            return 0.0
        return self._total_wins / self._total_parlays


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def collect_match_bets(
    all_proposals: List[BetProposal],
    max_per_match: int = 1,
) -> List[BetProposal]:
    """从单场建议池中收集用于串关的场次 (每场取最优方向)。"""
    match_bets: Dict[str, BetProposal] = {}
    for p in all_proposals:
        if p.match_id not in match_bets or p.value > match_bets[p.match_id].value:
            match_bets[p.match_id] = p
    return list(match_bets.values())