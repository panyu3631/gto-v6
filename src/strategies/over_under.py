"""
GTO-GameFlow v5.5 — 大小球策略引擎

基于泊松比分矩阵, 独立计算 Over/Under 各线的概率。
与 1X2 策略低相关性, 分散风险。

核心概念:
- 大小球线: 2.0/2.5/3.0/3.5 等
- 从 ScoreMatrix 推导总进球分布: P(总进球=k) = Σ P(home=i, away=j) 其中 i+j=k
- Over 概率 = P(总进球 > line)
- Under 概率 = P(总进球 < line)
- 整数线 (2.0/3.0): 等于时走水

使用方式:
    from src.strategies import OverUnderStrategy

    strategy = OverUnderStrategy(league_id="bundesliga")
    proposals = strategy.analyze(
        score_matrix=score_matrix,
        totals_odds={2.5: {"over": 1.90, "under": 2.00}},
        match_id="BAYvsDOR",
    )
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from ..data.models import ScoreMatrix, TotalsProposal, TotalsDistribution

logger = logging.getLogger(__name__)


# 标准大小球线
STANDARD_TOTALS_LINES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


class OverUnderStrategy:
    """
    大小球策略引擎。

    从比分概率矩阵推导总进球分布, 计算 Over/Under 概率。

    与 1X2 的独立性:
    - 1X2 关注胜平负方向, 大小球关注总进球数
    - 3:2 和 2:3 的 1X2 结果相反, 但大小球都是 Over 2.5
    - 1:0 和 0:0 的 1X2 结果不同, 但大小球都是 Under 2.5
    - 这种正交性使得大小球成为优秀的分散化策略

    参数:
        league_id: 联赛ID
        value_threshold: 最小价值阈值
        min_odds: 最小赔率
        max_odds: 最大赔率
    """

    def __init__(
        self,
        league_id: str = "",
        value_threshold: float = 0.015,
        confidence_threshold: float = 0.40,
        min_odds: float = 1.50,
        max_odds: float = 3.00,
        dispersion: float = 0.15,
    ):
        self.league_id = league_id
        self.value_threshold = value_threshold
        self.confidence_threshold = confidence_threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.dispersion = dispersion

    def analyze(
        self,
        score_matrix: ScoreMatrix,
        totals_odds: Dict[float, Dict[str, float]],
        match_id: str = "",
        league_id: str = "",
        kelly_discount: float = 0.25,
        strip_margin: bool = True,
    ) -> List[TotalsProposal]:
        """
        分析所有可用大小球线的投注机会。

        参数:
            score_matrix: 泊松比分概率矩阵
            totals_odds: {大小球线: {"over": 赔率, "under": 赔率}} 的字典
            match_id: 比赛ID
            league_id: 联赛ID
            kelly_discount: Kelly 折扣系数
            strip_margin: 是否剥离庄家 margin (合成赔率应设为 False)

        返回:
            TotalsProposal 列表 (按优先级排序)
        """
        # 构建总进球分布
        totals_dist = self._build_totals_distribution(score_matrix)

        proposals: List[TotalsProposal] = []

        for line, odds_dict in totals_odds.items():
            over_odds = odds_dict.get("over", 0)
            under_odds = odds_dict.get("under", 0)

            if not (self.min_odds <= over_odds <= self.max_odds
                    and self.min_odds <= under_odds <= self.max_odds):
                continue

            over_prob = totals_dist.over_prob(line)
            under_prob = totals_dist.under_prob(line)
            exact_prob = totals_dist.exact_prob(line)

            # Over 投注
            if over_odds > 0:
                prop = self._build_proposal(
                    match_id, line, "over", over_odds, over_prob,
                    exact_prob, totals_dist, league_id, kelly_discount,
                    opposite_odds=under_odds if strip_margin else 0.0,
                )
                if prop and prop.value >= self.value_threshold:
                    proposals.append(prop)

            # Under 投注
            if under_odds > 0:
                prop = self._build_proposal(
                    match_id, line, "under", under_odds, under_prob,
                    exact_prob, totals_dist, league_id, kelly_discount,
                    opposite_odds=over_odds if strip_margin else 0.0,
                )
                if prop and prop.value >= self.value_threshold:
                    proposals.append(prop)

        proposals.sort(key=lambda p: p.priority_score, reverse=True)
        return proposals

    def _build_totals_distribution(
        self,
        score_matrix: ScoreMatrix,
    ) -> TotalsDistribution:
        """
        从比分矩阵构建总进球分布 (含 over-dispersion 修正)。

        泊松分布假设均值=方差，但足球进球存在 over-dispersion。
        引入 15% 概率质量扩散到尾部，修正低估极端比分的偏差。
        """
        dist: Dict[int, float] = {}
        total_p = 0.0

        for (h, a), prob in score_matrix.matrix.items():
            total = h + a
            dist[total] = dist.get(total, 0.0) + prob
            total_p += prob

        # 归一化
        if total_p > 0:
            for k in dist:
                dist[k] /= total_p

        # 截断到合理范围 (0-10球)
        max_goals = min(10, max(dist.keys()) if dist else 5)
        for g in range(max_goals + 1):
            if g not in dist:
                dist[g] = 0.0

        # Over-dispersion 修正: 将概率质量从中心扩散到尾部
        # 模拟真实足球比分的 fat-tail 特性
        # v5.9.4 (FIX): 两阶段修正, 避免覆盖写入 bug
        # 第一阶段: 计算每个 bin 的净变化量
        original = dict(dist)
        delta: Dict[int, float] = {}
        for g in range(max_goals + 1):
            delta[g] = 0.0
        
        for g in range(max_goals + 1):
            leaked = original.get(g, 0.0) * self.dispersion
            delta[g] -= leaked  # 从当前 bin 减去
            # 扩散到相邻区间
            if g >= 2:
                delta[g - 2] = delta.get(g - 2, 0.0) + leaked * 0.15
            if g >= 1:
                delta[g - 1] = delta.get(g - 1, 0.0) + leaked * 0.25
            if g + 1 <= max_goals:
                delta[g + 1] = delta.get(g + 1, 0.0) + leaked * 0.25
            if g + 2 <= max_goals:
                delta[g + 2] = delta.get(g + 2, 0.0) + leaked * 0.15
            # 剩余 20% 泄漏到极端值 (0 和 max_goals)
            delta[0] = delta.get(0, 0.0) + leaked * 0.10
            delta[max_goals] = delta.get(max_goals, 0.0) + leaked * 0.10
        
        # 第二阶段: 一次性应用所有变化, 避免覆盖
        for g in range(max_goals + 1):
            dist[g] = max(0.0, original.get(g, 0.0) + delta.get(g, 0.0))
        
        # 重新归一化 (确保概率和为 1)
        total = sum(dist.values())
        if total > 0:
            for g in dist:
                dist[g] /= total

        # 计算平均进球数
        avg_goals = sum(k * p for k, p in dist.items())

        return TotalsDistribution(
            league_id=score_matrix.league_id,
            avg_goals=avg_goals,
            distribution=dist,
        )

    def _build_proposal(
        self,
        match_id: str,
        line: float,
        side: str,
        odds: float,
        target_prob: float,
        exact_prob: float,
        totals_dist: TotalsDistribution,
        league_id: str,
        kelly_discount: float,
        opposite_odds: float = 0.0,
    ) -> Optional[TotalsProposal]:
        """构建大小球投注建议"""
        # v5.9.4: 剥离庄家 margin, 获取公平隐含概率
        if opposite_odds > 0:
            total_implied = 1.0 / odds + 1.0 / opposite_odds
            if total_implied > 0:
                fair_implied = (1.0 / odds) / total_implied
            else:
                fair_implied = 1.0 / odds
        else:
            # 合成赔率: 直接使用隐含概率
            fair_implied = 1.0 / odds

        # 整数线处理: 走水概率折半
        is_integer_line = (line == int(line))
        if is_integer_line and exact_prob > 0:
            # 整数线: 概率 = target_prob + 0.5 * exact_prob (走水退本金, 相当于半赢)
            effective_prob = target_prob + 0.5 * exact_prob
        else:
            effective_prob = target_prob

        value = effective_prob - fair_implied

        if value < self.value_threshold:
            return None

        # 置信度过滤 (v5.9.4: 新增)
        confidence = self._compute_confidence(target_prob, odds, value, totals_dist)
        if confidence < self.confidence_threshold:
            return None

        # Kelly (v5.8: 整数线走水修正)
        b = odds - 1.0
        if is_integer_line and exact_prob > 0:
            # v5.8 (FIND-016): 整数线 Kelly 修正
            # P(win) = target_prob, P(push) = exact_prob, P(loss) = 1 - P(win) - P(push)
            # Kelly: f = (b * p_win - p_loss) / b
            # push 折半入有效概率, 但 Kelly 分母仍用 p_win + p_loss
            p_loss = 1.0 - target_prob - exact_prob
            f_kelly = max(0.0, (b * target_prob - p_loss) / max(b, 0.01))
        else:
            f_kelly = max(0.0, (b * effective_prob - (1 - effective_prob)) / max(b, 0.01))
        f_kelly *= kelly_discount
        kelly_stake = f_kelly * 10000

        # 优先级
        priority = f_kelly * max(0, value) * confidence

        return TotalsProposal(
            match_id=match_id,
            totals_line=line,
            side=side,
            odds=odds,
            over_prob=round(target_prob, 4),
            implied_prob=round(fair_implied, 4),
            value=round(value, 4),
            kelly_stake=round(kelly_stake, 2),
            adjusted_stake=round(kelly_stake, 2),
            priority_score=round(priority, 4),
            league_id=league_id or self.league_id,
            confidence=round(confidence, 4),  # v5.10.8
        )

    def _compute_confidence(
        self,
        prob: float,
        odds: float,
        value: float,
        totals_dist: TotalsDistribution,
    ) -> float:
        """
        大小球置信度计算。

        特殊考虑:
        - 联赛平均进球数: 进球数波动大的联赛置信度降低
        - 概率极端性: 接近 50% 时不确定性大
        """
        base = 0.7

        # 联赛进球波动
        avg = totals_dist.avg_goals
        variance = sum(
            (k - avg) ** 2 * p for k, p in totals_dist.distribution.items()
        )
        volatility = math.sqrt(max(variance, 0.1))
        # 波动越大, 置信度越低
        volatility_penalty = max(0.5, 1.0 - (volatility - 2.0) * 0.1)
        base *= volatility_penalty

        # 概率极端性
        prob_penalty = 1.0 - 2.0 * abs(prob - 0.5)
        base *= (0.6 + 0.4 * prob_penalty)

        # 价值奖励
        value_bonus = min(1.0, max(0.0, value * 10.0))
        base *= (0.8 + 0.2 * value_bonus)

        return min(1.0, max(0.1, base))

    # ═══════════════════════════════════════════════════════════
    # 模拟大小球赔率生成 (回测用)
    # ═══════════════════════════════════════════════════════════

    def generate_synthetic_odds(
        self,
        score_matrix: ScoreMatrix,
        margin: float = 0.065,
    ) -> Dict[float, Dict[str, float]]:
        """
        从比分矩阵生成模拟大小球赔率。

        参数:
            score_matrix: 比分概率矩阵
            margin: 庄家 margin (默认 6.5%，接近真实市场)

        返回:
            {大小球线: {"over": 赔率, "under": 赔率}}
        """
        totals_dist = self._build_totals_distribution(score_matrix)
        odds = {}

        for line in STANDARD_TOTALS_LINES:
            over_p = totals_dist.over_prob(line)
            under_p = totals_dist.under_prob(line)
            exact_p = totals_dist.exact_prob(line)

            # 整数线: 走水概率分配给两方
            if exact_p > 0:
                over_p += exact_p * 0.5
                under_p += exact_p * 0.5

            total = over_p + under_p
            if total <= 0:
                continue

            over_implied = over_p / total * (1 + margin)
            under_implied = under_p / total * (1 + margin)

            over_odds = 1.0 / max(over_implied, 0.01)
            under_odds = 1.0 / max(under_implied, 0.01)

            over_odds = min(3.00, max(1.50, over_odds))
            under_odds = min(3.00, max(1.50, under_odds))

            odds[line] = {"over": round(over_odds, 2), "under": round(under_odds, 2)}

        return odds

    # ═══════════════════════════════════════════════════════════
    # 结算
    # ═══════════════════════════════════════════════════════════

    def settle(
        self,
        proposal: TotalsProposal,
        actual_home_goals: int,
        actual_away_goals: int,
    ) -> Tuple[str, float]:
        """
        结算大小球投注。

        返回:
            (结算结果, 盈亏金额)
        """
        total_goals = actual_home_goals + actual_away_goals
        line = proposal.totals_line
        is_integer = (line == int(line))

        if proposal.side == "over":
            if total_goals > line:
                return "win", proposal.adjusted_stake * (proposal.odds - 1)
            elif is_integer and total_goals == line:
                return "push", 0.0
            else:
                return "loss", -proposal.adjusted_stake
        else:  # under
            if total_goals < line:
                return "win", proposal.adjusted_stake * (proposal.odds - 1)
            elif is_integer and total_goals == line:
                return "push", 0.0
            else:
                return "loss", -proposal.adjusted_stake