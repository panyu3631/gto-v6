"""
GTO-GameFlow v5.10 — 统一概率引擎 (UnifiedProbabilityEngine)

核心设计:
- 将 1X2/亚盘/大小球三种策略统一到同一个概率空间
- 所有策略的概率从 fused_probs (logit 70% + poisson 30% + BayesianShrinkage) 推导
- 亚盘 cover_prob = Σ P(score) × I(score_diff 覆盖让球线)
- 大小球 totals_prob = Σ P(score) × I(total_goals 与 totals_line 比较)
- 禁用合成赔率: 无真实赔率数据时策略不输出信号

数据流:
    MatchContext → FactorEngine → SignalDecomposer → logit_probs
    MatchContext → poisson_bridge → ScoreMatrix
    logit_probs + ScoreMatrix → dual_domain_fusion → fused_probs
    fused_probs → UnifiedBayesianShrinkage → final_probs
    final_probs → {1x2_probs, asian_probs, totals_probs}
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScoreMatrix:
    """泊松比分概率矩阵"""
    home_goals: List[int] = field(default_factory=list)
    away_goals: List[int] = field(default_factory=list)
    probs: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def prob(self, h: int, a: int) -> float:
        return self.probs.get((h, a), 0.0)

    def total_prob(self, total: int) -> float:
        return sum(self.prob(h, a) for h, a in self.probs if h + a == total)


@dataclass
class UnifiedProbs:
    """统一概率输出"""
    # 1X2 概率
    home_prob: float = 0.0
    draw_prob: float = 0.0
    away_prob: float = 0.0

    # 亚盘概率: {让球线: {"home": prob, "away": prob}}
    asian_probs: Dict[float, Dict[str, float]] = field(default_factory=dict)

    # 大小球概率: {大小球线: {"over": prob, "under": prob}}
    totals_probs: Dict[float, Dict[str, float]] = field(default_factory=dict)

    # 元数据
    elo_diff: float = 0.0
    avg_goals: float = 0.0
    home_advantage: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 统一概率引擎
# ═══════════════════════════════════════════════════════════════

class UnifiedProbabilityEngine:
    """
    统一概率引擎 v5.10。

    从 fused_probs (模型融合概率) 推导所有策略所需的概率分布。
    所有策略使用同一概率空间，确保跨策略可比性。

    使用方式:
        engine = UnifiedProbabilityEngine()
        result = engine.compute(
            fused_probs={"home": 0.45, "draw": 0.25, "away": 0.30},
            score_matrix=score_matrix,
            elo_diff=120,
            avg_goals=2.65,
            home_advantage=0.35,
        )
        # result.asian_probs 可用于亚盘策略
        # result.totals_probs 可用于大小球策略
    """

    # 标准亚盘让球线 (v5.10: 7 条核心线)
    STANDARD_HANDICAP_LINES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    # 标准大小球线
    STANDARD_TOTALS_LINES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    def __init__(
        self,
        over_dispersion: float = 0.15,
        handicap_line_range: Optional[List[float]] = None,
        totals_line_range: Optional[List[float]] = None,
    ):
        self.over_dispersion = over_dispersion
        self.handicap_lines = handicap_line_range or self.STANDARD_HANDICAP_LINES
        self.totals_lines = totals_line_range or self.STANDARD_TOTALS_LINES

    def compute(
        self,
        fused_probs: Dict[str, float],
        score_matrix: ScoreMatrix,
        elo_diff: float = 0.0,
        avg_goals: float = 2.5,
        home_advantage: float = 0.35,
    ) -> UnifiedProbs:
        """
        从融合概率和比分矩阵计算统一概率。

        参数:
            fused_probs: {"home": P_home, "draw": P_draw, "away": P_away}
            score_matrix: 泊松比分概率矩阵
            elo_diff: Elo 差值 (主队 - 客队)
            avg_goals: 场均进球
            home_advantage: 主场优势参数

        返回:
            UnifiedProbs
        """
        result = UnifiedProbs(
            home_prob=fused_probs.get("home", 0.0),
            draw_prob=fused_probs.get("draw", 0.0),
            away_prob=fused_probs.get("away", 0.0),
            elo_diff=elo_diff,
            avg_goals=avg_goals,
            home_advantage=home_advantage,
        )

        # 从 ScoreMatrix 推导亚盘概率
        result.asian_probs = self._compute_asian_probs(score_matrix)

        # 从 ScoreMatrix 推导大小球概率 (含 over-dispersion)
        result.totals_probs = self._compute_totals_probs(score_matrix)

        return result

    def _compute_asian_probs(
        self,
        score_matrix: ScoreMatrix,
    ) -> Dict[float, Dict[str, float]]:
        """
        从 ScoreMatrix 推导亚盘覆盖概率。

        对每条让球线 line:
            P(home covers) = Σ P(score) × I(home_goals - away_goals > line)
            P(away covers) = Σ P(score) × I(away_goals - home_goals > -line)

        对于半整数线 (0.25, 0.75, 1.25):
            P(home covers) = P(home - away > floor(line)) + 0.5 × P(home - away = floor(line))
            同理负方向。
        """
        asian_probs = {}
        for line in self.handicap_lines:
            home_prob = self._compute_cover_prob(score_matrix, line, side="home")
            away_prob = self._compute_cover_prob(score_matrix, line, side="away")
            asian_probs[line] = {"home": round(home_prob, 6), "away": round(away_prob, 6)}
        return asian_probs

    def _compute_cover_prob(
        self,
        score_matrix: ScoreMatrix,
        line: float,
        side: str,
    ) -> float:
        """
        计算指定让球线某侧的覆盖概率。

        亚盘结算规则 (以 home side, line=-0.75 即主让0.75为例):
        - 净胜0球: 全输
        - 净胜1球: 赢半 (一半算-0.5赢, 一半算-1.0走水)
        - 净胜2+球: 全赢

        quarter-line (0.25, 0.75, 1.25 ...):
          = 0.5 × cover_prob(floor(line)) + 0.5 × cover_prob(ceil(line))
        其中 floor/ceil 取到最近的半整数 (0, 0.5, 1.0, 1.5 ...)

        整数线 (0, 1, 2 ...):
          diff > line → 全赢, diff == line → 走水(半赢半输)

        半整数线 (0.5, 1.5 ...):
          diff > line → 全赢, diff <= line → 全输
        """
        # 判断是否为 quarter-line (0.25, 0.75, 1.25, ...)
        is_quarter = abs(line * 4 - round(line * 4)) < 0.01 and (round(line * 4) % 2 == 1)

        if is_quarter:
            # quarter-line: 拆分为两条半整数线的等权平均
            lower = math.floor(line * 2) / 2.0  # 向下取到半整数
            upper = math.ceil(line * 2) / 2.0    # 向上取到半整数
            p_lower = self._compute_half_line_cover(score_matrix, lower, side)
            p_upper = self._compute_half_line_cover(score_matrix, upper, side)
            return 0.5 * p_lower + 0.5 * p_upper
        else:
            return self._compute_half_line_cover(score_matrix, line, side)

    def _compute_half_line_cover(
        self,
        score_matrix: ScoreMatrix,
        line: float,
        side: str,
    ) -> float:
        """计算半整数线或整数线的覆盖概率"""
        prob = 0.0
        is_integer = abs(line - round(line)) < 0.01

        for (h, a), p in score_matrix.probs.items():
            if side == "home":
                diff = h - a
            else:
                diff = a - h

            if diff > line:
                prob += p
            elif is_integer and diff == line:
                prob += p * 0.5  # 整数线走水

        return prob

    def _compute_totals_probs(
        self,
        score_matrix: ScoreMatrix,
    ) -> Dict[float, Dict[str, float]]:
        """
        从 ScoreMatrix 推导大小球概率。

        对每条大小球线 line:
            P(over) = Σ P(score) × I(total_goals > line)
            P(under) = Σ P(score) × I(total_goals < line)

        整数线: exact = P(total = line), 走水概率从两侧各扣一半
        非整数线: exact = 0
        """
        # 构建总进球分布 (含 over-dispersion 修正)
        totals_dist = self._build_totals_distribution(score_matrix)

        totals_probs = {}
        for line in self.totals_lines:
            over_prob = 0.0
            under_prob = 0.0
            exact_prob = 0.0

            is_integer = line == int(line)

            for total, prob in totals_dist.items():
                if is_integer and total == int(line):
                    exact_prob += prob
                elif total > line:
                    over_prob += prob
                elif total < line:
                    under_prob += prob

            if is_integer and exact_prob > 0:
                # 整数线: 走水概率折半分配到两侧
                over_prob += exact_prob * 0.5
                under_prob += exact_prob * 0.5

            totals_probs[line] = {
                "over": round(over_prob, 6),
                "under": round(under_prob, 6),
                "exact": round(exact_prob, 6) if is_integer else 0.0,
            }

        return totals_probs

    def _build_totals_distribution(
        self,
        score_matrix: ScoreMatrix,
    ) -> Dict[int, float]:
        """
        从 ScoreMatrix 构建总进球数分布，含 over-dispersion 修正。

        over-dispersion 目的: 实际足球进球方差低于泊松预测 (under-dispersion),
        通过将部分概率质量从中心向两侧重新分配来修正。

        修正方式: 从每个进球数 g 取出 over_dispersion 比例的质量,
        按固定比例分配到邻居 (g-2, g-1, g+1, g+2) 和边界 (0, max_goals)。
        边界处归一化分配权重，确保 mass 严格守恒。
        """
        max_goals = max(h + a for h, a in score_matrix.probs.keys())
        dist: Dict[int, float] = {}
        for g in range(max_goals + 1):
            dist[g] = score_matrix.total_prob(g)

        # 归一化
        total = sum(dist.values())
        if total > 0:
            for g in dist:
                dist[g] /= total

        # Over-dispersion 修正 (两阶段，确保 mass 守恒)
        original = dict(dist)
        delta: Dict[int, float] = {g: 0.0 for g in range(max_goals + 1)}

        # 目标分配比例: g-2=15%, g-1=25%, g+1=25%, g+2=15%, 0=10%, max=10%
        # 边界处: 不可达位置的质量按比例重分配到可达位置
        base_weights = {-2: 0.15, -1: 0.25, 1: 0.25, 2: 0.15, "zero": 0.10, "max": 0.10}

        for g in range(max_goals + 1):
            leaked = original[g] * self.over_dispersion
            delta[g] -= leaked

            # 计算可达目标及对应权重
            targets = {}
            for offset, w in [(-2, 0.15), (-1, 0.25), (1, 0.25), (2, 0.15)]:
                t = g + offset
                if 0 <= t <= max_goals:
                    targets[t] = targets.get(t, 0.0) + w
            # 边界锚点
            if 0 != g and 0 not in targets:
                targets[0] = targets.get(0, 0.0) + 0.10
            elif 0 == g:
                # g=0 时 zero 锚点就是自身，质量已扣除，不再回加
                pass
            if max_goals != g and max_goals not in targets:
                targets[max_goals] = targets.get(max_goals, 0.0) + 0.10
            elif max_goals == g:
                pass

            # 归一化权重并分配
            w_total = sum(targets.values())
            if w_total > 0:
                for t, w in targets.items():
                    delta[t] += leaked * (w / w_total)

        for g in range(max_goals + 1):
            dist[g] = max(0.0, original[g] + delta[g])

        # 重新归一化
        total = sum(dist.values())
        if total > 0:
            for g in dist:
                dist[g] /= total

        return dist


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def build_score_matrix(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 8,
    rho: float = -0.10,
) -> ScoreMatrix:
    """
    从泊松 lambda 参数构建比分概率矩阵 (含 Dixon-Coles 校正)。

    参数:
        home_lambda: 主队预期进球
        away_lambda: 客队预期进球
        max_goals: 最大进球数
        rho: Dixon-Coles 相关系数 (负值=进球负相关, 典型-0.13~-0.05)
    """
    h_goals = list(range(max_goals + 1))
    a_goals = list(range(max_goals + 1))
    probs = {}

    for h in h_goals:
        h_p = poisson_pmf(h, home_lambda)
        for a in a_goals:
            a_p = poisson_pmf(a, away_lambda)
            tau = _dixon_coles_tau(h, a, home_lambda, away_lambda, rho)
            probs[(h, a)] = h_p * a_p * tau

    # 归一化
    total = sum(probs.values())
    if total > 0:
        for key in probs:
            probs[key] /= total

    return ScoreMatrix(home_goals=h_goals, away_goals=a_goals, probs=probs)


def _dixon_coles_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles tau 校正 (仅作用于低比分 x,y ∈ {0,1})"""
    if x >= 2 or y >= 2:
        return 1.0
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lam_h * lam_a * rho)
    elif x == 0 and y == 1:
        return max(0.0, 1.0 + lam_h * rho)
    elif x == 1 and y == 0:
        return max(0.0, 1.0 + lam_a * rho)
    elif x == 1 and y == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def poisson_pmf(k: int, lam: float) -> float:
    """泊松分布概率质量函数"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)