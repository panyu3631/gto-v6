"""
GTO v6.0 — 大小球动态线路选择器

核心逻辑:
1. 根据模型预期进球数，自动选择最佳线路
2. 同时计算5条线路的edge，选最大的
3. 联赛特化进球参数

线路选择规则:
| 预期进球 | 选线 | 理由 |
|----------|------|------|
| < 2.0    | 1.5  | 低进球比赛 |
| 2.0-2.3  | 2.0  | 中低进球 |
| 2.3-2.7  | 2.5  | 标准线 |
| 2.7-3.2  | 3.0  | 中高进球 |
| > 3.2    | 3.5  | 高进球比赛 |

使用方式:
    selector = DynamicTotalsSelector(league_id="premier_league")
    result = selector.select(
        home_lambda=1.4,
        away_lambda=1.2,
        market_odds_over=1.85,
        market_odds_under=1.95,
    )
    # result.best_line → 2.5
    # result.best_direction → "over"
    # result.edge → 0.05
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
class LineAnalysis:
    """单条线路分析结果"""
    line: float
    over_prob: float
    under_prob: float
    market_over_prob: float
    market_under_prob: float
    over_edge: float
    under_edge: float
    best_direction: str  # "over" / "under" / "skip"
    best_edge: float


@dataclass
class TotalsSelection:
    """大小球动态线路选择结果"""
    expected_goals: float
    best_line: float
    best_direction: str  # "over" / "under"
    best_edge: float
    model_prob: float
    market_prob: float
    all_lines: List[LineAnalysis] = field(default_factory=list)
    league_id: str = ""

    @property
    def is_valid(self) -> bool:
        """是否有有效的投注机会"""
        return self.best_edge > 0 and self.best_direction in ("over", "under")


# ═══════════════════════════════════════════════════════════════
# 联赛进球参数
# ═══════════════════════════════════════════════════════════════

LEAGUE_GOAL_PARAMS = {
    "premier_league": {
        "avg_goals": 2.65,
        "home_factor": 1.15,  # 主场进球乘数
        "away_factor": 0.87,  # 客场进球乘数
        "variance_scale": 1.0,
    },
    "la_liga": {
        "avg_goals": 2.55,
        "home_factor": 1.13,
        "away_factor": 0.88,
        "variance_scale": 0.95,
    },
    "bundesliga": {
        "avg_goals": 2.85,
        "home_factor": 1.18,
        "away_factor": 0.85,
        "variance_scale": 1.10,
    },
    "serie_a": {
        "avg_goals": 2.50,
        "home_factor": 1.12,
        "away_factor": 0.89,
        "variance_scale": 0.90,
    },
    "ligue_1": {
        "avg_goals": 2.60,
        "home_factor": 1.14,
        "away_factor": 0.87,
        "variance_scale": 1.00,
    },
}

DEFAULT_GOAL_PARAMS = {
    "avg_goals": 2.65,
    "home_factor": 1.15,
    "away_factor": 0.87,
    "variance_scale": 1.0,
}


# ═══════════════════════════════════════════════════════════════
# 标准线路
# ═══════════════════════════════════════════════════════════════

STANDARD_LINES = [1.5, 2.0, 2.5, 3.0, 3.5]


# ═══════════════════════════════════════════════════════════════
# 动态线路选择器
# ═══════════════════════════════════════════════════════════════

class DynamicTotalsSelector:
    """
    大小球动态线路选择器。

    根据预期进球和市场赔率，选择edge最大的线路和方向。

    使用方式:
        selector = DynamicTotalsSelector(league_id="premier_league")
        result = selector.select(
            home_lambda=1.4,
            away_lambda=1.2,
            market_odds_over={1.5: 1.30, 2.0: 1.55, 2.5: 1.85, 3.0: 2.20, 3.5: 2.80},
            market_odds_under={1.5: 3.40, 2.0: 2.40, 2.5: 1.95, 3.0: 1.65, 3.5: 1.40},
        )
    """

    def __init__(
        self,
        league_id: str = "",
        lines: Optional[List[float]] = None,
        min_edge: float = 0.03,
    ):
        """
        参数:
            league_id: 联赛ID
            lines: 可选线路列表 (默认 [1.5, 2.0, 2.5, 3.0, 3.5])
            min_edge: 最小edge阈值
        """
        self.league_id = league_id
        self.lines = lines or STANDARD_LINES
        self.min_edge = min_edge
        self.params = LEAGUE_GOAL_PARAMS.get(league_id, DEFAULT_GOAL_PARAMS)

    def select(
        self,
        home_lambda: float,
        away_lambda: float,
        market_odds_over: Dict[float, float],
        market_odds_under: Dict[float, float],
    ) -> TotalsSelection:
        """
        选择最佳线路。

        参数:
            home_lambda: 主队预期进球 (泊松lambda)
            away_lambda: 客队预期进球 (泊松lambda)
            market_odds_over: 各线路的over赔率 {line: odds}
            market_odds_under: 各线路的under赔率 {line: odds}

        返回:
            TotalsSelection
        """
        expected_goals = home_lambda + away_lambda
        all_lines = []

        best_edge = 0.0
        best_line = 2.5
        best_direction = "over"
        best_model_prob = 0.0
        best_market_prob = 0.0

        for line in self.lines:
            analysis = self._analyze_line(
                line, home_lambda, away_lambda,
                market_odds_over.get(line, 0.0),
                market_odds_under.get(line, 0.0),
            )
            all_lines.append(analysis)

            # 更新最佳选择
            if analysis.best_direction == "over" and analysis.over_edge > best_edge:
                best_edge = analysis.over_edge
                best_line = line
                best_direction = "over"
                best_model_prob = analysis.over_prob
                best_market_prob = analysis.market_over_prob
            elif analysis.best_direction == "under" and analysis.under_edge > best_edge:
                best_edge = analysis.under_edge
                best_line = line
                best_direction = "under"
                best_model_prob = analysis.under_prob
                best_market_prob = analysis.market_under_prob

        return TotalsSelection(
            expected_goals=expected_goals,
            best_line=best_line,
            best_direction=best_direction,
            best_edge=best_edge,
            model_prob=best_model_prob,
            market_prob=best_market_prob,
            all_lines=all_lines,
            league_id=self.league_id,
        )

    def select_simple(
        self,
        home_lambda: float,
        away_lambda: float,
        market_odds_over_25: float,
        market_odds_under_25: float,
    ) -> TotalsSelection:
        """
        简化版: 只用2.5线的市场赔率，动态选择线路。

        参数:
            home_lambda: 主队预期进球
            away_lambda: 客队预期进球
            market_odds_over_25: 2.5 over 赔率
            market_odds_under_25: 2.5 under 赔率

        返回:
            TotalsSelection
        """
        # 从2.5线赔率推算其他线路的近似赔率
        expected_goals = home_lambda + away_lambda
        
        # 构建各线路的近似市场赔率
        market_odds_over = {}
        market_odds_under = {}
        
        for line in self.lines:
            # 基于2.5线赔率和预期进球推算
            over_prob_25 = 1.0 / market_odds_over_25 if market_odds_over_25 > 1 else 0.5
            under_prob_25 = 1.0 / market_odds_under_25 if market_odds_under_25 > 1 else 0.5
            
            # 调整因子: 基于线路与2.5的差异
            diff = line - 2.5
            # 每0.5球差异约调整8-10%
            adjustment = diff * 0.18
            
            over_prob_line = max(0.05, min(0.95, over_prob_25 + adjustment))
            under_prob_line = 1.0 - over_prob_line
            
            # 转回赔率 (含边际)
            margin = 1.0 / market_odds_over_25 + 1.0 / market_odds_under_25 - 1.0
            market_odds_over[line] = 1.0 / (over_prob_line * (1 + margin * 0.5)) if over_prob_line > 0 else 10.0
            market_odds_under[line] = 1.0 / (under_prob_line * (1 + margin * 0.5)) if under_prob_line > 0 else 10.0
        
        return self.select(
            home_lambda, away_lambda,
            market_odds_over, market_odds_under,
        )

    def _analyze_line(
        self,
        line: float,
        home_lambda: float,
        away_lambda: float,
        market_odds_over: float,
        market_odds_under: float,
    ) -> LineAnalysis:
        """分析单条线路"""
        # 计算模型概率 (泊松分布)
        total_lambda = home_lambda + away_lambda
        over_prob = self._poisson_over_prob(total_lambda, line)
        under_prob = 1.0 - over_prob

        # 计算市场隐含概率
        if market_odds_over > 1 and market_odds_under > 1:
            margin = 1.0 / market_odds_over + 1.0 / market_odds_under
            market_over_prob = (1.0 / market_odds_over) / margin
            market_under_prob = (1.0 / market_odds_under) / margin
        else:
            market_over_prob = 0.5
            market_under_prob = 0.5

        # 计算edge
        over_edge = over_prob - market_over_prob
        under_edge = under_prob - market_under_prob

        # 选择最佳方向
        if over_edge > under_edge and over_edge > self.min_edge:
            best_direction = "over"
            best_edge = over_edge
        elif under_edge > over_edge and under_edge > self.min_edge:
            best_direction = "under"
            best_edge = under_edge
        else:
            best_direction = "skip"
            best_edge = 0.0

        return LineAnalysis(
            line=line,
            over_prob=round(over_prob, 6),
            under_prob=round(under_prob, 6),
            market_over_prob=round(market_over_prob, 6),
            market_under_prob=round(market_under_prob, 6),
            over_edge=round(over_edge, 6),
            under_edge=round(under_edge, 6),
            best_direction=best_direction,
            best_edge=round(best_edge, 6),
        )

    def _poisson_over_prob(self, lam: float, line: float) -> float:
        """计算泊松分布 P(goals > line)"""
        if lam <= 0:
            return 0.0
        
        prob = 0.0
        for k in range(int(line) + 1, 20):
            prob += self._poisson_pmf(k, lam)
        return prob

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """泊松分布概率质量函数"""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def select_totals_line(
    league_id: str,
    home_lambda: float,
    away_lambda: float,
    market_odds_over_25: float,
    market_odds_under_25: float,
    min_edge: float = 0.03,
) -> TotalsSelection:
    """
    便捷函数: 选择最佳大小球线路。

    参数:
        league_id: 联赛ID
        home_lambda: 主队预期进球
        away_lambda: 客队预期进球
        market_odds_over_25: 2.5 over 赔率
        market_odds_under_25: 2.5 under 赔率
        min_edge: 最小edge阈值

    返回:
        TotalsSelection
    """
    selector = DynamicTotalsSelector(league_id=league_id, min_edge=min_edge)
    return selector.select_simple(
        home_lambda, away_lambda,
        market_odds_over_25, market_odds_under_25,
    )
