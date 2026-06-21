"""
GTO v6.0 — 亚盘策略模块

核心功能:
1. 从CSV读取真实让球数 (AHh字段)
2. 基于比分矩阵精确计算覆盖概率
3. 处理 quarter-line (0.25, 0.75, 1.25)
4. 处理走水 (整数线)

亚盘结算规则:
- 半整数线 (0.5, 1.5): 赢/输，无走水
- 整数线 (0, 1, 2): 赢/输/走水
- quarter线 (0.25, 0.75): 拆分为两条半整数线

使用方式:
    module = AsianHandicapModule(league_id="premier_league")
    result = module.evaluate(
        score_matrix=score_matrix,
        handicap_line=-0.75,
        home_odds=1.90,
        away_odds=1.95,
    )
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
class AsianHandicapResult:
    """亚盘评估结果"""
    handicap_line: float
    home_cover_prob: float
    away_cover_prob: float
    void_prob: float  # 走水概率
    market_home_prob: float
    market_away_prob: float
    home_edge: float
    away_edge: float
    best_direction: str  # "home" / "away" / "skip"
    best_edge: float
    home_odds: float
    away_odds: float

    @property
    def is_valid(self) -> bool:
        return self.best_edge > 0 and self.best_direction in ("home", "away")


@dataclass
class AsianHandicapLine:
    """亚盘线路分析"""
    line: float
    home_prob: float
    away_prob: float
    void_prob: float
    home_edge: float
    away_edge: float


# ═══════════════════════════════════════════════════════════════
# 亚盘模块
# ═══════════════════════════════════════════════════════════════

class AsianHandicapModule:
    """
    亚盘策略模块。

    基于比分矩阵精确计算覆盖概率。

    使用方式:
        module = AsianHandicapModule(league_id="premier_league")
        result = module.evaluate(
            score_matrix={(0,0): 0.08, (1,0): 0.12, ...},
            handicap_line=-0.75,
            home_odds=1.90,
            away_odds=1.95,
        )
    """

    # 标准亚盘线路
    STANDARD_LINES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                       0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    def __init__(
        self,
        league_id: str = "",
        min_edge: float = 0.03,
    ):
        self.league_id = league_id
        self.min_edge = min_edge

    def evaluate(
        self,
        score_matrix: Dict[Tuple[int, int], float],
        handicap_line: float,
        home_odds: float,
        away_odds: float,
    ) -> AsianHandicapResult:
        """
        评估单条亚盘线路。

        参数:
            score_matrix: 比分概率矩阵 {(h, a): prob}
            handicap_line: 让球数 (负数=主队让球)
            home_odds: 主队赔率
            away_odds: 客队赔率

        返回:
            AsianHandicapResult
        """
        # 计算覆盖概率
        home_cover, away_cover, void_prob = self._compute_cover_probs(
            score_matrix, handicap_line
        )

        # 市场隐含概率
        if home_odds > 1 and away_odds > 1:
            margin = 1.0 / home_odds + 1.0 / away_odds
            market_home = (1.0 / home_odds) / margin
            market_away = (1.0 / away_odds) / margin
        else:
            market_home = 0.5
            market_away = 0.5

        # 计算edge
        home_edge = home_cover - market_home
        away_edge = away_cover - market_away

        # 选择方向
        if home_edge > away_edge and home_edge > self.min_edge:
            best_direction = "home"
            best_edge = home_edge
        elif away_edge > home_edge and away_edge > self.min_edge:
            best_direction = "away"
            best_edge = away_edge
        else:
            best_direction = "skip"
            best_edge = 0.0

        return AsianHandicapResult(
            handicap_line=handicap_line,
            home_cover_prob=round(home_cover, 6),
            away_cover_prob=round(away_cover, 6),
            void_prob=round(void_prob, 6),
            market_home_prob=round(market_home, 6),
            market_away_prob=round(market_away, 6),
            home_edge=round(home_edge, 6),
            away_edge=round(away_edge, 6),
            best_direction=best_direction,
            best_edge=round(best_edge, 6),
            home_odds=home_odds,
            away_odds=away_odds,
        )

    def find_best_line(
        self,
        score_matrix: Dict[Tuple[int, int], float],
        lines_odds: Dict[float, Tuple[float, float]],
    ) -> Optional[AsianHandicapResult]:
        """
        在多条线路中找最佳edge。

        参数:
            score_matrix: 比分概率矩阵
            lines_odds: 各线路赔率 {line: (home_odds, away_odds)}

        返回:
            最佳 AsianHandicapResult
        """
        best_result = None

        for line, (home_odds, away_odds) in lines_odds.items():
            result = self.evaluate(score_matrix, line, home_odds, away_odds)
            if result.is_valid:
                if best_result is None or result.best_edge > best_result.best_edge:
                    best_result = result

        return best_result

    def _compute_cover_probs(
        self,
        score_matrix: Dict[Tuple[int, int], float],
        handicap_line: float,
    ) -> Tuple[float, float, float]:
        """
        计算覆盖概率。

        参数:
            score_matrix: 比分概率矩阵
            handicap_line: 让球数

        返回:
            (home_cover, away_cover, void_prob)
        """
        # 判断是否为 quarter-line
        is_quarter = abs(handicap_line * 4 - round(handicap_line * 4)) < 0.01 and (round(handicap_line * 4) % 2 == 1)

        if is_quarter:
            # quarter-line: 拆分为两条半整数线
            lower = math.floor(handicap_line * 2) / 2.0
            upper = math.ceil(handicap_line * 2) / 2.0
            h1, a1, v1 = self._compute_half_line(score_matrix, lower)
            h2, a2, v2 = self._compute_half_line(score_matrix, upper)
            return (h1 + h2) / 2, (a1 + a2) / 2, (v1 + v2) / 2
        else:
            return self._compute_half_line(score_matrix, handicap_line)

    def _compute_half_line(
        self,
        score_matrix: Dict[Tuple[int, int], float],
        line: float,
    ) -> Tuple[float, float, float]:
        """计算半整数线或整数线的覆盖概率"""
        home_cover = 0.0
        away_cover = 0.0
        void_prob = 0.0
        is_integer = abs(line - round(line)) < 0.01

        for (h, a), prob in score_matrix.items():
            # 主队净胜球 (考虑让球)
            diff = (h - a) + line  # line为负=主队让球

            if diff > 0:
                home_cover += prob
            elif diff < 0:
                away_cover += prob
            else:
                if is_integer:
                    void_prob += prob  # 走水
                else:
                    # 半整数线不会出现平局
                    home_cover += prob * 0.5
                    away_cover += prob * 0.5

        return home_cover, away_cover, void_prob


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def evaluate_asian_handicap(
    score_matrix: Dict[Tuple[int, int], float],
    handicap_line: float,
    home_odds: float,
    away_odds: float,
    min_edge: float = 0.03,
) -> AsianHandicapResult:
    """
    便捷函数: 评估亚盘。

    参数:
        score_matrix: 比分概率矩阵
        handicap_line: 让球数
        home_odds: 主队赔率
        away_odds: 客队赔率
        min_edge: 最小edge

    返回:
        AsianHandicapResult
    """
    module = AsianHandicapModule(min_edge=min_edge)
    return module.evaluate(score_matrix, handicap_line, home_odds, away_odds)
