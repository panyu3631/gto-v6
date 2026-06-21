"""
GTO-GameFlow v5.5 — 亚洲让球盘策略引擎

亚洲盘口 (Asian Handicap) 从比分概率矩阵直接推导让球盘跑出概率。

核心概念:
- 让球线 (Handicap Line): 主队让N球, 如 -0.5/-1.0/-1.5 (主让) 或 +0.25/+0.5 (客让)
- 水位 (Odds): 亚盘赔率, 通常接近 1.85-2.05
- 结算规则: 全赢/赢半/走水/输半/全输

让球线类型:
- 0.0 (平手): 平局走水, 胜者赢
- 0.25 (平半): 主胜→全赢, 平局→输半, 主负→全输
- 0.5 (半球): 主胜→全赢, 平局/主负→全输
- 0.75 (半一): 主胜1球→赢半, 主胜2+→全赢, 平局/主负→全输
- 1.0 (一球): 主胜1球→走水, 主胜2+→全赢, 平局/主负→全输
- 1.25 (一球球半): 主胜1球→输半, 主胜2+→全赢, 平局/主负→全输
- 1.5 (球半): 主胜1球→全输, 主胜2+→全赢
- 以此类推...

使用方式:
    from src.strategies import AsianHandicapStrategy

    strategy = AsianHandicapStrategy(league_id="bundesliga")
    proposals = strategy.analyze(
        score_matrix=score_matrix,    # 从泊松模型获得
        handicap_odds={0.5: {"home": 1.92, "away": 1.98}},  # 让半球盘
        match_id="BAYvsDOR",
    )
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..data.models import (
    ScoreMatrix, AsianHandicapProposal, AsianHandicapResult,
    TotalsDistribution,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 让球线类型
# ═══════════════════════════════════════════════════════════════

class HandicapType(Enum):
    """让球线结构类型"""
    FLAT = "flat"           # 0.0 (平手)
    QUARTER = "quarter"     # 0.25/0.75/1.25... (含半份)
    HALF = "half"           # 0.5/1.5/2.5... (半球/球半)
    WHOLE = "whole"         # 1.0/2.0... (一球/两球)


@dataclass
class HandicapLine:
    """让球线定义"""
    line: float              # 让球线数值 (正数=主让, 负数=客让)
    htype: HandicapType      # 线类型

    @property
    def whole_part(self) -> int:
        """整数部分 (如 1.25 → 1)"""
        return int(abs(self.line))

    @property
    def fraction_part(self) -> float:
        """小数部分 (如 1.25 → 0.25)"""
        return abs(self.line) - int(abs(self.line))


# 标准让球盘 (v5.9.4: 裁剪至 7 条核心线, 减少极端线噪声)
STANDARD_HANDICAP_LINES = [
    HandicapLine(0.0, HandicapType.FLAT),
    HandicapLine(0.25, HandicapType.QUARTER),
    HandicapLine(0.5, HandicapType.HALF),
    HandicapLine(0.75, HandicapType.QUARTER),
    HandicapLine(1.0, HandicapType.WHOLE),
    HandicapLine(1.25, HandicapType.QUARTER),
    HandicapLine(1.5, HandicapType.HALF),
]


# ═══════════════════════════════════════════════════════════════
# 亚盘策略引擎
# ═══════════════════════════════════════════════════════════════

class AsianHandicapStrategy:
    """
    亚洲让球盘策略引擎。

    从比分概率矩阵 (ScoreMatrix) 直接推导让球盘跑出概率:
    1. 对每个可能的比分, 计算 "让球后比分"
    2. 根据让球线类型, 判断全赢/赢半/走水/输半/全输
    3. 汇总概率

    参数:
        league_id: 联赛ID
        value_threshold: 最小价值阈值 (默认 0.02, 比1X2的0.03更宽松)
        min_odds: 最小赔率
        max_odds: 最大赔率
    """

    def __init__(
        self,
        league_id: str = "",
        value_threshold: float = 0.015,
        confidence_threshold: float = 0.35,
        min_odds: float = 1.50,
        max_odds: float = 3.00,
    ):
        self.league_id = league_id
        self.value_threshold = value_threshold
        self.confidence_threshold = confidence_threshold
        self.min_odds = min_odds
        self.max_odds = max_odds

    def analyze(
        self,
        score_matrix: ScoreMatrix,
        handicap_odds: Dict[float, Dict[str, float]],
        match_id: str = "",
        league_id: str = "",
        kelly_discount: float = 0.25,
        strip_margin: bool = True,
    ) -> List[AsianHandicapProposal]:
        """
        分析所有可用让球盘的投注机会。

        参数:
            score_matrix: 泊松比分概率矩阵
            handicap_odds: {让球线: {"home": 赔率, "away": 赔率}} 的字典
            match_id: 比赛ID
            league_id: 联赛ID
            kelly_discount: Kelly 折扣系数
            strip_margin: 是否剥离庄家 margin (合成赔率应设为 False)

        返回:
            AsianHandicapProposal 列表 (按优先级排序)
        """
        proposals: List[AsianHandicapProposal] = []

        for line, odds_dict in handicap_odds.items():
            home_odds = odds_dict.get("home", 0)
            away_odds = odds_dict.get("away", 0)

            if not (self.min_odds <= home_odds <= self.max_odds
                    and self.min_odds <= away_odds <= self.max_odds):
                continue

            home_prob = self._calculate_cover_probability(score_matrix, line, "home")
            away_prob = self._calculate_cover_probability(score_matrix, line, "away")

            # 主场投注
            if home_odds > 0:
                prop = self._build_proposal(
                    match_id, line, "home", home_odds, home_prob,
                    league_id, kelly_discount,
                    opposite_odds=away_odds if strip_margin else 0.0,
                )
                if prop and prop.value >= self.value_threshold:
                    proposals.append(prop)

            # 客场投注
            if away_odds > 0:
                prop = self._build_proposal(
                    match_id, line, "away", away_odds, away_prob,
                    league_id, kelly_discount,
                    opposite_odds=home_odds if strip_margin else 0.0,
                )
                if prop and prop.value >= self.value_threshold:
                    proposals.append(prop)

        # 按优先级排序
        proposals.sort(key=lambda p: p.priority_score, reverse=True)
        return proposals

    def _calculate_cover_probability(
        self,
        score_matrix: ScoreMatrix,
        handicap_line: float,
        side: str,
    ) -> float:
        """
        计算亚盘跑出概率。

        从比分矩阵中, 对每个比分 (h, a):
        - 让球后比分: h' = h - handicap_line (主让), a' = a
        - 判断 h' vs a' 的结果: 全赢/赢半/走水/输半/全输

        参数:
            score_matrix: 比分概率矩阵
            handicap_line: 让球线 (正数=主让)
            side: "home" 或 "away"

        返回:
            跑出概率 (含赢半的50%折合)
        """
        htype = self._get_handicap_type(handicap_line)
        total_prob = 0.0

        for (h, a), prob in score_matrix.matrix.items():
            if prob <= 0:
                continue

            result = self._evaluate_handicap_result(h, a, handicap_line, htype, side)
            if result == AsianHandicapResult.FULL_WIN:
                total_prob += prob
            elif result == AsianHandicapResult.HALF_WIN:
                total_prob += prob * 0.5
            elif result == AsianHandicapResult.PUSH:
                total_prob += prob * 0.0  # 走水不计入
            # HALF_LOSS 和 FULL_LOSS 不计入

        return total_prob

    def _evaluate_handicap_result(
        self,
        home_goals: int,
        away_goals: int,
        handicap_line: float,
        htype: HandicapType,
        side: str,
    ) -> AsianHandicapResult:
        """
        判断单个比分在给定让球线下的结算结果。

        核心逻辑:
        - 让球后主队得分 = home_goals - handicap_line
        - adjusted_diff = (让球后主队得分) - away_goals
        - 根据 side 和 htype 判断结果

        Quarter 线 (0.25/0.75) 的特殊处理:
        - 0.25 线: 相当于 [-0.5, 0] 两个线的平均
        - 0.75 线: 相当于 [-1.0, -0.5] 两个线的平均
        """
        # 调整后主队得分 (handicap_line 为负时主队让球)
        adjusted_home = home_goals + handicap_line
        diff = adjusted_home - away_goals

        if side == "away":
            diff = -diff  # 客队视角反转

        if htype == HandicapType.FLAT:
            # 平手盘: diff>0全赢, diff=0走水, diff<0全输
            if diff > 0:
                return AsianHandicapResult.FULL_WIN
            elif diff == 0:
                return AsianHandicapResult.PUSH
            else:
                return AsianHandicapResult.FULL_LOSS

        elif htype == HandicapType.HALF:
            # 半球/球半: diff>0全赢, diff<0全输 (无走水)
            if diff > 0:
                return AsianHandicapResult.FULL_WIN
            else:
                return AsianHandicapResult.FULL_LOSS

        elif htype == HandicapType.WHOLE:
            # 一球/两球: diff>0全赢, diff=0走水, diff<0全输
            if diff > 0:
                return AsianHandicapResult.FULL_WIN
            elif diff == 0:
                return AsianHandicapResult.PUSH
            else:
                return AsianHandicapResult.FULL_LOSS

        elif htype == HandicapType.QUARTER:
            # 平半/半一/一球球半 — 拆分为两个分量分别评估
            # 0.25 = 0.0 + 0.5  (平手 + 半球)
            # 0.75 = 0.5 + 1.0  (半球 + 一球)
            # 1.25 = 1.0 + 1.5  (一球 + 球半)
            #
            # v5.10.5 修复: 保留 handicap_line 符号
            # handicap_line 为负 → 主队让球 → lower/upper 为正 (主队让X球)
            # handicap_line 为正 → 主队受让 → lower/upper 为负 (主队受让X球)
            # _evaluate_single_line 约定: adjusted_home = home_goals - line
            #   正 line → 主队让球 (adjusted_home 减少)
            #   负 line → 主队受让 (adjusted_home 增加)
            sign = -1.0 if handicap_line > 0 else 1.0
            whole = int(abs(handicap_line))
            frac = abs(handicap_line) - whole

            if frac == 0.25:
                lower = sign * whole          # 0.25 → 0.0
                upper = sign * (whole + 0.5)  # 0.25 → 0.5
            else:  # 0.75
                lower = sign * (whole + 0.5)  # 0.75 → 0.5
                upper = sign * (whole + 1.0)  # 0.75 → 1.0

            # 评估两个分量
            lower_result = self._evaluate_single_line(home_goals, away_goals, lower, side)
            upper_result = self._evaluate_single_line(home_goals, away_goals, upper, side)

            return self._combine_quarter_results(lower_result, upper_result)

        return AsianHandicapResult.FULL_LOSS

    def _evaluate_single_line(
        self,
        home_goals: int,
        away_goals: int,
        line: float,
        side: str,
    ) -> str:
        """
        评估单一线 (非 quarter) 的结果。

        约定: line > 0 → 主队让球, line < 0 → 主队受让
        adjusted_home = home_goals - line
        (与主函数 _evaluate_handicap_result 的约定相反:
          主函数: adjusted_home = home_goals + handicap_line, 负=主让)

        返回: "win", "push", "loss"
        """
        adjusted_home = home_goals - line
        diff = adjusted_home - away_goals
        if side == "away":
            diff = -diff

        if diff > 0:
            return "win"
        elif diff == 0:
            return "push"
        else:
            return "loss"

    def _combine_quarter_results(
        self,
        lower: str,
        upper: str,
    ) -> AsianHandicapResult:
        """
        合并两个分量线的结果。

        规则:
        - 两赢 → FULL_WIN
        - 一赢一走 → HALF_WIN
        - 两走 → PUSH
        - 一输一走 → HALF_LOSS
        - 两输 → FULL_LOSS
        """
        if lower == "win" and upper == "win":
            return AsianHandicapResult.FULL_WIN
        elif (lower == "win" and upper == "push") or (lower == "push" and upper == "win"):
            return AsianHandicapResult.HALF_WIN
        elif lower == "push" and upper == "push":
            return AsianHandicapResult.PUSH
        elif (lower == "loss" and upper == "push") or (lower == "push" and upper == "loss"):
            return AsianHandicapResult.HALF_LOSS
        else:
            return AsianHandicapResult.FULL_LOSS

    def _get_handicap_type(self, line: float) -> HandicapType:
        """根据让球线数值判断类型"""
        abs_line = abs(line)
        frac = abs_line - int(abs_line)

        if frac == 0.0:
            if int(abs_line) == 0:
                return HandicapType.FLAT
            return HandicapType.WHOLE
        elif frac == 0.5:
            return HandicapType.HALF
        else:
            return HandicapType.QUARTER

    def _build_proposal(
        self,
        match_id: str,
        handicap_line: float,
        side: str,
        odds: float,
        cover_prob: float,
        league_id: str,
        kelly_discount: float,
        opposite_odds: float = 0.0,
    ) -> Optional[AsianHandicapProposal]:
        """构建亚盘投注建议"""
        # v5.9.4: 剥离庄家 margin, 获取公平隐含概率
        # 合成赔率 (opposite_odds=0) 不剥离 margin (已含 margin)
        if opposite_odds > 0:
            total_implied = 1.0 / odds + 1.0 / opposite_odds
            if total_implied > 0:
                fair_implied = (1.0 / odds) / total_implied
            else:
                fair_implied = 1.0 / odds
        else:
            # 合成赔率: 直接使用隐含概率
            fair_implied = 1.0 / odds

        value = cover_prob - fair_implied

        if value < self.value_threshold:
            return None

        # 置信度过滤 (v5.9.4: 新增)
        confidence = self._compute_confidence(cover_prob, odds, value)
        if confidence < self.confidence_threshold:
            return None

        # Kelly 计算
        b = odds - 1.0
        f_kelly = max(0.0, (b * cover_prob - (1 - cover_prob)) / max(b, 0.01))
        f_kelly *= kelly_discount
        kelly_stake = f_kelly * 10000  # 标准化到 10000 基准

        # 优先级: Kelly × value × 置信度
        priority = f_kelly * max(0, value) * confidence

        return AsianHandicapProposal(
            match_id=match_id,
            handicap_line=handicap_line,
            side=side,
            odds=odds,
            cover_prob=round(cover_prob, 4),
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
        cover_prob: float,
        odds: float,
        value: float,
    ) -> float:
        """
        亚盘置信度计算。

        比 1X2 更保守: 亚盘本质上是二选一 (去掉平局),
        但让球线引入的复杂性降低了置信度。
        """
        # 基础置信度
        base = 0.7

        # 概率极端性惩罚 (接近 50% 时不确定性最大)
        prob_penalty = 1.0 - 2.0 * abs(cover_prob - 0.5)
        base *= (0.6 + 0.4 * prob_penalty)

        # 价值大小奖励
        value_bonus = min(1.0, max(0.0, value * 10.0))
        base *= (0.8 + 0.2 * value_bonus)

        # 赔率合理性 (偏离 1.90 太远降低置信度)
        odds_penalty = 1.0 - abs(odds - 1.90) / 2.0
        base *= max(0.5, odds_penalty)

        return min(1.0, max(0.1, base))

    # ═══════════════════════════════════════════════════════════
    # 从泊松模型直接生成标准让球盘赔率 (模拟)
    # ═══════════════════════════════════════════════════════════

    def generate_synthetic_odds(
        self,
        score_matrix: ScoreMatrix,
        margin: float = 0.065,
    ) -> Dict[float, Dict[str, float]]:
        """
        从比分矩阵生成模拟亚盘赔率。

        用于回测: 当没有真实亚盘数据时, 用泊松模型推导的"公平盘"
        加上 margin 生成模拟赔率。

        参数:
            score_matrix: 比分概率矩阵
            margin: 庄家 margin (默认 2.5%)

        返回:
            {让球线: {"home": 赔率, "away": 赔率}}
        """
        odds = {}
        for hl in STANDARD_HANDICAP_LINES:
            home_prob = self._calculate_cover_probability(score_matrix, hl.line, "home")
            away_prob = self._calculate_cover_probability(score_matrix, hl.line, "away")

            total = home_prob + away_prob
            if total <= 0:
                continue

            # 加入 margin
            home_implied = home_prob / total * (1 + margin)
            away_implied = away_prob / total * (1 + margin)

            home_odds = 1.0 / max(home_implied, 0.01)
            away_odds = 1.0 / max(away_implied, 0.01)

            # 限制赔率范围
            home_odds = min(3.00, max(1.50, home_odds))
            away_odds = min(3.00, max(1.50, away_odds))

            odds[hl.line] = {"home": round(home_odds, 2), "away": round(away_odds, 2)}

        return odds

    # ═══════════════════════════════════════════════════════════
    # 结算
    # ═══════════════════════════════════════════════════════════

    def settle(
        self,
        proposal: AsianHandicapProposal,
        actual_home_goals: int,
        actual_away_goals: int,
    ) -> Tuple[AsianHandicapResult, float]:
        """
        结算亚盘投注。

        返回:
            (结算结果, 盈亏金额)
        """
        htype = self._get_handicap_type(proposal.handicap_line)
        result = self._evaluate_handicap_result(
            actual_home_goals, actual_away_goals,
            proposal.handicap_line, htype, proposal.side,
        )

        profit_map = {
            AsianHandicapResult.FULL_WIN: proposal.adjusted_stake * (proposal.odds - 1),
            AsianHandicapResult.HALF_WIN: proposal.adjusted_stake * (proposal.odds - 1) / 2,
            AsianHandicapResult.PUSH: 0.0,
            AsianHandicapResult.HALF_LOSS: -proposal.adjusted_stake / 2,
            AsianHandicapResult.FULL_LOSS: -proposal.adjusted_stake,
        }

        return result, profit_map.get(result, 0.0)