"""
GTO-GameFlow v5.0 资金管理与投注引擎

实现规范文档第8-9章：
- 8.3: 优先级排序 priority_score = f_actual × value × confidence
- 8.3b: confidence 四因子合成公式
- 8.4: 硬性过滤 (value≥0.03, confidence≥0.6, odds≥1.05, odds≤10.0)
- 9.1-9.2: Kelly 公式 + 分数 Kelly
- 9.4: 多注分配 score_i = value_i × confidence_i × edge_i
"""
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.data.models import (
    BetProposal, BetSelection, BetPlacement, BetResult,
    BankrollState, ProbabilityDistribution,
)
from src.config.settings import config as global_config


def compute_confidence(
    data_completeness: float = 0.8,
    factor_activation_rate: float = 0.8,
    dispersion_penalty: float = 0.1,
    match_phase: float = 1.0,
    weights: Optional[Tuple[float, float, float, float]] = None,
) -> float:
    """
    置信度评分计算 — 规范第8.3b节。

    confidence = w_data × data_completeness + w_factor × factor_activation_rate
               + w_dispersion × dispersion_penalty + w_phase × match_phase

    v5.2: 支持联赛特定权重 (默认: 0.4/0.3/0.2/0.1)

    其中:
    - data_completeness: 可用数据源数/预期数据源数 (0~1)
    - factor_activation_rate: 激活因子数/41 (0~1)
    - dispersion_penalty: 1 − min(1, odds_std / 0.15)，赔率离散度惩罚
    - match_phase: 赛季初0.85 / 赛季中1.0 / 赛季末0.95
    """
    if weights is None:
        w_data, w_factor, w_dispersion, w_phase = 0.4, 0.3, 0.2, 0.1
    else:
        w_data, w_factor, w_dispersion, w_phase = weights

    confidence = (
        w_data * data_completeness +
        w_factor * factor_activation_rate +
        w_dispersion * dispersion_penalty +
        w_phase * match_phase
    )
    return max(0.0, min(1.0, confidence))


class BankrollManager:
    """
    资金管理器 — 管理资金状态、Kelly 计算、多注分配。

    规范第9章:
    - f = (b × p − q) / b  (标准 Kelly)
    - f_actual = f_kelly × kelly_discount  (1/4 Kelly)
    - stake_i = f_i × bankroll × scoring_proportion_i
    - Σ stake_i ≤ bankroll × 0.20

    v5.9: 固定基数模式 (fixed_base=True)
    - 所有投注以初始资金为基准计算，不做复利累积
    - 避免资金指数膨胀，更贴近真实长期投注场景
    """

    def __init__(self, initial_bankroll: float = None, fixed_base: bool = True):
        cfg = global_config.bankroll
        self.state = BankrollState(
            balance=initial_bankroll if initial_bankroll else cfg.initial_bankroll,
            peak_balance=initial_bankroll if initial_bankroll else cfg.initial_bankroll,
        )
        self.kelly_discount = cfg.kelly_fraction
        self.max_exposure = cfg.max_total_exposure
        self.single_bet_max = cfg.single_bet_max_ratio
        # v5.9: 固定基数模式
        self.fixed_base = fixed_base
        self._initial_bankroll = initial_bankroll if initial_bankroll else cfg.initial_bankroll

    def _get_base_bankroll(self) -> float:
        """获取计算投注额的基础资金"""
        if self.fixed_base:
            return self._initial_bankroll
        return self.state.balance

    # ================================================================
    # Kelly 公式
    # ================================================================
    def compute_kelly(
        self,
        model_prob: float,
        odds: float,
        discount: Optional[float] = None,
        is_away: bool = False,
        match_phase: float = 1.0,
    ) -> float:
        """
        v5.8: 计算 Kelly 投注比例 (动态折扣 + 客场贴现 + 赛季阶段调整)。

        f = (b × p − q) / b
        b = odds − 1  (净赔率)
        p = model_prob
        q = 1 − p

        v5.8 动态折扣:
        - 赛季初期 (match_phase < 0.3): discount × 0.85 (数据不足, 更保守)
        - 赛季中期 (0.3 ≤ phase < 0.7): discount × 1.0
        - 赛季末期 (phase ≥ 0.7): discount × 1.15 (数据充足, 可适度激进)

        v5.8 客场贴现 (FIND-012):
        - 客场比赛不确定性更高, Kelly 额外打 9 折

        返回: f_actual = f_kelly × discount × phase_adj × away_adj
        """
        if discount is None:
            discount = self.kelly_discount

        if odds <= 0:
            return 0.0

        b = odds - 1.0
        if b < 1e-6:
            return 0.0

        p = max(0.0001, min(0.9999, model_prob))
        q = 1.0 - p

        f_kelly = (b * p - q) / b
        if f_kelly <= 0:
            return 0.0

        # v5.8: 赛季阶段调整
        if match_phase < 0.3:
            phase_adj = 0.85
        elif match_phase < 0.7:
            phase_adj = 1.0
        else:
            phase_adj = 1.15

        # v5.8: 客场贴现
        away_adj = 0.90 if is_away else 1.0

        f_actual = f_kelly * discount * phase_adj * away_adj
        return max(0.0, f_actual)

    # ================================================================
    # Stage 6: 优先级排序 — 规范第8.3节
    # ================================================================
    def compute_priority_score(
        self,
        value: float,
        model_prob: float,
        implied_prob: float,
        confidence: float,
        odds: float,
    ) -> float:
        """
        计算投注优先级评分 — 规范第8.3节。

        priority_score = f_actual × value × confidence

        其中:
        - f_actual: 经折扣后的 Kelly 建议注额比例 (初始估计值)
        - value: 模型概率相对于市场隐含概率的价值
        - confidence: 模型置信度评分 (0~1)
        """
        if implied_prob <= 0:
            return 0.0

        # 初始 Kelly 估计值 (用于优先级排序，规范第8.3节)
        b = odds - 1.0
        if b < 1e-6:
            return 0.0
        p = max(0.0001, min(0.9999, model_prob))
        q = 1.0 - p
        f_kelly_initial = (b * p - q) / b
        if f_kelly_initial <= 0:
            return 0.0
        f_actual = f_kelly_initial * self.kelly_discount

        score = f_actual * value * confidence
        return max(0.0, score)

    def sort_by_priority(
        self,
        proposals: List[BetProposal],
    ) -> List[BetProposal]:
        """按优先级评分降序排列投注建议"""
        proposals.sort(key=lambda x: x.priority_score, reverse=True)
        return proposals

    # ================================================================
    # Stage 7: 资金分配 — 规范第9.4节
    # ================================================================
    def compute_allocation_score(
        self,
        model_prob: float,
        implied_prob: float,
        confidence: float,
    ) -> float:
        """
        计算资金分配评分 — 规范第9.4节。

        score_i = value_i × confidence_i × edge_i

        其中:
        - value_i = model_prob / implied_prob − 1  (价值比率)
        - confidence_i = 模型置信度 (0~1)
        - edge_i = max(value_i, 0)  (边际优势)
        """
        if implied_prob <= 0:
            return 0.0

        value_ratio = (model_prob / implied_prob) - 1.0
        edge = max(value_ratio, 0.0)

        score = value_ratio * confidence * edge
        return max(0.0, score)

    def allocate_stakes(
        self,
        proposals: List[BetProposal],
        kelly_discount: Optional[float] = None,
        match_phase: float = 1.0,
        strategy_weights: Optional[Dict[str, float]] = None,
    ) -> List[BetProposal]:
        """
        v5.8: 多注资金分配 — 规范第9.4节 (单层Kelly优化)。

        v5.10.10: 策略权重 — 高ROI策略获得更多资金分配
        - strategy_weights: {"1x2": 1.2, "asian_handicap": 1.0, "over_under": 1.0}
        - 权重乘入 priority_score, 间接影响 scoring_proportion

        v5.8 (FIND-011): 去除双重折扣
        - 旧: 全Kelly → 1/4 Kelly → scoring_proportion (三层折扣)
        - 新: 全Kelly → scoring_proportion (两层折扣)
        - 默认 kelly_discount=1.0 (全Kelly), 仅通过 scoring_proportion 分配

        流程:
        1. 计算每个投注的全额 Kelly 比例 f_i (kelly_discount=1.0)
        2. 计算评分比例 scoring_proportion_i = score_i / Σ score_j
        3. 计算 stake_i = f_i × bankroll × scoring_proportion_i
        4. 若总 stake > bankroll × 0.20，按等比例缩减
        5. 四舍五入到 2 位小数
        """
        if not proposals:
            return []

        # v5.8: 默认使用全Kelly (1.0), 不再预先打1/4折扣
        discount = kelly_discount if kelly_discount is not None else 1.0
        bankroll = self._get_base_bankroll()  # v5.9: 固定基数模式

        # v5.10.10: 策略权重 — 高ROI策略获得更多资金
        weights = strategy_weights or {}

        total_score = 0.0
        for p in proposals:
            sw = weights.get(p.strategy_type, 1.0)
            total_score += p.priority_score * sw

        if total_score < 1e-10:
            return []

        total_stake = 0.0
        for p in proposals:
            sw = weights.get(p.strategy_type, 1.0)
            scoring_proportion = p.priority_score * sw / total_score
            # v5.8: 全Kelly, 仅通过 scoring_proportion 分配
            is_away = (p.selection == BetSelection.AWAY_WIN)
            f_actual = self.compute_kelly(
                p.model_prob, p.odds, discount=1.0,
                is_away=is_away, match_phase=match_phase,
            )
            p.kelly_stake = f_actual * bankroll
            p.adjusted_stake = p.kelly_stake * scoring_proportion
            total_stake += p.adjusted_stake

        # 总曝光上限 = bankroll × 20%
        max_total = bankroll * self.max_exposure
        if total_stake > max_total:
            scale = max_total / total_stake
            for p in proposals:
                p.adjusted_stake *= scale

        # 单注上限 = bankroll × 5%
        single_max = bankroll * self.single_bet_max
        for p in proposals:
            p.adjusted_stake = min(p.adjusted_stake, single_max)

        return proposals

    # ================================================================
    # 资金状态管理
    # ================================================================
    def record_bet(self, bet: BetPlacement):
        """记录一笔投注"""
        self.state.total_staked += bet.stake
        self.state.total_bets += 1

    def settle_bet(
        self,
        bet: BetPlacement,
        result: BetResult,
        profit_loss: float,
    ):
        """结算一笔投注，更新所有资金状态和熔断相关计数"""
        bet.result = result
        bet.profit_loss = profit_loss

        self.state.balance += profit_loss
        self.state.total_returned += profit_loss

        if result == BetResult.WIN:
            self.state.total_wins += 1
            self.state.consecutive_losses = 0
        elif result == BetResult.LOSS:
            self.state.consecutive_losses += 1
            # 更新日/周/月累计亏损 (供熔断检查使用)
            self.state.daily_loss += abs(profit_loss)
            self.state.weekly_loss += abs(profit_loss)
            self.state.monthly_loss += abs(profit_loss)

        # 更新峰值和回撤
        self.state.peak_balance = max(self.state.peak_balance, self.state.balance)
        if self.state.peak_balance > 0:
            self.state.max_drawdown = max(
                self.state.max_drawdown,
                (self.state.peak_balance - self.state.balance) / self.state.peak_balance,
            )

    def reset_daily_loss(self):
        """重置当日亏损累计 (每日 00:00 调用)"""
        self.state.daily_loss = 0.0

    def reset_weekly_loss(self):
        """重置当周亏损累计 (每周一 00:00 调用)"""
        self.state.weekly_loss = 0.0

    def reset_monthly_loss(self):
        """重置当月亏损累计 (每月 1 日 00:00 调用)"""
        self.state.monthly_loss = 0.0

    def reset_all_window_losses(self):
        """重置所有时间窗口亏损累计 (冷启动/人工重置)"""
        self.state.daily_loss = 0.0
        self.state.weekly_loss = 0.0
        self.state.monthly_loss = 0.0


# ================================================================
# 投注建议生成 + 硬性过滤 — 规范第8.4节
# ================================================================

def generate_bet_proposals(
    value_results: Dict[str, Dict[str, float]],
    match_id: str,
    league_id: str,
    factor_count: int = 41,
    data_source_count: int = 5,
    odds_std: float = 0.05,
    match_phase: float = 1.0,
    threshold: Optional[float] = None,
    precomputed_confidence: Optional[float] = None,
    confidence_threshold: float = 0.6,
) -> List[BetProposal]:
    """
    从价值评估结果生成投注建议列表，并执行硬性过滤。

    规范第8.4节硬性过滤:
    - value ≥ 0.03
    - confidence ≥ 0.6
    - odds ≥ 1.05
    - odds ≤ 10.0

    参数:
        precomputed_confidence: 若提供，则跳过内部置信度计算，直接使用该值 (避免重复计算)
    """
    if threshold is None:
        threshold = global_config.pipeline.min_value_threshold

    proposals = []
    outcome_map = {
        "home": BetSelection.HOME_WIN,
        "draw": BetSelection.DRAW,
        "away": BetSelection.AWAY_WIN,
    }

    for outcome, data in value_results.items():
        value = data["value"]
        odds = data["odds"]

        # 硬性过滤: value
        if value < threshold:
            continue

        # 硬性过滤: 赔率范围
        if odds < global_config.pipeline.default_odds_min:
            continue
        if odds > global_config.pipeline.default_odds_max:
            continue

        # 计算 confidence (规范第8.3b节)
        if precomputed_confidence is not None:
            confidence = precomputed_confidence
        else:
            data_completeness = min(1.0, data_source_count / 5.0)
            factor_activation_rate = min(1.0, factor_count / 41.0)
            dispersion_penalty = 1.0 - min(1.0, odds_std / 0.15)
            confidence = compute_confidence(
                data_completeness=data_completeness,
                factor_activation_rate=factor_activation_rate,
                dispersion_penalty=dispersion_penalty,
                match_phase=match_phase,
            )

        # 硬性过滤: confidence
        if confidence < confidence_threshold:
            continue

        proposals.append(BetProposal(
            match_id=match_id,
            selection=outcome_map[outcome],
            odds=odds,
            model_prob=data["model_prob"],
            implied_prob=data["implied_prob"],
            value=value,
            kelly_stake=0.0,
            adjusted_stake=0.0,
            priority_score=0.0,
            confidence=confidence,  # v5.10.5: 存储置信度
            league_id=league_id,
        ))

    return proposals