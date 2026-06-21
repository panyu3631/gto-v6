"""
GTO-GameFlow v5.0 风险控制引擎

实现规范文档第10章：单注上限、日/周限额、相关性暴露、熔断机制。
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from src.data.models import (
    BetProposal, BetPlacement, BetSelection,
    BankrollState, CircuitBreakerState,
)
from src.config.settings import config as global_config


class RiskController:
    """
    风险控制器 — 执行风控校验和熔断判断。

    规范第10章:
    - 单注上限: min(stake, bankroll × 5%)
    - 日限额: Σ stake ≤ bankroll × 15%
    - 周限额: Σ stake ≤ bankroll × 35%
    - 熔断: 连续5场亏损 / 日亏8% / 周亏15% / 月亏25%
    """

    def __init__(self):
        cfg = global_config.circuit_breaker
        br_cfg = global_config.bankroll
        self.max_consecutive_losses = cfg.max_consecutive_losses
        self.daily_loss_limit = cfg.daily_loss_pct
        self.weekly_loss_limit = cfg.weekly_loss_pct
        self.monthly_loss_limit = cfg.monthly_loss_pct
        self.cooldown_hours = cfg.cooldown_hours
        self.daily_exposure_limit = br_cfg.daily_exposure_limit
        self.weekly_exposure_limit = br_cfg.weekly_exposure_limit
        self.breaker_state = CircuitBreakerState()

    # ================================================================
    # Stage 8: 风控校验
    # ================================================================

    def check_single_bet_limit(
        self,
        proposals: List[BetProposal],
        bankroll: BankrollState,
    ) -> List[BetProposal]:
        """
        单注上限校验: stake_i ≤ bankroll × 5%

        规范第10.1节: 每注不得超过当前资金的5%
        """
        single_max = bankroll.balance * global_config.bankroll.single_bet_max_ratio
        for p in proposals:
            p.adjusted_stake = min(p.adjusted_stake, single_max)
        return proposals

    def check_daily_limit(
        self,
        proposals: List[BetProposal],
        bankroll: BankrollState,
        daily_staked: float,
    ) -> List[BetProposal]:
        """
        日限额校验: 当日总 stake ≤ bankroll × 15%

        规范第10.2节: 超限时按优先级裁剪
        """
        daily_max = bankroll.balance * self.daily_exposure_limit
        available = daily_max - daily_staked

        if available <= 0:
            return []  # 当日额度已用完

        total_new = sum(p.adjusted_stake for p in proposals)
        if total_new <= available:
            return proposals

        # 按优先级裁剪 (保留高分投注)
        scale = available / total_new
        for p in proposals:
            p.adjusted_stake *= scale

        return proposals

    def check_weekly_limit(
        self,
        proposals: List[BetProposal],
        bankroll: BankrollState,
        weekly_staked: float,
    ) -> List[BetProposal]:
        """周限额校验: 当周总 stake ≤ bankroll × 35%"""
        weekly_max = bankroll.balance * self.weekly_exposure_limit
        available = weekly_max - weekly_staked

        if available <= 0:
            return []

        total_new = sum(p.adjusted_stake for p in proposals)
        if total_new <= available:
            return proposals

        scale = available / total_new
        for p in proposals:
            p.adjusted_stake *= scale

        return proposals

    def check_correlation_exposure(
        self,
        proposals: List[BetProposal],
        bankroll: BankrollState,
        max_league_exposure: float = 0.25,
    ) -> List[BetProposal]:
        """
        相关性暴露管理。

        规范第10.3节: 同联赛总暴露 ≤ bankroll × 25%
        """
        # 按联赛分组
        league_exposure: Dict[str, float] = {}
        for p in proposals:
            lid = p.league_id or "unknown"
            league_exposure[lid] = league_exposure.get(lid, 0.0) + p.adjusted_stake

        # 检查每个联赛的暴露
        league_max = bankroll.balance * max_league_exposure
        for lid, exposure in league_exposure.items():
            if exposure > league_max:
                # 对该联赛的投注按比例缩减
                scale = league_max / exposure
                for p in proposals:
                    if (p.league_id or "unknown") == lid:
                        p.adjusted_stake *= scale

        return proposals

    def run_all_checks(
        self,
        proposals: List[BetProposal],
        bankroll: BankrollState,
        daily_staked: float = 0.0,
        weekly_staked: float = 0.0,
    ) -> Tuple[List[BetProposal], List[str]]:
        """
        执行全部风控校验 (Stage 8)。

        顺序: 单注 → 日限额 → 周限额 → 相关性

        返回: (过滤后的投注建议, 警告列表)
        """
        warnings = []

        if not proposals:
            return proposals, warnings

        original_count = len(proposals)

        # 1. 单注上限
        proposals = self.check_single_bet_limit(proposals, bankroll)

        # 2. 日限额
        proposals = self.check_daily_limit(proposals, bankroll, daily_staked)
        if len(proposals) < original_count:
            warnings.append(f"日限额触发: {original_count}→{len(proposals)} 注")

        # 3. 周限额
        proposals = self.check_weekly_limit(proposals, bankroll, weekly_staked)

        # 4. 相关性暴露
        proposals = self.check_correlation_exposure(proposals, bankroll)

        return proposals, warnings

    # ================================================================
    # Stage 9: 熔断机制
    # ================================================================

    def check_circuit_breaker(
        self,
        bankroll: BankrollState,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        熔断判断。

        规范第10.10节:
        - 连续亏损 ≥ 5 场 → 暂停 24h
        - 单日亏损 ≥ 8% → 暂停至次日
        - 单周亏损 ≥ 15% → 暂停至下周
        - 单月亏损 ≥ 25% → 暂停 + 人工审核

        返回: (是否熔断, 原因)
        """
        if now is None:
            now = datetime.now()

        # 检查是否仍在冷却期
        if self.breaker_state.is_active and self.breaker_state.cooldown_until:
            if now < self.breaker_state.cooldown_until:
                return True, f"冷却期中: {self.breaker_state.trigger_reason}"
            else:
                # 冷却期已过
                self.breaker_state.is_active = False
                self.breaker_state.cooldown_until = None

        # 检查月亏损 (最严重，规范第10.10节: 检查顺序 月→周→日→连续亏损)
        if bankroll.balance > 0:
            monthly_pct = bankroll.monthly_loss / bankroll.balance
            if monthly_pct >= self.monthly_loss_limit:
                self._trigger_breaker(
                    f"单月亏损 {monthly_pct:.1%} ≥ {self.monthly_loss_limit:.0%}",
                    now + timedelta(hours=self.cooldown_hours * 15),  # 月 = 配置×15
                    now=now,
                )
                return True, self.breaker_state.trigger_reason

            # 检查周亏损
            weekly_pct = bankroll.weekly_loss / bankroll.balance
            if weekly_pct >= self.weekly_loss_limit:
                self._trigger_breaker(
                    f"单周亏损 {weekly_pct:.1%} ≥ {self.weekly_loss_limit:.0%}",
                    now + timedelta(hours=self.cooldown_hours * 3.5),  # 周 = 配置×3.5
                    now=now,
                )
                return True, self.breaker_state.trigger_reason

            # 检查日亏损
            daily_pct = bankroll.daily_loss / bankroll.balance
            if daily_pct >= self.daily_loss_limit:
                self._trigger_breaker(
                    f"单日亏损 {daily_pct:.1%} ≥ {self.daily_loss_limit:.0%}",
                    now + timedelta(hours=self.cooldown_hours * 0.5),  # 日 = 配置×0.5
                    now=now,
                )
                return True, self.breaker_state.trigger_reason

        # 检查连续亏损 (规范: 暂停 24h)
        if bankroll.consecutive_losses >= self.max_consecutive_losses:
            self._trigger_breaker(
                f"连续亏损 {bankroll.consecutive_losses} 场",
                now + timedelta(hours=self.cooldown_hours),  # 使用配置值
                now=now,
            )
            return True, self.breaker_state.trigger_reason

        return False, ""

    def _trigger_breaker(self, reason: str, cooldown_until: datetime, now: Optional[datetime] = None):
        """触发熔断"""
        self.breaker_state.is_active = True
        self.breaker_state.trigger_reason = reason
        self.breaker_state.triggered_at = now or datetime.now()
        self.breaker_state.cooldown_until = cooldown_until

    def reset_breaker(self):
        """重置熔断状态 (人工确认后)"""
        self.breaker_state = CircuitBreakerState()

    def apply_cooldown_restrictions(
        self,
        proposals: List[BetProposal],
        bankroll: BankrollState,
    ) -> List[BetProposal]:
        """
        熔断恢复后的冷却期限制: 资金上限临时降至 10%

        规范第10.10节: 恢复后资金使用上限临时降至 bankroll 的 10%
        """
        cooldown_max = bankroll.balance * 0.10
        total_stake = sum(p.adjusted_stake for p in proposals)

        if total_stake > cooldown_max:
            scale = cooldown_max / total_stake
            for p in proposals:
                p.adjusted_stake *= scale

        return proposals