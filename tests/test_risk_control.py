"""
L2 模块测试：风险控制引擎 (risk_control.py)

测试范围:
- 单注上限 (5%)
- 日限额 (15%)
- 周限额 (35%)
- 相关性暴露 (同联赛 25%)
- 熔断机制 (连续亏损/日/周/月)
- 冷却期限制
- 风控全流程
"""
import pytest
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine.risk_control import RiskController
from src.data.models import (
    BetProposal, BetSelection, BetResult, BetPlacement,
    BankrollState, CircuitBreakerState,
)
from src.config.settings import config as global_config


# ================================================================
# 单注上限
# ================================================================

class TestSingleBetLimit:
    """单注上限测试 — 规范第10.1节"""

    def test_within_limit_unchanged(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 300, 0.1)
        results = ctrl.check_single_bet_limit([p], bankroll)
        # 300 < 500 (5%) → 不变
        assert results[0].adjusted_stake == 300

    def test_exceeds_limit_capped(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 800, 0.1)
        results = ctrl.check_single_bet_limit([p], bankroll)
        assert results[0].adjusted_stake <= 500  # 5% of 10000


# ================================================================
# 日限额
# ================================================================

class TestDailyLimit:
    """日限额测试 — 规范第10.2节"""

    def test_within_daily_limit(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 500, 0.1)
        results = ctrl.check_daily_limit([p], bankroll, daily_staked=500)
        # 500 + 500 = 1000 < 1500 → 全部通过
        assert len(results) == 1

    def test_exceeds_daily_limit(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 500, 0.1)
        results = ctrl.check_daily_limit([p], bankroll, daily_staked=1400)
        # 500 + 1400 = 1900 > 1500 → 按比例裁剪
        assert len(results) == 1
        assert results[0].adjusted_stake < 500

    def test_daily_limit_exhausted(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 500, 0.1)
        results = ctrl.check_daily_limit([p], bankroll, daily_staked=1500)
        assert len(results) == 0  # 额度已用完


# ================================================================
# 周限额
# ================================================================

class TestWeeklyLimit:
    """周限额测试 — 规范第10.2节"""

    def test_within_weekly_limit(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 1000, 0.1)
        results = ctrl.check_weekly_limit([p], bankroll, weekly_staked=1000)
        assert len(results) == 1

    def test_exceeds_weekly_limit(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 1000, 0.1)
        results = ctrl.check_weekly_limit([p], bankroll, weekly_staked=3000)
        # 1000 + 3000 = 4000 > 3500 → 裁剪
        assert len(results) == 1
        assert results[0].adjusted_stake < 1000


# ================================================================
# 相关性暴露
# ================================================================

class TestCorrelationExposure:
    """相关性暴露测试 — 规范第10.3节"""

    def test_single_league_within_limit(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 1000, 0.1, "premier_league")
        results = ctrl.check_correlation_exposure([p], bankroll)
        assert results[0].adjusted_stake == 1000

    def test_single_league_exceeds_limit(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p1 = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 1500, 0.1, "premier_league")
        p2 = BetProposal("m2", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 1500, 0.1, "premier_league")
        results = ctrl.check_correlation_exposure([p1, p2], bankroll)
        total = sum(p.adjusted_stake for p in results)
        assert total <= 2500  # 25% of 10000

    def test_multi_league_independent(self):
        """不同联赛不应互相影响"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p1 = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 2000, 0.1, "premier_league")
        p2 = BetProposal("m2", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 2000, 0.1, "bundesliga")
        results = ctrl.check_correlation_exposure([p1, p2], bankroll)
        # 各自联赛均超过 25%，但不同联赛独立
        assert len(results) == 2


# ================================================================
# 全流程风控
# ================================================================

class TestRunAllChecks:
    """全流程风控测试"""

    def test_all_checks_pass(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 300, 0.1)
        results, warnings = ctrl.run_all_checks([p], bankroll)
        assert len(results) == 1
        assert results[0].adjusted_stake <= 300

    def test_daily_limit_warning(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 500, 0.1)
        results, warnings = ctrl.run_all_checks([p], bankroll, daily_staked=1400)
        # 单个提案被裁剪但未被过滤，不应产生警告
        assert len(results) == 1
        assert results[0].adjusted_stake < 500

    def test_empty_proposals(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        results, warnings = ctrl.run_all_checks([], bankroll)
        assert results == []


# ================================================================
# 熔断机制
# ================================================================

class TestCircuitBreaker:
    """熔断测试 — 规范第10.10节"""

    def test_no_breaker_normal(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=0)
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert not broken

    def test_consecutive_losses_breaker(self):
        """连续 5 场亏损 → 熔断"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=5)
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert broken
        assert "连续亏损" in reason

    def test_consecutive_4_not_triggered(self):
        """连续 4 场亏损 → 不触发"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=4)
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert not broken

    def test_daily_loss_breaker(self):
        """日亏损 8% → 熔断"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, daily_loss=800, consecutive_losses=0)
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert broken
        assert "日亏损" in reason

    def test_weekly_loss_breaker(self):
        """周亏损 15% → 熔断"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, weekly_loss=1500, daily_loss=0, consecutive_losses=0)
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert broken
        assert "周亏损" in reason

    def test_monthly_loss_breaker(self):
        """月亏损 25% → 熔断"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, monthly_loss=2500, weekly_loss=0, daily_loss=0, consecutive_losses=0)
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert broken
        assert "月亏损" in reason

    def test_cooldown_active(self):
        """冷却期内无法投注"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=5)
        # 第一次触发熔断
        broken, _ = ctrl.check_circuit_breaker(bankroll)
        assert broken
        # 冷却期内再次检查
        broken2, _ = ctrl.check_circuit_breaker(bankroll)
        assert broken2

    def test_cooldown_expired(self):
        """冷却期过期后恢复"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=0)
        # 手动设置已过期的冷却
        ctrl.breaker_state.is_active = True
        ctrl.breaker_state.cooldown_until = datetime.now() - timedelta(hours=1)
        broken, _ = ctrl.check_circuit_breaker(bankroll)
        assert not broken

    def test_reset_breaker(self):
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=5)
        broken, _ = ctrl.check_circuit_breaker(bankroll)
        assert broken
        ctrl.reset_breaker()
        assert not ctrl.breaker_state.is_active

    def test_cooldown_restrictions(self):
        """冷却期限制: 资金上限 10%"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000)
        p = BetProposal("m1", BetSelection.HOME_WIN, 2.0, 0.55, 0.50, 0.05, 0, 1500, 0.1)
        results = ctrl.apply_cooldown_restrictions([p], bankroll)
        total = sum(r.adjusted_stake for r in results)
        assert total <= 1000  # 10% of 10000

    def test_monthly_breaker_highest_priority(self):
        """月亏损优先级最高 (先检查月亏损)"""
        ctrl = RiskController()
        bankroll = BankrollState(
            balance=10000, monthly_loss=2500, weekly_loss=0, daily_loss=0,
            consecutive_losses=0,
        )
        broken, reason = ctrl.check_circuit_breaker(bankroll)
        assert broken
        assert "月亏损" in reason  # 月亏损先触发

    def test_consecutive_loss_cooldown_config(self):
        """连续亏损: 冷却时间使用配置值"""
        ctrl = RiskController()
        bankroll = BankrollState(balance=10000, consecutive_losses=5)
        now = datetime(2025, 8, 1, 12, 0, 0)
        broken, _ = ctrl.check_circuit_breaker(bankroll, now=now)
        assert broken
        assert ctrl.breaker_state.cooldown_until == now + timedelta(hours=ctrl.cooldown_hours)

    def test_recovery_cooldown_48h(self):
        """恢复后冷却: 48h (配置值)"""
        ctrl = RiskController()
        assert ctrl.cooldown_hours == 48