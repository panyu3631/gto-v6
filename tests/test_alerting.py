"""
GTO-GameFlow v5.5 — 告警系统测试

测试内容:
1. AlertManager 基础功能 (创建、发送、冷却、频率限制)
2. PipelineAlertAdapter 告警检测
3. 邮件通知器配置验证
4. 集成到回测流程

用法:
    python tests/test_alerting.py           # 仅 dry-run (不发送邮件)
    python tests/test_alerting.py --send    # 发送测试邮件
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_alerting")


def test_alert_creation():
    """测试告警创建"""
    from src.monitoring import Alert, AlertLevel, AlertType

    alert = Alert(
        level=AlertLevel.CRITICAL,
        alert_type=AlertType.CIRCUIT_BREAKER,
        title="熔断测试",
        message="这是一条测试告警",
        metadata={"league": "serie_a", "loss": "8.5%"},
    )
    assert alert.level == AlertLevel.CRITICAL
    assert alert.alert_type == AlertType.CIRCUIT_BREAKER
    assert alert.metadata["league"] == "serie_a"
    assert not alert.sent
    assert isinstance(alert.timestamp, datetime)
    logger.info("PASS: test_alert_creation")


def test_email_notifier_structure():
    """测试邮件通知器结构"""
    from src.monitoring import EmailNotifier

    notifier = EmailNotifier(
        smtp_host="smtp.qq.com",
        smtp_port=465,
        sender="test@qq.com",
        password="test_auth_code",
        recipients=["test@qq.com"],
    )
    assert notifier.smtp_host == "smtp.qq.com"
    assert notifier.smtp_port == 465
    assert notifier.use_ssl is True
    logger.info("PASS: test_email_notifier_structure")


def test_email_notifier_no_config():
    """测试未配置时的安全处理"""
    from src.monitoring import EmailNotifier, Alert, AlertLevel, AlertType

    notifier = EmailNotifier()  # 空配置
    alert = Alert(level=AlertLevel.INFO, alert_type=AlertType.SYSTEM, title="测试", message="test")
    result = notifier.send(alert)
    assert result is False
    assert "配置不完整" in notifier._last_error
    logger.info("PASS: test_email_notifier_no_config")


def test_alert_manager_cooldown():
    """测试告警管理器冷却机制"""
    from src.monitoring import AlertManager, Alert, AlertLevel, AlertType

    mgr = AlertManager(
        notifier=None,  # dry-run
        cooldown_seconds=2,
        max_daily_sends=100,
        enabled=True,
    )

    alert = Alert(level=AlertLevel.WARN, alert_type=AlertType.DRAWDOWN, title="回撤测试", message="测试")

    # 第一次发送成功
    result1 = mgr.send(alert)
    assert result1 is True, "第一次发送应该成功"

    # 立即第二次发送应被冷却抑制
    result2 = mgr.send(alert)
    assert result2 is False, "冷却期内第二次发送应被抑制"

    # CRITICAL 级别绕过冷却
    critical = Alert(level=AlertLevel.CRITICAL, alert_type=AlertType.CIRCUIT_BREAKER, title="熔断", message="测试")
    result3 = mgr.send(critical)
    assert result3 is True, "CRITICAL 级别应绕过冷却"

    # 等待冷却期后恢复
    time.sleep(2.5)
    result4 = mgr.send(Alert(level=AlertLevel.WARN, alert_type=AlertType.DRAWDOWN, title="回撤2", message="测试"))
    assert result4 is True, "冷却期后应恢复发送"

    logger.info("PASS: test_alert_manager_cooldown")


def test_alert_manager_daily_limit():
    """测试每日发送上限"""
    from src.monitoring import AlertManager, Alert, AlertLevel, AlertType

    mgr = AlertManager(notifier=None, max_daily_sends=5, enabled=True)

    # 发送 5 条成功
    for i in range(5):
        alert = Alert(level=AlertLevel.INFO, alert_type=AlertType.SYSTEM, title=f"测试{i}", message="test")
        result = mgr.send(alert, bypass_cooldown=True)
        assert result is True, f"第 {i + 1} 条应成功"

    # 第 6 条被抑制
    alert6 = Alert(level=AlertLevel.INFO, alert_type=AlertType.SYSTEM, title="测试6", message="test")
    result6 = mgr.send(alert6, bypass_cooldown=True)
    assert result6 is False, "超过日上限应被抑制"

    logger.info("PASS: test_alert_manager_daily_limit")


def test_alert_manager_disabled():
    """测试告警禁用"""
    from src.monitoring import AlertManager, Alert, AlertLevel, AlertType

    mgr = AlertManager(notifier=None, enabled=False)
    alert = Alert(level=AlertLevel.CRITICAL, alert_type=AlertType.CIRCUIT_BREAKER, title="测试", message="test")
    result = mgr.send(alert)
    assert result is False, "禁用时应返回 False"
    logger.info("PASS: test_alert_manager_disabled")


def test_pipeline_adapter_circuit_breaker():
    """测试流水线适配器 — 熔断告警"""
    from src.monitoring import PipelineAlertAdapter, AlertManager, AlertLevel, AlertType

    mgr = AlertManager(notifier=None, enabled=True)
    adapter = PipelineAlertAdapter(mgr)

    # 模拟熔断
    alert = adapter.check_circuit_breaker(
        circuit_broken=True,
        circuit_reason="单日亏损 8.5% 超过 8% 阈值",
        league_id="premier_league",
    )
    assert alert is not None
    assert alert.level == AlertLevel.CRITICAL
    assert alert.alert_type == AlertType.CIRCUIT_BREAKER
    assert "单日亏损" in alert.message
    logger.info("PASS: test_pipeline_adapter_circuit_breaker")

    # 未熔断时不产生告警
    alert2 = adapter.check_circuit_breaker(circuit_broken=False, circuit_reason="")
    assert alert2 is None
    logger.info("PASS: test_pipeline_adapter_no_circuit_breaker")


def test_pipeline_adapter_drawdown():
    """测试流水线适配器 — 回撤告警"""
    from src.monitoring import PipelineAlertAdapter, AlertManager, AlertLevel, AlertType
    from src.data.models import BankrollState

    mgr = AlertManager(notifier=None, enabled=True)
    adapter = PipelineAlertAdapter(mgr, drawdown_warn_pct=0.20, drawdown_critical_pct=0.30)

    state = BankrollState(balance=7500.0)
    state.peak_balance = 10000.0
    state.max_drawdown = 0.25
    state.consecutive_losses = 3

    # 回撤 25% → WARN
    alert = adapter.check_drawdown(current_drawdown=0.25, bankroll_state=state, league_id="premier_league")
    assert alert is not None
    assert alert.level == AlertLevel.WARN
    assert alert.alert_type == AlertType.DRAWDOWN
    logger.info("PASS: test_pipeline_adapter_drawdown_warn")

    # 回撤 35% → CRITICAL
    state.max_drawdown = 0.35
    adapter._last_drawdown_alert = None  # 重置冷却
    alert2 = adapter.check_drawdown(current_drawdown=0.35, bankroll_state=state, league_id="premier_league")
    assert alert2 is not None
    assert alert2.level == AlertLevel.CRITICAL
    logger.info("PASS: test_pipeline_adapter_drawdown_critical")

    # 回撤 < 20% → 无告警
    alert3 = adapter.check_drawdown(current_drawdown=0.10, bankroll_state=state)
    assert alert3 is None
    logger.info("PASS: test_pipeline_adapter_drawdown_none")


def test_pipeline_adapter_abnormal_odds():
    """测试流水线适配器 — 异常赔率告警"""
    from src.monitoring import PipelineAlertAdapter, AlertManager, AlertLevel, AlertType

    mgr = AlertManager(notifier=None, enabled=True)
    adapter = PipelineAlertAdapter(mgr, odd_std_warn=0.08)

    # 赔率标准差超标
    alert = adapter.check_abnormal_odds(
        odds_std=0.12,
        league_id="bundesliga",
        match_id="BAYvsDOR",
        home_odds=1.80,
        draw_odds=3.50,
        away_odds=4.50,
    )
    assert alert is not None
    assert alert.level == AlertLevel.WARN
    assert alert.alert_type == AlertType.ABNORMAL_ODDS
    assert "0.12" in alert.message
    logger.info("PASS: test_pipeline_adapter_abnormal_odds")

    # 正常赔率
    alert2 = adapter.check_abnormal_odds(odds_std=0.03, league_id="bundesliga", match_id="BAYvsDOR")
    assert alert2 is None
    logger.info("PASS: test_pipeline_adapter_abnormal_odds_normal")


def test_pipeline_adapter_pipeline_errors():
    """测试流水线适配器 — 流水线错误告警"""
    from src.monitoring import PipelineAlertAdapter, AlertManager, AlertLevel, AlertType

    mgr = AlertManager(notifier=None, enabled=True)
    adapter = PipelineAlertAdapter(mgr)

    alert = adapter.check_pipeline_errors(
        errors=["Division by zero in Poisson calculation", "Factor weight NaN detected"],
        league_id="serie_a",
        match_id="MILvsJUV",
    )
    assert alert is not None
    assert alert.level == AlertLevel.CRITICAL
    assert alert.alert_type == AlertType.PIPELINE_ERROR
    assert "2 个错误" in alert.message
    logger.info("PASS: test_pipeline_adapter_pipeline_errors")

    # 无错误
    alert2 = adapter.check_pipeline_errors(errors=[], league_id="serie_a")
    assert alert2 is None
    logger.info("PASS: test_pipeline_adapter_pipeline_errors_none")


def test_pipeline_adapter_large_bets():
    """测试流水线适配器 — 大额投注告警"""
    from src.monitoring import PipelineAlertAdapter, AlertManager, AlertLevel, AlertType
    from src.data.models import BankrollState, BetProposal, BetSelection

    mgr = AlertManager(notifier=None, enabled=True)
    adapter = PipelineAlertAdapter(mgr, large_bet_ratio=0.04)

    state = BankrollState(balance=10000.0)

    # 模拟大额投注 (5% 资金)
    proposal = BetProposal(
        match_id="TEST001",
        selection=BetSelection.HOME_WIN,
        odds=2.10,
        model_prob=0.55,
        implied_prob=0.48,
        value=0.07,
        kelly_stake=500.0,
        adjusted_stake=500.0,  # 5% of balance
        priority_score=0.8,
    )

    alerts = adapter.check_large_bets(
        proposals=[proposal],
        bankroll_state=state,
        league_id="la_liga",
        match_id="TEST001",
    )
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARN
    assert alerts[0].alert_type == AlertType.LARGE_BET
    logger.info("PASS: test_pipeline_adapter_large_bets")

    # 小投注不触发
    proposal2 = BetProposal(
        match_id="TEST002",
        selection=BetSelection.DRAW,
        odds=3.50,
        model_prob=0.32,
        implied_prob=0.28,
        value=0.04,
        kelly_stake=100.0,
        adjusted_stake=100.0,  # 1% of balance
        priority_score=0.6,
    )
    alerts2 = adapter.check_large_bets(
        proposals=[proposal2],
        bankroll_state=state,
        league_id="la_liga",
        match_id="TEST002",
    )
    assert len(alerts2) == 0
    logger.info("PASS: test_pipeline_adapter_large_bets_normal")


def test_daily_summary():
    """测试每日摘要"""
    from src.monitoring import PipelineAlertAdapter, AlertManager, AlertLevel, AlertType
    from src.data.models import BankrollState

    mgr = AlertManager(notifier=None, enabled=True)
    adapter = PipelineAlertAdapter(mgr)

    state = BankrollState(balance=11500.0)
    state.peak_balance = 12000.0
    state.max_drawdown = 0.04

    alert = adapter.send_daily_summary(
        date_str="2026-06-17",
        bankroll_state=state,
        daily_bets=12,
        daily_wins=8,
        daily_profit=350.0,
        league_id="serie_a",
    )
    assert alert is not None
    assert alert.level == AlertLevel.INFO
    assert alert.alert_type == AlertType.DAILY_SUMMARY
    assert "12 注" in alert.message
    assert "8 注" in alert.message
    logger.info("PASS: test_daily_summary")


def test_create_qq_email_manager():
    """测试 QQ 邮箱告警管理器工厂函数"""
    from src.monitoring import create_qq_email_alert_manager

    mgr = create_qq_email_alert_manager(
        sender="739252249@qq.com",
        auth_code="test_code_placeholder",
        recipients=["739252249@qq.com"],
        cooldown_seconds=600,
        max_daily_sends=30,
        enabled=True,
    )
    assert mgr.enabled is True
    assert mgr.cooldown_seconds == 600
    assert mgr.max_daily_sends == 30
    assert mgr.notifier is not None
    assert mgr.notifier.smtp_host == "smtp.qq.com"
    assert mgr.notifier.smtp_port == 465
    assert mgr.notifier.sender == "739252249@qq.com"
    logger.info("PASS: test_create_qq_email_manager")


def test_alert_stats():
    """测试告警统计"""
    from src.monitoring import AlertManager, Alert, AlertLevel, AlertType

    mgr = AlertManager(notifier=None, max_daily_sends=100, enabled=True)

    for i in range(5):
        mgr.send(Alert(level=AlertLevel.WARN, alert_type=AlertType.DRAWDOWN,
                       title=f"测试{i}", message="test"), bypass_cooldown=True)

    stats = mgr.stats
    assert stats["enabled"] is True
    assert stats["total_sent"] == 5
    assert stats["daily_count"] == 5
    logger.info(f"PASS: test_alert_stats — {stats}")


def test_email_format():
    """测试邮件 HTML 格式化"""
    from src.monitoring import EmailNotifier, Alert, AlertLevel, AlertType

    notifier = EmailNotifier(sender="test@qq.com", password="xxx", recipients=["test@qq.com"])

    alert = Alert(
        level=AlertLevel.CRITICAL,
        alert_type=AlertType.CIRCUIT_BREAKER,
        title="熔断触发 — 英超",
        message="单日亏损 8.5% 超过 8% 阈值，已触发熔断。",
        metadata={"league": "premier_league", "daily_loss_pct": "8.5%"},
    )

    subject = notifier._format_subject(alert)
    assert "熔断" in subject
    assert "[GTO-v5.5]" in subject

    html = notifier._format_html(alert)
    assert "熔断触发" in html
    assert "8.5%" in html
    assert "premier_league" in html
    assert "#ff4757" in html  # CRITICAL 颜色
    logger.info("PASS: test_email_format")


def test_batch_format():
    """测试批量邮件格式化"""
    from src.monitoring import EmailNotifier, Alert, AlertLevel, AlertType

    notifier = EmailNotifier(sender="test@qq.com", password="xxx", recipients=["test@qq.com"])

    alerts = [
        Alert(level=AlertLevel.WARN, alert_type=AlertType.DRAWDOWN, title="回撤警告", message="英超回撤 22%"),
        Alert(level=AlertLevel.INFO, alert_type=AlertType.SYSTEM, title="系统正常", message="所有检查通过"),
    ]
    body = notifier._render_batch_body(alerts)
    assert "回撤警告" in body
    assert "系统正常" in body
    logger.info("PASS: test_batch_format")


# ═══════════════════════════════════════════════════════════════
# 发送测试邮件 (需 --send 参数)
# ═══════════════════════════════════════════════════════════════

def send_test_email():
    """发送一封测试邮件到 QQ 邮箱 (需要有效的授权码)"""
    from src.monitoring import create_qq_email_alert_manager, Alert, AlertLevel, AlertType

    # 要求用户输入授权码
    print("\n" + "=" * 60)
    print("  发送测试邮件到 739252249@qq.com")
    print("=" * 60)
    print()
    print("需要 QQ 邮箱 SMTP 授权码 (非 QQ 密码!)")
    print("获取方式: QQ邮箱 → 设置 → 账户 → POP3/SMTP服务 → 生成授权码")
    print()

    auth_code = input("请输入授权码: ").strip()
    if not auth_code:
        print("未输入授权码，跳过邮件发送测试。")
        return

    mgr = create_qq_email_alert_manager(
        sender="739252249@qq.com",
        auth_code=auth_code,
        recipients=["739252249@qq.com"],
        cooldown_seconds=0,
        max_daily_sends=10,
        enabled=True,
    )

    alert = Alert(
        level=AlertLevel.INFO,
        alert_type=AlertType.SYSTEM,
        title="GTO-GameFlow v5.5 告警系统测试",
        message=(
            "这是一封测试邮件，确认告警通知系统配置正确。\n\n"
            "如果您收到此邮件，说明告警系统已正确配置！\n\n"
            "系统将自动发送以下类型的告警:\n"
            "• 熔断触发 — 日/周/月亏损超标\n"
            "• 大额投注 — 单注超过资金 4%\n"
            "• 异常赔率 — 赔率离散度超标\n"
            "• 回撤警告 — 回撤超过 20%/30%\n"
            "• 每日摘要 — 当日投注汇总\n"
            "• 流水线错误 — 系统异常"
        ),
        metadata={
            "版本": "v5.5",
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "测试": "通过",
        },
    )

    print("正在发送测试邮件...")
    success = mgr.send(alert, bypass_cooldown=True)

    if success:
        print("测试邮件发送成功! 请检查邮箱 739252249@qq.com")
        print(f"通知器统计: {mgr.notifier.stats}")
    else:
        print(f"测试邮件发送失败: {mgr.notifier._last_error}")
        print("可能原因:")
        print("1. 授权码错误 — 请确认是 SMTP 授权码，非 QQ 密码")
        print("2. 未开启 SMTP 服务 — 请在 QQ 邮箱设置中开启")
        print("3. 网络问题 — 请检查能否访问 smtp.qq.com")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GTO-GameFlow v5.5 告警系统测试")
    parser.add_argument("--send", action="store_true", help="发送测试邮件")
    args = parser.parse_args()

    print("=" * 60)
    print("  GTO-GameFlow v5.5 — 告警系统测试")
    print("=" * 60)
    print()

    tests = [
        ("告警创建", test_alert_creation),
        ("邮件通知器结构", test_email_notifier_structure),
        ("空配置安全处理", test_email_notifier_no_config),
        ("告警冷却机制", test_alert_manager_cooldown),
        ("每日发送上限", test_alert_manager_daily_limit),
        ("告警禁用", test_alert_manager_disabled),
        ("熔断告警适配", test_pipeline_adapter_circuit_breaker),
        ("回撤告警适配", test_pipeline_adapter_drawdown),
        ("异常赔率告警适配", test_pipeline_adapter_abnormal_odds),
        ("流水线错误告警适配", test_pipeline_adapter_pipeline_errors),
        ("大额投注告警适配", test_pipeline_adapter_large_bets),
        ("每日摘要", test_daily_summary),
        ("QQ邮箱工厂函数", test_create_qq_email_manager),
        ("告警统计", test_alert_stats),
        ("邮件格式化", test_email_format),
        ("批量格式化", test_batch_format),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAIL: {name} — {e}")
            failed += 1

    print()
    print(f"结果: {passed} 通过, {failed} 失败, {len(tests)} 总计")
    print()

    if args.send:
        send_test_email()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())