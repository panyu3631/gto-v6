"""
GTO-GameFlow v5.5 — 结构化日志与审计追踪测试

测试内容:
1. 结构化日志: JSON 格式、上下文追踪、Stage 计时
2. 审计追踪: 投注生命周期 (proposal → execution → settlement)
3. 流水线剖析器: 耗时统计、联赛统计
4. 集成: orchestrator 日志注入
"""

import sys
import os
import json
import time
import logging
import tempfile
from datetime import datetime
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ═══════════════════════════════════════════════════════════════
# 测试 1: JSON 格式化器
# ═══════════════════════════════════════════════════════════════

def test_json_formatter_basic():
    """测试 JSON 格式化器基本输出"""
    from src.monitoring.structured_logger import JSONFormatter, get_logger

    logger = get_logger("test.formatter")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("test_event some message", extra={"key1": "value1", "key2": 123})

    output = stream.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.formatter"
    assert parsed["event"] == "test_event"
    assert parsed["message"] == "some message"
    assert parsed["key1"] == "value1"
    assert parsed["key2"] == 123
    assert "timestamp" in parsed
    assert "trace_id" in parsed

    logger.removeHandler(handler)
    print("PASS: test_json_formatter_basic")


def test_json_formatter_with_context():
    """测试 JSON 格式化器 + LogContext"""
    from src.monitoring.structured_logger import (
        JSONFormatter, LogContext, get_logger, StageTimer,
    )

    logger = get_logger("test.context")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    with LogContext(match_id="BAYvsDOR", league_id="bundesliga", stage="S4"):
        with StageTimer("S4"):
            time.sleep(0.01)
        logger.info("poisson_complete", extra={"home_goals": 1.5, "away_goals": 0.8})

    output = stream.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["match_id"] == "BAYvsDOR"
    assert parsed["league_id"] == "bundesliga"
    assert parsed["stage"] == "S4"
    assert parsed["home_goals"] == 1.5
    assert parsed["away_goals"] == 0.8
    assert "stage_timings" in parsed
    assert "S4" in parsed["stage_timings"]

    logger.removeHandler(handler)
    print("PASS: test_json_formatter_with_context")


def test_log_context_nesting():
    """测试 LogContext 嵌套"""
    from src.monitoring.structured_logger import LogContext

    LogContext.clear()
    assert LogContext.all() == {}

    with LogContext(league_id="bundesliga"):
        assert LogContext.get("league_id") == "bundesliga"
        with LogContext(match_id="BAYvsDOR", stage="S1"):
            assert LogContext.get("league_id") == "bundesliga"
            assert LogContext.get("match_id") == "BAYvsDOR"
            assert LogContext.get("stage") == "S1"
        # 退出内层后恢复
        assert LogContext.get("league_id") == "bundesliga"
        assert LogContext.get("match_id") is None

    LogContext.clear()
    assert LogContext.all() == {}
    print("PASS: test_log_context_nesting")


# ═══════════════════════════════════════════════════════════════
# 测试 2: 结构化 Logger
# ═══════════════════════════════════════════════════════════════

def test_structured_logger_audit():
    """测试 audit() 方法"""
    from src.monitoring.structured_logger import (
        get_logger, JSONFormatter, AUDIT_LEVEL,
    )

    logger = get_logger("test.audit")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    handler.setLevel(AUDIT_LEVEL)
    logger.addHandler(handler)
    logger.setLevel(AUDIT_LEVEL)

    logger.audit("bet_executed", bet_id="B001", stake=250.0, odds=2.10)

    output = stream.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["level"] == "AUDIT"
    assert parsed["event"] == "bet_executed"
    assert parsed["bet_id"] == "B001"
    assert parsed["stake"] == 250.0
    assert parsed["odds"] == 2.1

    logger.removeHandler(handler)
    print("PASS: test_structured_logger_audit")


def test_structured_logger_event():
    """测试 event() 方法"""
    from src.monitoring.structured_logger import get_logger, JSONFormatter

    logger = get_logger("test.event")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.event("stage_complete", stage="S4", duration_ms=12.3, factors=41)

    output = stream.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["event"] == "stage_complete"
    assert parsed["stage"] == "S4"
    assert parsed["duration_ms"] == 12.3
    assert parsed["factors"] == 41

    logger.removeHandler(handler)
    print("PASS: test_structured_logger_event")


def test_stage_timer():
    """测试 StageTimer 计时"""
    from src.monitoring.structured_logger import StageTimer

    with StageTimer("S1_factor_calc") as timer:
        time.sleep(0.05)

    assert timer.elapsed_ms > 0
    assert timer.stage_name == "S1_factor_calc"
    print(f"PASS: test_stage_timer — {timer.elapsed_ms:.1f}ms")


# ═══════════════════════════════════════════════════════════════
# 测试 3: 审计追踪
# ═══════════════════════════════════════════════════════════════

def test_audit_trail_bet_lifecycle():
    """测试投注完整生命周期审计"""
    from src.monitoring.audit_trail import BetAuditTrail

    trail = BetAuditTrail(max_in_memory=100)

    # Proposal
    trail.record_proposal(
        bet_id="B2026001",
        match_id="BAYvsDOR",
        league_id="bundesliga",
        selection="home_win",
        odds=2.10,
        model_prob=0.55,
        implied_prob=0.48,
        value=0.07,
        kelly_stake=125.0,
        adjusted_stake=100.0,
        priority_score=0.85,
    )

    # Execution
    trail.record_execution(
        bet_id="B2026001",
        match_id="BAYvsDOR",
        league_id="bundesliga",
        selection="home_win",
        odds=2.10,
        stake=100.0,
        balance_before=10000.0,
    )

    # Settlement (WIN)
    trail.record_settlement(
        bet_id="B2026001",
        match_id="BAYvsDOR",
        league_id="bundesliga",
        result="WIN",
        profit_loss=110.0,
        balance_after=10110.0,
        consecutive_losses=0,
    )

    history = trail.get_bet_history("B2026001")
    assert len(history) == 3
    assert history[0]["event"] == "bet_proposed"
    assert history[1]["event"] == "bet_executed"
    assert history[2]["event"] == "bet_settled"
    assert history[2]["profit_loss"] == 110.0

    print("PASS: test_audit_trail_bet_lifecycle")


def test_audit_trail_system_events():
    """测试系统事件审计 (熔断/风控)"""
    from src.monitoring.audit_trail import BetAuditTrail

    trail = BetAuditTrail()

    trail.record_circuit_breaker(
        league_id="premier_league",
        reason="单日亏损 8.5% 超过 8% 阈值",
        balance=9150.0,
        consecutive_losses=5,
    )

    trail.record_risk_limit(
        league_id="serie_a",
        limit_type="single_bet_limit",
        details="stake 600 > max 500",
    )

    trail.record_circuit_breaker_reset(league_id="premier_league")

    sys_events = trail.get_system_events()
    assert len(sys_events) == 3
    assert sys_events[0]["event"] == "circuit_breaker_activated"
    assert sys_events[1]["event"] == "risk_limit_enforced"
    assert sys_events[2]["event"] == "circuit_breaker_reset"

    print("PASS: test_audit_trail_system_events")


def test_audit_trail_stats():
    """测试审计统计"""
    from src.monitoring.audit_trail import BetAuditTrail

    trail = BetAuditTrail()

    for i in range(10):
        bid = f"B{i:04d}"
        trail.record_proposal(bid, f"M{i}", "bundesliga", "home_win", 2.0, 0.55, 0.48, 0.07, 100, 80, 0.8)
        trail.record_execution(bid, f"M{i}", "bundesliga", "home_win", 2.0, 80, 10000)
        result = "WIN" if i < 6 else "LOSS"
        profit = 80 * 1.0 if i < 6 else -80
        trail.record_settlement(bid, f"M{i}", "bundesliga", result, profit, 10000 + profit)

    stats = trail.get_stats()
    assert stats["total_bets"] == 10
    assert stats["settled"] == 10
    assert stats["won"] == 6
    assert stats["lost"] == 4
    assert stats["win_rate"] == 60.0
    assert stats["total_staked"] == 800.0
    assert stats["total_profit"] == 160.0
    assert stats["roi"] == 20.0

    print(f"PASS: test_audit_trail_stats — {stats}")


def test_audit_trail_export():
    """测试审计数据导出 JSONL"""
    from src.monitoring.audit_trail import BetAuditTrail

    trail = BetAuditTrail()
    trail.record_proposal("B001", "M1", "bundesliga", "home_win", 2.0, 0.55, 0.48, 0.07, 100, 80, 0.8)
    trail.record_execution("B001", "M1", "bundesliga", "home_win", 2.0, 80, 10000)
    trail.record_settlement("B001", "M1", "bundesliga", "WIN", 80, 10080)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        fpath = f.name

    count = trail.export_jsonl(fpath)
    assert count == 3

    with open(fpath) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 3
    assert lines[0]["event"] == "bet_proposed"

    os.unlink(fpath)
    print("PASS: test_audit_trail_export")


def test_audit_trail_void():
    """测试投注无效记录"""
    from src.monitoring.audit_trail import BetAuditTrail

    trail = BetAuditTrail()
    trail.record_void("B001", "M1", "ligue_1", "比赛延期")
    history = trail.get_bet_history("B001")
    assert len(history) == 1
    assert history[0]["event"] == "bet_voided"
    assert history[0]["reason"] == "比赛延期"
    print("PASS: test_audit_trail_void")


def test_audit_trail_balance_change():
    """测试资金变更记录"""
    from src.monitoring.audit_trail import BetAuditTrail

    trail = BetAuditTrail()
    trail.record_balance_change(
        league_id="serie_a",
        balance=11500.0,
        peak=12000.0,
        drawdown=0.04,
        reason="daily_summary",
    )

    sys_events = trail.get_system_events()
    assert len(sys_events) == 1
    assert sys_events[0]["event"] == "balance_changed"
    assert sys_events[0]["balance"] == 11500.0
    print("PASS: test_audit_trail_balance_change")


# ═══════════════════════════════════════════════════════════════
# 测试 4: 流水线剖析器
# ═══════════════════════════════════════════════════════════════

def test_pipeline_profiler_basic():
    """测试剖析器基本功能"""
    from src.monitoring.pipeline_profiler import PipelineProfiler

    profiler = PipelineProfiler()

    for i in range(5):
        profiler.start_run(match_id=f"M{i}", league_id="bundesliga")
        time.sleep(0.005)
        profiler.record_stage("S1", 1.2 + i * 0.1)
        profiler.record_stage("S4", 5.5 + i * 0.5)
        profiler.record_bet_result("bundesliga", i < 3)  # 3 wins, 2 losses
        profiler.end_run(bets_placed=2, bets_settled=2, has_error=(i == 4))

    for i in range(3):
        profiler.start_run(match_id=f"M{i + 5}", league_id="serie_a")
        time.sleep(0.003)
        profiler.record_stage("S1", 0.8)
        profiler.record_stage("S4", 3.2)
        profiler.record_bet_result("serie_a", i < 2)  # 2 wins, 1 loss
        profiler.end_run(bets_placed=1, bets_settled=1, has_error=False)

    stats = profiler.get_stats()

    # 整体统计
    assert stats["totals"]["total_runs"] == 8
    assert stats["totals"]["total_bets"] == 13
    assert stats["totals"]["total_errors"] == 1

    # Stage 统计
    assert "S1" in stats["stage_timings"]
    assert stats["stage_timings"]["S1"]["count"] == 8

    # 联赛统计
    assert stats["league_stats"]["bundesliga"]["runs"] == 5
    assert stats["league_stats"]["bundesliga"]["bets"] == 10
    assert stats["league_stats"]["bundesliga"]["wins"] == 3
    assert stats["league_stats"]["serie_a"]["runs"] == 3
    assert stats["league_stats"]["serie_a"]["bets"] == 3
    assert stats["league_stats"]["serie_a"]["wins"] == 2

    # 运行统计
    assert stats["run_stats"]["count"] == 8
    assert stats["run_stats"]["total_errors"] == 1

    print("PASS: test_pipeline_profiler_basic")


def test_pipeline_profiler_summary():
    """测试剖析器摘要输出"""
    from src.monitoring.pipeline_profiler import PipelineProfiler

    profiler = PipelineProfiler()
    profiler.start_run(match_id="M1", league_id="bundesliga")
    profiler.record_stage("S1", 1.5)
    profiler.record_stage("S4", 6.0)
    profiler.record_bet_result("bundesliga", True)
    profiler.end_run(bets_placed=1, bets_settled=1)

    summary = profiler.get_summary()
    assert "GTO-GameFlow" in summary
    assert "bundesliga" in summary
    assert "S1" in summary
    print("PASS: test_pipeline_profiler_summary")


def test_pipeline_profiler_reset():
    """测试剖析器重置"""
    from src.monitoring.pipeline_profiler import PipelineProfiler

    profiler = PipelineProfiler()
    profiler.start_run(match_id="M1", league_id="bundesliga")
    profiler.record_stage("S1", 1.0)
    profiler.end_run(bets_placed=1, bets_settled=1)

    assert profiler.get_stats()["totals"]["total_runs"] == 1

    profiler.reset()
    assert profiler.get_stats()["totals"]["total_runs"] == 0
    print("PASS: test_pipeline_profiler_reset")


# ═══════════════════════════════════════════════════════════════
# 测试 5: 集成 — 日志系统 setup
# ═══════════════════════════════════════════════════════════════

def test_setup_logging():
    """测试日志系统初始化"""
    from src.monitoring.structured_logger import setup_logging, get_logger, shutdown_logging

    tmpdir = tempfile.mkdtemp()
    log_file = os.path.join(tmpdir, "test.jsonl")

    setup_logging(level=logging.DEBUG, log_file=log_file, log_dir=tmpdir)

    logger = get_logger("test.setup")
    logger.info("setup_test", extra={"success": True})

    # 验证文件已创建
    assert os.path.exists(log_file)
    with open(log_file) as f:
        line = f.readline()
        parsed = json.loads(line)
        assert parsed["event"] == "setup_test"
        assert parsed["success"] is True

    # 验证审计目录
    audit_dir = os.path.join(tmpdir, "audit")
    assert os.path.isdir(audit_dir)

    shutdown_logging()

    # 清理
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("PASS: test_setup_logging")


def test_event_levels():
    """测试各日志级别"""
    from src.monitoring.structured_logger import (
        get_logger, JSONFormatter, AUDIT_LEVEL,
    )

    logger = get_logger("test.levels")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.debug("debug_event", extra={"detail": "low"})
    logger.info("info_event", extra={"detail": "normal"})
    logger.warning("warn_event", extra={"detail": "high"})
    logger.error("error_event", extra={"detail": "critical"})
    logger.audit("audit_event", detail="forever")

    lines = [json.loads(l) for l in stream.getvalue().strip().split("\n") if l]
    levels = [l["level"] for l in lines]

    assert "DEBUG" in levels
    assert "INFO" in levels
    assert "WARNING" in levels
    assert "ERROR" in levels
    assert "AUDIT" in levels

    logger.removeHandler(handler)
    print("PASS: test_event_levels")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GTO-GameFlow v5.5 — 结构化日志与审计追踪测试")
    print("=" * 60)
    print()

    tests = [
        # JSON 格式化器
        ("JSON 基本输出", test_json_formatter_basic),
        ("JSON + 上下文", test_json_formatter_with_context),
        ("上下文嵌套", test_log_context_nesting),
        # StructuredLogger
        ("audit() 方法", test_structured_logger_audit),
        ("event() 方法", test_structured_logger_event),
        ("StageTimer", test_stage_timer),
        # 审计追踪
        ("投注生命周期", test_audit_trail_bet_lifecycle),
        ("系统事件审计", test_audit_trail_system_events),
        ("审计统计", test_audit_trail_stats),
        ("审计导出 JSONL", test_audit_trail_export),
        ("投注无效", test_audit_trail_void),
        ("资金变更记录", test_audit_trail_balance_change),
        # 流水线剖析器
        ("剖析器基本功能", test_pipeline_profiler_basic),
        ("剖析器摘要", test_pipeline_profiler_summary),
        ("剖析器重置", test_pipeline_profiler_reset),
        # 集成
        ("日志级别", test_event_levels),
        ("日志系统初始化", test_setup_logging),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"结果: {passed} 通过, {failed} 失败, {len(tests)} 总计")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())