"""
GTO-GameFlow v5.5 — 结构化日志系统

特性:
- JSON 格式日志输出 (stdout + 文件)
- 上下文追踪 (match_id, league_id, stage, trace_id)
- 自动耗时统计 (Stage 计时)
- 审计事件标记 (AUDIT 级别)
- 与 Python logging 完全兼容

使用方式:
    from src.monitoring.structured_logger import get_logger, LogContext

    logger = get_logger(__name__)

    with LogContext(match_id="BAYvsDOR", league_id="bundesliga", stage="S4"):
        logger.info("poisson_bridge_complete", home_goals_exp=1.55, away_goals_exp=0.87)

    logger.audit("bet_executed", bet_id="B2026001", stake=250.0, odds=2.10)
"""

import json
import logging
import os
import sys
import time
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════
# 自定义日志级别
# ═══════════════════════════════════════════════════════════════

AUDIT_LEVEL = 25  # 介于 INFO(20) 和 WARNING(30) 之间
logging.addLevelName(AUDIT_LEVEL, "AUDIT")


# ═══════════════════════════════════════════════════════════════
# 上下文追踪
# ═══════════════════════════════════════════════════════════════

class LogContext:
    """
    线程本地日志上下文 — 跨函数传递 match_id/league_id/stage 等信息。

    使用方式:
        with LogContext(match_id="BAYvsDOR", league_id="bundesliga", stage="S4"):
            logger.info("processing")
    """

    _storage = threading.local()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._saved = None

    def __enter__(self):
        self._saved = dict(getattr(self._storage, 'data', {}))
        current = dict(self._saved)
        current.update(self.kwargs)
        self._storage.data = current
        return self

    def __exit__(self, *args):
        self._storage.data = self._saved

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return getattr(cls._storage, 'data', {}).get(key, default)

    @classmethod
    def all(cls) -> Dict[str, Any]:
        return dict(getattr(cls._storage, 'data', {}))

    @classmethod
    def set(cls, **kwargs):
        current = dict(getattr(cls._storage, 'data', {}))
        current.update(kwargs)
        cls._storage.data = current

    @classmethod
    def clear(cls):
        cls._storage.data = {}


# ═══════════════════════════════════════════════════════════════
# 耗时追踪器
# ═══════════════════════════════════════════════════════════════

class StageTimer:
    """Stage 计时器 — 记录每个流水线阶段的耗时"""

    _timers = threading.local()

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        # 存储到线程本地，供 JSONFormatter 使用
        if not hasattr(self._timers, 'data'):
            self._timers.data = {}
        self._timers.data[self.stage_name] = self.elapsed_ms

    @classmethod
    def pop_stage_timing(cls) -> Dict[str, float]:
        """获取当前线程所有 Stage 耗时并清空"""
        data = dict(getattr(cls._timers, 'data', {}))
        cls._timers.data = {}
        return data


# ═══════════════════════════════════════════════════════════════
# JSON 格式化器
# ═══════════════════════════════════════════════════════════════

class JSONFormatter(logging.Formatter):
    """
    结构化 JSON 日志格式化器。

    输出格式:
    {
      "timestamp": "2026-06-17T10:30:00.123Z",
      "level": "INFO",
      "logger": "src.pipeline.orchestrator",
      "event": "poisson_bridge_complete",
      "trace_id": "a1b2c3d4",
      "match_id": "BAYvsDOR",
      "league_id": "bundesliga",
      "stage": "S4",
      "home_goals_exp": 1.55,
      "away_goals_exp": 0.87,
      "duration_ms": 12.3
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.") + f"{record.msecs:03.0f}Z",
            "level": record.levelname,
            "logger": record.name,
        }

        # 事件名 (取 msg 的第一段，或直接用 msg)
        msg = record.getMessage()
        if " " in msg and not msg.startswith("{"):
            log_entry["event"] = msg.split(" ")[0]
            log_entry["message"] = " ".join(msg.split(" ")[1:])
        else:
            log_entry["event"] = msg
            log_entry["message"] = ""

        # 注入 LogContext
        ctx = LogContext.all()
        for key in ("match_id", "league_id", "stage", "trace_id", "season", "matchday"):
            if key in ctx:
                log_entry[key] = ctx[key]

        # 注入 trace_id (无上下文时自动生成)
        if "trace_id" not in log_entry:
            log_entry["trace_id"] = getattr(record, 'trace_id', None) or str(uuid.uuid4())[:8]

        # 注入 Stage 耗时
        timings = StageTimer.pop_stage_timing()
        if timings:
            log_entry["stage_timings"] = timings

        # 注入 extra 字段
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "trace_id",
            ) and not key.startswith("_"):
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    log_entry[key] = value

        # 异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════════════════════
# 增强 Logger
# ═══════════════════════════════════════════════════════════════

class StructuredLogger(logging.Logger):
    """
    结构化 Logger — 扩展标准 logging.Logger，增加 audit() 和 event() 方法。

    使用方式:
        logger = get_logger(__name__)
        logger.audit("bet_settled", bet_id="B001", profit=150.0)
        logger.event("stage_complete", stage="S4", duration_ms=12.3)
    """

    def audit(self, event: str, **kwargs):
        """
        审计日志 — 记录所有关键业务事件 (投注执行、结算、熔断、资金变更)。

        自动标记为 AUDIT 级别，确保审计事件不被过滤。
        """
        if self.isEnabledFor(AUDIT_LEVEL):
            extra = kwargs.copy()
            extra["_audit"] = True
            self._log(AUDIT_LEVEL, event, (), extra=extra)

    def event(self, event: str, **kwargs):
        """结构化事件日志 — 用 event 名 + 结构化字段代替散落的字符串"""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, event, (), extra=kwargs)

    def stage_enter(self, stage: str, **kwargs):
        """流水线 Stage 入口"""
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, f"stage_enter {stage}", (), extra=kwargs)

    def stage_exit(self, stage: str, duration_ms: float = 0.0, **kwargs):
        """流水线 Stage 出口"""
        if self.isEnabledFor(logging.INFO):
            extra = kwargs.copy()
            extra["duration_ms"] = round(duration_ms, 2)
            self._log(logging.INFO, f"stage_exit {stage}", (), extra=extra)


# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════

_log_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    json_stdout: bool = True,
    json_file: bool = True,
) -> None:
    """
    初始化结构化日志系统。

    参数:
        level: 日志级别
        log_file: 日志文件路径 (None 则自动生成)
        log_dir: 日志目录
        json_stdout: 是否 JSON 格式输出到 stdout
        json_file: 是否 JSON 格式输出到文件
    """
    global _log_initialized
    if _log_initialized:
        return

    # 注册自定义 Logger 类
    logging.setLoggerClass(StructuredLogger)

    root = logging.getLogger()
    root.setLevel(level)

    # 清除已有 handlers
    root.handlers.clear()

    # 禁用现有非结构化 handler 的传播
    root.propagate = False

    # 1. stdout handler
    if json_stdout:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(JSONFormatter())
        root.addHandler(console)

    # 2. 文件 handler
    if json_file:
        os.makedirs(log_dir, exist_ok=True)
        if log_file is None:
            log_file = os.path.join(
                log_dir,
                f"gto-gameflow-{datetime.now().strftime('%Y%m%d')}.jsonl",
            )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # 3. 审计专用文件 handler (始终记录 AUDIT 级别)
    audit_dir = os.path.join(log_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    audit_file = os.path.join(
        audit_dir,
        f"audit-{datetime.now().strftime('%Y%m%d')}.jsonl",
    )
    audit_handler = logging.FileHandler(audit_file, encoding="utf-8")
    audit_handler.setLevel(AUDIT_LEVEL)
    audit_handler.setFormatter(JSONFormatter())
    audit_handler.addFilter(lambda r: getattr(r, '_audit', False))
    root.addHandler(audit_handler)

    _log_initialized = True


def get_logger(name: str) -> "StructuredLogger":
    """获取结构化 Logger — 如果 setup_logging() 未调用，返回普通 Logger 的包装"""
    logger = logging.getLogger(name)
    if not isinstance(logger, StructuredLogger):
        # 动态添加 audit/event 方法到普通 Logger
        pass  # 通过 monkey-patch 实现，见下方
    return logger  # type: ignore


# Monkey-patch logging.Logger 以支持 audit/event 方法
def _audit(self, event: str, **kwargs):
    """审计日志"""
    if self.isEnabledFor(AUDIT_LEVEL):
        extra = kwargs.copy()
        extra["_audit"] = True
        self._log(AUDIT_LEVEL, event, (), extra=extra)


def _event(self, event: str, **kwargs):
    """结构化事件日志"""
    if self.isEnabledFor(logging.INFO):
        self._log(logging.INFO, event, (), extra=kwargs)


def _stage_enter(self, stage: str, **kwargs):
    """流水线 Stage 入口"""
    if self.isEnabledFor(logging.DEBUG):
        self._log(logging.DEBUG, f"stage_enter {stage}", (), extra=kwargs)


def _stage_exit(self, stage: str, duration_ms: float = 0.0, **kwargs):
    """流水线 Stage 出口"""
    if self.isEnabledFor(logging.INFO):
        extra = kwargs.copy()
        extra["duration_ms"] = round(duration_ms, 2)
        self._log(logging.INFO, f"stage_exit {stage}", (), extra=extra)


# 为所有 Logger 实例添加方法
logging.Logger.audit = _audit         # type: ignore
logging.Logger.event = _event         # type: ignore
logging.Logger.stage_enter = _stage_enter  # type: ignore
logging.Logger.stage_exit = _stage_exit    # type: ignore


def shutdown_logging():
    """关闭日志系统 — 关闭所有 handler 并清除"""
    root = logging.getLogger()
    for h in list(root.handlers):
        h.close()
    root.handlers.clear()
    logging.shutdown()


# 便捷别名
logger = get_logger("gto-gameflow")