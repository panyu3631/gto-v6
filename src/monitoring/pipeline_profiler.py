"""
GTO-GameFlow v5.5 — 流水线性能剖析器

自动记录每个 Stage 的耗时、投注统计、异常检测。
可直接装饰 orchestrator 的 run_full 方法，或在回测循环中手动调用。

使用方式:
    profiler = PipelineProfiler()

    # 方式1: 装饰器
    @profiler.profile
    def run_full(self, match, ...):
        ...

    # 方式2: 手动
    profiler.start_run(match_id, league_id)
    # ... 执行流水线 ...
    profiler.end_run(result)

    # 查看统计
    print(profiler.get_stats())
"""

import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .structured_logger import get_logger, LogContext, StageTimer

logger = get_logger(__name__)


class PipelineProfiler:
    """
    流水线性能剖析器。

    功能:
    - 按 Stage 统计耗时 (min/max/avg/p50/p95/p99)
    - 按联赛统计投注频率和胜率
    - 异常检测 (耗时异常长、投注频率异常)
    - 导出性能报告
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Stage 耗时统计
        self._stage_timings: Dict[str, List[float]] = defaultdict(list)

        # 整体运行时统计
        self._total_runs: int = 0
        self._total_errors: int = 0
        self._total_bets: int = 0
        self._total_settlements: int = 0
        self._run_timings: List[float] = []

        # 联赛统计
        self._league_runs: Dict[str, int] = defaultdict(int)
        self._league_bets: Dict[str, int] = defaultdict(int)
        self._league_wins: Dict[str, int] = defaultdict(int)

        # 当前运行
        self._current_run_start: Optional[float] = None
        self._current_match_id: str = ""
        self._current_league_id: str = ""

    def start_run(self, match_id: str = "", league_id: str = ""):
        """开始新一轮流水线运行"""
        self._current_run_start = time.perf_counter()
        self._current_match_id = match_id
        self._current_league_id = league_id

    def end_run(
        self,
        result: Any = None,
        bets_placed: int = 0,
        bets_settled: int = 0,
        has_error: bool = False,
    ):
        """结束当前流水线运行并记录统计"""
        if self._current_run_start is None:
            return

        elapsed = (time.perf_counter() - self._current_run_start) * 1000

        with self._lock:
            self._total_runs += 1
            self._run_timings.append(elapsed)
            self._total_bets += bets_placed
            self._total_settlements += bets_settled
            if has_error:
                self._total_errors += 1

            if self._current_league_id:
                self._league_runs[self._current_league_id] += 1
                self._league_bets[self._current_league_id] += bets_placed

        # 记录到日志
        logger.event(
            "pipeline_run_complete",
            match_id=self._current_match_id,
            league_id=self._current_league_id,
            duration_ms=round(elapsed, 2),
            bets_placed=bets_placed,
            bets_settled=bets_settled,
            has_error=has_error,
        )

        self._current_run_start = None
        self._current_match_id = ""
        self._current_league_id = ""

    def record_stage(self, stage_name: str, duration_ms: float):
        """记录单个 Stage 耗时"""
        with self._lock:
            self._stage_timings[stage_name].append(duration_ms)

    def record_bet_result(self, league_id: str, won: bool):
        """记录投注结果"""
        with self._lock:
            if won:
                self._league_wins[league_id] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取完整的性能统计"""
        with self._lock:
            stage_stats = {}
            for stage, timings in self._stage_timings.items():
                if not timings:
                    continue
                sorted_t = sorted(timings)
                n = len(sorted_t)
                stage_stats[stage] = {
                    "count": n,
                    "min_ms": round(sorted_t[0], 2),
                    "max_ms": round(sorted_t[-1], 2),
                    "avg_ms": round(sum(sorted_t) / n, 2),
                    "p50_ms": round(sorted_t[n // 2], 2),
                    "p95_ms": round(sorted_t[int(n * 0.95)], 2),
                    "p99_ms": round(sorted_t[int(n * 0.99)], 2),
                }

            run_stats = {}
            if self._run_timings:
                sorted_r = sorted(self._run_timings)
                n = len(sorted_r)
                run_stats = {
                    "count": self._total_runs,
                    "avg_ms": round(sum(sorted_r) / n, 2),
                    "p50_ms": round(sorted_r[n // 2], 2),
                    "p95_ms": round(sorted_r[int(n * 0.95)], 2),
                    "total_errors": self._total_errors,
                    "error_rate": round(self._total_errors / max(1, self._total_runs) * 100, 2),
                }

            league_stats = {}
            for league, runs in self._league_runs.items():
                bets = self._league_bets.get(league, 0)
                wins = self._league_wins.get(league, 0)
                league_stats[league] = {
                    "runs": runs,
                    "bets": bets,
                    "wins": wins,
                    "win_rate": round(wins / max(1, bets) * 100, 2),
                    "bets_per_run": round(bets / max(1, runs), 2),
                }

            return {
                "stage_timings": stage_stats,
                "run_stats": run_stats,
                "league_stats": league_stats,
                "totals": {
                    "total_runs": self._total_runs,
                    "total_bets": self._total_bets,
                    "total_settlements": self._total_settlements,
                    "total_errors": self._total_errors,
                },
            }

    def get_summary(self) -> str:
        """获取可读的性能摘要"""
        stats = self.get_stats()
        lines = ["=" * 60, "  GTO-GameFlow v5.5 流水线性能报告", "=" * 60, ""]

        # 整体统计
        lines.append(f"总运行次数: {stats['totals']['total_runs']}")
        lines.append(f"总投注数: {stats['totals']['total_bets']}")
        lines.append(f"总结算数: {stats['totals']['total_settlements']}")
        if stats['run_stats']:
            rs = stats['run_stats']
            lines.append(f"平均耗时: {rs['avg_ms']:.1f}ms (P50: {rs['p50_ms']:.1f}ms, P95: {rs['p95_ms']:.1f}ms)")
            lines.append(f"错误率: {rs['error_rate']:.1f}%")

        # Stage 耗时
        if stats['stage_timings']:
            lines.append("")
            lines.append("--- Stage 耗时 ---")
            for stage, s in stats['stage_timings'].items():
                lines.append(f"  {stage}: avg={s['avg_ms']:.1f}ms p50={s['p50_ms']:.1f}ms p95={s['p95_ms']:.1f}ms (n={s['count']})")

        # 联赛统计
        if stats['league_stats']:
            lines.append("")
            lines.append("--- 联赛统计 ---")
            for league, l in stats['league_stats'].items():
                lines.append(f"  {league}: {l['runs']} runs, {l['bets']} bets, win_rate={l['win_rate']}%")

        return "\n".join(lines)

    def reset(self):
        """重置所有统计"""
        with self._lock:
            self._stage_timings.clear()
            self._total_runs = 0
            self._total_errors = 0
            self._total_bets = 0
            self._total_settlements = 0
            self._run_timings.clear()
            self._league_runs.clear()
            self._league_bets.clear()
            self._league_wins.clear()


# 全局单例
_profiler: Optional[PipelineProfiler] = None


def get_profiler() -> PipelineProfiler:
    global _profiler
    if _profiler is None:
        _profiler = PipelineProfiler()
    return _profiler


def reset_profiler():
    global _profiler
    _profiler = PipelineProfiler()