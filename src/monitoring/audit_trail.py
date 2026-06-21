"""
GTO-GameFlow v5.5 — 投注审计追踪

完整记录每笔投注的完整生命周期:
Proposal → Execution → Settlement → Void

存储:
- JSONL 审计日志文件 (logs/audit/audit-YYYYMMDD.jsonl)
- 内存审计追踪 (可通过 API 查询)

审计事件类型:
- bet_proposed: 投注建议生成
- bet_executed: 投注执行
- bet_settled: 投注结算 (WIN/LOSS/VOID)
- bet_voided: 投注无效
- balance_changed: 资金变更
- circuit_breaker_activated: 熔断激活
- circuit_breaker_reset: 熔断重置
- risk_limit_enforced: 风控限制触发
"""

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .structured_logger import get_logger, LogContext

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 审计事件数据模型
# ═══════════════════════════════════════════════════════════════

class BetAuditTrail:
    """
    投注审计追踪 — 记录每笔投注的完整生命周期。

    使用方式:
        trail = BetAuditTrail()
        trail.record_proposal(bet_id, proposal_data)
        trail.record_execution(bet_id, execution_data)
        trail.record_settlement(bet_id, settlement_data)
        trail.get_bet_history(bet_id)  # 完整生命周期
    """

    def __init__(self, max_in_memory: int = 10000):
        self._bets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._max_in_memory = max_in_memory
        self._total_events: int = 0

    # ── 记录方法 ─────────────────────────────────────────────

    def record_proposal(
        self,
        bet_id: str,
        match_id: str,
        league_id: str,
        selection: str,
        odds: float,
        model_prob: float,
        implied_prob: float,
        value: float,
        kelly_stake: float,
        adjusted_stake: float,
        priority_score: float,
        **extra,
    ):
        """记录投注建议生成"""
        event = {
            "event": "bet_proposed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bet_id": bet_id,
            "match_id": match_id,
            "league_id": league_id,
            "selection": selection,
            "odds": round(odds, 4),
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "value": round(value, 4),
            "kelly_stake": round(kelly_stake, 2),
            "adjusted_stake": round(adjusted_stake, 2),
            "priority_score": round(priority_score, 4),
            **extra,
        }
        self._append(bet_id, event)
        logger.audit(
            "bet_proposed",
            bet_id=bet_id,
            match_id=match_id,
            league_id=league_id,
            selection=selection,
            odds=round(odds, 4),
            adjusted_stake=round(adjusted_stake, 2),
            value=round(value, 4),
        )

    def record_execution(
        self,
        bet_id: str,
        match_id: str,
        league_id: str,
        selection: str,
        odds: float,
        stake: float,
        balance_before: float,
        **extra,
    ):
        """记录投注执行"""
        event = {
            "event": "bet_executed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bet_id": bet_id,
            "match_id": match_id,
            "league_id": league_id,
            "selection": selection,
            "odds": round(odds, 4),
            "stake": round(stake, 2),
            "balance_before": round(balance_before, 2),
            **extra,
        }
        self._append(bet_id, event)
        logger.audit(
            "bet_executed",
            bet_id=bet_id,
            match_id=match_id,
            league_id=league_id,
            selection=selection,
            odds=round(odds, 4),
            stake=round(stake, 2),
            balance_before=round(balance_before, 2),
        )

    def record_settlement(
        self,
        bet_id: str,
        match_id: str,
        league_id: str,
        result: str,
        profit_loss: float,
        balance_after: float,
        consecutive_losses: int = 0,
        **extra,
    ):
        """记录投注结算"""
        event = {
            "event": "bet_settled",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bet_id": bet_id,
            "match_id": match_id,
            "league_id": league_id,
            "result": result,
            "profit_loss": round(profit_loss, 2),
            "balance_after": round(balance_after, 2),
            "consecutive_losses": consecutive_losses,
            **extra,
        }
        self._append(bet_id, event)
        logger.audit(
            "bet_settled",
            bet_id=bet_id,
            match_id=match_id,
            league_id=league_id,
            result=result,
            profit_loss=round(profit_loss, 2),
            balance_after=round(balance_after, 2),
            consecutive_losses=consecutive_losses,
        )

    def record_void(
        self,
        bet_id: str,
        match_id: str,
        league_id: str,
        reason: str,
        **extra,
    ):
        """记录投注无效"""
        event = {
            "event": "bet_voided",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bet_id": bet_id,
            "match_id": match_id,
            "league_id": league_id,
            "reason": reason,
            **extra,
        }
        self._append(bet_id, event)
        logger.audit(
            "bet_voided",
            bet_id=bet_id,
            match_id=match_id,
            league_id=league_id,
            reason=reason,
        )

    def record_circuit_breaker(
        self,
        league_id: str,
        reason: str,
        balance: float,
        consecutive_losses: int,
        **extra,
    ):
        """记录熔断激活"""
        event = {
            "event": "circuit_breaker_activated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "league_id": league_id,
            "reason": reason,
            "balance": round(balance, 2),
            "consecutive_losses": consecutive_losses,
            **extra,
        }
        self._append("__system__", event)
        logger.audit(
            "circuit_breaker_activated",
            league_id=league_id,
            reason=reason,
            balance=round(balance, 2),
            consecutive_losses=consecutive_losses,
        )

    def record_circuit_breaker_reset(self, league_id: str, **extra):
        """记录熔断重置"""
        event = {
            "event": "circuit_breaker_reset",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "league_id": league_id,
            **extra,
        }
        self._append("__system__", event)
        logger.audit("circuit_breaker_reset", league_id=league_id)

    def record_risk_limit(
        self,
        league_id: str,
        limit_type: str,
        details: str,
        **extra,
    ):
        """记录风控限制触发"""
        event = {
            "event": "risk_limit_enforced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "league_id": league_id,
            "limit_type": limit_type,
            "details": details,
            **extra,
        }
        self._append("__system__", event)
        logger.audit(
            "risk_limit_enforced",
            league_id=league_id,
            limit_type=limit_type,
            details=details,
        )

    def record_balance_change(
        self,
        league_id: str,
        balance: float,
        peak: float,
        drawdown: float,
        reason: str = "",
        **extra,
    ):
        """记录资金变更"""
        event = {
            "event": "balance_changed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "league_id": league_id,
            "balance": round(balance, 2),
            "peak_balance": round(peak, 2),
            "max_drawdown": round(drawdown, 4),
            "reason": reason,
            **extra,
        }
        self._append("__system__", event)
        logger.audit(
            "balance_changed",
            league_id=league_id,
            balance=round(balance, 2),
            peak=round(peak, 2),
            drawdown=round(drawdown, 4),
            reason=reason,
        )

    # ── 查询方法 ─────────────────────────────────────────────

    def get_bet_history(self, bet_id: str) -> List[Dict[str, Any]]:
        """获取单笔投注的完整生命周期"""
        with self._lock:
            return list(self._bets.get(bet_id, []))

    def get_all_bets(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有投注的审计追踪"""
        with self._lock:
            return dict(self._bets)

    def get_bet_ids(self) -> Set[str]:
        """获取所有 bet_id (不含系统事件)"""
        with self._lock:
            return {k for k in self._bets if k != "__system__"}

    def get_system_events(self) -> List[Dict[str, Any]]:
        """获取系统事件 (熔断/风控)"""
        with self._lock:
            return list(self._bets.get("__system__", []))

    def get_stats(self) -> Dict[str, Any]:
        """获取审计统计"""
        with self._lock:
            bet_ids = self.get_bet_ids()
            total = len(bet_ids)
            settled = 0
            won = 0
            lost = 0
            total_profit = 0.0
            total_staked = 0.0

            for bid in bet_ids:
                events = self._bets[bid]
                for e in events:
                    if e["event"] == "bet_executed":
                        total_staked += e.get("stake", 0)
                    elif e["event"] == "bet_settled":
                        settled += 1
                        profit = e.get("profit_loss", 0)
                        total_profit += profit
                        if profit > 0:
                            won += 1
                        elif profit < 0:
                            lost += 1

            return {
                "total_bets": total,
                "settled": settled,
                "won": won,
                "lost": lost,
                "voided": total - settled,
                "total_staked": round(total_staked, 2),
                "total_profit": round(total_profit, 2),
                "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0,
                "win_rate": round(won / settled * 100, 2) if settled > 0 else 0,
                "total_events": self._total_events,
                "system_events": len(self.get_system_events()),
            }

    def export_jsonl(self, filepath: str) -> int:
        """导出全部审计数据为 JSONL 文件"""
        count = 0
        with open(filepath, "w", encoding="utf-8") as f:
            with self._lock:
                for bet_id, events in self._bets.items():
                    for event in events:
                        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                        count += 1
        return count

    # ── 内部方法 ─────────────────────────────────────────────

    def _append(self, key: str, event: Dict[str, Any]):
        with self._lock:
            self._bets[key].append(event)
            self._total_events += 1
            # 内存限制
            if self._total_events > self._max_in_memory * 2:
                # 清理最旧的系统事件
                sys_events = self._bets.get("__system__", [])
                if len(sys_events) > 100:
                    self._bets["__system__"] = sys_events[-50:]


# ═══════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════

_audit_trail: Optional[BetAuditTrail] = None


def get_audit_trail() -> BetAuditTrail:
    """获取全局审计追踪单例"""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = BetAuditTrail()
    return _audit_trail


def reset_audit_trail():
    """重置审计追踪 (测试用)"""
    global _audit_trail
    _audit_trail = BetAuditTrail()