"""
投注决策日志系统 (Phase 5.0)

持久化每笔投注的完整决策链，支持:
- 赛后复盘和策略归因
- 异常检测 (连续亏损自动暂停)
- Walk-Forward 验证的 traceability
- JSON 格式，人类可读 + 机器可解析
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class BetDecision:
    """单笔投注的完整决策记录"""
    # 标识
    bet_id: str
    match_id: str
    match_desc: str  # "主队 vs 客队"
    league_id: str
    season: str
    strategy: str  # "1x2" | "asian" | "over_under" | "parlay"
    selection: str  # BetSelection.value

    # 赔率
    odds: float
    odds_source: str  # "pinnacle" | "market_avg" | "bet365" | "synthetic"

    # 概率
    model_prob: float
    implied_prob: float
    value: float

    # 置信度
    confidence: float
    factor_count: int
    data_source_count: int

    # 资金
    kelly_stake: float
    adjusted_stake: float
    kelly_fraction: float
    bankroll_before: float

    # 风控
    risk_approved: bool
    risk_reason: str = ""

    # 结果 (赛后填充)
    outcome: str = ""  # "win" | "loss" | "push" | "half_win" | "half_loss" | "pending"
    profit_loss: float = 0.0
    settled: bool = False

    # 因子贡献 (Top 5)
    factor_contributions: Dict[str, float] = field(default_factory=dict)

    # 时间戳
    decision_time: str = ""
    settlement_time: str = ""

    # 元数据
    margin_estimate: float = 0.0  # 庄家 margin
    home_advantage_used: float = 0.0
    elo_home: float = 1500.0
    elo_away: float = 1500.0


class BetLogger:
    """
    投注日志系统。

    用法:
        logger = BetLogger("path/to/logs")
        logger.log_decision(decision)
        logger.log_settlement(bet_id, outcome, profit_loss)
        logger.check_anomaly(max_consecutive_losses=8)  # 异常检测
    """

    def __init__(self, log_dir: str, session_id: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._decisions: List[BetDecision] = []
        self._pending: Dict[str, BetDecision] = {}
        self._consecutive_losses = 0
        self._total_bets = 0
        self._total_wins = 0
        self._anomaly_paused = False

    def log_decision(self, decision: BetDecision):
        """记录投注决策"""
        decision.decision_time = datetime.now().isoformat()
        self._decisions.append(decision)
        self._pending[decision.bet_id] = decision
        self._total_bets += 1

    def log_settlement(self, bet_id: str, outcome: str, profit_loss: float):
        """
        记录结算结果。

        参数:
            bet_id: 投注 ID
            outcome: "win" | "loss" | "push" | "half_win" | "half_loss"
            profit_loss: 盈亏金额
        """
        if bet_id in self._pending:
            decision = self._pending.pop(bet_id)
            decision.outcome = outcome
            decision.profit_loss = profit_loss
            decision.settled = True
            decision.settlement_time = datetime.now().isoformat()

            if outcome in ("win", "half_win"):
                self._total_wins += 1
                self._consecutive_losses = 0
            elif outcome == "push":
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1

    def check_anomaly(self, max_consecutive_losses: int = 8) -> bool:
        """
        检查是否需要暂停策略。

        返回:
            True 如果触发异常 (建议暂停)
        """
        if self._consecutive_losses >= max_consecutive_losses:
            if not self._anomaly_paused:
                self._anomaly_paused = True
                self._flush("anomaly_paused")
            return True
        if self._anomaly_paused:
            self._anomaly_paused = False
        return False

    def get_rolling_roi(self, window: int = 20) -> float:
        """最近 N 注的 ROI"""
        recent = [d for d in self._decisions[-window:] if d.settled]
        if not recent:
            return 0.0
        total_staked = sum(d.adjusted_stake for d in recent)
        total_returned = sum(d.profit_loss + d.adjusted_stake for d in recent if d.outcome in ("win", "half_win"))
        total_returned += sum(d.adjusted_stake * 0.5 for d in recent if d.outcome == "half_win")
        total_returned += sum(0.0 for d in recent if d.outcome == "loss")
        total_returned += sum(d.adjusted_stake for d in recent if d.outcome == "push")
        return (total_returned - total_staked) / max(total_staked, 0.01)

    def get_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        settled = [d for d in self._decisions if d.settled]
        total_staked = sum(d.adjusted_stake for d in settled)
        total_returned = sum(d.profit_loss + d.adjusted_stake for d in settled if d.outcome in ("win",))
        total_returned += sum(d.adjusted_stake * 0.5 + d.profit_loss for d in settled if d.outcome == "half_win")
        total_returned += sum(d.adjusted_stake for d in settled if d.outcome == "push")

        by_strategy: Dict[str, Dict] = {}
        for d in settled:
            s = d.strategy
            if s not in by_strategy:
                by_strategy[s] = {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0}
            by_strategy[s]["bets"] += 1
            by_strategy[s]["staked"] += d.adjusted_stake
            if d.outcome in ("win", "half_win"):
                by_strategy[s]["wins"] += 1
                by_strategy[s]["returned"] += d.profit_loss + d.adjusted_stake

        return {
            "session_id": self.session_id,
            "total_bets": len(settled),
            "total_wins": self._total_wins,
            "win_rate": self._total_wins / max(len(settled), 1),
            "total_staked": total_staked,
            "total_returned": total_returned,
            "roi": (total_returned - total_staked) / max(total_staked, 0.01),
            "consecutive_losses": self._consecutive_losses,
            "anomaly_paused": self._anomaly_paused,
            "by_strategy": by_strategy,
        }

    def _flush(self, suffix: str = ""):
        """持久化到磁盘"""
        stats = self.get_stats()
        filename = f"bet_log_{self.session_id}"
        if suffix:
            filename += f"_{suffix}"
        filepath = os.path.join(self.log_dir, f"{filename}.json")

        data = {
            "stats": stats,
            "decisions": [asdict(d) for d in self._decisions],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def flush(self):
        """手动持久化"""
        self._flush()

    def __del__(self):
        try:
            self._flush()
        except Exception:
            pass


def create_bet_id(league_id: str, season: str, matchup_count: int, strategy: str) -> str:
    """生成唯一投注 ID"""
    return f"{league_id}_{season}_{matchup_count:04d}_{strategy}"