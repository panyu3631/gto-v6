"""
GTO-GameFlow v5.5 — 流水线告警适配器

在流水线执行过程中自动检测告警条件并发送通知。
作为 orchestrator 和 AlertManager 之间的桥梁。

集成点:
- 熔断触发 → CRITICAL 告警
- 大额投注 → WARN 告警
- 异常赔率 → WARN 告警
- 回撤超标 → WARN/CRITICAL 告警
- 流水线错误 → CRITICAL 告警
- 每日摘要 → INFO 告警
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .alerting import AlertManager, Alert, AlertLevel, AlertType
from ..utils.i18n import cn_selection, cn_league

if TYPE_CHECKING:
    from ..pipeline.orchestrator import PipelineResult
    from ..engine.bankroll import BankrollState, BetProposal, BetPlacement

logger = logging.getLogger(__name__)


class PipelineAlertAdapter:
    """
    流水线告警适配器。

    在 orchestrator 的每个关键节点调用对应的检查方法，
    自动生成告警并通过 AlertManager 发送。

    使用方式:
        adapter = PipelineAlertAdapter(alert_manager, config)
        result = orchestrator.run_full(match)
        adapter.check_after_run(result, bankroll_state)
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        drawdown_warn_pct: float = 0.20,
        drawdown_critical_pct: float = 0.30,
        large_bet_ratio: float = 0.04,
        odd_std_warn: float = 0.08,
    ):
        self.alert_mgr = alert_manager
        self.drawdown_warn_pct = drawdown_warn_pct
        self.drawdown_critical_pct = drawdown_critical_pct
        self.large_bet_ratio = large_bet_ratio
        self.odd_std_warn = odd_std_warn

        # 追踪状态
        self._last_drawdown_alert: Optional[datetime] = None
        self._drawdown_alert_cooldown: int = 3600  # 回撤告警冷却 1 小时

    # ═══════════════════════════════════════════════════════
    # 熔断告警
    # ═══════════════════════════════════════════════════════

    def check_circuit_breaker(
        self,
        circuit_broken: bool,
        circuit_reason: str,
        league_id: str = "",
        bankroll_state: Optional["BankrollState"] = None,
    ) -> Optional[Alert]:
        """
        检查熔断状态并发送告警。

        在 orchestrator.run_stages_6_9() 的 Stage 9 之后调用。
        """
        if not circuit_broken:
            return None

        # 构建详细消息
        details = ""
        if bankroll_state:
            details = (
                f"当前资金: ¥{bankroll_state.balance:,.2f} | "
                f"连续亏损: {bankroll_state.consecutive_losses} 场 | "
                f"最大回撤: {bankroll_state.max_drawdown * 100:.1f}%"
            )

        alert = Alert(
            level=AlertLevel.CRITICAL,
            alert_type=AlertType.CIRCUIT_BREAKER,
            title=f"熔断触发 — {league_id or '全局'}",
            message=f"熔断原因: {circuit_reason}\n\n{details}\n\n系统已自动暂停投注，请人工审核后手动恢复。",
            metadata={
                "league": league_id or "global",
                "reason": circuit_reason,
                "balance": f"¥{bankroll_state.balance:,.2f}" if bankroll_state else "N/A",
                "consecutive_losses": str(bankroll_state.consecutive_losses) if bankroll_state else "N/A",
            },
        )
        self.alert_mgr.send(alert, bypass_cooldown=True)
        return alert

    # ═══════════════════════════════════════════════════════
    # 大额投注告警
    # ═══════════════════════════════════════════════════════

    def check_large_bets(
        self,
        proposals: List["BetProposal"],
        bankroll_state: "BankrollState",
        league_id: str = "",
        match_id: str = "",
    ) -> List[Alert]:
        """
        检查是否有大额投注并发送告警。

        在 orchestrator.run_stages_6_9() 的 Stage 8 (资金分配) 之后调用。
        """
        alerts = []
        if not bankroll_state.balance:
            return alerts

        for p in proposals:
            ratio = p.adjusted_stake / bankroll_state.balance
            if ratio >= self.large_bet_ratio:
                alert = Alert(
                    level=AlertLevel.WARN,
                    alert_type=AlertType.LARGE_BET,
                    title=f"大额投注 — {league_id} {match_id}",
                    message=(
                        f"投注金额 ¥{p.adjusted_stake:,.2f} 占资金 {ratio * 100:.1f}% "
                        f"(阈值 {self.large_bet_ratio * 100:.0f}%)\n"
                        f"选项: {cn_selection(p.selection.value)} | 赔率: {p.odds:.2f} | "
                        f"模型概率: {p.model_prob:.2%} | 价值: {p.value:.3f}"
                    ),
                    metadata={
                        "league": league_id,
                        "match_id": match_id,
                        "stake": f"¥{p.adjusted_stake:,.2f}",
                        "ratio": f"{ratio * 100:.1f}%",
                        "selection": p.selection.value,
                        "odds": f"{p.odds:.2f}",
                        "model_prob": f"{p.model_prob:.2%}",
                        "value": f"{p.value:.3f}",
                    },
                )
                self.alert_mgr.send(alert)
                alerts.append(alert)

        return alerts

    # ═══════════════════════════════════════════════════════
    # 异常赔率告警
    # ═══════════════════════════════════════════════════════

    def check_abnormal_odds(
        self,
        odds_std: float,
        league_id: str = "",
        match_id: str = "",
        home_odds: float = 0,
        draw_odds: float = 0,
        away_odds: float = 0,
    ) -> Optional[Alert]:
        """
        检查赔率异常并发送告警。

        在 generate_bet_proposals() 中赔率离散度超标时调用。
        """
        if odds_std < self.odd_std_warn:
            return None

        alert = Alert(
            level=AlertLevel.WARN,
            alert_type=AlertType.ABNORMAL_ODDS,
            title=f"赔率异常离散 — {league_id} {match_id}",
            message=(
                f"赔率标准差 {odds_std:.3f} 超过阈值 {self.odd_std_warn:.3f}\n"
                f"赔率: 主 {home_odds:.2f} / 平 {draw_odds:.2f} / 客 {away_odds:.2f}\n"
                f"建议人工复核赔率数据源。"
            ),
            metadata={
                "league": league_id,
                "match_id": match_id,
                "odds_std": f"{odds_std:.4f}",
                "home_odds": f"{home_odds:.2f}",
                "draw_odds": f"{draw_odds:.2f}",
                "away_odds": f"{away_odds:.2f}",
            },
        )
        self.alert_mgr.send(alert)
        return alert

    # ═══════════════════════════════════════════════════════
    # 回撤告警
    # ═══════════════════════════════════════════════════════

    def check_drawdown(
        self,
        current_drawdown: float,
        bankroll_state: "BankrollState",
        league_id: str = "",
    ) -> Optional[Alert]:
        """
        检查回撤是否超标并发送告警。

        在 settle_bet() 之后调用，每次结算后检查。
        """
        if current_drawdown < self.drawdown_warn_pct:
            return None

        # 回撤告警有独立冷却 (1小时)，避免每笔亏损都发
        now = datetime.now()
        if self._last_drawdown_alert:
            if (now - self._last_drawdown_alert).total_seconds() < self._drawdown_alert_cooldown:
                return None

        is_critical = current_drawdown >= self.drawdown_critical_pct
        level = AlertLevel.CRITICAL if is_critical else AlertLevel.WARN

        alert = Alert(
            level=level,
            alert_type=AlertType.DRAWDOWN,
            title=f"回撤{'严重' if is_critical else ''}警告 — {league_id or '全局'}",
            message=(
                f"当前回撤: {current_drawdown * 100:.1f}% "
                f"(警告线: {self.drawdown_warn_pct * 100:.0f}%"
                f"{' / 严重线: ' + str(self.drawdown_critical_pct * 100) + '%' if is_critical else ''})\n"
                f"当前资金: ¥{bankroll_state.balance:,.2f} | "
                f"峰值: ¥{bankroll_state.peak_balance:,.2f} | "
                f"连续亏损: {bankroll_state.consecutive_losses} 场"
            ),
            metadata={
                "league": league_id or "global",
                "drawdown": f"{current_drawdown * 100:.1f}%",
                "balance": f"¥{bankroll_state.balance:,.2f}",
                "peak": f"¥{bankroll_state.peak_balance:,.2f}",
                "consecutive_losses": str(bankroll_state.consecutive_losses),
            },
        )
        self.alert_mgr.send(alert, bypass_cooldown=is_critical)
        self._last_drawdown_alert = now
        return alert

    # ═══════════════════════════════════════════════════════
    # 流水线错误告警
    # ═══════════════════════════════════════════════════════

    def check_pipeline_errors(
        self,
        errors: List[str],
        league_id: str = "",
        match_id: str = "",
    ) -> Optional[Alert]:
        """检查流水线错误并发送告警"""
        if not errors:
            return None

        alert = Alert(
            level=AlertLevel.CRITICAL,
            alert_type=AlertType.PIPELINE_ERROR,
            title=f"流水线错误 — {league_id} {match_id}",
            message=(
                f"流水线执行过程中发生 {len(errors)} 个错误:\n\n"
                + "\n".join(f"• {e}" for e in errors[:5])
                + ("\n..." if len(errors) > 5 else "")
            ),
            metadata={
                "league": league_id,
                "match_id": match_id,
                "error_count": str(len(errors)),
                "first_error": errors[0] if errors else "",
            },
        )
        self.alert_mgr.send(alert, bypass_cooldown=True)
        return alert

    # ═══════════════════════════════════════════════════════
    # 每日摘要
    # ═══════════════════════════════════════════════════════

    def send_daily_summary(
        self,
        date_str: str,
        bankroll_state: "BankrollState",
        daily_bets: int = 0,
        daily_wins: int = 0,
        daily_profit: float = 0.0,
        league_id: str = "",
    ) -> Optional[Alert]:
        """发送每日投注摘要 (INFO 级别)"""
        win_rate = f"{daily_wins / daily_bets * 100:.1f}%" if daily_bets > 0 else "N/A"

        alert = Alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.DAILY_SUMMARY,
            title=f"每日摘要 — {league_id or '全局'} {date_str}",
            message=(
                f"投注: {daily_bets} 注 | 胜: {daily_wins} 注 | 胜率: {win_rate}\n"
                f"日盈亏: ¥{daily_profit:+,.2f} | "
                f"当前资金: ¥{bankroll_state.balance:,.2f} | "
                f"回撤: {bankroll_state.max_drawdown * 100:.1f}%"
            ),
            metadata={
                "league": league_id or "global",
                "date": date_str,
                "bets": str(daily_bets),
                "wins": str(daily_wins),
                "win_rate": win_rate,
                "profit": f"¥{daily_profit:+,.2f}",
                "balance": f"¥{bankroll_state.balance:,.2f}",
                "drawdown": f"{bankroll_state.max_drawdown * 100:.1f}%",
            },
        )
        self.alert_mgr.send(alert, bypass_cooldown=True)
        return alert

    # ═══════════════════════════════════════════════════════
    # 综合检查 (单次流水线运行后调用)
    # ═══════════════════════════════════════════════════════

    def check_after_run(
        self,
        result: "PipelineResult",
        bankroll_state: "BankrollState",
        daily_staked: float = 0.0,
        weekly_staked: float = 0.0,
    ) -> List[Alert]:
        """
        流水线运行后综合检查 — 一键检查所有告警条件。

        在 orchestrator.run_full() 之后调用。
        """
        alerts: List[Alert] = []

        # 1. 熔断检查
        if result.circuit_broken:
            a = self.check_circuit_breaker(
                circuit_broken=True,
                circuit_reason=result.circuit_reason,
                league_id=result.league_id,
                bankroll_state=bankroll_state,
            )
            if a:
                alerts.append(a)

        # 2. 大额投注检查
        if result.proposals:
            a_list = self.check_large_bets(
                proposals=result.proposals,
                bankroll_state=bankroll_state,
                league_id=result.league_id,
                match_id=getattr(result, 'match_id', ''),
            )
            alerts.extend(a_list)

        # 3. 流水线错误检查
        if result.errors:
            a = self.check_pipeline_errors(
                errors=result.errors,
                league_id=result.league_id,
                match_id=getattr(result, 'match_id', ''),
            )
            if a:
                alerts.append(a)

        # 4. 回撤检查
        if bankroll_state.max_drawdown > 0:
            a = self.check_drawdown(
                current_drawdown=bankroll_state.max_drawdown,
                bankroll_state=bankroll_state,
                league_id=result.league_id,
            )
            if a:
                alerts.append(a)

        return alerts