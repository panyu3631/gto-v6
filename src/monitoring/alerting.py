"""
GTO-GameFlow v5.5 — 告警通知系统

支持:
- 邮件通知 (QQ邮箱 SMTP)
- 告警分级: INFO / WARN / CRITICAL
- 告警类型: 熔断触发 / 大额投注 / 异常赔率 / 回撤警告 / 系统错误
- 频率限制: 同类型告警冷却期、每日最大发送数
- 批量聚合: 可配置时间窗口内聚合发送

使用方式:
    manager = AlertManager(EmailNotifier(
        smtp_host="smtp.qq.com",
        smtp_port=465,
        sender="739252249@qq.com",
        password="<授权码>",
        recipients=["739252249@qq.com"],
    ))
    manager.send(Alert(
        level=AlertLevel.WARN,
        alert_type=AlertType.CIRCUIT_BREAKER,
        title="熔断触发 — 英超",
        message="单日亏损 8.5% 超过 8% 阈值，已触发熔断。",
        metadata={"league": "premier_league", "daily_loss_pct": 8.5},
    ))
"""

import smtplib
import time
import threading
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"          # 信息通知 (日常摘要、系统健康)
    WARN = "warn"          # 警告 (回撤超限、赔率异常)
    CRITICAL = "critical"  # 严重 (熔断触发、系统错误)


class AlertType(Enum):
    """告警类型"""
    CIRCUIT_BREAKER = "circuit_breaker"    # 熔断触发/恢复
    LARGE_BET = "large_bet"                # 大额投注
    ABNORMAL_ODDS = "abnormal_odds"        # 异常赔率
    DRAWDOWN = "drawdown"                  # 回撤警告
    DAILY_SUMMARY = "daily_summary"        # 每日摘要
    SYSTEM = "system"                      # 系统状态
    PIPELINE_ERROR = "pipeline_error"      # 流水线错误


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """告警消息"""
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    metadata: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    sent: bool = False


# ═══════════════════════════════════════════════════════════════
# 邮件通知器
# ═══════════════════════════════════════════════════════════════

class EmailNotifier:
    """
    邮件通知器 — 通过 QQ SMTP 发送告警邮件。

    参数:
        smtp_host: SMTP 服务器地址 (QQ: smtp.qq.com)
        smtp_port: SMTP 端口 (QQ SSL: 465)
        sender: 发件人邮箱地址
        password: SMTP 授权码 (QQ邮箱设置 → 账户 → POP3/SMTP 服务 → 生成授权码)
        recipients: 收件人邮箱列表
        use_ssl: 是否使用 SSL 连接
    """

    def __init__(
        self,
        smtp_host: str = "smtp.qq.com",
        smtp_port: int = 465,
        sender: str = "",
        password: str = "",
        recipients: Optional[List[str]] = None,
        use_ssl: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipients = recipients or []
        self.use_ssl = use_ssl
        self._last_error: Optional[str] = None
        self._sent_count: int = 0
        self._fail_count: int = 0

    def send(self, alert: Alert) -> bool:
        """
        发送单条告警邮件。

        返回:
            True 表示发送成功，False 表示发送失败。
        """
        if not self.sender or not self.password or not self.recipients:
            self._last_error = "邮件配置不完整: 缺少 sender/password/recipients"
            logger.warning(self._last_error)
            return False

        try:
            subject = self._format_subject(alert)
            html_body = self._format_html(alert)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
                server.starttls()

            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.recipients, msg.as_string())
            server.quit()

            self._sent_count += 1
            self._last_error = None
            logger.info(f"告警邮件已发送: {alert.title} → {self.recipients}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            self._fail_count += 1
            self._last_error = f"SMTP 认证失败，请检查授权码: {e}"
            logger.error(self._last_error)
            return False
        except smtplib.SMTPConnectError as e:
            self._fail_count += 1
            self._last_error = f"SMTP 连接失败 ({self.smtp_host}:{self.smtp_port}): {e}"
            logger.error(self._last_error)
            return False
        except Exception as e:
            self._fail_count += 1
            self._last_error = f"邮件发送异常: {e}"
            logger.error(self._last_error)
            return False

    def send_batch(self, alerts: List[Alert]) -> bool:
        """批量发送聚合告警 (合并为一封邮件)"""
        if not alerts:
            return True
        if len(alerts) == 1:
            return self.send(alerts[0])

        # 取最高级别作为标题
        max_level = max(alerts, key=lambda a: {
            AlertLevel.INFO: 0, AlertLevel.WARN: 1, AlertLevel.CRITICAL: 2
        }[a.level])

        merged = Alert(
            level=max_level.level,
            alert_type=AlertType.SYSTEM,
            title=f"GTO-GameFlow 告警聚合 ({len(alerts)}条)",
            message=self._render_batch_body(alerts),
            metadata={"batch_count": len(alerts)},
        )
        return self.send(merged)

    def _format_subject(self, alert: Alert) -> str:
        level_icon = {
            AlertLevel.INFO: "ℹ",
            AlertLevel.WARN: "⚠",
            AlertLevel.CRITICAL: "🔴",
        }
        type_cn = {
            AlertType.CIRCUIT_BREAKER: "熔断",
            AlertType.LARGE_BET: "大额投注",
            AlertType.ABNORMAL_ODDS: "异常赔率",
            AlertType.DRAWDOWN: "回撤",
            AlertType.DAILY_SUMMARY: "每日摘要",
            AlertType.SYSTEM: "系统",
            AlertType.PIPELINE_ERROR: "流水线错误",
        }
        return f"{level_icon.get(alert.level, '')} [GTO-v5.5] [{type_cn.get(alert.alert_type, alert.alert_type.value)}] {alert.title}"

    def _format_html(self, alert: Alert) -> str:
        level_color = {
            AlertLevel.INFO: "#00d4aa",
            AlertLevel.WARN: "#ffa502",
            AlertLevel.CRITICAL: "#ff4757",
        }
        color = level_color.get(alert.level, "#7b8ca0")
        ts = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        meta_html = ""
        if alert.metadata:
            meta_rows = "".join(
                f"<tr><td style='padding:4px 12px;color:#7b8ca0;font-size:13px'>{k}</td>"
                f"<td style='padding:4px 12px;color:#e8ecf1;font-size:13px;font-weight:600'>{v}</td></tr>"
                for k, v in alert.metadata.items()
            )
            meta_html = f"""
            <table style="width:100%;border-collapse:collapse;margin-top:12px">
                {meta_rows}
            </table>"""

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background:#0f1923;padding:24px;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif">
<div style="max-width:600px;margin:0 auto;background:#1a2332;border-radius:8px;border:1px solid #2a3a4a;overflow:hidden">
    <div style="background:{color};padding:16px 24px">
        <h2 style="margin:0;color:#0f1923;font-size:18px">{alert.title}</h2>
    </div>
    <div style="padding:24px">
        <p style="color:#e8ecf1;font-size:15px;line-height:1.8;margin:0 0 16px 0">{alert.message}</p>
        {meta_html}
        <hr style="border:none;border-top:1px solid #2a3a4a;margin:20px 0">
        <p style="color:#7b8ca0;font-size:12px;margin:0">
            GTO-GameFlow v5.5 自动告警系统 &nbsp;|&nbsp; 级别: {alert.level.value.upper()} &nbsp;|&nbsp; {ts}
        </p>
    </div>
</div>
</body>
</html>"""

    def _render_batch_body(self, alerts: List[Alert]) -> str:
        lines = []
        for i, a in enumerate(alerts, 1):
            level_icon = {AlertLevel.CRITICAL: "🔴", AlertLevel.WARN: "⚠", AlertLevel.INFO: "ℹ"}.get(a.level, "")
            lines.append(f"<p style='color:#e8ecf1;margin:8px 0'>{level_icon} <strong>#{i}</strong> {a.title}: {a.message}</p>")
        return "\n".join(lines)

    @property
    def stats(self) -> Dict:
        return {
            "sent_count": self._sent_count,
            "fail_count": self._fail_count,
            "last_error": self._last_error,
        }


# ═══════════════════════════════════════════════════════════════
# 告警管理器
# ═══════════════════════════════════════════════════════════════

class AlertManager:
    """
    告警管理核心 — 分级、频率限制、聚合发送。

    功能:
    - 同类型告警冷却: 避免短时间内重复发送同类告警
    - 日发送上限: 每日最多发送 N 封邮件
    - 批量聚合: 可配置时间窗口，将窗口内的告警合并发送
    - 异步发送: 邮件发送在后台线程中执行，不阻塞主流程

    参数:
        notifier: 通知器实例 (EmailNotifier)
        cooldown_seconds: 同类型告警冷却时间 (秒)
        max_daily_sends: 每日最大发送数
        batch_window_seconds: 批量聚合时间窗口 (秒)，0 表示不聚合
        enabled: 是否启用告警
    """

    def __init__(
        self,
        notifier: Optional[EmailNotifier] = None,
        cooldown_seconds: int = 900,       # 15 分钟冷却
        max_daily_sends: int = 50,
        batch_window_seconds: int = 0,     # 默认不聚合，即时发送
        enabled: bool = True,
    ):
        self.notifier = notifier
        self.cooldown_seconds = cooldown_seconds
        self.max_daily_sends = max_daily_sends
        self.batch_window_seconds = batch_window_seconds
        self.enabled = enabled

        # 内部状态
        self._last_sent: Dict[AlertType, datetime] = {}   # 每种类型上次发送时间
        self._daily_count: int = 0
        self._daily_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._alert_history: List[Alert] = []
        self._lock = threading.Lock()

    def send(
        self,
        alert: Alert,
        bypass_cooldown: bool = False,
    ) -> bool:
        """
        发送告警 (如果启用且未超过频率限制)。

        返回:
            True 表示已发送，False 表示被抑制 (冷却/超限/禁用)。
        """
        if not self.enabled:
            logger.debug(f"告警已禁用，跳过: {alert.title}")
            return False

        with self._lock:
            # 重置每日计数
            now = datetime.now()
            if now.date() > self._daily_reset.date():
                self._daily_count = 0
                self._daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # 日发送上限检查
            if self._daily_count >= self.max_daily_sends:
                logger.warning(f"告警已达日发送上限 ({self.max_daily_sends})，跳过: {alert.title}")
                return False

            # 同类型冷却检查 (CRITICAL 级别绕过冷却)
            if not bypass_cooldown and alert.level != AlertLevel.CRITICAL:
                last_time = self._last_sent.get(alert.alert_type)
                if last_time and (now - last_time).total_seconds() < self.cooldown_seconds:
                    remaining = self.cooldown_seconds - (now - last_time).total_seconds()
                    logger.debug(f"告警类型 {alert.alert_type.value} 在冷却中 ({remaining:.0f}s)，跳过: {alert.title}")
                    return False

            # 更新状态
            self._last_sent[alert.alert_type] = now
            self._daily_count += 1
            self._alert_history.append(alert)

            # 裁剪历史 (保留最近 1000 条)
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-500:]

        # 异步发送邮件
        if self.notifier:
            t = threading.Thread(target=self._do_send, args=(alert,), daemon=True)
            t.start()
        else:
            logger.info(f"[告警-未发送] {alert.level.value.upper()}: {alert.title} — {alert.message}")

        return True

    def _do_send(self, alert: Alert):
        """在后台线程中发送邮件"""
        success = self.notifier.send(alert)
        alert.sent = success

    def send_batch(self, alerts: List[Alert]) -> bool:
        """
        批量发送聚合告警。

        如果配置了 batch_window_seconds > 0，则所有告警合并为一封邮件。
        否则逐条发送。
        """
        if not alerts:
            return True

        if self.batch_window_seconds > 0 and self.notifier:
            return self.notifier.send_batch(alerts)

        results = [self.send(a) for a in alerts]
        return all(results)

    def send_if_configured(self, alert: Alert) -> bool:
        """仅在通知器已配置时发送"""
        if self.notifier and self.notifier.sender and self.notifier.password:
            return self.send(alert)
        return False

    @property
    def history(self) -> List[Alert]:
        """返回最近的告警历史"""
        return list(self._alert_history)

    @property
    def stats(self) -> Dict:
        """返回告警系统统计"""
        return {
            "enabled": self.enabled,
            "daily_count": self._daily_count,
            "max_daily_sends": self.max_daily_sends,
            "total_sent": len(self._alert_history),
            "notifier_stats": self.notifier.stats if self.notifier else None,
        }


# ═══════════════════════════════════════════════════════════════
# 便捷工厂函数
# ═══════════════════════════════════════════════════════════════

def create_qq_email_alert_manager(
    sender: str,
    auth_code: str,
    recipients: Optional[List[str]] = None,
    cooldown_seconds: int = 900,
    max_daily_sends: int = 50,
    enabled: bool = True,
) -> AlertManager:
    """
    创建 QQ 邮箱告警管理器。

    参数:
        sender: 发件人 QQ 邮箱地址
        auth_code: QQ 邮箱 SMTP 授权码 (非 QQ 密码！)
        recipients: 收件人列表 (默认同 sender)
        cooldown_seconds: 同类型告警冷却 (秒)
        max_daily_sends: 每日最大发送数
        enabled: 是否启用
    """
    notifier = EmailNotifier(
        smtp_host="smtp.qq.com",
        smtp_port=465,
        sender=sender,
        password=auth_code,
        recipients=recipients or [sender],
        use_ssl=True,
    )
    return AlertManager(
        notifier=notifier,
        cooldown_seconds=cooldown_seconds,
        max_daily_sends=max_daily_sends,
        enabled=enabled,
    )