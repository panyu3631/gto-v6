"""
GTO-GameFlow v5.0 全局配置

所有可调参数集中管理，支持环境变量覆盖。
"""
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class League(Enum):
    """支持的联赛枚举"""
    EPL = "premier_league"       # 英超
    LA_LIGA = "la_liga"          # 西甲
    BUNDESLIGA = "bundesliga"    # 德甲
    SERIE_A = "serie_a"          # 意甲
    LIGUE_1 = "ligue_1"          # 法甲


class BetType(Enum):
    """投注类型"""
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


class Result(Enum):
    """比赛结果"""
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


@dataclass
class BankrollConfig:
    """资金管理配置"""
    initial_bankroll: float = 10000.0          # 初始资金
    kelly_fraction: float = 0.25               # 半凯利折扣 (1/4 Kelly)
    max_total_exposure: float = 0.20           # 总投注上限 (20% of bankroll)
    single_bet_max_ratio: float = 0.05         # 单注上限 (5% of bankroll)
    daily_exposure_limit: float = 0.15         # 单日投注暴露上限 (15%)
    weekly_exposure_limit: float = 0.35        # 单周投注暴露上限 (35%)


@dataclass
class CircuitBreakerConfig:
    """熔断机制配置"""
    max_consecutive_losses: int = 5            # 连续亏损场次阈值
    daily_loss_pct: float = 0.08               # 单日亏损比例阈值
    weekly_loss_pct: float = 0.15              # 单周亏损比例阈值
    monthly_loss_pct: float = 0.25               # 单月亏损比例阈值 (规范第10.10节: 25%)
    cooldown_hours: int = 48                   # 冷却时间 (小时) (规范第10.10节: 48h)


@dataclass
class PipelineConfig:
    """计算流水线配置"""
    stage_count: int = 9                       # 9阶段流水线
    default_odds_min: float = 1.05             # 最低赔率
    default_odds_max: float = 10.0             # 最高赔率 (规范第8.4节: odds > 10.0 过滤)
    score_matrix_max: int = 5                  # 比分矩阵截断 (0:0 至 5:5)
    min_value_threshold: float = 0.02         # v5.10.5: 降低至0.02 (模型从市场先验出发，边际信号较小)


@dataclass
class AlertingConfig:
    """告警通知配置"""
    enabled: bool = True
    email_sender: str = ""                       # 发件人邮箱
    email_auth_code: str = ""                    # SMTP 授权码
    email_recipients: List[str] = field(default_factory=list)  # 收件人列表
    smtp_host: str = "smtp.qq.com"              # SMTP 服务器
    smtp_port: int = 465                         # SMTP 端口 (SSL)
    cooldown_seconds: int = 900                  # 同类型告警冷却 (15分钟)
    max_daily_sends: int = 50                    # 每日最大发送数
    # 告警阈值
    drawdown_warn_pct: float = 0.20              # 回撤警告阈值
    drawdown_critical_pct: float = 0.30          # 回撤严重阈值
    large_bet_ratio: float = 0.04                # 大额投注阈值 (单注 > 4% 资金)
    odd_std_warn: float = 0.08                   # 赔率离散度警告阈值


@dataclass
class GlobalConfig:
    """全局配置"""
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    supported_leagues: List[League] = field(default_factory=lambda: list(League))
    version: str = "5.9.2"
    # 数据库
    db_url: str = field(default_factory=lambda: os.getenv(
        "GTO_DB_URL", "postgresql://localhost:5432/gto_gameflow"
    ))
    # API 配置
    api_football_key: Optional[str] = field(default_factory=lambda: os.getenv("API_FOOTBALL_KEY"))
    football_data_key: Optional[str] = field(default_factory=lambda: os.getenv("FOOTBALL_DATA_KEY"))
    # 日志
    log_level: str = field(default_factory=lambda: os.getenv("GTO_LOG_LEVEL", "INFO"))


# 全局单例
config = GlobalConfig()