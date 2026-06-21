"""
GTO-GameFlow v5.5 核心数据模型 — 支持多策略（1X2 / 亚盘 / 大小球）

定义系统中所有核心数据结构，使用 dataclass + Pydantic 风格。
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class StrategyType(str, Enum):
    """策略类型"""
    ONE_X_TWO = "1x2"               # 欧赔胜平负
    ASIAN_HANDICAP = "asian_handicap"  # 亚洲让球盘
    OVER_UNDER = "over_under"       # 大小球


class BetSelection(str, Enum):
    """投注选择 — 1X2 / 大小球"""
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    OVER = "over"               # 大小球: 大球
    UNDER = "under"             # 大小球: 小球


class AsianHandicapResult(str, Enum):
    """亚盘结算结果"""
    FULL_WIN = "full_win"      # 全赢
    HALF_WIN = "half_win"      # 赢半
    PUSH = "push"              # 走水
    HALF_LOSS = "half_loss"    # 输半
    FULL_LOSS = "full_loss"    # 全输


class BetResult(str, Enum):
    """1X2 结算结果"""
    WIN = "win"
    LOSS = "loss"
    VOID = "void"
    PENDING = "pending"


# FactorCategory 定义在 src.factors.registry 中，此处仅引用
# from src.factors.registry import FactorCategory


@dataclass
class FactorDelta:
    """单个因子的 delta 值（对胜平负三个方向的调整）"""
    factor_id: str
    delta_home: float = 0.0
    delta_draw: float = 0.0
    delta_away: float = 0.0


@dataclass
class MatchContext:
    """单场比赛的完整上下文"""
    match_id: str
    league_id: str
    season: str
    matchday: int
    kickoff_time: datetime

    # 球队
    home_team: str
    away_team: str

    # Elo
    home_elo: float = 1500.0
    away_elo: float = 1500.0

    # 赔率 (1X2)
    odds_home: float = 1.0
    odds_draw: float = 1.0
    odds_away: float = 1.0

    # 进阶数据 (可选)
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None

    # 扩展数据
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProbabilityDistribution:
    """胜平负概率分布"""
    prob_home: float
    prob_draw: float
    prob_away: float

    def __post_init__(self):
        total = self.prob_home + self.prob_draw + self.prob_away
        if total <= 0:
            raise ValueError(f"概率分布总和必须大于0: {self}")
        # 归一化
        self.prob_home /= total
        self.prob_draw /= total
        self.prob_away /= total

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.prob_home, self.prob_draw, self.prob_away)


@dataclass
class ScoreMatrix:
    """比分概率矩阵 (0:0 至 N:N)"""
    league_id: str
    max_goals: int
    matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def get_probability(self, home_goals: int, away_goals: int) -> float:
        return self.matrix.get((home_goals, away_goals), 0.0)


@dataclass
class BetProposal:
    """单注投注建议 — 支持多策略类型"""
    match_id: str
    selection: BetSelection
    odds: float
    model_prob: float           # 模型概率
    implied_prob: float         # 赔率隐含概率
    value: float                # 价值 = model_prob - implied_prob
    kelly_stake: float          # Kelly 建议投注额
    adjusted_stake: float       # 风控调整后投注额
    priority_score: float       # 优先级排序分数
    confidence: float = 0.0     # v5.10.5: 置信度评分 (规范第8.3b节)
    league_id: str = ""
    strategy_type: str = "1x2"  # 策略类型: "1x2"/"asian_handicap"/"over_under"
    # 亚盘专用字段
    handicap_line: float = 0.0  # 让球线 (正数=主队让球, 如 0.5 表示主让半球)
    # 大小球专用字段
    totals_line: float = 0.0    # 大小球线 (如 2.5)
    # 多策略组合权重
    strategy_weight: float = 1.0  # MPT 分配的策略内权重


@dataclass
class AsianHandicapProposal:
    """亚盘投注建议"""
    match_id: str
    handicap_line: float        # 让球线 (正数=主队让球)
    side: str                   # "home" 或 "away" — 投注主队或客队
    odds: float                 # 亚盘赔率（水位）
    cover_prob: float           # 模型计算的跑出概率
    implied_prob: float         # 赔率隐含概率
    value: float                # 价值
    kelly_stake: float
    adjusted_stake: float
    priority_score: float
    league_id: str = ""
    strategy_type: str = "asian_handicap"
    strategy_weight: float = 1.0
    confidence: float = 0.0     # v5.10.8: 亚盘置信度 (_compute_confidence)


@dataclass
class TotalsProposal:
    """大小球投注建议"""
    match_id: str
    totals_line: float          # 大小球线 (如 2.5)
    side: str                   # "over" 或 "under"
    odds: float
    over_prob: float            # 超过大小球线的概率
    implied_prob: float
    value: float
    kelly_stake: float
    adjusted_stake: float
    priority_score: float
    league_id: str = ""
    strategy_type: str = "over_under"
    strategy_weight: float = 1.0
    confidence: float = 0.0     # v5.10.8: 大小球置信度


@dataclass
class TotalsDistribution:
    """总进球数概率分布"""
    league_id: str
    avg_goals: float
    distribution: Dict[int, float] = field(default_factory=dict)  # 总进球数 → 概率

    def over_prob(self, line: float) -> float:
        """计算超过指定线的概率"""
        prob = 0.0
        for goals, p in self.distribution.items():
            if goals > line:
                prob += p
            elif goals == int(line) and line != int(line):
                # 非整数线 (如 2.5): 不存在等于的情况
                pass
            elif goals == int(line) and line == int(line):
                # 整数线 (如 2.0): 等于时走水，不计入 over
                pass
        return prob

    def under_prob(self, line: float) -> float:
        """计算低于指定线的概率"""
        prob = 0.0
        for goals, p in self.distribution.items():
            if goals < line:
                prob += p
        return prob

    def exact_prob(self, line: float) -> float:
        """恰好等于指定线的概率 (整数线走水)"""
        if line != int(line):
            return 0.0
        return self.distribution.get(int(line), 0.0)


@dataclass
class StrategyAllocation:
    """MPT 多策略权重分配"""
    strategy_type: str
    weight: float               # 资金分配权重 (0-1)
    expected_return: float      # 预期回报率
    volatility: float           # 波动率 (标准差)
    sharpe: float               # 夏普比率
    allocation: float           # 本轮资金分配额

    @property
    def allocation_pct(self) -> float:
        return self.weight * 100


@dataclass
class StrategyPortfolio:
    """多策略投资组合"""
    allocations: List[StrategyAllocation] = field(default_factory=list)
    total_expected_return: float = 0.0
    total_volatility: float = 0.0
    portfolio_sharpe: float = 0.0
    total_bankroll: float = 0.0
    total_allocated: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def unallocated(self) -> float:
        return self.total_bankroll - self.total_allocated


@dataclass
class BetPlacement:
    """已执行的投注记录"""
    bet_id: str
    match_id: str
    selection: BetSelection
    odds: float
    stake: float
    placed_at: datetime
    result: BetResult = BetResult.PENDING
    profit_loss: float = 0.0
    league_id: str = ""


@dataclass
class BankrollState:
    """资金状态快照"""
    balance: float
    total_staked: float = 0.0
    total_returned: float = 0.0
    total_bets: int = 0
    total_wins: int = 0
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0
    peak_balance: float = 0.0
    max_drawdown: float = 0.0

    @property
    def roi(self) -> float:
        if self.total_staked == 0:
            return 0.0
        return (self.total_returned - self.total_staked) / self.total_staked

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.total_wins / self.total_bets


@dataclass
class CircuitBreakerState:
    """熔断机制状态"""
    is_active: bool = False
    trigger_reason: str = ""
    triggered_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    consecutive_losses: int = 0