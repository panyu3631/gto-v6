"""
GTO-GameFlow v5.10 — 市场真实化集成器 (MarketRealismIntegrator)

将 BookmakerBehavior、TransactionCost、LiquidityCheck、DynamicHomeAdvantage
统一集成到单一入口，供统一管道调用。

v5.9 问题:
- 各模块在 Walk-Forward 测试中分散调用 (lines 292-297, 466-470, etc.)
- 缺乏统一的调用接口，参数传递不一致
- 交易成本扣除逻辑分散在多处

v5.10 统一方案:
- 单一入口: process_match() 处理一场比赛的完整市场真实化
- 单一入口: process_settlement() 处理一笔投注的完整结算
- 联赛校准: 每个联赛的限额/成本参数内聚

数据流:
    MatchOddsBundle → process_match() → adjusted_odds → pipeline
    BetResult → process_settlement() → net_profit

使用方式:
    integrator = MarketRealismIntegrator(league_id="premier_league")
    adjusted = integrator.process_match(odds_bundle)
    net_profit = integrator.process_settlement(profit, stake, strategy)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .market_realism import (
    BookmakerBehavior, TransactionCost, LiquidityCheck,
    get_dynamic_home_advantage,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class MarketAdjustment:
    """市场真实化调整结果"""
    # 调整后赔率
    adjusted_home: float
    adjusted_draw: float
    adjusted_away: float

    # 是否跳过此比赛
    skip: bool = False
    skip_reason: str = ""

    # Steam move 检测
    steam_detected: bool = False
    kelly_penalty: float = 1.0

    # 主场优势
    home_advantage: float = 0.35

    # 联赛限额
    league_limit_1x2: float = 5000.0
    league_limit_asian: float = 7500.0
    league_limit_totals: float = 6000.0


@dataclass
class SettlementResult:
    """结算结果"""
    gross_profit: float
    net_profit: float
    costs: float
    liquidity_ok: bool


# ═══════════════════════════════════════════════════════════════
# 市场真实化集成器
# ═══════════════════════════════════════════════════════════════

class MarketRealismIntegrator:
    """
    市场真实化集成器 (v5.10)。

    统一处理:
    - 庄家赔率调整 (BookmakerBehavior)
    - 交易成本扣除 (TransactionCost)
    - 流动性验证 (LiquidityCheck)
    - 动态主场优势 (DynamicHomeAdvantage)

    使用方式:
        integrator = MarketRealismIntegrator(league_id="premier_league")
        adj = integrator.process_match(odds_home=2.10, odds_draw=3.50, odds_away=3.20, season="2023-24")
        net = integrator.process_settlement(profit=100, stake=500, strategy="1x2")
    """

    def __init__(
        self,
        league_id: str = "",
        favorite_bias: float = 0.02,
        longshot_bias: float = 0.03,
        withdrawal_fee: float = 0.015,
        fx_spread: float = 0.008,
        default_limit: float = 5000.0,
    ):
        """
        参数:
            league_id: 联赛ID
            favorite_bias: 热门方 bias 幅度
            longshot_bias: 冷门方 bias 幅度
            withdrawal_fee: 提款手续费
            fx_spread: 汇率差
            default_limit: 默认投注限额
        """
        self.league_id = league_id

        # 子模块
        self.bookmaker = BookmakerBehavior(
            favorite_bias=favorite_bias,
            longshot_bias=longshot_bias,
        )
        self.tx_cost = TransactionCost(
            withdrawal_fee=withdrawal_fee,
            fx_spread=fx_spread,
        )
        self.liquidity = LiquidityCheck(default_limit=default_limit)

    def process_match(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        season: str = "",
        market_sentiment: Optional[float] = None,
        recent_home_win_rate: Optional[float] = None,
    ) -> MarketAdjustment:
        """
        处理一场比赛的市场真实化。

        1. 庄家赔率调整
        2. 动态主场优势计算
        3. 联赛限额查询

        参数:
            odds_home/draw/away: 原始赔率
            season: 赛季字符串
            market_sentiment: 市场情绪
            recent_home_win_rate: 近期主场胜率

        返回:
            MarketAdjustment
        """
        # 庄家赔率调整
        bm_adj = self.bookmaker.adjust_odds(odds_home, odds_draw, odds_away, market_sentiment)

        # 动态主场优势
        home_adv = get_dynamic_home_advantage(
            self.league_id, season, recent_home_win_rate,
        )

        # 联赛限额
        limit_1x2 = self.liquidity.get_max_stake(self.league_id, "1x2")
        limit_asian = self.liquidity.get_max_stake(self.league_id, "asian")
        limit_totals = self.liquidity.get_max_stake(self.league_id, "over_under")

        return MarketAdjustment(
            adjusted_home=bm_adj.adjusted_home,
            adjusted_draw=bm_adj.adjusted_draw,
            adjusted_away=bm_adj.adjusted_away,
            skip=bm_adj.skip_recommendation,
            skip_reason=bm_adj.skip_reason,
            steam_detected=bm_adj.steam_detected,
            kelly_penalty=self.bookmaker.get_kelly_penalty(bm_adj.steam_detected),
            home_advantage=home_adv,
            league_limit_1x2=limit_1x2,
            league_limit_asian=limit_asian,
            league_limit_totals=limit_totals,
        )

    def process_settlement(
        self,
        profit: float,
        stake: float,
        strategy: str = "1x2",
    ) -> SettlementResult:
        """
        处理一笔投注的结算。

        1. 交易成本扣除
        2. 流动性验证 (用于日志)

        参数:
            profit: 原始利润 (正=盈利, 负=亏损)
            stake: 投注金额
            strategy: 策略类型

        返回:
            SettlementResult
        """
        # 流动性验证
        can_exec, reason = self.liquidity.check(self.league_id, strategy, stake)

        # 交易成本
        net_profit = self.tx_cost.apply_costs(profit, stake)
        costs = profit - net_profit if profit > 0 else 0.0

        return SettlementResult(
            gross_profit=profit,
            net_profit=net_profit,
            costs=costs,
            liquidity_ok=can_exec,
        )

    def check_liquidity(self, stake: float, strategy: str = "1x2") -> Tuple[bool, str]:
        """检查流动性"""
        return self.liquidity.check(self.league_id, strategy, stake)

    def get_max_stake(self, strategy: str = "1x2") -> float:
        """获取最大投注额"""
        return self.liquidity.get_max_stake(self.league_id, strategy)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def create_integrator_for_league(league_id: str) -> MarketRealismIntegrator:
    """为指定联赛创建市场真实化集成器"""
    return MarketRealismIntegrator(league_id=league_id)