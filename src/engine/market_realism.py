"""
庄家行为建模与交易成本 (Phase 5.0)

三个核心模块:
1. BookmakerBehavior — 建模庄家赔率调整的非对称性
2. TransactionCost — 交易摩擦成本 (提款手续费、汇率损失)
3. LiquidityCheck — 市场流动性验证 (投注限额)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BookmakerAdjustment:
    """
    庄家赔率调整模型。

    真实市场中，庄家根据投注流动态调整赔率:
    - Favorite-Longshot Bias: 热门方赔率被压低，冷门方赔率被抬高
    - 调整幅度与原始赔率、margin 大小相关
    """
    # 调整后的赔率
    adjusted_home: float
    adjusted_draw: float
    adjusted_away: float

    # 调整量
    home_adjustment: float  # 正数=赔率提高, 负数=赔率降低
    draw_adjustment: float
    away_adjustment: float

    # 是否触发跳过 (赔率调整过大，不建议投注)
    skip_recommendation: bool = False
    skip_reason: str = ""

    # 赔率是否有剧烈变动 (steam move)
    steam_detected: bool = False


class BookmakerBehavior:
    """
    庄家行为建模。

    模拟庄家对赔率的非对称调整:
    - 热门方 (odds < 2.0): 赔率被压低 1-3% (favorite-longshot bias)
    - 冷门方 (odds > 4.0): 赔率被抬高 1-3%
    - 赔率剧烈变动 (steam move): 表示"聪明钱"流入，降低 Kelly 分数
    """

    def __init__(
        self,
        favorite_bias: float = 0.02,      # 热门方 bias 幅度
        longshot_bias: float = 0.03,      # 冷门方 bias 幅度
        steam_threshold: float = 0.05,    # 赔率变动 >= 5% 视为 steam
        steam_kelly_penalty: float = 0.5, # steam 时 Kelly 打 5 折
    ):
        self.favorite_bias = favorite_bias
        self.longshot_bias = longshot_bias
        self.steam_threshold = steam_threshold
        self.steam_kelly_penalty = steam_kelly_penalty

    def adjust_odds(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        market_sentiment: Optional[float] = None,  # -1.0 (强烈看空) 到 +1.0 (强烈看多)
    ) -> BookmakerAdjustment:
        """
        调整赔率以反映真实市场可获得性。

        参数:
            odds_home/draw/away: 原始赔率
            market_sentiment: 市场情绪 (可选)

        返回:
            BookmakerAdjustment 含调整后赔率
        """
        adj_h, h_adj = self._adjust_single(odds_home, market_sentiment, 1.0)
        adj_d, d_adj = self._adjust_single(odds_draw, market_sentiment, 0.0)
        adj_a, a_adj = self._adjust_single(odds_away, market_sentiment, -1.0)

        # 检测 steam move
        max_adj = max(abs(h_adj), abs(d_adj), abs(a_adj))
        steam = max_adj >= self.steam_threshold

        # 如果赔率被大幅压低，建议跳过
        skip = False
        reason = ""
        if adj_h < 1.05 or adj_d < 1.05 or adj_a < 1.05:
            skip = True
            reason = "adjusted odds below minimum (1.05)"

        return BookmakerAdjustment(
            adjusted_home=adj_h,
            adjusted_draw=adj_d,
            adjusted_away=adj_a,
            home_adjustment=h_adj,
            draw_adjustment=d_adj,
            away_adjustment=a_adj,
            skip_recommendation=skip,
            skip_reason=reason,
            steam_detected=steam,
        )

    def _adjust_single(
        self,
        odds: float,
        sentiment: Optional[float],
        side: float,
    ) -> tuple[float, float]:
        """对单边赔率应用调整"""
        if odds <= 0:
            return odds, 0.0

        # Favorite-Longshot Bias: 低赔率被压低，高赔率被抬高
        if odds < 2.0:
            bias = self.favorite_bias
        elif odds > 4.0:
            bias = self.longshot_bias
        else:
            bias = (self.favorite_bias + self.longshot_bias) / 2

        # 市场情绪叠加
        if sentiment is not None:
            bias *= (1.0 + sentiment * side * 0.5)

        adjustment = -bias  # 赔率降低
        adjusted = odds * (1.0 + adjustment)
        return max(1.02, adjusted), adjustment

    def get_kelly_penalty(self, steam_detected: bool) -> float:
        """获取 Kelly 惩罚系数"""
        return self.steam_kelly_penalty if steam_detected else 1.0


class TransactionCost:
    """
    交易成本模型。

    模拟实战中的交易摩擦:
    - 提款手续费: 1-2%
    - 汇率转换损失: 0.5-1%
    - 资金冻结期: 机会成本
    """

    def __init__(
        self,
        withdrawal_fee: float = 0.015,     # 提款手续费 1.5%
        fx_spread: float = 0.008,          # 汇率差 0.8%
        min_stake_threshold: float = 10.0, # 低于此金额不扣除手续费
    ):
        self.withdrawal_fee = withdrawal_fee
        self.fx_spread = fx_spread
        self.min_stake_threshold = min_stake_threshold

    def apply_costs(self, profit: float, stake: float) -> float:
        """
        扣除交易成本后的净利润。

        参数:
            profit: 原始利润 (可为负)
            stake: 投注金额

        返回:
            扣除成本后的净利润
        """
        if profit <= 0:
            return profit  # 亏损不扣手续费

        if stake < self.min_stake_threshold:
            return profit  # 小额不扣

        # 提款手续费
        net = profit * (1.0 - self.withdrawal_fee)

        # 汇率损失
        net *= (1.0 - self.fx_spread)

        return net

    def get_effective_roi(self, gross_roi: float, avg_stake: float) -> float:
        """从毛 ROI 估算有效 ROI"""
        if avg_stake < self.min_stake_threshold:
            return gross_roi
        cost_rate = self.withdrawal_fee + self.fx_spread
        return gross_roi - cost_rate


class LiquidityCheck:
    """
    市场流动性验证。

    博彩商对特定联赛/盘口有投注限额:
    - 英超: 限额最高 (€10,000+)
    - 次级联赛: 限额较低 (€1,000-5,000)
    - 亚洲盘口: 限额通常比 1X2 高
    """

    # 联赛限额 (欧元)
    LEAGUE_LIMITS = {
        "premier_league": 10000,
        "la_liga": 8000,
        "bundesliga": 8000,
        "serie_a": 6000,
        "ligue_1": 5000,
    }

    # 策略限额乘数
    STRATEGY_MULTIPLIER = {
        "1x2": 1.0,
        "asian": 1.5,    # 亚盘限额更高
        "over_under": 1.2,
        "parlay": 0.5,   # 串关限额更低
    }

    def __init__(self, default_limit: float = 5000.0):
        self.default_limit = default_limit

    def check(self, league_id: str, strategy: str, stake: float) -> tuple[bool, str]:
        """
        检查投注金额是否在市场流动性范围内。

        返回:
            (可执行, 原因说明)
        """
        league_limit = self.LEAGUE_LIMITS.get(league_id, self.default_limit)
        multiplier = self.STRATEGY_MULTIPLIER.get(strategy, 1.0)
        effective_limit = league_limit * multiplier

        if stake <= effective_limit:
            return True, f"stake {stake:.0f} within limit {effective_limit:.0f}"
        else:
            return False, f"stake {stake:.0f} exceeds limit {effective_limit:.0f}"

    def get_max_stake(self, league_id: str, strategy: str) -> float:
        """获取最大可投注金额"""
        league_limit = self.LEAGUE_LIMITS.get(league_id, self.default_limit)
        multiplier = self.STRATEGY_MULTIPLIER.get(strategy, 1.0)
        return league_limit * multiplier


# 全局预计算参数
HOME_ADVANTAGE_DYNAMIC = {
    "premier_league": {"normal": 0.38, "covid": 0.22, "post_covid": 0.35},
    "la_liga": {"normal": 0.42, "covid": 0.25, "post_covid": 0.38},
    "bundesliga": {"normal": 0.45, "covid": 0.28, "post_covid": 0.40},
    "serie_a": {"normal": 0.40, "covid": 0.24, "post_covid": 0.36},
    "ligue_1": {"normal": 0.35, "covid": 0.20, "post_covid": 0.30},
}


def get_dynamic_home_advantage(
    league_id: str,
    season: str,
    recent_home_win_rate: Optional[float] = None,
) -> float:
    """
    动态主场优势校准。

    根据赛季和近期主场胜率动态调整主场优势参数:
    - 2019/20 和 2020/21: COVID-19 空场期，主场优势大幅降低
    - 2021/22 及之后: 逐渐恢复但未完全回到 pre-COVID 水平
    - 如果提供了近期主场胜率，用 EWMA 平滑调整

    参数:
        league_id: 联赛 ID
        season: 赛季字符串 (如 "2019-20")
        recent_home_win_rate: 近期主场胜率 (可选，用于动态调整)
    """
    defaults = HOME_ADVANTAGE_DYNAMIC.get(league_id, {"normal": 0.38, "covid": 0.22, "post_covid": 0.35})

    # 基于赛季的静态调整
    if season in ("2019-20", "2020-21"):
        base = defaults["covid"]
    elif season in ("2021-22", "2022-23"):
        base = defaults["post_covid"]
    else:
        base = defaults["normal"]

    # 基于近期主场胜率的动态调整 (EWMA)
    if recent_home_win_rate is not None:
        # 主场胜率 50% → 不调整，>50% → 上调，<50% → 下调
        deviation = recent_home_win_rate - 0.50
        base += deviation * 0.15  # 缓慢调整
        base = max(0.15, min(0.55, base))  # 限制范围

    return base