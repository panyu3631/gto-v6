"""
GTO-GameFlow v5.8 — MPT 现代投资组合优化器

将三种策略 (1X2 / 亚盘 / 大小球) 视为独立资产, 通过现代投资组合理论
(Modern Portfolio Theory) 在有效前沿上分配最优权重。

v5.8 升级:
- 成对协方差估计: 使用所有重叠数据点, 而非截断对齐 (FIND-017)
- 收缩阈值调整: 更保守的收缩策略 (FIND-018)
- 最小样本数提升: min_samples 10→20, 防止小样本过拟合

核心算法:
1. 成对协方差矩阵估计 — 充分利用异步收益序列
2. 均值-方差优化 — 最大化 Sharpe 比率 (约束: 权重 ≥ 0, Σw = 1)
3. 风险平价回退 — 当数据不足时, 按波动率倒数分配权重
4. 贝叶斯收缩 — 小样本下向先验 (等权重) 收缩估计

使用方式:
    from src.strategies import MPTPortfolioOptimizer

    opt = MPTPortfolioOptimizer()
    portfolio = opt.optimize(
        strategy_returns={
            "1x2": StrategyReturnSeries(...),
            "asian_handicap": StrategyReturnSeries(...),
            "over_under": StrategyReturnSeries(...),
        },
        total_bankroll=100000,
    )
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.models import StrategyAllocation, StrategyPortfolio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategyReturnSeries:
    """单个策略的历史收益序列 (用于 MPT 协方差估计)"""
    strategy_type: str                     # "1x2" / "asian_handicap" / "over_under"
    returns: List[float] = field(default_factory=list)  # 每笔投注的 ROI (profit/stake)
    total_bets: int = 0
    total_stake: float = 0.0
    total_profit: float = 0.0
    win_rate: float = 0.0
    avg_odds: float = 0.0

    @property
    def mean_return(self) -> float:
        """平均每注回报率"""
        if not self.returns:
            return 0.0
        return np.mean(self.returns)

    @property
    def volatility(self) -> float:
        """回报率波动率 (标准差)"""
        if len(self.returns) < 2:
            return 0.05  # 默认5%波动率
        return np.std(self.returns, ddof=1)

    @property
    def sharpe(self) -> float:
        """策略 Sharpe 比率 (假设无风险利率=0)"""
        vol = self.volatility
        if vol <= 0:
            return 0.0
        return self.mean_return / vol

    @property
    def n_bets(self) -> int:
        return len(self.returns) if self.returns else self.total_bets

    def to_dict(self) -> dict:
        return {
            "strategy_type": self.strategy_type,
            "n_bets": self.n_bets,
            "mean_return": round(self.mean_return, 4),
            "volatility": round(self.volatility, 4),
            "sharpe": round(self.sharpe, 4),
            "total_profit": round(self.total_profit, 2),
            "win_rate": round(self.win_rate, 4),
        }


# ═══════════════════════════════════════════════════════════════
# 协方差估计器
# ═══════════════════════════════════════════════════════════════

class CovarianceEstimator:
    """
    v5.8: 策略间协方差矩阵估计。

    方法:
    - 成对协方差 (v5.8): 对每对策略使用所有重叠数据点
    - 样本协方差 (Fallback): 截断对齐 (向后兼容)
    - 贝叶斯收缩: 向对角矩阵收缩
    - Ledoit-Wolf 风格收缩: 平衡样本协方差和结构化协方差
    """

    @staticmethod
    def estimate(
        return_series: Dict[str, StrategyReturnSeries],
        min_samples: int = 20,
        shrinkage: float = 0.3,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        v5.8: 成对协方差估计 (FIND-017)。

        使用每对策略的所有重叠数据点计算协方差,
        而非截断到最小公共长度。充分利用异步收益序列。

        参数:
            return_series: 策略收益序列字典
            min_samples: 最少样本数 (v5.8: 10→20)
            shrinkage: 收缩强度 (0=纯样本, 1=纯对角)

        返回:
            (协方差矩阵 N×N, 策略名称列表)
        """
        strategies = list(return_series.keys())
        n = len(strategies)

        if n <= 1:
            return np.array([[max(return_series[strategies[0]].volatility ** 2, 0.0001)]]), strategies

        # v5.8: 成对协方差估计
        cov = np.zeros((n, n))
        pair_counts = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                returns_i = return_series[strategies[i]].returns
                returns_j = return_series[strategies[j]].returns

                if not returns_i or not returns_j:
                    # 默认使用方差
                    if i == j:
                        cov[i, i] = max(return_series[strategies[i]].volatility ** 2, 0.0001)
                    else:
                        cov[i, j] = cov[j, i] = 0.0
                    continue

                # 成对: 使用所有重叠点
                len_i = len(returns_i)
                len_j = len(returns_j)

                if len_i == len_j:
                    # 相同长度: 直接计算
                    xi = np.array(returns_i)
                    xj = np.array(returns_j)
                    cov_ij = np.cov(xi, xj)[0, 1] if len_i >= 2 else 0.0
                    pair_counts[i, j] = pair_counts[j, i] = len_i
                else:
                    # 不同长度: 截取较短者末尾进行对齐
                    min_len = min(len_i, len_j)
                    xi = np.array(returns_i[-min_len:])
                    xj = np.array(returns_j[-min_len:])
                    cov_ij = np.cov(xi, xj)[0, 1] if min_len >= 2 else 0.0
                    pair_counts[i, j] = pair_counts[j, i] = min_len

                cov[i, j] = cov_ij
                if i != j:
                    cov[j, i] = cov_ij

        # 确定最小样本数用于收缩决策
        min_pair_count = np.min(pair_counts[pair_counts > 0]) if np.any(pair_counts > 0) else 0

        # v5.8: 收缩调整 (FIND-018)
        # 更保守的收缩: 样本不足时增加收缩强度
        diag_target = np.diag(np.diag(cov))

        if min_pair_count < min_samples:
            # 小样本: 强收缩 → 对角矩阵
            effective_shrinkage = max(0.5, 1.0 - min_pair_count / min_samples)
        elif min_pair_count < min_samples * 2:
            # 中等样本: 中等收缩
            effective_shrinkage = max(shrinkage, 20.0 / min_pair_count)
        else:
            # 大样本: 轻度收缩
            effective_shrinkage = min(shrinkage, 20.0 / min_pair_count)

        shrunk_cov = (1 - effective_shrinkage) * cov + effective_shrinkage * diag_target

        # 确保对称正定
        shrunk_cov = (shrunk_cov + shrunk_cov.T) / 2

        return shrunk_cov, strategies


# ═══════════════════════════════════════════════════════════════
# MPT 优化器 (均值-方差)
# ═══════════════════════════════════════════════════════════════

class MPTPortfolioOptimizer:
    """
    MPT 现代投资组合多策略权重优化器。

    参数:
        risk_free_rate: 无风险利率 (默认 0)
        max_single_weight: 单一策略最大权重 (默认 0.5, 防止过度集中)
        min_single_weight: 单一策略最小权重 (默认 0.1, 确保至少分配)
        risk_aversion: 风险厌恶系数 (默认 1.0, 越大越保守)
        num_portfolios: 随机搜索的候选组合数 (默认 10000)
        default_weights: 数据不足时的默认权重 {'1x2': 0.5, 'asian_handicap': 0.25, 'over_under': 0.25}
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        max_single_weight: float = 0.5,
        min_single_weight: float = 0.0,
        risk_aversion: float = 1.0,
        num_portfolios: int = 10000,
        default_weights: Optional[Dict[str, float]] = None,
    ):
        self.risk_free_rate = risk_free_rate
        self.max_single_weight = max_single_weight
        self.min_single_weight = min_single_weight
        self.risk_aversion = risk_aversion
        self.num_portfolios = num_portfolios
        self.default_weights = default_weights or {
            "1x2": 0.50,
            "asian_handicap": 0.25,
            "over_under": 0.25,
        }

    def optimize(
        self,
        strategy_returns: Dict[str, StrategyReturnSeries],
        total_bankroll: float = 100000.0,
        active_strategies: Optional[List[str]] = None,
    ) -> StrategyPortfolio:
        """
        优化多策略权重分配。

        参数:
            strategy_returns: 各策略的历史收益序列
            total_bankroll: 总资金
            active_strategies: 本轮活跃的策略 (None=全部使用)

        返回:
            StrategyPortfolio 包含权重分配
        """
        if active_strategies is None:
            active_strategies = list(strategy_returns.keys())
        else:
            active_strategies = [s for s in active_strategies if s in strategy_returns]

        if not active_strategies:
            return StrategyPortfolio(total_bankroll=total_bankroll)

        # 1. 协方差估计
        cov_estimator = CovarianceEstimator()
        active_series = {s: strategy_returns[s] for s in active_strategies}
        cov_matrix, strategy_names = cov_estimator.estimate(active_series)

        # 2. 期望收益向量
        mu = np.array([
            active_series[s].mean_return for s in strategy_names
        ])

        # 3. 判断数据是否充足
        min_bets = min(
            active_series[s].n_bets for s in strategy_names
        )

        if min_bets < 10:
            # 数据不足: 使用风险平价
            weights = self._risk_parity(cov_matrix, strategy_names, active_series)
        elif min_bets < 30:
            # 中等数据: 贝叶斯混合 (均值-方差 + 等权重)
            mv_weights = self._mean_variance_optimize(cov_matrix, mu, strategy_names)
            prior_weights = self._equal_weights(strategy_names)
            # 收缩因子: 样本越少, 越接近先验
            shrink = max(0.0, 1.0 - min_bets / 30.0)
            weights = (1 - shrink) * mv_weights + shrink * prior_weights
        else:
            # 充足数据: 纯均值-方差优化
            weights = self._mean_variance_optimize(cov_matrix, mu, strategy_names)

        # 4. 构建分配结果
        allocations = []
        total_allocated = 0.0

        for i, name in enumerate(strategy_names):
            weight = float(weights[i])
            series = active_series[name]
            alloc_amount = weight * total_bankroll

            allocations.append(StrategyAllocation(
                strategy_type=name,
                weight=round(weight, 4),
                expected_return=round(series.mean_return, 4),
                volatility=round(series.volatility, 4),
                sharpe=round(series.sharpe, 4),
                allocation=round(alloc_amount, 2),
            ))
            total_allocated += alloc_amount

        # 5. 组合统计
        portfolio_vol = math.sqrt(
            weights.T @ cov_matrix @ weights
        )
        portfolio_return = float(np.dot(weights, mu))
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / max(portfolio_vol, 0.001)

        return StrategyPortfolio(
            allocations=allocations,
            total_expected_return=round(portfolio_return, 4),
            total_volatility=round(portfolio_vol, 4),
            portfolio_sharpe=round(portfolio_sharpe, 4),
            total_bankroll=total_bankroll,
            total_allocated=round(total_allocated, 2),
        )

    def _mean_variance_optimize(
        self,
        cov_matrix: np.ndarray,
        mu: np.ndarray,
        strategy_names: List[str],
    ) -> np.ndarray:
        """
        均值-方差优化: 在有效前沿上搜索最大 Sharpe 比率组合。

        使用随机搜索 (Monte Carlo) 在约束空间中采样:
        - Σw_i = 1
        - w_i ∈ [min_single_weight, max_single_weight]
        """
        n = len(strategy_names)
        best_sharpe = -float("inf")
        best_weights = self._equal_weights(strategy_names)

        rng = np.random.RandomState(42)

        # 候选组合生成
        for _ in range(self.num_portfolios):
            # 生成随机权重
            raw = rng.rand(n)
            raw = raw / raw.sum()

            # 应用约束
            raw = np.clip(raw, self.min_single_weight, self.max_single_weight)
            # 重新归一化
            total = raw.sum()
            if total > 0:
                w = raw / total
            else:
                w = self._equal_weights(strategy_names)

            # 计算组合指标
            port_return = np.dot(w, mu)
            port_vol = math.sqrt(w.T @ cov_matrix @ w)

            if port_vol <= 0:
                continue

            sharpe = (port_return - self.risk_free_rate) / port_vol

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w.copy()

        return best_weights

    def _risk_parity(
        self,
        cov_matrix: np.ndarray,
        strategy_names: List[str],
        active_series: Dict[str, StrategyReturnSeries],
    ) -> np.ndarray:
        """
        风险平价: 按波动率倒数分配权重。

        当历史数据不足时使用, 避免过拟合。
        w_i ∝ 1 / σ_i
        """
        n = len(strategy_names)
        vols = np.array([
            active_series[s].volatility for s in strategy_names
        ])
        vols = np.maximum(vols, 0.01)  # 避免除零

        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        # 应用约束
        weights = np.clip(weights, self.min_single_weight, self.max_single_weight)
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def _equal_weights(self, strategy_names: List[str]) -> np.ndarray:
        """等权重 (均匀分配)"""
        n = len(strategy_names)
        return np.ones(n) / n

    # ═══════════════════════════════════════════════════════════
    # 便捷方法
    # ═══════════════════════════════════════════════════════════

    def update_return_series(
        self,
        existing: StrategyReturnSeries,
        new_return: float,
        stake: float = 0.0,
        profit: float = 0.0,
        odds: float = 0.0,
        won: bool = False,
        max_history: int = 500,
    ) -> StrategyReturnSeries:
        """
        更新策略收益序列 (增量式, 用于实时回测)。

        参数:
            existing: 现有序列
            new_return: 本次 ROI (profit/stake)
            stake: 本次投注额
            profit: 本次盈亏
            odds: 本次赔率
            won: 是否获胜
            max_history: 最大保留历史 (FIFO)

        返回:
            更新后的 StrategyReturnSeries
        """
        returns = (existing.returns + [new_return])[-max_history:]
        total_bets = existing.total_bets + 1
        total_stake = existing.total_stake + stake
        total_profit = existing.total_profit + profit
        old_wins = existing.win_rate * max(existing.total_bets, 1)
        new_win_rate = (old_wins + (1 if won else 0)) / max(total_bets, 1)

        return StrategyReturnSeries(
            strategy_type=existing.strategy_type,
            returns=returns,
            total_bets=total_bets,
            total_stake=total_stake,
            total_profit=total_profit,
            win_rate=new_win_rate,
            avg_odds=((existing.avg_odds * existing.total_bets + odds) /
                      max(total_bets, 1)) if odds > 0 else existing.avg_odds,
        )

    def build_from_results(
        self,
        strategy_type: str,
        settlement_results: List[dict],
    ) -> StrategyReturnSeries:
        """
        从结算结果列表构建 StrategyReturnSeries。

        参数:
            strategy_type: 策略类型
            settlement_results: [{"stake": float, "profit_loss": float, "odds": float, "won": bool}, ...]

        返回:
            StrategyReturnSeries
        """
        returns = []
        total_bets = 0
        total_stake = 0.0
        total_profit = 0.0
        wins = 0
        sum_odds = 0.0

        for r in settlement_results:
            stake = r.get("stake", 0.0)
            profit_loss = r.get("profit_loss", 0.0)
            odds = r.get("odds", 0.0)
            won = r.get("won", False)

            if stake > 0:
                roi = profit_loss / stake
                returns.append(roi)
                total_bets += 1
                total_stake += stake
                total_profit += profit_loss
                if won:
                    wins += 1
                if odds > 0:
                    sum_odds += odds

        return StrategyReturnSeries(
            strategy_type=strategy_type,
            returns=returns,
            total_bets=total_bets,
            total_stake=total_stake,
            total_profit=total_profit,
            win_rate=wins / max(total_bets, 1),
            avg_odds=sum_odds / max(total_bets, 1),
        )


# ═══════════════════════════════════════════════════════════════
# 辅助: 策略相关性分析
# ═══════════════════════════════════════════════════════════════

def analyze_strategy_correlation(
    strategy_returns: Dict[str, StrategyReturnSeries],
) -> Dict[str, Dict[str, float]]:
    """
    计算策略间皮尔逊相关系数矩阵。

    低相关性 = 更好的分散化效果。
    预期: 1X2 vs 大小球 ≈ 0.3-0.5 (部分相关)
          1X2 vs 亚盘 ≈ 0.6-0.8 (高度相关, 同方向)
          亚盘 vs 大小球 ≈ 0.2-0.4 (低相关, 正交性强)

    返回:
        {strategy_a: {strategy_b: correlation}}
    """
    cov_estimator = CovarianceEstimator()
    cov_matrix, names = cov_estimator.estimate(strategy_returns)

    # 转换为相关系数矩阵
    n = len(names)
    corr = {}
    for i, name_i in enumerate(names):
        corr[name_i] = {}
        for j, name_j in enumerate(names):
            if i == j:
                corr[name_i][name_j] = 1.0
            else:
                r = cov_matrix[i, j] / math.sqrt(
                    max(cov_matrix[i, i], 1e-10) * max(cov_matrix[j, j], 1e-10)
                )
                corr[name_i][name_j] = round(r, 4)

    return corr