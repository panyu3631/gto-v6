"""
GTO-GameFlow v5.9 — 多策略编排器

统一调度四种策略 (1X2 / 亚盘 / 大小球 / 串关), 通过 MPT 优化权重分配,
将多策略结果聚合为统一的投注组合。

编排流程:
1. 接收 MatchContext + ScoreMatrix → 并行执行三种策略分析
2. 收集各策略的投注建议 (proposals)
3. 查询历史收益序列 → MPT 权重优化
4. 按 MPT 权重缩放各策略的 Kelly 投注额
5. 聚合为统一的 MultiStrategyResult
6. 批量收集后生成串关组合 (跨场次)

使用方式:
    from src.strategies import StrategyOrchestrator

    orch = StrategyOrchestrator(league_id="bundesliga")
    result = orch.run(
        match=match_context,
        score_matrix=score_matrix,
        handicap_odds={0.5: {"home": 1.92, "away": 1.98}},
        totals_odds={2.5: {"over": 1.90, "under": 2.00}},
    )
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..data.models import (
    ScoreMatrix, MatchContext, BetProposal, BetSelection,
    AsianHandicapProposal, TotalsProposal,
    StrategyAllocation, StrategyPortfolio, StrategyType,
)
from .asian_handicap import AsianHandicapStrategy
from .over_under import OverUnderStrategy
from .parlay import ParlayStrategy, ParlayProposal, ParlaySettlement
from .mpt_portfolio import MPTPortfolioOptimizer, StrategyReturnSeries, CovarianceEstimator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class MultiStrategyResult:
    """多策略编排结果"""
    match_id: str
    league_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # 各策略的独立结果
    x2_proposals: List[BetProposal] = field(default_factory=list)
    asian_proposals: List[AsianHandicapProposal] = field(default_factory=list)
    totals_proposals: List[TotalsProposal] = field(default_factory=list)

    # MPT 组合
    portfolio: Optional[StrategyPortfolio] = None

    # 聚合后的统一投注建议 (1X2 格式, 兼容现有 pipeline)
    unified_proposals: List[BetProposal] = field(default_factory=list)

    # 元数据
    active_strategies: List[str] = field(default_factory=list)
    strategy_count: int = 0
    total_value: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total_proposals(self) -> int:
        return (
            len(self.x2_proposals) +
            len(self.asian_proposals) +
            len(self.totals_proposals)
        )

    def summary(self) -> dict:
        """生成摘要信息"""
        return {
            "match_id": self.match_id,
            "league_id": self.league_id,
            "active_strategies": self.active_strategies,
            "strategy_count": self.strategy_count,
            "total_proposals": self.total_proposals,
            "x2_count": len(self.x2_proposals),
            "asian_count": len(self.asian_proposals),
            "totals_count": len(self.totals_proposals),
            "unified_count": len(self.unified_proposals),
            "total_value": round(self.total_value, 4),
            "portfolio": self.portfolio_summary() if self.portfolio else None,
            "warnings": self.warnings,
        }

    def portfolio_summary(self) -> Optional[dict]:
        """MPT 组合摘要"""
        if not self.portfolio:
            return None
        return {
            "sharpe": self.portfolio.portfolio_sharpe,
            "expected_return": self.portfolio.total_expected_return,
            "volatility": self.portfolio.total_volatility,
            "total_bankroll": self.portfolio.total_bankroll,
            "total_allocated": self.portfolio.total_allocated,
            "allocations": [
                {
                    "strategy": a.strategy_type,
                    "weight": round(a.weight * 100, 1),
                    "allocation": a.allocation,
                    "sharpe": a.sharpe,
                }
                for a in self.portfolio.allocations
            ],
        }


# ═══════════════════════════════════════════════════════════════
# 策略编排器
# ═══════════════════════════════════════════════════════════════

class StrategyOrchestrator:
    """
    多策略编排器 — 统一调度 1X2 / 亚盘 / 大小球。

    参数:
        league_id: 联赛ID
        enable_asian: 是否启用亚盘策略
        enable_over_under: 是否启用大小球策略
        mpt_optimizer: MPT 优化器实例 (None=创建默认)
        asian_config: 亚盘策略配置
        over_under_config: 大小球策略配置
        default_weights: 数据不足时的默认权重
    """

    def __init__(
        self,
        league_id: str = "",
        enable_asian: bool = True,
        enable_over_under: bool = True,
        mpt_optimizer: Optional[MPTPortfolioOptimizer] = None,
        asian_config: Optional[dict] = None,
        over_under_config: Optional[dict] = None,
        default_weights: Optional[Dict[str, float]] = None,
    ):
        self.league_id = league_id
        self.enable_asian = enable_asian
        self.enable_over_under = enable_over_under

        # 策略实例
        self.asian_strategy = AsianHandicapStrategy(
            league_id=league_id,
            **(asian_config or {}),
        )
        self.over_under_strategy = OverUnderStrategy(
            league_id=league_id,
            **(over_under_config or {}),
        )

        # MPT 优化器
        self.mpt = mpt_optimizer or MPTPortfolioOptimizer(
            default_weights=default_weights,
        )

        # 历史收益序列 (跨比赛累积)
        self._return_series: Dict[str, StrategyReturnSeries] = {
            "1x2": StrategyReturnSeries(strategy_type="1x2"),
            "asian_handicap": StrategyReturnSeries(strategy_type="asian_handicap"),
            "over_under": StrategyReturnSeries(strategy_type="over_under"),
        }

    def run(
        self,
        match: MatchContext,
        score_matrix: ScoreMatrix,
        handicap_odds: Optional[Dict[float, Dict[str, float]]] = None,
        totals_odds: Optional[Dict[float, Dict[str, float]]] = None,
        x2_proposals: Optional[List[BetProposal]] = None,
        kelly_discount: float = 0.25,
        total_bankroll: float = 100000.0,
        strip_margin_asian: bool = True,
        strip_margin_totals: bool = True,
    ) -> MultiStrategyResult:
        """
        执行多策略编排。

        参数:
            match: 比赛上下文
            score_matrix: 泊松比分概率矩阵
            handicap_odds: 亚盘赔率数据 (None=生成模拟赔率)
            totals_odds: 大小球赔率数据 (None=生成模拟赔率)
            x2_proposals: 已有的 1X2 投注建议 (None=跳过, 从外部 pipeline 传入)
            kelly_discount: Kelly 折扣系数
            total_bankroll: 总资金
            strip_margin_asian: 亚盘是否剥离庄家 margin (合成赔率应设为 False)
            strip_margin_totals: 大小球是否剥离庄家 margin (合成赔率应设为 False)

        返回:
            MultiStrategyResult
        """
        result = MultiStrategyResult(
            match_id=match.match_id,
            league_id=match.league_id,
        )

        active_strategies = []

        # ── 1X2: 从外部传入 (已有 pipeline 产出) ──
        if x2_proposals:
            result.x2_proposals = x2_proposals
            active_strategies.append("1x2")

        # ── 亚盘 ──
        if self.enable_asian:
            try:
                # v5.10: 禁用合成赔率 — 无真实赔率时亚盘策略不输出
                if handicap_odds is None or len(handicap_odds) == 0:
                    result.warnings.append("asian_handicap skipped: no real odds available")
                else:
                    asian_proposals = self.asian_strategy.analyze(
                        score_matrix=score_matrix,
                        handicap_odds=handicap_odds,
                        match_id=match.match_id,
                        league_id=match.league_id,
                        kelly_discount=kelly_discount,
                        strip_margin=strip_margin_asian,
                    )
                    result.asian_proposals = asian_proposals
                    if asian_proposals:
                        active_strategies.append("asian_handicap")
            except Exception as e:
                logger.warning(f"亚盘策略执行失败: {e}")
                result.warnings.append(f"asian_handicap error: {e}")

        # ── 大小球 ──
        if self.enable_over_under:
            try:
                # v5.10: 禁用合成赔率 — 无真实赔率时大小球策略不输出
                if totals_odds is None or len(totals_odds) == 0:
                    result.warnings.append("over_under skipped: no real odds available")
                else:
                    totals_proposals = self.over_under_strategy.analyze(
                        score_matrix=score_matrix,
                        totals_odds=totals_odds,
                        match_id=match.match_id,
                        league_id=match.league_id,
                        kelly_discount=kelly_discount,
                        strip_margin=strip_margin_totals,
                    )
                    result.totals_proposals = totals_proposals
                    if totals_proposals:
                        active_strategies.append("over_under")
            except Exception as e:
                logger.warning(f"大小球策略执行失败: {e}")
                result.warnings.append(f"over_under error: {e}")

        result.active_strategies = active_strategies
        result.strategy_count = len(active_strategies)

        # ── MPT 权重优化 ──
        if len(active_strategies) >= 2:
            result.portfolio = self.mpt.optimize(
                strategy_returns={
                    s: self._return_series[s]
                    for s in active_strategies
                },
                total_bankroll=total_bankroll,
                active_strategies=active_strategies,
            )
        elif len(active_strategies) == 1:
            # 单一策略: 100% 分配
            s = active_strategies[0]
            series = self._return_series[s]
            result.portfolio = StrategyPortfolio(
                allocations=[
                    StrategyAllocation(
                        strategy_type=s,
                        weight=1.0,
                        expected_return=series.mean_return,
                        volatility=series.volatility,
                        sharpe=series.sharpe,
                        allocation=total_bankroll,
                    )
                ],
                total_expected_return=series.mean_return,
                total_volatility=series.volatility,
                portfolio_sharpe=series.sharpe,
                total_bankroll=total_bankroll,
                total_allocated=total_bankroll,
            )

        # ── 聚合统一投注建议 ──
        result.unified_proposals = self._aggregate_proposals(result, total_bankroll)

        # 计算总价值
        result.total_value = sum(
            max(0, p.value) for p in result.unified_proposals
        )

        return result

    def _aggregate_proposals(
        self,
        result: MultiStrategyResult,
        total_bankroll: float,
    ) -> List[BetProposal]:
        """
        将多策略投注建议聚合为统一的 BetProposal 列表。

        聚合规则:
        1. 1X2 proposals: 直接保留 (已有 BetProposal 格式)
        2. 亚盘 proposals: 转换为 BetProposal (映射到最接近的 1X2 方向)
        3. 大小球 proposals: 转换为 BetProposal (新类型标记)
        4. 按 MPT 权重缩放 Kelly stake
        """
        unified: List[BetProposal] = []

        # 获取各策略的 MPT 权重
        weights = self._get_strategy_weights(result)

        # ── 1X2: 直接保留 ──
        x2_weight = weights.get("1x2", 1.0)
        for p in result.x2_proposals:
            p.strategy_weight = x2_weight
            p.adjusted_stake = p.kelly_stake * x2_weight
            p.strategy_type = "1x2"
            unified.append(p)

        # ── 亚盘: 转换为 BetProposal ──
        asian_weight = weights.get("asian_handicap", 0.0)
        for p in result.asian_proposals:
            bp = self._asian_to_bet_proposal(p, asian_weight)
            unified.append(bp)

        # ── 大小球: 转换为 BetProposal ──
        totals_weight = weights.get("over_under", 0.0)
        for p in result.totals_proposals:
            bp = self._totals_to_bet_proposal(p, totals_weight)
            unified.append(bp)

        # 按优先级排序
        unified.sort(key=lambda x: x.priority_score, reverse=True)

        return unified

    def _get_strategy_weights(
        self,
        result: MultiStrategyResult,
    ) -> Dict[str, float]:
        """从 MPT portfolio 提取各策略权重"""
        if not result.portfolio or not result.portfolio.allocations:
            return self.mpt.default_weights

        weights = {}
        for alloc in result.portfolio.allocations:
            weights[alloc.strategy_type] = alloc.weight

        return weights

    def _asian_to_bet_proposal(
        self,
        p: AsianHandicapProposal,
        weight: float,
    ) -> BetProposal:
        """将亚盘投注建议转换为 BetProposal (兼容现有 pipeline)"""
        # 亚盘方向映射到最接近的 1X2 方向
        if p.side == "home":
            selection = BetSelection.HOME_WIN
        else:
            selection = BetSelection.AWAY_WIN

        return BetProposal(
            match_id=p.match_id,
            selection=selection,
            odds=p.odds,
            model_prob=p.cover_prob,
            implied_prob=p.implied_prob,
            value=p.value,
            kelly_stake=p.kelly_stake,
            adjusted_stake=p.kelly_stake * weight,
            priority_score=p.priority_score * weight,
            league_id=p.league_id,
            strategy_type="asian_handicap",
            handicap_line=p.handicap_line,
            strategy_weight=weight,
        )

    def _totals_to_bet_proposal(
        self,
        p: TotalsProposal,
        weight: float,
    ) -> BetProposal:
        """将大小球投注建议转换为 BetProposal (兼容现有 pipeline)"""
        if p.side == "over":
            selection = BetSelection.OVER
        else:
            selection = BetSelection.UNDER

        return BetProposal(
            match_id=p.match_id,
            selection=selection,
            odds=p.odds,
            model_prob=p.over_prob,
            implied_prob=p.implied_prob,
            value=p.value,
            kelly_stake=p.kelly_stake,
            adjusted_stake=p.kelly_stake * weight,
            priority_score=p.priority_score * weight,
            league_id=p.league_id,
            strategy_type="over_under",
            totals_line=p.totals_line,
            strategy_weight=weight,
        )

    # ═══════════════════════════════════════════════════════════
    # 收益序列管理 (跨比赛累积)
    # ═══════════════════════════════════════════════════════════

    def record_settlement(
        self,
        strategy_type: str,
        roi: float,
        stake: float = 0.0,
        profit: float = 0.0,
        odds: float = 0.0,
        won: bool = False,
    ):
        """
        记录一笔结算结果, 更新对应策略的收益序列。

        应在每场比赛结算后调用, 用于 MPT 的增量更新。
        """
        if strategy_type not in self._return_series:
            logger.warning(f"未知策略类型: {strategy_type}")
            return

        self._return_series[strategy_type] = self.mpt.update_return_series(
            self._return_series[strategy_type],
            new_return=roi,
            stake=stake,
            profit=profit,
            odds=odds,
            won=won,
        )

    def get_return_series(
        self,
        strategy_type: str,
    ) -> Optional[StrategyReturnSeries]:
        """获取策略的当前收益序列"""
        return self._return_series.get(strategy_type)

    def get_all_series(self) -> Dict[str, StrategyReturnSeries]:
        """获取所有策略的收益序列"""
        return dict(self._return_series)

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """获取策略间相关系数矩阵"""
        from .mpt_portfolio import analyze_strategy_correlation
        return analyze_strategy_correlation(self._return_series)


# ═══════════════════════════════════════════════════════════════
# v5.9: 串关批量管理器
# ═══════════════════════════════════════════════════════════════

class ParlayBatchManager:
    """
    串关批量管理器 — 跨场次收集单场投注，批量生成串关组合。

    与单场策略不同，串关需要在累积一定场次后才能生成组合。
    此管理器维护一个待处理池，每轮比赛后增量生成串关。

    使用方式:
        mgr = ParlayBatchManager()
        # 每轮比赛后调用
        mgr.add_match_bets(match_id, proposals)
        parlays = mgr.generate_batch(bankroll)
        # 结算时
        settlement = mgr.settle_parlay(parlay_id, match_results)
    """

    def __init__(
        self,
        max_legs: int = 2,
        min_single_value: float = 0.03,
        min_combined_value: float = 0.05,
        kelly_discount: float = 0.25,
        max_batch_size: int = 20,
    ):
        self.parlay_strategy = ParlayStrategy(
            max_legs=max_legs,
            min_single_value=min_single_value,
            min_combined_value=min_combined_value,
            kelly_discount=kelly_discount,
        )
        self.max_batch_size = max_batch_size

        # 待处理池: {match_id: [BetProposal, ...]}
        self._pending_pool: Dict[str, List[BetProposal]] = {}
        # 活跃串关: {parlay_id: ParlayProposal}
        self._active_parlays: Dict[str, ParlayProposal] = {}
        # 已结算串关: {parlay_id: ParlaySettlement}
        self._settled_parlays: Dict[str, ParlaySettlement] = {}
        # 串关收益序列
        self._parlay_returns: List[float] = []

    def add_match_bets(
        self,
        match_id: str,
        proposals: List[BetProposal],
    ):
        """添加一场比赛的单场投注到待处理池"""
        if proposals:
            self._pending_pool[match_id] = proposals

    def generate_batch(
        self,
        bankroll: float,
    ) -> List[ParlayProposal]:
        """
        从待处理池生成串关组合。

        池中积累超过 max_batch_size 场后，取最近的一批生成串关。
        生成后清空池，避免重复组合。
        """
        if len(self._pending_pool) < 2:
            return []

        # 收集所有单场投注
        all_bets = []
        for bets in self._pending_pool.values():
            all_bets.extend(bets)

        # 生成串关
        parlays = self.parlay_strategy.generate_parlays(
            single_bets=all_bets,
            bankroll=bankroll,
        )

        # 应用总曝光上限
        parlays = self.parlay_strategy.apply_stake_cap(parlays, bankroll)

        # 限制数量: 取优先级最高的前 N 个
        parlays = parlays[:self.max_batch_size]

        # 注册到活跃列表
        for p in parlays:
            self._active_parlays[p.parlay_id] = p

        # 清空已使用的池
        if len(self._pending_pool) >= self.max_batch_size:
            self._pending_pool.clear()

        return parlays

    def settle_parlay(
        self,
        parlay_id: str,
        match_results: Dict[str, Tuple[str, str]],
    ) -> Optional[ParlaySettlement]:
        """结算单个串关"""
        if parlay_id not in self._active_parlays:
            return None

        proposal = self._active_parlays[parlay_id]
        settlement = self.parlay_strategy.settle(proposal, match_results)

        self._settled_parlays[parlay_id] = settlement
        del self._active_parlays[parlay_id]

        # 记录收益
        roi = settlement.profit / settlement.stake if settlement.stake > 0 else 0.0
        self._parlay_returns.append(roi)

        return settlement

    def settle_all_ready(
        self,
        match_results: Dict[str, Tuple[str, str]],
    ) -> List[ParlaySettlement]:
        """
        结算所有已完成的串关。

        返回已结算的串关列表。
        """
        settled = []
        to_remove = []
        for pid, proposal in self._active_parlays.items():
            ready = all(leg.match_id in match_results for leg in proposal.legs)
            if ready:
                settlement = self.parlay_strategy.settle(proposal, match_results)
                self._settled_parlays[pid] = settlement
                to_remove.append(pid)
                settled.append(settlement)
                roi = settlement.profit / settlement.stake if settlement.stake > 0 else 0.0
                self._parlay_returns.append(roi)

        for pid in to_remove:
            del self._active_parlays[pid]

        return settled

    @property
    def active_count(self) -> int:
        return len(self._active_parlays)

    @property
    def settled_count(self) -> int:
        return len(self._settled_parlays)

    @property
    def parlay_roi(self) -> float:
        if not self._parlay_returns:
            return 0.0
        return sum(self._parlay_returns) / len(self._parlay_returns)

    @property
    def parlay_win_rate(self) -> float:
        if not self._settled_parlays:
            return 0.0
        wins = sum(1 for s in self._settled_parlays.values() if s.won)
        return wins / len(self._settled_parlays)

    def get_stats(self) -> dict:
        return {
            "active": self.active_count,
            "settled": self.settled_count,
            "roi": round(self.parlay_roi, 4),
            "win_rate": round(self.parlay_win_rate, 4),
            "total_profit": round(sum(self._parlay_returns), 2),
        }


# ═══════════════════════════════════════════════════════════════
# 便捷工厂函数
# ═══════════════════════════════════════════════════════════════

def create_orchestrator(
    league_id: str,
    enable_asian: bool = True,
    enable_over_under: bool = True,
    total_bankroll: float = 100000.0,
) -> StrategyOrchestrator:
    """
    创建策略编排器。

    按联赛特征调整默认参数:
    - 德甲/荷甲: 高进球联赛, 大小球策略权重更高
    - 意甲: 低进球联赛, 亚盘策略更适用
    - 英超: 均衡, 三种策略均等
    """
    league_lower = league_id.lower()

    # 默认权重按联赛特征调整
    if league_lower in ("bundesliga", "eredivisie"):
        # 高进球: 大小球策略更有效
        default_weights = {"1x2": 0.40, "asian_handicap": 0.25, "over_under": 0.35}
    elif league_lower in ("serie_a", "ligue_1"):
        # 低进球: 亚盘精细分析更有效
        default_weights = {"1x2": 0.45, "asian_handicap": 0.35, "over_under": 0.20}
    else:
        # 均衡 (英超/西甲)
        default_weights = {"1x2": 0.50, "asian_handicap": 0.25, "over_under": 0.25}

    return StrategyOrchestrator(
        league_id=league_id,
        enable_asian=enable_asian,
        enable_over_under=enable_over_under,
        default_weights=default_weights,
    )