"""
GTO v6.0 — 策略插件接口

所有策略必须实现此接口，确保：
1. 统一的输入输出格式
2. 独立的投注逻辑
3. 可插拔的策略系统
4. 独立的回测能力

使用方式:
    class MyStrategy(BaseStrategy):
        def evaluate(self, context: StrategyContext) -> List[BetProposal]:
            return [BetProposal(...)]
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """策略上下文 — 策略计算所需的全部数据"""
    match_id: str
    league_id: str
    home_team: str
    away_team: str
    
    # 模型概率
    home_prob: float = 0.0
    draw_prob: float = 0.0
    away_prob: float = 0.0
    
    # 比分矩阵
    score_matrix: Dict[tuple, float] = field(default_factory=dict)
    
    # 预期进球
    home_lambda: float = 0.0
    away_lambda: float = 0.0
    
    # 市场赔率
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    
    # 大小球赔率
    odds_over: Dict[float, float] = field(default_factory=dict)
    odds_under: Dict[float, float] = field(default_factory=dict)
    
    # 亚盘赔率
    handicap_line: float = 0.0
    odds_home_ah: float = 0.0
    odds_away_ah: float = 0.0
    
    # 资金
    bankroll: float = 10000.0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BetProposal:
    """投注提案"""
    match_id: str
    strategy: str          # "1x2" / "over_under" / "asian_handicap"
    direction: str         # "home" / "draw" / "away" / "over" / "under"
    odds: float
    model_prob: float
    market_prob: float
    value: float
    kelly_stake: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """策略基类 — 所有策略必须继承此类"""
    
    strategy_id: str = ""
    strategy_name: str = ""
    
    @abstractmethod
    def evaluate(self, context: StrategyContext) -> List[BetProposal]:
        """
        评估策略，生成投注提案。
        
        参数:
            context: 策略上下文
        
        返回:
            投注提案列表
        """
        pass
    
    def is_available(self, context: StrategyContext) -> bool:
        """检查策略是否有足够数据运行"""
        return True


class Strategy1X2(BaseStrategy):
    """1X2策略"""
    
    strategy_id = "1x2"
    strategy_name = "胜平负"
    
    def __init__(self, threshold: float = 0.05, min_odds: float = 1.20, max_odds: float = 10.0):
        self.threshold = threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
    
    def evaluate(self, context: StrategyContext) -> List[BetProposal]:
        proposals = []
        
        probs = {
            "home": context.home_prob,
            "draw": context.draw_prob,
            "away": context.away_prob,
        }
        odds_map = {
            "home": context.odds_home,
            "draw": context.odds_draw,
            "away": context.odds_away,
        }
        
        for direction in ["home", "draw", "away"]:
            prob = probs[direction]
            odds = odds_map[direction]
            
            if odds < self.min_odds or odds > self.max_odds:
                continue
            
            market_prob = 1.0 / odds
            value = prob - market_prob
            
            if value >= self.threshold:
                b = odds - 1
                kelly = max(0, (b * prob - (1 - prob)) / b) if b > 0 else 0
                
                proposals.append(BetProposal(
                    match_id=context.match_id,
                    strategy="1x2",
                    direction=direction,
                    odds=odds,
                    model_prob=prob,
                    market_prob=market_prob,
                    value=value,
                    kelly_stake=kelly,
                    confidence=min(1.0, value / self.threshold),
                ))
        
        return proposals


class StrategyOverUnder(BaseStrategy):
    """大小球策略"""
    
    strategy_id = "over_under"
    strategy_name = "大小球"
    
    def __init__(self, threshold: float = 0.08, lines: List[float] = None):
        self.threshold = threshold
        self.lines = lines or [1.5, 2.0, 2.5, 3.0, 3.5]
    
    def evaluate(self, context: StrategyContext) -> List[BetProposal]:
        proposals = []
        
        if not context.score_matrix:
            return proposals
        
        for line in self.lines:
            over_prob = sum(p for (h, a), p in context.score_matrix.items() if h + a > line)
            under_prob = 1.0 - over_prob
            
            over_odds = context.odds_over.get(line, 0)
            under_odds = context.odds_under.get(line, 0)
            
            for direction, prob, odds in [("over", over_prob, over_odds), ("under", under_prob, under_odds)]:
                if odds <= 1:
                    continue
                
                market_prob = 1.0 / odds
                value = prob - market_prob
                
                if value >= self.threshold:
                    b = odds - 1
                    kelly = max(0, (b * prob - (1 - prob)) / b) if b > 0 else 0
                    
                    proposals.append(BetProposal(
                        match_id=context.match_id,
                        strategy="over_under",
                        direction=direction,
                        odds=odds,
                        model_prob=prob,
                        market_prob=market_prob,
                        value=value,
                        kelly_stake=kelly,
                        confidence=min(1.0, value / self.threshold),
                        metadata={"line": line},
                    ))
        
        return proposals


class StrategyAsianHandicap(BaseStrategy):
    """亚盘策略"""
    
    strategy_id = "asian_handicap"
    strategy_name = "亚盘"
    
    def __init__(self, threshold: float = 0.08):
        self.threshold = threshold
    
    def evaluate(self, context: StrategyContext) -> List[BetProposal]:
        proposals = []
        
        if not context.score_matrix or context.handicap_line == 0:
            return proposals
        
        # 计算覆盖概率
        home_cover = 0.0
        away_cover = 0.0
        
        for (h, a), p in context.score_matrix.items():
            diff = (h - a) + context.handicap_line
            if diff > 0:
                home_cover += p
            elif diff < 0:
                away_cover += p
        
        directions = {
            "home": (home_cover, context.odds_home_ah),
            "away": (away_cover, context.odds_away_ah),
        }
        
        for direction, (prob, odds) in directions.items():
            if odds <= 1:
                continue
            
            market_prob = 1.0 / odds
            value = prob - market_prob
            
            if value >= self.threshold:
                b = odds - 1
                kelly = max(0, (b * prob - (1 - prob)) / b) if b > 0 else 0
                
                proposals.append(BetProposal(
                    match_id=context.match_id,
                    strategy="asian_handicap",
                    direction=direction,
                    odds=odds,
                    model_prob=prob,
                    market_prob=market_prob,
                    value=value,
                    kelly_stake=kelly,
                    confidence=min(1.0, value / self.threshold),
                    metadata={"handicap_line": context.handicap_line},
                ))
        
        return proposals


class StrategyManager:
    """策略管理器 — 管理所有策略"""
    
    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
    
    def register(self, strategy: BaseStrategy):
        """注册策略"""
        self._strategies[strategy.strategy_id] = strategy
        logger.debug(f"策略已注册: {strategy.strategy_id}")
    
    def get(self, strategy_id: str) -> Optional[BaseStrategy]:
        """获取策略"""
        return self._strategies.get(strategy_id)
    
    def evaluate_all(self, context: StrategyContext) -> List[BetProposal]:
        """评估所有策略"""
        proposals = []
        for strategy_id, strategy in self._strategies.items():
            try:
                if strategy.is_available(context):
                    proposals.extend(strategy.evaluate(context))
            except Exception as e:
                logger.warning(f"策略 {strategy_id} 评估失败: {e}")
        return proposals
    
    def evaluate_strategy(self, strategy_id: str, context: StrategyContext) -> List[BetProposal]:
        """评估指定策略"""
        strategy = self._strategies.get(strategy_id)
        if strategy:
            try:
                if strategy.is_available(context):
                    return strategy.evaluate(context)
            except Exception as e:
                logger.warning(f"策略 {strategy_id} 评估失败: {e}")
        return []
    
    def list_strategies(self) -> List[str]:
        """列出所有策略"""
        return list(self._strategies.keys())


# 全局策略管理器
_manager = None


def get_strategy_manager() -> StrategyManager:
    """获取全局策略管理器"""
    global _manager
    if _manager is None:
        _manager = StrategyManager()
        # 注册默认策略
        _manager.register(Strategy1X2())
        _manager.register(StrategyOverUnder())
        _manager.register(StrategyAsianHandicap())
    return _manager
