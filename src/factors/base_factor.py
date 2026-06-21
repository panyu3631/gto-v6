"""
GTO v6.0 — 因子接口标准化

所有因子必须实现此接口，确保：
1. 统一的输入输出格式
2. 独立的计算逻辑
3. 可插拔的因子系统
4. 独立的测试能力

使用方式:
    class MyFactor(BaseFactor):
        def compute(self, inputs: FactorInputs) -> Dict[str, float]:
            return {"home": 0.1, "draw": 0.0, "away": -0.1}
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FactorInputs:
    """因子计算所需的统一输入"""
    # 基础数据
    elo_diff: float = 0.0
    home_team: str = ""
    away_team: str = ""
    league_id: str = ""
    
    # 比赛统计
    home_shots: int = 0
    away_shots: int = 0
    home_sot: int = 0
    away_sot: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_yellows: int = 0
    away_yellows: int = 0
    home_ht_goals: int = 0
    away_ht_goals: int = 0
    
    # 赔率
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    opening_odds_home: float = 0.0
    opening_odds_draw: float = 0.0
    opening_odds_away: float = 0.0
    
    # 历史数据
    recent_results: List[float] = None
    h2h_results: List[float] = None
    match_history: List[Dict] = None
    
    # 积分榜
    home_rank: int = 10
    away_rank: int = 10
    home_points: int = 0
    away_points: int = 0
    
    # 外部数据
    xg_home: float = 0.0
    xg_away: float = 0.0
    weather_impact: float = 0.0
    coach_changed_home: bool = False
    coach_changed_away: bool = False
    is_promoted_home: bool = False
    is_promoted_away: bool = False
    
    def __post_init__(self):
        if self.recent_results is None:
            self.recent_results = []
        if self.h2h_results is None:
            self.h2h_results = []
        if self.match_history is None:
            self.match_history = []


@dataclass
class FactorOutput:
    """因子计算输出"""
    factor_id: str
    deltas: Dict[str, float]  # {"home": delta, "draw": delta, "away": delta}
    confidence: float = 1.0   # 因子置信度 (0-1)
    metadata: Dict[str, Any] = None  # 附加信息
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseFactor(ABC):
    """因子基类 — 所有因子必须继承此类"""
    
    factor_id: str = ""
    factor_name: str = ""
    category: str = ""  # base, enhanced, league, stats
    
    @abstractmethod
    def compute(self, inputs: FactorInputs) -> FactorOutput:
        """
        计算因子值。
        
        参数:
            inputs: 因子输入数据
        
        返回:
            FactorOutput 包含 home/draw/away 的 delta 值
        """
        pass
    
    def is_available(self, inputs: FactorInputs) -> bool:
        """
        检查因子是否有足够数据计算。
        
        参数:
            inputs: 因子输入数据
        
        返回:
            True 如果有足够数据，False 否则
        """
        return True
    
    def get_default_output(self) -> FactorOutput:
        """获取默认输出（数据不足时使用）"""
        return FactorOutput(
            factor_id=self.factor_id,
            deltas={"home": 0.0, "draw": 0.0, "away": 0.0},
            confidence=0.0,
            metadata={"reason": "insufficient_data"}
        )


class FactorRegistry:
    """因子注册中心 — 管理所有因子"""
    
    def __init__(self):
        self._factors: Dict[str, BaseFactor] = {}
    
    def register(self, factor: BaseFactor):
        """注册因子"""
        self._factors[factor.factor_id] = factor
        logger.debug(f"因子已注册: {factor.factor_id}")
    
    def get(self, factor_id: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self._factors.get(factor_id)
    
    def compute_all(self, inputs: FactorInputs) -> Dict[str, FactorOutput]:
        """计算所有已注册因子"""
        results = {}
        for factor_id, factor in self._factors.items():
            try:
                if factor.is_available(inputs):
                    output = factor.compute(inputs)
                else:
                    output = factor.get_default_output()
                results[factor_id] = output
            except Exception as e:
                logger.warning(f"因子 {factor_id} 计算失败: {e}")
                results[factor_id] = factor.get_default_output()
        return results
    
    def compute_enabled(self, inputs: FactorInputs, enabled_factors: List[str]) -> Dict[str, FactorOutput]:
        """计算指定的因子"""
        results = {}
        for factor_id in enabled_factors:
            factor = self._factors.get(factor_id)
            if factor:
                try:
                    if factor.is_available(inputs):
                        output = factor.compute(inputs)
                    else:
                        output = factor.get_default_output()
                    results[factor_id] = output
                except Exception as e:
                    logger.warning(f"因子 {factor_id} 计算失败: {e}")
                    results[factor_id] = factor.get_default_output()
        return results
    
    def list_factors(self) -> List[str]:
        """列出所有已注册因子"""
        return list(self._factors.keys())
    
    def list_by_category(self, category: str) -> List[str]:
        """按类别列出因子"""
        return [fid for fid, f in self._factors.items() if f.category == category]


# 全局因子注册表
_registry = None


def get_factor_registry() -> FactorRegistry:
    """获取全局因子注册表"""
    global _registry
    if _registry is None:
        _registry = FactorRegistry()
    return _registry
