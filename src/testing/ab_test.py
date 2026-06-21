"""
GTO v6.0 — A/B测试框架

支持多配置并行测试，找出最优参数组合。

使用方式:
    ab = ABTest()
    ab.add_variant("baseline", baseline_config)
    ab.add_variant("optimized", optimized_config)
    results = ab.run(matches)
    ab.compare()
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class VariantResult:
    """变体结果"""
    variant_id: str
    total_bets: int = 0
    wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    profit: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    bets: List[Dict] = field(default_factory=list)


@dataclass
class ABTestResult:
    """A/B测试结果"""
    variants: Dict[str, VariantResult]
    winner: str = ""
    confidence: float = 0.0
    summary: str = ""


class ABTest:
    """A/B测试框架"""
    
    def __init__(self):
        self._variants: Dict[str, Dict] = {}
        self._results: Dict[str, VariantResult] = {}
    
    def add_variant(self, variant_id: str, config: Dict):
        """添加测试变体"""
        self._variants[variant_id] = config
        logger.info(f"变体已添加: {variant_id}")
    
    def run_variant(self, variant_id: str, matches: List, predict_fn, **kwargs) -> VariantResult:
        """运行单个变体"""
        config = self._variants.get(variant_id, {})
        result = VariantResult(variant_id=variant_id)
        
        bankroll = config.get("bankroll", 10000)
        peak = bankroll
        
        for match in matches:
            # 使用变体配置预测
            prediction = predict_fn(match, config)
            
            if not prediction or prediction.get("value", 0) < config.get("threshold", 0.05):
                continue
            
            # 下注
            odds = prediction.get("odds", 1.0)
            stake = min(
                bankroll * prediction.get("kelly", 0) * config.get("kelly_fraction", 0.25),
                bankroll * config.get("max_stake_pct", 0.03)
            )
            
            if stake < 10:
                continue
            
            # 结算
            won = prediction.get("won", False)
            profit = stake * (odds - 1) if won else -stake
            bankroll += profit
            
            result.total_bets += 1
            result.total_staked += stake
            if won:
                result.wins += 1
                result.total_returned += stake + profit
            result.profit += profit
            
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak if peak > 0 else 0
            if drawdown > result.max_drawdown:
                result.max_drawdown = drawdown
            
            result.bets.append({
                "match": match.get("id", ""),
                "direction": prediction.get("direction", ""),
                "odds": odds,
                "stake": stake,
                "won": won,
                "profit": profit,
            })
        
        # 计算统计
        if result.total_bets > 0:
            result.win_rate = result.wins / result.total_bets
            result.roi = result.profit / result.total_staked if result.total_staked > 0 else 0
        
        self._results[variant_id] = result
        return result
    
    def run_all(self, matches: List, predict_fn, **kwargs) -> Dict[str, VariantResult]:
        """运行所有变体"""
        results = {}
        for variant_id in self._variants:
            results[variant_id] = self.run_variant(variant_id, matches, predict_fn, **kwargs)
        return results
    
    def compare(self) -> ABTestResult:
        """比较变体结果"""
        if not self._results:
            return ABTestResult(variants={})
        
        # 按ROI排序
        sorted_variants = sorted(
            self._results.items(),
            key=lambda x: x[1].roi,
            reverse=True
        )
        
        winner_id = sorted_variants[0][0]
        winner_result = sorted_variants[0][1]
        
        # 计算置信度（简化版）
        if len(sorted_variants) >= 2:
            roi_diff = sorted_variants[0][1].roi - sorted_variants[1][1].roi
            # 简化的置信度计算
            confidence = min(0.99, roi_diff * 10 + 0.5)
        else:
            confidence = 0.5
        
        # 生成摘要
        summary_lines = []
        for vid, result in sorted_variants:
            summary_lines.append(
                f"  {vid}: ROI={result.roi:+.1%} 胜率={result.win_rate:.1%} "
                f"投注={result.total_bets} 回撤={result.max_drawdown:.1%}"
            )
        
        summary = f"最优变体: {winner_id} (ROI={winner_result.roi:+.1%})\n" + "\n".join(summary_lines)
        
        return ABTestResult(
            variants=dict(self._results),
            winner=winner_id,
            confidence=confidence,
            summary=summary,
        )
    
    def get_result(self, variant_id: str) -> Optional[VariantResult]:
        """获取变体结果"""
        return self._results.get(variant_id)
    
    def export_results(self, path: str):
        """导出结果到文件"""
        export = {}
        for vid, result in self._results.items():
            export[vid] = {
                "total_bets": result.total_bets,
                "wins": result.wins,
                "win_rate": result.win_rate,
                "total_staked": result.total_staked,
                "total_returned": result.total_returned,
                "profit": result.profit,
                "roi": result.roi,
                "max_drawdown": result.max_drawdown,
            }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2)
        
        logger.info(f"结果已导出: {path}")


# 预定义的变体配置
VARIANT_CONFIGS = {
    "baseline": {
        "bankroll": 10000,
        "threshold": 0.05,
        "kelly_fraction": 0.25,
        "max_stake_pct": 0.03,
        "consensus": 3,
    },
    "conservative": {
        "bankroll": 10000,
        "threshold": 0.08,
        "kelly_fraction": 0.20,
        "max_stake_pct": 0.02,
        "consensus": 4,
    },
    "aggressive": {
        "bankroll": 10000,
        "threshold": 0.03,
        "kelly_fraction": 0.30,
        "max_stake_pct": 0.05,
        "consensus": 3,
    },
    "high_conviction": {
        "bankroll": 10000,
        "threshold": 0.10,
        "kelly_fraction": 0.15,
        "max_stake_pct": 0.02,
        "consensus": 4,
    },
}


def create_ab_test(preset: str = "all") -> ABTest:
    """创建A/B测试"""
    ab = ABTest()
    
    if preset == "all":
        for vid, config in VARIANT_CONFIGS.items():
            ab.add_variant(vid, config)
    elif preset in VARIANT_CONFIGS:
        ab.add_variant(preset, VARIANT_CONFIGS[preset])
    
    return ab
