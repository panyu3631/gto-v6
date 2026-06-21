"""
GTO-GameFlow v5.10 — 统一贝叶斯后验收缩 (UnifiedBayesianShrinkage)

合并 SignalDecomposer (v5.6) + PriorShrinkage (v5.3a) 为单一模块，
消除 "双重 Elo 抑制" 问题。

v5.9 问题:
- SignalDecomposer 将 Elo 相关因子 delta 抑制 (阶段2)
- PriorShrinkage 再将模型概率向市场回缩 (阶段7)
- 双重抑制导致模型过度依赖市场，失去独立预测能力

v5.10 统一方案:
- 单次分解: 将原始因子 delta 分解为 Elo分量 + 非Elo分量
- 单次收缩: 对非Elo分量计算的概率应用贝叶斯收缩
- 联赛校准: 每个联赛独立学习最优 alpha 参数
- 两阶段收缩: 仅对非Elo概率的极端部分进行收缩

数据流:
    raw_factor_deltas → decompose → orthogonal_deltas → compute_probs → shrink → final_probs

使用方式:
    from src.engine.unified_bayesian_shrinkage import UnifiedBayesianShrinkage

    ubs = UnifiedBayesianShrinkage(league_id="premier_league")
    result = ubs.process(
        factor_deltas=raw_deltas,
        market_probs=(0.40, 0.28, 0.32),
        elo_diff=120,
        home_advantage=0.35,
    )
    # result.final_probs: (0.43, 0.26, 0.31) — 单次收缩后的最终概率
    # result.orthogonal_deltas: 可用于后续 logit 累加
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 因子分类 (与 SignalDecomposer 一致)
# ═══════════════════════════════════════════════════════════════

ELO_DIRECT_FACTORS = {"F1"}

ELO_DERIVED_FACTORS = {
    "F7", "F8", "F19", "F27", "F29", "F33", "F37", "F39", "F40",
}

ELO_CORRELATED_FACTORS = {
    "F3", "F5", "F20", "F38",
}

INDEPENDENT_FACTORS = {
    "F2", "F4", "F6", "F9", "F10", "F11", "F12", "F13",
    "F15", "F16", "F17", "F18",
    "F21", "F22", "F23", "F24", "F25", "F26",
    "F28", "F30", "F31", "F32", "F34", "F35", "F36",
    "F41",
}

# 默认 Elo 解释比率
DEFAULT_ELO_RATIO = {
    "elo_direct": 1.00,
    "elo_derived": 0.85,
    "elo_correlated": 0.60,
}

# ═══════════════════════════════════════════════════════════════
# 联赛校准的收缩参数
# ═══════════════════════════════════════════════════════════════

# 各联赛的 alpha 参数 (势均力敌时的模型权重)
# 来源: 训练窗口 Walk-Forward 网格搜索最优值
# v5.10.1: 提高 alpha_low 从 0.10-0.12 到 0.25-0.30
# 避免过度收缩导致模型与市场无差异
LEAGUE_ALPHA_PARAMS = {
    "premier_league": {"alpha_high": 0.60, "alpha_low": 0.30},
    "la_liga":        {"alpha_high": 0.55, "alpha_low": 0.28},
    "bundesliga":     {"alpha_high": 0.55, "alpha_low": 0.28},
    "serie_a":        {"alpha_high": 0.52, "alpha_low": 0.25},
    "ligue_1":        {"alpha_high": 0.52, "alpha_low": 0.25},
}

# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class ShrinkageResult:
    """统一贝叶斯收缩结果"""
    # 最终概率 (已收缩)
    final_probs: Tuple[float, float, float] = (0.33, 0.34, 0.33)

    # 正交化后的因子 delta (用于 logit 累加和 ScoreMatrix 构建)
    orthogonal_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 分解统计
    total_raw_magnitude: float = 0.0
    total_orthogonal_magnitude: float = 0.0
    signal_retention_ratio: float = 0.0

    # 收缩参数
    alpha_used: float = 0.5
    model_probs_before_shrink: Tuple[float, float, float] = (0.33, 0.34, 0.33)
    elo_explained_ratios: Dict[str, float] = field(default_factory=dict)

    # Elo 分量
    elo_driven_probs: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    non_elo_probs: Tuple[float, float, float] = (0.33, 0.34, 0.33)


# ═══════════════════════════════════════════════════════════════
# 统一贝叶斯收缩引擎
# ═══════════════════════════════════════════════════════════════

class UnifiedBayesianShrinkage:
    """
    统一贝叶斯后验收缩 (v5.10)。

    一次完成信号分解 + 贝叶斯收缩，消除双重 Elo 抑制。

    核心逻辑:
    1. 因子分解: 将 raw_deltas 分为 Elo分量 和 非Elo分量
    2. 非Elo概率: 仅从非Elo分量计算模型概率
    3. 贝叶斯收缩: 对非Elo概率应用联赛校准的 alpha 收缩
    4. 输出: 最终概率 + 正交化 delta

    使用方式:
        ubs = UnifiedBayesianShrinkage(league_id="premier_league")
        result = ubs.process(factor_deltas, market_probs, elo_diff)
        # result.orthogonal_deltas → 用于 logit_accumulation
        # result.final_probs → 最终投注概率
    """

    def __init__(
        self,
        league_id: str = "",
        alpha_high: Optional[float] = None,
        alpha_low: Optional[float] = None,
        elo_suppression: float = 1.0,
    ):
        """
        参数:
            league_id: 联赛ID (用于查找联赛校准参数)
            alpha_high: 势均力敌时的模型权重 (None=使用联赛默认值)
            alpha_low: 实力悬殊时的模型权重 (None=使用联赛默认值)
            elo_suppression: 全局 Elo 抑制强度 (1.0=完全按比率抑制)
        """
        self.league_id = league_id
        self.elo_suppression = elo_suppression

        # 联赛参数
        league_params = LEAGUE_ALPHA_PARAMS.get(league_id, {})
        self.alpha_high = alpha_high if alpha_high is not None else league_params.get("alpha_high", 0.50)
        self.alpha_low = alpha_low if alpha_low is not None else league_params.get("alpha_low", 0.10)

        # 因子分类 Elo 比率
        self.elo_ratios = dict(DEFAULT_ELO_RATIO)

    def process(
        self,
        factor_deltas: Dict[str, Dict[str, float]],
        market_probs: Tuple[float, float, float],
        elo_diff: float,
        home_advantage: float = 0.35,
    ) -> ShrinkageResult:
        """
        一次处理: 分解 + 收缩。

        参数:
            factor_deltas: 原始因子 delta {fid: {"home": d, "draw": d, "away": d}}
            market_probs: 市场隐含概率 (home, draw, away) — 已归一化
            elo_diff: Elo 差值 (home_elo - away_elo)
            home_advantage: 主场优势参数

        返回:
            ShrinkageResult
        """
        result = ShrinkageResult()

        # ── 阶段 1: 分解原始 delta 为 Elo 分量 + 非Elo 分量 ──
        result.orthogonal_deltas = self._decompose(factor_deltas, result)

        # ── 阶段 2: 计算动态 alpha ──
        alpha = self._compute_alpha(elo_diff)
        result.alpha_used = alpha

        # ── 阶段 3: 从非Elo分量推导模型概率 ──
        # 使用正交化 delta 的 magnitude 估算非Elo概率
        non_elo_home, non_elo_draw, non_elo_away = self._compute_non_elo_probs(
            result.orthogonal_deltas, market_probs, elo_diff, home_advantage
        )
        result.non_elo_probs = (non_elo_home, non_elo_draw, non_elo_away)
        result.model_probs_before_shrink = (non_elo_home, non_elo_draw, non_elo_away)

        # ── 阶段 4: 贝叶斯收缩 (单次) ──
        final_home = (1.0 - alpha) * market_probs[0] + alpha * non_elo_home
        final_draw = (1.0 - alpha) * market_probs[1] + alpha * non_elo_draw
        final_away = (1.0 - alpha) * market_probs[2] + alpha * non_elo_away

        # 归一化
        total = final_home + final_draw + final_away
        if total > 0:
            final_home /= total
            final_draw /= total
            final_away /= total

        result.final_probs = (final_home, final_draw, final_away)

        return result

    def _decompose(
        self,
        factor_deltas: Dict[str, Dict[str, float]],
        result: ShrinkageResult,
    ) -> Dict[str, Dict[str, float]]:
        """
        将原始因子 delta 分解，保留非Elo分量的正交 delta。

        与 SignalDecomposer 逻辑一致，但去除了独立的 PriorShrinkage 步骤。
        """
        orthogonal: Dict[str, Dict[str, float]] = {}

        for fid, deltas in factor_deltas.items():
            if fid in ELO_DIRECT_FACTORS:
                ratio = self.elo_ratios["elo_direct"] * self.elo_suppression
                result.elo_explained_ratios[fid] = ratio
                retention = 1.0 - ratio
                if retention > 0.001:
                    orthogonal[fid] = {k: v * retention for k, v in deltas.items()}

            elif fid in ELO_DERIVED_FACTORS:
                ratio = self.elo_ratios["elo_derived"] * self.elo_suppression
                result.elo_explained_ratios[fid] = ratio
                retention = 1.0 - ratio
                if retention > 0.001:
                    orthogonal[fid] = {k: v * retention for k, v in deltas.items()}

            elif fid in ELO_CORRELATED_FACTORS:
                ratio = self.elo_ratios["elo_correlated"] * self.elo_suppression
                result.elo_explained_ratios[fid] = ratio
                retention = 1.0 - ratio
                if retention > 0.001:
                    orthogonal[fid] = {k: v * retention for k, v in deltas.items()}

            else:
                # 独立因子: 完全保留
                orthogonal[fid] = dict(deltas)
                result.elo_explained_ratios[fid] = 0.0

        # 统计
        result.total_raw_magnitude = self._magnitude(factor_deltas)
        result.total_orthogonal_magnitude = self._magnitude(orthogonal)
        if result.total_raw_magnitude > 0:
            result.signal_retention_ratio = (
                result.total_orthogonal_magnitude / result.total_raw_magnitude
            )

        return orthogonal

    def _compute_alpha(self, elo_diff: float) -> float:
        """
        根据 Elo 差距计算动态收缩参数 α。

        α 是模型概率在最终融合中的权重。
        |elo_diff| 越大 → 市场定价越充分 → α 越小 → 模型权重越低。

        v5.10.1: 提高最低 alpha 确保模型保有一定独立性
        """
        abs_diff = abs(elo_diff)

        if abs_diff < 50:
            return self.alpha_high
        elif abs_diff < 150:
            t = (abs_diff - 50) / 100.0
            return self.alpha_high + t * (0.45 - self.alpha_high)
        elif abs_diff < 300:
            t = (abs_diff - 150) / 150.0
            return 0.45 + t * (self.alpha_low - 0.45)
        else:
            return self.alpha_low

    def _compute_non_elo_probs(
        self,
        orthogonal_deltas: Dict[str, Dict[str, float]],
        market_probs: Tuple[float, float, float],
        elo_diff: float,
        home_advantage: float,
    ) -> Tuple[float, float, float]:
        """
        从正交化 delta 推导非Elo概率。

        方法:
        1. 计算正交 delta 的净方向: 对各因子 delta 求和得到总调整量
        2. 以市场概率为基准，应用正交调整量
        3. 归一化

        这取代了原来的两阶段: SignalDecomposer → logit_accumulation → PriorShrinkage
        """
        # 聚合正交 delta
        adj_home = 0.0
        adj_draw = 0.0
        adj_away = 0.0

        for fid, deltas in orthogonal_deltas.items():
            adj_home += deltas.get("home", 0.0)
            adj_draw += deltas.get("draw", 0.0)
            adj_away += deltas.get("away", 0.0)

        # 从市场概率出发，应用正交调整
        # 使用 softmax 风格的调整 (避免概率超出[0,1])
        raw_home = max(0.01, market_probs[0] + adj_home * 0.3)
        raw_draw = max(0.01, market_probs[1] + adj_draw * 0.3)
        raw_away = max(0.01, market_probs[2] + adj_away * 0.3)

        # 归一化
        total = raw_home + raw_draw + raw_away
        non_elo_home = raw_home / total
        non_elo_draw = raw_draw / total
        non_elo_away = raw_away / total

        # v5.10.8: 防偏移约束 — 模型概率不得偏离市场超过 8pp
        # 因子系统性偏向客胜时，累积偏差可达 10pp+，导致 0 注主胜/平局
        # 限制单边最大偏离，确保模型能预测所有三种结果
        max_deviation = 0.08
        non_elo_home = max(market_probs[0] - max_deviation, min(market_probs[0] + max_deviation, non_elo_home))
        non_elo_draw = max(market_probs[1] - max_deviation, min(market_probs[1] + max_deviation, non_elo_draw))
        non_elo_away = max(market_probs[2] - max_deviation, min(market_probs[2] + max_deviation, non_elo_away))

        # 重新归一化
        total = non_elo_home + non_elo_draw + non_elo_away
        return (
            non_elo_home / total,
            non_elo_draw / total,
            non_elo_away / total,
        )

    def _magnitude(self, deltas: Dict[str, Dict[str, float]]) -> float:
        """计算 delta 的总幅度 (L2 范数)"""
        total = 0.0
        for fid, d in deltas.items():
            for outcome in ("home", "draw", "away"):
                total += d.get(outcome, 0.0) ** 2
        return math.sqrt(total)

    def get_factor_class(self, factor_id: str) -> str:
        """返回单个因子的分类"""
        if factor_id in ELO_DIRECT_FACTORS:
            return "elo_direct"
        elif factor_id in ELO_DERIVED_FACTORS:
            return "elo_derived"
        elif factor_id in ELO_CORRELATED_FACTORS:
            return "elo_correlated"
        elif factor_id in INDEPENDENT_FACTORS:
            return "independent"
        return "unknown"


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def create_shrinkage_for_league(league_id: str) -> UnifiedBayesianShrinkage:
    """为指定联赛创建校准后的收缩器"""
    return UnifiedBayesianShrinkage(league_id=league_id)


def get_league_alpha(league_id: str) -> Tuple[float, float]:
    """获取联赛校准的 alpha 参数"""
    params = LEAGUE_ALPHA_PARAMS.get(league_id, {"alpha_high": 0.50, "alpha_low": 0.10})
    return params["alpha_high"], params["alpha_low"]