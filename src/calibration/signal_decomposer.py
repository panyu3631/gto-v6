"""
GTO-GameFlow v5.6 信号正交分解引擎

将因子 delta 分解为 Elo 驱动分量与正交分量，解决模型-市场同源问题。

v5.6 升级:
- 真正正交分解: 使用回归残差替代硬抑制 (FIND-005)
- 可校准: calibrate_from_history() 从历史数据学习 Elo-因子关系
- F9 (xg_diff) 从 ELO_DERIVED 移至 INDEPENDENT (FIND-008)
- 因子 delta 标准化 (z-score) 支持 (FIND-004)

核心思路:
- 模型与市场共享 Elo 作为信息源 → 模型输出与市场定价高度共线
- value = P_model - P_market 不是真正的"信息优势"，而是 Elo 信号的重复核算
- 解决方案: 只保留非 Elo 因子贡献的增量信息 (正交分量)

因子分类:
- ELO_DIRECT:   直接使用 Elo 或 elo_diff (F1)
- ELO_DERIVED:  从 Elo 派生 (F7, F8, F19, F27, F29, F33, F37, F39, F40)
- ELO_CORRELATED: 在模拟环境中与 Elo 高度相关 (F3, F5, F20, F38)
- INDEPENDENT:  独立于 Elo (F2, F4, F6, F9-F13, F15-F18, F21-F26, F28, F30-F32, F34-F36, F41)
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# v6.0: 从 registry 读取因子分类，不再硬编码
from src.factors.registry import (
    EloCategory, get_factors_by_elo_category,
)

# 向后兼容: 保留旧变量名
ELO_DIRECT_FACTORS = get_factors_by_elo_category(EloCategory.DIRECT)
ELO_DERIVED_FACTORS = get_factors_by_elo_category(EloCategory.DERIVED)
ELO_CORRELATED_FACTORS = get_factors_by_elo_category(EloCategory.CORRELATED)
INDEPENDENT_FACTORS = get_factors_by_elo_category(EloCategory.INDEPENDENT)

# 所有与 Elo 相关的因子 (将被正交分解)
ELO_AFFECTED_FACTORS = ELO_DIRECT_FACTORS | ELO_DERIVED_FACTORS | ELO_CORRELATED_FACTORS


# ================================================================
# v5.6: 默认 Elo 解释比率 (未校准时使用)
# ================================================================

# 每个因子类别的默认 Elo 解释比率
# residual = delta * (1 - elo_explained_ratio)
DEFAULT_ELO_EXPLAINED_RATIO = {
    "elo_direct": 1.00,      # 100% 由 Elo 解释 → 完全剔除
    "elo_derived": 0.85,     # 85% 由 Elo 解释 → 保留 15%
    "elo_correlated": 0.60,  # 60% 由 Elo 解释 → 保留 40%
}


@dataclass
class DecompositionResult:
    """信号分解结果"""
    # 原始 delta (全部因子)
    raw_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 正交 delta (仅非 Elo 因子 + Elo 相关因子的残差)
    orthogonal_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Elo 驱动 delta (被剔除的分量)
    elo_driven_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 统计
    total_raw_magnitude: float = 0.0
    total_orthogonal_magnitude: float = 0.0
    signal_retention_ratio: float = 0.0  # orthogonal / raw
    elo_affected_count: int = 0
    independent_count: int = 0
    # v5.6: 每个因子的 Elo 解释比率 (用于可解释性)
    elo_explained_ratios: Dict[str, float] = field(default_factory=dict)


class SignalDecomposer:
    """
    信号正交分解器 (v5.6: 回归残差版)。

    将 40 个因子的 delta 分解为:
    - Elo 驱动分量: 可由 Elo 差异解释的信号 → 剔除 (或部分剔除)
    - 正交分量: 非 Elo 因子贡献的增量信息 → 保留

    两种模式:
    1. 未校准模式 (默认): 使用类别级别的 Elo 解释比率
    2. 校准模式: 使用 calibrate_from_history() 学习到的回归系数

    使用方法:
        decomposer = SignalDecomposer()
        # 可选: 从历史数据校准
        decomposer.calibrate_from_history(historical_data)
        result = decomposer.decompose(factor_deltas, elo_diff)

    然后在 logit_accumulation 中使用 result.orthogonal_deltas 替代原始 factor_deltas。
    """

    def __init__(
        self,
        elo_suppression: float = 1.0,
        elo_explained_ratios: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            elo_suppression: 全局 Elo 抑制强度 (0.0 = 不抑制, 1.0 = 完全剔除)
                            用作乘法因子应用于所有 Elo 解释比率
            elo_explained_ratios: 自定义类别级 Elo 解释比率
                                  {"elo_direct": 1.0, "elo_derived": 0.85, "elo_correlated": 0.60}
        """
        self.elo_suppression = elo_suppression
        self.elo_explained_ratios = elo_explained_ratios or dict(DEFAULT_ELO_EXPLAINED_RATIO)

        # v5.6: 回归系数存储 (校准后填充)
        # {factor_id: {"home": (α, β), "draw": (α, β), "away": (α, β)}}
        self._regression_coeffs: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._is_calibrated: bool = False
        self._calibration_stats: Dict[str, Dict] = {}

    # ================================================================
    # v5.6: 历史数据校准
    # ================================================================

    def calibrate_from_history(
        self,
        history: List[Tuple[float, Dict[str, Dict[str, float]]]],
        min_samples: int = 30,
    ) -> Dict[str, Dict]:
        """
        从历史数据学习 Elo-因子回归关系。

        Args:
            history: [(elo_diff, {fid: {"home": d, "draw": d, "away": d}}), ...]
            min_samples: 最少样本数，低于此阈值不校准

        Returns:
            校准统计信息 {fid: {"home_r2": ..., "draw_r2": ..., "away_r2": ..., "n": ...}}
        """
        if len(history) < min_samples:
            return {}

        # 按因子聚合数据
        factor_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for elo_diff, deltas in history:
            for fid, outcomes in deltas.items():
                if fid not in ELO_AFFECTED_FACTORS:
                    continue
                for outcome in ("home", "draw", "away"):
                    d = outcomes.get(outcome, 0.0)
                    factor_data[fid][outcome].append((elo_diff, d))

        # 对每个因子做线性回归
        for fid, outcome_data in factor_data.items():
            self._regression_coeffs[fid] = {}
            self._calibration_stats[fid] = {}

            for outcome, pairs in outcome_data.items():
                if len(pairs) < min_samples:
                    continue

                xs = np.array([p[0] for p in pairs])
                ys = np.array([p[1] for p in pairs])

                # 线性回归: y = α + β * x
                # β = Cov(x,y) / Var(x)
                x_mean = np.mean(xs)
                y_mean = np.mean(ys)
                cov = np.mean((xs - x_mean) * (ys - y_mean))
                var = np.var(xs)

                if var > 1e-10:
                    beta = cov / var
                    alpha = y_mean - beta * x_mean

                    # R² = 1 - SS_res / SS_tot
                    y_pred = alpha + beta * xs
                    ss_res = np.sum((ys - y_pred) ** 2)
                    ss_tot = np.sum((ys - y_mean) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

                    self._regression_coeffs[fid][outcome] = (alpha, beta)
                    self._calibration_stats[fid][outcome] = {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "r2": float(r2),
                        "n": len(pairs),
                    }

        if self._regression_coeffs:
            self._is_calibrated = True

        return dict(self._calibration_stats)

    # ================================================================
    # 核心分解方法
    # ================================================================

    def decompose(
        self,
        factor_deltas: Dict[str, Dict[str, float]],
        elo_diff: float,
    ) -> DecompositionResult:
        """
        分解因子 delta。

        v5.6: 校准模式下使用回归残差，未校准模式下使用类别级解释比率。

        Args:
            factor_deltas: {factor_id: {"home": delta, "draw": delta, "away": delta}}
            elo_diff: home_elo - away_elo

        Returns:
            DecompositionResult with orthogonal_deltas ready for logit accumulation
        """
        result = DecompositionResult()
        result.raw_deltas = factor_deltas

        for fid, deltas in factor_deltas.items():
            if fid in ELO_DIRECT_FACTORS:
                # Elo 直接因子: 完全剔除 (或部分保留取决于 suppression)
                result.elo_driven_deltas[fid] = deltas
                result.elo_affected_count += 1

                if self._is_calibrated and fid in self._regression_coeffs:
                    # 校准模式: 使用回归残差
                    result.orthogonal_deltas[fid] = self._compute_residuals(
                        fid, deltas, elo_diff
                    )
                    result.elo_explained_ratios[fid] = self._compute_r2_mean(fid)
                else:
                    # 未校准: 使用类别级比率
                    ratio = self.elo_explained_ratios.get("elo_direct", 1.0) * self.elo_suppression
                    result.elo_explained_ratios[fid] = ratio
                    retention = 1.0 - ratio
                    if retention > 0.001:
                        result.orthogonal_deltas[fid] = {
                            k: v * retention for k, v in deltas.items()
                        }

            elif fid in ELO_DERIVED_FACTORS:
                # Elo 派生因子: 部分剔除
                result.elo_driven_deltas[fid] = deltas
                result.elo_affected_count += 1

                if self._is_calibrated and fid in self._regression_coeffs:
                    result.orthogonal_deltas[fid] = self._compute_residuals(
                        fid, deltas, elo_diff
                    )
                    result.elo_explained_ratios[fid] = self._compute_r2_mean(fid)
                else:
                    ratio = self.elo_explained_ratios.get("elo_derived", 0.85) * self.elo_suppression
                    result.elo_explained_ratios[fid] = ratio
                    retention = 1.0 - ratio
                    if retention > 0.001:
                        result.orthogonal_deltas[fid] = {
                            k: v * retention for k, v in deltas.items()
                        }

            elif fid in ELO_CORRELATED_FACTORS:
                # Elo 相关因子: 部分剔除 (保留更多)
                result.elo_driven_deltas[fid] = deltas
                result.elo_affected_count += 1

                if self._is_calibrated and fid in self._regression_coeffs:
                    result.orthogonal_deltas[fid] = self._compute_residuals(
                        fid, deltas, elo_diff
                    )
                    result.elo_explained_ratios[fid] = self._compute_r2_mean(fid)
                else:
                    ratio = self.elo_explained_ratios.get("elo_correlated", 0.60) * self.elo_suppression
                    result.elo_explained_ratios[fid] = ratio
                    retention = 1.0 - ratio
                    if retention > 0.001:
                        result.orthogonal_deltas[fid] = {
                            k: v * retention for k, v in deltas.items()
                        }

            else:
                # 独立因子: 完全保留
                result.orthogonal_deltas[fid] = deltas
                result.independent_count += 1
                result.elo_explained_ratios[fid] = 0.0

        # 计算统计量
        result.total_raw_magnitude = self._compute_magnitude(factor_deltas)
        result.total_orthogonal_magnitude = self._compute_magnitude(result.orthogonal_deltas)
        if result.total_raw_magnitude > 0:
            result.signal_retention_ratio = result.total_orthogonal_magnitude / result.total_raw_magnitude

        return result

    def _compute_residuals(
        self,
        fid: str,
        deltas: Dict[str, float],
        elo_diff: float,
    ) -> Dict[str, float]:
        """
        使用校准后的回归系数计算残差: residual = delta - (α + β * elo_diff)
        """
        residuals = {}
        coeffs = self._regression_coeffs.get(fid, {})

        for outcome in ("home", "draw", "away"):
            d = deltas.get(outcome, 0.0)

            if outcome in coeffs:
                alpha, beta = coeffs[outcome]
                elo_driven = alpha + beta * elo_diff
                residual = d - elo_driven * self.elo_suppression
            else:
                # 该 outcome 无校准数据，保留原始值
                residual = d

            residuals[outcome] = residual

        return residuals

    def _compute_r2_mean(self, fid: str) -> float:
        """计算因子的平均 R² (Elo 解释比率)"""
        stats = self._calibration_stats.get(fid, {})
        if not stats:
            return 0.0
        r2_values = [s["r2"] for s in stats.values()]
        return float(np.mean(r2_values)) if r2_values else 0.0

    # ================================================================
    # v5.6: 因子 delta 标准化 (z-score)
    # ================================================================

    @staticmethod
    def standardize_deltas(
        factor_deltas: Dict[str, Dict[str, float]],
        factor_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        对因子 delta 进行 z-score 标准化。

        防止高 magnitude 因子主导 logit 累加。

        Args:
            factor_deltas: {fid: {"home": d, "draw": d, "away": d}}
            factor_stats: {fid: {"home": (mean, std), "draw": (mean, std), "away": (mean, std)}}
                          如果为 None，则使用全局均值/标准差

        Returns:
            标准化后的 deltas
        """
        standardized = {}

        # 收集所有 deltas 用于计算全局统计量
        all_values = []
        for fid, deltas in factor_deltas.items():
            for outcome in ("home", "draw", "away"):
                all_values.append(deltas.get(outcome, 0.0))

        global_mean = float(np.mean(all_values)) if all_values else 0.0
        global_std = float(np.std(all_values)) if all_values else 1.0

        for fid, deltas in factor_deltas.items():
            standardized[fid] = {}
            for outcome in ("home", "draw", "away"):
                d = deltas.get(outcome, 0.0)

                if factor_stats and fid in factor_stats and outcome in factor_stats[fid]:
                    mean, std = factor_stats[fid][outcome]
                else:
                    mean, std = global_mean, global_std

                if abs(std) < 1e-10:
                    standardized[fid][outcome] = 0.0
                else:
                    standardized[fid][outcome] = (d - mean) / std

        return standardized

    @staticmethod
    def compute_factor_statistics(
        all_deltas: List[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        从历史数据计算每个因子的统计量 (mean, std)。

        Args:
            all_deltas: [{"F1": {"home": 0.1, ...}, ...}, ...]

        Returns:
            {fid: {"home": (mean, std), "draw": (mean, std), "away": (mean, std)}}
        """
        # 收集数据
        factor_values: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for deltas in all_deltas:
            for fid, outcomes in deltas.items():
                for outcome in ("home", "draw", "away"):
                    factor_values[fid][outcome].append(outcomes.get(outcome, 0.0))

        # 计算统计量
        stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for fid, outcome_data in factor_values.items():
            stats[fid] = {}
            for outcome, values in outcome_data.items():
                if len(values) >= 2:
                    mean = float(np.mean(values))
                    std = float(np.std(values, ddof=1))
                else:
                    mean = 0.0
                    std = 1.0
                stats[fid][outcome] = (mean, std)

        return stats

    def _compute_magnitude(self, deltas: Dict[str, Dict[str, float]]) -> float:
        """计算 delta 的总幅度 (L2 范数)"""
        total = 0.0
        for fid, d in deltas.items():
            for outcome in ("home", "draw", "away"):
                total += d.get(outcome, 0.0) ** 2
        return math.sqrt(total)


# ================================================================
# 贝叶斯先验收缩
# ================================================================

class PriorShrinkage:
    """
    贝叶斯先验收缩器。

    核心思想: 当 Elo 差距很大时，市场已经充分定价了 Elo 信息。
    此时模型输出应该向市场先验收缩，避免过度自信。

    P_shrunk = (1 - α) × P_market + α × P_model

    动态 α 调度:
    - |elo_diff| < 50:   α = 0.50  (势均力敌，市场定价不充分，模型权重高)
    - 50 ≤ |elo_diff| < 150: α = 0.35
    - 150 ≤ |elo_diff| < 300: α = 0.20
    - |elo_diff| ≥ 300:  α = 0.10  (实力悬殊，市场已充分定价，模型权重低)
    """

    def __init__(
        self,
        alpha_high: float = 0.50,   # 势均力敌时的 α
        alpha_low: float = 0.10,    # 实力悬殊时的 α
    ):
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low

    def compute_alpha(self, elo_diff: float) -> float:
        """
        根据 Elo 差距计算动态收缩参数 α。

        α 决定模型概率在最终融合中的权重。
        """
        abs_diff = abs(elo_diff)

        if abs_diff < 50:
            return self.alpha_high
        elif abs_diff < 150:
            t = (abs_diff - 50) / 100.0
            return self.alpha_high + t * (0.35 - self.alpha_high)
        elif abs_diff < 300:
            t = (abs_diff - 150) / 150.0
            return 0.35 + t * (0.20 - 0.35)
        else:
            return self.alpha_low

    def shrink(
        self,
        model_probs: Tuple[float, float, float],  # (home, draw, away)
        market_probs: Tuple[float, float, float],
        elo_diff: float,
    ) -> Tuple[float, float, float]:
        """
        对模型概率应用贝叶斯先验收缩。

        Args:
            model_probs: (P_home, P_draw, P_away) 模型输出
            market_probs: (P_home, P_draw, P_away) 市场隐含概率
            elo_diff: Elo 差异

        Returns:
            (P_home_shrunk, P_draw_shrunk, P_away_shrunk)
        """
        alpha = self.compute_alpha(elo_diff)

        shrunk = tuple(
            (1.0 - alpha) * mp + alpha * model_p
            for mp, model_p in zip(market_probs, model_probs)
        )

        return shrunk


# ================================================================
# 便捷函数
# ================================================================

def get_factor_classification() -> Dict[str, List[str]]:
    """返回因子分类汇总"""
    return {
        "elo_direct": sorted(ELO_DIRECT_FACTORS),
        "elo_derived": sorted(ELO_DERIVED_FACTORS),
        "elo_correlated": sorted(ELO_CORRELATED_FACTORS),
        "independent": sorted(INDEPENDENT_FACTORS),
        "elo_affected": sorted(ELO_AFFECTED_FACTORS),
    }


def get_factor_class(factor_id: str) -> str:
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