"""
GTO-GameFlow v6.0 自适应裁剪 + 市场异常检测

v5.10 核心问题:
- 固定 5pp 裁剪: 无论模型多确信，最多偏离市场 5pp → 模型永远无法表达独立观点
- 没有市场效率检测: 对英超焦点战和法甲保级战一视同仁
- 没有"价值被低估"的检测: 无法识别市场定价错误

v6.0 根本性重构:
1. 自适应裁剪: 基于因子激活率、信息质量、市场效率动态调整裁剪阈值
2. 市场异常检测: 前置检测低效率市场，自动调整模型权重
3. 价值洼地检测: 识别市场系统性低估的选择
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketEfficiencyProfile:
    """市场效率画像"""
    is_efficient: bool           # 是否高效市场
    is_high_attention: bool      # 是否高关注度比赛
    efficiency_score: float      # 效率评分 (0-1, 1=高效)
    model_weight_boost: float    # 模型权重加成 (0-1)
    clip_allowance: float        # 裁剪宽松度 (0-1, 1=完全宽松)
    value_signal_decay: float    # 价值信号衰减 (0-1, 1=无衰减)
    league_tier: int             # 联赛层级 (1=顶级, 2=次级, 3=其他)

    @property
    def max_clip_pp(self) -> float:
        """v6.0.1: 最大允许偏离百分点 — 保守回到5pp (与v5.10.8一致)"""
        return 0.05  # v6.0.1: 固定5pp, 暂时关闭自适应


@dataclass
class ValueDepressionAlert:
    """价值洼地检测告警"""
    outcome: str                    # 被低估的结果 (home/draw/away)
    depression_score: float         # 低估程度 (0-1)
    model_prob: float               # 模型概率
    implied_prob: float             # 市场隐含概率
    raw_value: float                # 原始价值 (裁剪前)
    clipped_value: float            # 裁剪后价值
    signals: list                   # 触发信号列表


class MarketAnomalyDetector:
    """
    市场异常检测器 — 前置层。

    回答三个问题:
    1. 当前市场效率如何？(高效 → 多听市场，低效 → 多听模型)
    2. 模型应该被允许偏离市场多远？
    3. 是否存在被市场系统性低估的选择？
    """

    # 联赛关注度分级
    LEAGUE_TIER = {
        "EPL": 1, "LA_LIGA": 1, "BUNDESLIGA": 1, "SERIE_A": 1, "LIGUE_1": 1,
        "CHAMPIONSHIP": 2, "BUNDESLIGA_2": 2, "SERIE_B": 2, "LIGUE_2": 2,
    }

    def __init__(self, league_id: str):
        self.league_id = league_id
        self.league_tier = self.LEAGUE_TIER.get(league_id, 3)

    def assess_market_efficiency(
        self,
        odds_std: float = 0.05,
        season_phase: float = 0.5,
        elo_diff: float = 0.0,
        is_derby: bool = False,
        is_primetime: bool = False,
        market_efficiency_metric: float = 0.5,
        factor_activation_rate: float = 0.5,
        information_quality: float = 0.5,
    ) -> MarketEfficiencyProfile:
        """
        评估市场效率并生成画像。

        低效市场特征:
        - 赔率离散度高 (博彩公司分歧大)
        - 赛季初 (数据不足，市场定价不准确)
        - 低关注度比赛 (非黄金时段，非焦点对决)
        - 德比战 (情绪因素大，基本面定价不准)
        - 低级联赛 (信息不充分)

        Args:
            odds_std: 赔率离散度 (跨公司标准差)
            season_phase: 赛季阶段 (0=开局, 1=末段)
            elo_diff: Elo差值
            is_derby: 是否德比
            is_primetime: 是否黄金时段
            market_efficiency_metric: 市场效率度量 (历史Brier分数)
            factor_activation_rate: 因子激活率
            information_quality: 信息质量评分
        """
        # === 效率评分 ===
        # 起始: 市场效率度量
        efficiency = market_efficiency_metric

        # 赔率离散度惩罚: 离散度越高，市场越不确定
        dispersion_penalty = min(odds_std / 0.15, 1.0) * 0.3
        efficiency -= dispersion_penalty

        # 赛季初惩罚: 赛季初市场定价不准
        early_season_penalty = (1.0 - season_phase) * 0.15
        efficiency -= early_season_penalty

        # 德比惩罚: 德比战情绪因素大
        if is_derby:
            efficiency -= 0.1

        # 低关注度惩罚: 非黄金时段 = 市场关注度低
        if not is_primetime:
            efficiency -= 0.05

        # 联赛层级调整
        if self.league_tier == 2:
            efficiency -= 0.1
        elif self.league_tier == 3:
            efficiency -= 0.2

        # 实力悬殊调整: 实力差距大时市场更准
        if abs(elo_diff) > 300:
            efficiency += 0.05

        efficiency = max(0.1, min(1.0, efficiency))

        # === 判断高效/低效 ===
        is_efficient = efficiency > 0.50  # v6.0.1: 进一步降低阈值 0.55→0.50, 更偏袒"高效"
        is_high_attention = self.league_tier == 1 and is_primetime

        # === 模型权重加成 ===
        # 低效市场 → 提高模型权重
        model_weight_boost = (1.0 - efficiency) * 0.5  # 最多 +50%

        # === 裁剪宽松度 ===
        # 低效市场 → 放宽裁剪 (但幅度更大)
        # 因子激活率高 → 放宽裁剪 (信息充分)
        # 信息质量高 → 放宽裁剪
        clip_allowance = (
            (1.0 - efficiency) * 0.30 +           # v6.0.1: 低效市场: 最多 +30% (进一步降低)
            factor_activation_rate * 0.15 +        # 因子激活: 最多 +15% (降低)
            information_quality * 0.05             # 信息质量: 最多 +5% (降低)
        )
        clip_allowance = max(0.0, min(1.0, clip_allowance))

        # === 价值信号衰减 ===
        # 高效市场 → 价值信号衰减大 (市场定价准，难找到价值)
        value_signal_decay = efficiency * 0.8 + 0.2  # 0.2~1.0

        return MarketEfficiencyProfile(
            is_efficient=is_efficient,
            is_high_attention=is_high_attention,
            efficiency_score=efficiency,
            model_weight_boost=model_weight_boost,
            clip_allowance=clip_allowance,
            value_signal_decay=value_signal_decay,
            league_tier=self.league_tier,
        )

    def detect_value_depression(
        self,
        model_probs: Dict[str, float],
        market_probs: Dict[str, float],
        efficiency_profile: MarketEfficiencyProfile,
        factor_activation_rate: float = 0.5,
        information_quality: float = 0.5,
        odds_std: float = 0.05,
    ) -> list:
        """
        检测市场系统性低估的选择。

        判断标准:
        1. 模型概率 > 市场概率 + 阈值
        2. 市场效率低 (有定价错误的可能)
        3. 不是"明显热门" (博彩公司不会在热门上犯错)
        4. 平局方向额外加分 (平局是最容易被低估的)

        Returns:
            List of ValueDepressionAlert
        """
        alerts = []

        # 动态阈值: 低效市场可以降低阈值
        base_threshold = 0.02  # 基础 2pp
        if not efficiency_profile.is_efficient:
            base_threshold *= 0.7  # 低效市场降低到 1.4pp
        if not efficiency_profile.is_high_attention:
            base_threshold *= 0.8  # 低关注度进一步降低

        for outcome in ("home", "draw", "away"):
            model_p = model_probs.get(outcome, 0.0)
            market_p = market_probs.get(outcome, 0.0)
            raw_value = model_p - market_p

            # 排除明显热门 (博彩公司不会在热门上犯错)
            if market_p > 0.55:
                # 热门方需要更强的信号
                adjusted_threshold = base_threshold * 1.5
            elif market_p < 0.20:
                # 冷门方可能被低估
                adjusted_threshold = base_threshold * 0.6
            else:
                adjusted_threshold = base_threshold

            if raw_value <= adjusted_threshold:
                continue

            # 计算低估程度
            # 额外加分项:
            # - 低效市场: +20%
            # - 平局方向: +30% (平局最容易被低估)
            # - 冷门方向: +15% (市场倾向低估冷门)
            depression_score = raw_value / 0.15  # 归一化到 0-1
            if not efficiency_profile.is_efficient:
                depression_score *= 1.2
            if outcome == "draw":
                depression_score *= 1.3
            if market_p < 0.25:
                depression_score *= 1.15
            depression_score = min(1.0, depression_score)

            # 收集信号
            signals = []
            if not efficiency_profile.is_efficient:
                signals.append(f"市场低效(效率={efficiency_profile.efficiency_score:.2f})")
            if outcome == "draw":
                signals.append("平局方向(最容易被低估)")
            if market_p < 0.25:
                signals.append(f"冷门方向(市场隐含={market_p:.2%})")
            if factor_activation_rate > 0.6:
                signals.append(f"因子激活率高({factor_activation_rate:.1%})")
            if information_quality > 0.6:
                signals.append(f"信息质量高({information_quality:.1%})")

            # 裁剪后价值
            max_clip = efficiency_profile.max_clip_pp
            clipped_value = max(-max_clip, min(max_clip, raw_value))

            alerts.append(ValueDepressionAlert(
                outcome=outcome,
                depression_score=depression_score,
                model_prob=model_p,
                implied_prob=market_p,
                raw_value=raw_value,
                clipped_value=clipped_value,
                signals=signals,
            ))

        # 按低估程度排序
        alerts.sort(key=lambda x: x.depression_score, reverse=True)
        return alerts


class AdaptiveClipping:
    """
    自适应裁剪 — 替代固定 5pp 裁剪。

    核心逻辑:
    - 市场高效 + 因子激活率低 → 保守裁剪 (3-5pp)
    - 市场低效 + 因子激活率高 + 信息质量好 → 激进裁剪 (10-15pp)
    - 平局方向 → 额外宽松 (平局最难预测，市场定价最不准确)
    """

    def __init__(self):
        # 基础裁剪参数
        self.min_clip = 0.03   # 最保守: 3pp
        self.max_clip = 0.15   # 最激进: 15pp
        self.draw_bonus = 0.03 # 平局额外宽松: +3pp

    def compute_clip_limit(
        self,
        efficiency_profile: MarketEfficiencyProfile,
        factor_activation_rate: float = 0.5,
        factor_diversity: float = 0.5,    # 因子多样性 (激活维度数/总维度数)
        information_quality: float = 0.5,
        outcome: str = "home",
    ) -> float:
        """
        计算当前比赛的裁剪上限。

        Args:
            efficiency_profile: 市场效率画像
            factor_activation_rate: 因子激活率 (0-1)
            factor_diversity: 因子多样性 (不同维度激活比例)
            information_quality: 信息质量 (0-1)
            outcome: 投注方向 (home/draw/away)

        Returns:
            最大允许偏离的百分点 (如 0.08 = 8pp)
        """
        # 基础: 从效率画像获取
        clip = efficiency_profile.max_clip_pp

        # 因子多样性调整: 多个独立维度一致 → 增强信心
        if factor_diversity > 0.6:
            clip += 0.02  # 多维一致 +2pp

        # 因子激活率调整: 激活率高 → 更多信息 → 更自信
        if factor_activation_rate > 0.7:
            clip += 0.02

        # 信息质量调整
        if information_quality > 0.7:
            clip += 0.01

        # 平局方向额外宽松 (平局市场定价最不透明)
        if outcome == "draw":
            clip += self.draw_bonus

        # 上限
        clip = min(clip, self.max_clip)

        # 下限 (确保至少有一定自由度)
        clip = max(clip, self.min_clip)

        return clip

    def apply_clip(
        self,
        model_prob: float,
        market_prob: float,
        clip_limit: float,
    ) -> float:
        """
        将模型概率裁剪到 [market_prob - clip_limit, market_prob + clip_limit]。
        """
        lower = market_prob - clip_limit
        upper = market_prob + clip_limit
        return max(lower, min(upper, model_prob))

    def clip_probs(
        self,
        model_probs: Dict[str, float],
        market_probs: Dict[str, float],
        efficiency_profile: MarketEfficiencyProfile,
        factor_activation_rate: float = 0.5,
        factor_diversity: float = 0.5,
        information_quality: float = 0.5,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        对三个方向的概率分别应用自适应裁剪。

        Returns:
            (clipped_probs, clip_limits) — 裁剪后概率和每个方向的裁剪上限
        """
        clipped = {}
        clip_limits = {}

        for outcome in ("home", "draw", "away"):
            limit = self.compute_clip_limit(
                efficiency_profile=efficiency_profile,
                factor_activation_rate=factor_activation_rate,
                factor_diversity=factor_diversity,
                information_quality=information_quality,
                outcome=outcome,
            )
            clip_limits[outcome] = limit
            clipped[outcome] = self.apply_clip(
                model_prob=model_probs.get(outcome, 0.33),
                market_prob=market_probs.get(outcome, 0.33),
                clip_limit=limit,
            )

        # 归一化
        total = sum(clipped.values())
        if total > 0:
            for outcome in clipped:
                clipped[outcome] /= total

        return clipped, clip_limits


# ================================================================
# 便捷函数
# ================================================================

def compute_factor_diversity(activated_dimensions: set, total_dimensions: int = 12) -> float:
    """计算因子多样性: 激活的维度数 / 总维度数"""
    if total_dimensions == 0:
        return 0.0
    return len(activated_dimensions) / total_dimensions


def compute_information_quality(
    factor_activation_rate: float,
    data_completeness: float,
    odds_std: float,
    season_phase: float,
) -> float:
    """
    计算信息质量评分。

    高质量特征:
    - 因子激活率高 (数据充分)
    - 数据完整性高
    - 赔率离散度低 (市场一致，信息可靠)
    - 赛季中后期 (数据积累充分)
    """
    quality = (
        factor_activation_rate * 0.35 +
        data_completeness * 0.25 +
        (1.0 - min(odds_std / 0.15, 1.0)) * 0.20 +  # 低离散度 = 高质量
        season_phase * 0.20
    )
    return max(0.1, min(1.0, quality))