"""
GTO-GameFlow v5.6 概率引擎

实现规范文档第6章：logit累加、sigmoid归一化、泊松桥接、价值计算。

v5.6 升级:
- 因子权重应用: logit_accumulation 中按因子权重加权累加 (FIND-006)
- Softmax 温度参数: sigmoid_normalization 增加 temperature 控制 (FIND-007)
- F10 解耦: 使用 market_probs 先验时自动跳过 F10 避免双重计入 (FIND-002)
"""
import math
import numpy as np
from typing import Dict, Tuple, Optional
from src.config.league_params import get_league_params


def _poisson_pmf(k: int, lam: float) -> float:
    """泊松分布PMF — 自建实现，去除scipy依赖"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dixon_coles_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    """
    Dixon-Coles tau 校正函数。

    标准独立泊松模型假设两队进球独立，但足球中低比分结果
    (0:0, 1:0, 0:1, 1:1) 存在显著的相关性:
    - 一队进球后，另一队更保守 → 进球负相关
    - 0:0 和 1:1 被低估，1:0 和 0:1 被高估

    tau 校正仅作用于低比分 (x,y ∈ {0,1}):
    τ(0,0) = 1 - λ_h * λ_a * ρ
    τ(0,1) = 1 + λ_h * ρ
    τ(1,0) = 1 + λ_a * ρ
    τ(1,1) = 1 - ρ
    τ(x,y) = 1  (x≥2 or y≥2)

    典型 ρ 值: -0.13 ~ -0.05 (负值表示进球负相关)
    来源: Dixon & Coles (1997), "Modelling Association Football Scores"
    """
    if x >= 2 or y >= 2:
        return 1.0
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lam_home * lam_away * rho)
    elif x == 0 and y == 1:
        return max(0.0, 1.0 + lam_home * rho)
    elif x == 1 and y == 0:
        return max(0.0, 1.0 + lam_away * rho)
    elif x == 1 and y == 1:
        return max(0.0, 1.0 - rho)
    return 1.0
from src.data.models import ProbabilityDistribution, ScoreMatrix


class ProbabilityEngine:
    """
    概率计算引擎 — 将因子delta转换为胜平负概率分布。

    流水线:
    Stage 2: logit_accumulation  — 从市场先验出发，累加因子delta
    Stage 3: sigmoid_normalization — 将logit转换回概率空间 (含温度参数)
    Stage 4: poisson_bridge       — Dual-Domain架构: 泊松比分模型独立计算
    Stage 5: value_calculation    — 计算模型概率与赔率隐含概率的差值
    """

    def __init__(self, league_id: str):
        self.league_id = league_id
        self.params = get_league_params(league_id)

    # ================================================================
    # Stage 2: logit_accumulation (v5.6: 因子权重 + F10解耦)
    # ================================================================
    def logit_accumulation(
        self,
        market_probs: Dict[str, float],
        factor_deltas: Dict[str, Dict[str, float]],
        uniform_prior: bool = True,
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        从市场先验(或均匀先验)出发，累加所有因子delta到logit空间。

        v5.6 规范化: logit(p_k) = logit(p_market_k) + Σ w_i × delta_i_k

        Args:
            market_probs: 市场隐含概率 {"home": 0.45, "draw": 0.28, "away": 0.27}
            factor_deltas: 正交化后的因子delta
            uniform_prior: True=均匀先验, False=市场先验
            factor_weights: 因子权重映射 {fid: weight}, 默认1.0
        """
        logits = {}
        for outcome in ("home", "draw", "away"):
            if uniform_prior:
                # 均匀先验 logit(0.33) = 0
                base_logit = 0.0
            else:
                mp = market_probs.get(outcome, 0.33)
                mp = max(0.001, min(0.999, mp))
                base_logit = math.log(mp / (1.0 - mp))

            # v5.6: 累加加权因子 delta
            total_delta = 0.0
            for fid, deltas in factor_deltas.items():
                # v5.6 (FIND-002): 使用市场先验时跳过 F10 避免双重计入
                if not uniform_prior and fid == "F10":
                    continue

                # v5.6 (FIND-006): 应用因子权重
                weight = 1.0
                if factor_weights:
                    weight = factor_weights.get(fid, 1.0)

                total_delta += deltas.get(outcome, 0.0) * weight

            logits[outcome] = base_logit + total_delta

        return logits

    # ================================================================
    # Stage 3: sigmoid_normalization (v5.6: 温度参数)
    # ================================================================
    def sigmoid_normalization(
        self,
        logits: Dict[str, float],
        temperature: float = 1.0,
    ) -> ProbabilityDistribution:
        """
        将 logit 值通过 softmax 转换回概率空间。

        v5.6 (FIND-007): P_k = exp(logit_k / T) / Σ exp(logit_j / T)

        Args:
            logits: {"home": 0.5, "draw": -0.2, "away": -0.8}
            temperature: 温度参数, T>1 平滑分布, T<1 锐化分布, T=1 标准 softmax

        Returns:
            ProbabilityDistribution
        """
        t = max(0.1, temperature)  # 防止温度 ≤ 0

        exp_values = {
            outcome: math.exp(logit / t)
            for outcome, logit in logits.items()
        }
        total = sum(exp_values.values())

        return ProbabilityDistribution(
            prob_home=exp_values["home"] / total,
            prob_draw=exp_values["draw"] / total,
            prob_away=exp_values["away"] / total,
        )

    def logit_to_probability(
        self,
        market_probs: Dict[str, float],
        factor_deltas: Dict[str, Dict[str, float]],
        factor_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ) -> ProbabilityDistribution:
        """Stage 2 + Stage 3 组合调用"""
        logits = self.logit_accumulation(
            market_probs, factor_deltas,
            uniform_prior=False, factor_weights=factor_weights,
        )
        return self.sigmoid_normalization(logits, temperature)

    # ================================================================
    # Stage 4: poisson_bridge — Dual-Domain架构
    # ================================================================
    def poisson_bridge(
        self,
        home_elo: float,
        away_elo: float,
        factor_deltas: Dict[str, Dict[str, float]],
        max_goals: int = 5,
    ) -> Tuple[ProbabilityDistribution, ScoreMatrix]:
        """
        使用泊松模型独立计算比分矩阵，并通过Dual-Domain架构与logit模型融合。

        步骤:
        1. 从Elo + 因子计算预期进球 λ_home, λ_away
        2. 构建比分概率矩阵 (0:0 至 max_goals:max_goals)
        3. 从比分矩阵推导胜平负概率 (泊松域)
        4. 融合 logit域概率与泊松域概率

        返回: (融合后的ProbabilityDistribution, ScoreMatrix)
        """
        # === Step 1: 计算预期进球 ===
        # 基础 λ: 从联赛场均进球和Elo差推导
        base_lambda = self.params.avg_goals / 2.0  # 每队基础进球期望
        elo_factor = (home_elo - away_elo) / 400.0

        # 主场优势调整
        lambda_home = base_lambda * (1.0 + elo_factor * 0.5 + self.params.home_advantage * 0.3)
        lambda_away = base_lambda * (1.0 - elo_factor * 0.5)

        # 因子delta对预期进球的影响
        # 进攻型因子 (F9, F29) 提升进球期望
        # 防守型因子 (F8) 降低进球期望
        for fid, deltas in factor_deltas.items():
            if fid in ("F9", "F29"):  # xG, 大小球趋势
                lambda_home += deltas.get("home", 0.0) * 0.5
                lambda_away += deltas.get("away", 0.0) * 0.5
            elif fid == "F8":  # 进球/失球差
                lambda_home += deltas.get("home", 0.0) * 0.3
                lambda_away -= deltas.get("home", 0.0) * 0.3

        # 泊松相关性校正 (v5.2: 非对称 — 按实力加权分配)
        # 强队应该获得更多进球增量，弱队获得更少
        total_elo = home_elo + away_elo
        if total_elo > 0:
            home_weight = home_elo / total_elo
            away_weight = away_elo / total_elo
        else:
            home_weight = away_weight = 0.5
        lambda_home += self.params.poisson_delta * home_weight * 2.0
        lambda_away += self.params.poisson_delta * away_weight * 2.0

        # 确保非负
        lambda_home = max(0.1, lambda_home)
        lambda_away = max(0.1, lambda_away)

        # v5.10.8: 校准 lambda 使预期总进球等于联赛场均
        # 原始公式高估了总进球 (home_advantage * 0.3 + poisson_delta * 2.0 提供额外偏移)
        # 泊松分布假设 mean=variance，但足球进球存在负相关 (一队进球后另一队更保守)
        # 导致实际总进球方差低于泊松预测，因此降低校准目标 7%
        expected_total = lambda_home + lambda_away
        if expected_total > 0:
            scale = self.params.avg_goals * 0.93 / expected_total
            lambda_home *= scale
            lambda_away *= scale

        # === Step 2: 构建比分矩阵 (Dixon-Coles 校正) ===
        # v5.11: 使用 Dixon-Coles 模型替代独立泊松
        # ρ (rho) 校正低比分的联合概率，反映进球负相关性
        # 典型值: ρ ∈ [-0.13, -0.05]，联赛场均越高 ρ 越接近 0
        # 来源: Dixon & Coles (1997)
        rho = -0.10  # 默认值 (五大联赛回测最优区间)
        # 联赛特化: 高进球联赛相关性更弱
        if self.params.avg_goals > 2.9:
            rho = -0.07  # 德甲: 高进球，相关性弱
        elif self.params.avg_goals < 2.6:
            rho = -0.13  # 意甲: 低进球，相关性强

        matrix = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                base_prob = _poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)
                tau = _dixon_coles_tau(h, a, lambda_home, lambda_away, rho)
                matrix[(h, a)] = base_prob * tau

        # 归一化 (截断部分重新分配)
        total = sum(matrix.values())
        if total > 0:
            for key in matrix:
                matrix[key] /= total

        score_matrix = ScoreMatrix(
            league_id=self.league_id,
            max_goals=max_goals,
            matrix=matrix,
        )

        # === Step 3: 从比分矩阵推导胜平负概率 ===
        poisson_home = 0.0
        poisson_draw = 0.0
        poisson_away = 0.0
        for (h, a), prob in matrix.items():
            if h > a:
                poisson_home += prob
            elif h == a:
                poisson_draw += prob
            else:
                poisson_away += prob

        poisson_probs = ProbabilityDistribution(
            prob_home=poisson_home,
            prob_draw=poisson_draw,
            prob_away=poisson_away,
        )

        return poisson_probs, score_matrix

    def dual_domain_fusion(
        self,
        logit_probs: ProbabilityDistribution,
        poisson_probs: ProbabilityDistribution,
        fusion_weight: float = 0.3,
        data_quality: Optional[float] = None,
        odds_std: Optional[float] = None,
    ) -> ProbabilityDistribution:
        """
        Dual-Domain融合: logit模型(70%) + 泊松模型(30%)。

        v5.2: 融合权重动态化
        - 基础权重: α = fusion_weight (默认0.3)
        - 数据质量调整: 低质量时提高泊松权重 (泊松依赖较少数据)
        - 赔率离散度调整: 高离散度时提高泊松权重 (市场不确定时泊松更稳定)

        P_fused = (1 - α) × P_logit + α × P_poisson
        """
        alpha = fusion_weight

        # 动态调整 (v5.2)
        if data_quality is not None:
            # 数据质量越低，泊松权重越高 (泊松模型对数据依赖性更低)
            quality_adjust = (1.0 - data_quality) * 0.15
            alpha += quality_adjust

        if odds_std is not None:
            # 赔率离散度越高，市场不确定性越大，泊松模型更稳定
            dispersion_adjust = min(odds_std / 0.15, 1.0) * 0.10
            alpha += dispersion_adjust

        # 泊松权重范围: 0.15 ~ 0.55
        alpha = max(0.15, min(0.55, alpha))

        return ProbabilityDistribution(
            prob_home=(1 - alpha) * logit_probs.prob_home + alpha * poisson_probs.prob_home,
            prob_draw=(1 - alpha) * logit_probs.prob_draw + alpha * poisson_probs.prob_draw,
            prob_away=(1 - alpha) * logit_probs.prob_away + alpha * poisson_probs.prob_away,
        )

    # ================================================================
    # Stage 5: value_calculation
    # ================================================================
    def calculate_value(
        self,
        model_probs: ProbabilityDistribution,
        odds: Dict[str, float],
        calibration_discount: float = 0.15,
        calibration_multiplier: float = 1.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        计算每个投注选项的价值 = 模型概率 - 赔率隐含概率。

        步骤:
        1. 计算 overround = 1/odds_home + 1/odds_draw + 1/odds_away
        2. 去庄家margin: implied_prob = (1 / odds) / overround
        3. value = model_prob - implied_prob

        calibration_discount: 模型概率向市场回撤的基础比例 (0.15=15%回撤)
        calibration_multiplier: 联赛差异化校准强度 (1.0=基准, 1.8=法甲强校准)

        返回: {
            "home": {"model_prob": 0.45, "implied_prob": 0.42, "value": 0.03},
            "draw": {...},
            "away": {...}
        }
        """
        # 计算庄家 overround (margin)
        overround = 0.0
        for outcome in ("home", "draw", "away"):
            odd = odds.get(outcome, 1.0)
            if odd > 0:
                overround += 1.0 / odd

        if overround <= 0:
            overround = 1.0  # 防御

        result = {}
        for outcome in ("home", "draw", "away"):
            model_p = getattr(model_probs, f"prob_{outcome}")
            odd = odds.get(outcome, 1.0)
            if odd <= 0:
                odd = 1.0
            implied = (1.0 / odd) / overround
            # v5.10.9: 渐进式校准 — 模型概率越高, 回撤越大 (解决0.6-0.8区间系统性高估)
            # 基础折扣: calibration_discount (联赛参数)
            # 渐进因子: 模型概率/0.5, 在0.5时为1.0, 在0.8时为1.6
            # 联赛乘数: calibration_multiplier (法甲1.8, 意甲1.3, 英超1.2, 其他1.0)
            progressive = 1.0 + max(0.0, (model_p - 0.5) / 0.5) * 0.6
            effective_discount = min(calibration_discount * progressive * calibration_multiplier, 0.40)
            model_weight = 1.0 - effective_discount
            calibrated_p = model_p * model_weight + implied * effective_discount
            value = calibrated_p - implied

            result[outcome] = {
                "model_prob": calibrated_p,
                "implied_prob": implied,
                "value": value,
                "odds": odd,
            }

        return result

    def find_value_opportunities(
        self,
        value_results: Dict[str, Dict[str, float]],
        threshold: float = 0.005,
    ) -> list:
        """
        筛选价值 > 阈值的投注机会。
        返回: 按价值降序排列的 [(outcome, data), ...]
        """
        opportunities = []
        for outcome, data in value_results.items():
            if data["value"] > threshold:
                opportunities.append((outcome, data))
        opportunities.sort(key=lambda x: x[1]["value"], reverse=True)
        return opportunities