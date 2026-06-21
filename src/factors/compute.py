"""
GTO-GameFlow v5.0 因子计算引擎

实现规范文档第4章(F1-F18)、第5章(F19-F32)、第12章(F33-F41)的全部因子计算公式。
"""
import math
import numpy as np
from typing import Dict, Optional, List
from src.config.league_params import LeagueParams, get_league_params
from src.factors.registry import FactorDefinition, get_active_factors, get_factor_weight, is_factor_disabled
from src.factors.factor_inputs import FactorInputs


class FactorComputationEngine:
    """因子计算引擎 — 计算每个因子的 delta 值"""

    def __init__(self, league_id: str, weight_multipliers: Optional[Dict[str, float]] = None):
        self.league_id = league_id
        self.params: LeagueParams = get_league_params(league_id)
        self.active_factors: Dict[str, FactorDefinition] = get_active_factors(league_id)
        # v5.3b: 可选的权重乘数覆盖 (用于校准)
        self.weight_multipliers = weight_multipliers or {}

    def compute_all(
        self,
        elo_diff: float,
        xi_rating: float,
        recent_results: List[float],  # 最近5场结果 [3,3,1,0,1] = 胜胜平负平
        h2h_results: List[float],     # 最近5场交锋 [1,3,0,3,1]
        matches_7d: int,
        rank_diff: int,
        goal_diff: float,
        xg_diff: float,
        market_probs: Dict[str, float],  # {"home": 0.45, "draw": 0.28, "away": 0.27}
        opening_probs: Optional[Dict[str, float]] = None,
        weather: float = 0.0,
        ref_yellow_rate: float = 0.0,
        coach_change_effect: float = 0.0,
        fatigue_penalty: float = 0.0,
        rotation_risk: float = 0.0,
        derby_boost: float = 0.0,
        # 增强因子
        style_matchup_score: float = 0.5,
        streak_momentum: float = 0.0,
        player_form: float = 6.5,
        market_sentiment: float = 0.0,
        odds_std: float = 0.05,
        nlp_sentiment: float = 0.0,
        time_decay_factor: float = 1.0,
        league_strength_bias: float = 0.0,
        poisson_correction: float = 0.0,
        handicap_depth: float = 0.0,
        totals_trend: float = 0.0,
        value_signal: float = 0.0,
        contrarian_signal: float = 0.0,
        market_efficiency: float = 0.0,
        # 联赛特定
        motivation_boost: float = 0.0,
        financial_gap_effect: float = 0.0,
        winter_break_effect: float = 0.0,
        christmas_fatigue: float = 0.0,
        complacency_effect: float = 0.0,
        streak_momentum_league: float = 0.0,
        position_advantage: float = 0.0,
        promoted_team_delta: float = 0.0,
        schedule_advantage: float = 0.0,
        derby_intensity: float = 0.0,
        # v5.10.8: 比赛统计衍生因子
        ht_momentum: float = 0.0,
        shot_eff_diff: float = 0.0,
        territorial_dominance: float = 0.0,
        discipline_index: float = 0.0,
        odds_drift: float = 0.0,
        market_disagreement: float = 0.0,
        referee_home_bias: float = 0.0,
        comeback_resilience: float = 0.0,
        streak_momentum_enriched: float = 0.0,
        goal_volatility: float = 0.0,
        corner_dominance: float = 0.0,
        sot_rate_diff: float = 0.0,
        ah_odds_drift: float = 0.0,
        totals_odds_drift: float = 0.0,
        # v5.11: 平局专属因子
        draw_tactical_matchup: float = 0.0,
        draw_goal_expectancy: float = 0.0,
        draw_team_tendency: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        计算所有因子的 delta 值。
        返回: {factor_id: {"home": delta_home, "draw": delta_draw, "away": delta_away}}
        """
        raw_deltas = {}

        # ================================================
        # 第4章：基础因子 F1-F18
        # ================================================

        # F1: ELO评分 — delta = k × (elo_diff / 400), k=0.5
        f1_k = 0.5
        f1_val = f1_k * (elo_diff / 400.0)
        raw_deltas["F1"] = {"home": f1_val, "draw": 0.0, "away": -f1_val}

        # F2: 核心伤停 — delta = (xi_rating - 6.0) × 0.08
        f2_val = (xi_rating - 6.0) * 0.08
        raw_deltas["F2"] = {"home": f2_val, "draw": 0.0, "away": -f2_val}

        # F3: 近期状态 — delta = (ewma_form - 1.5) × 0.15
        # ewma: 指数加权移动平均, α = time_decay_alpha
        f3_alpha = self.params.time_decay_alpha
        f3_ewma = self._compute_ewma(recent_results, f3_alpha)
        f3_val = (f3_ewma - 1.5) * 0.15
        raw_deltas["F3"] = {"home": f3_val, "draw": 0.0, "away": -f3_val}

        # F4: 主客场 — delta_home = home_advantage × 0.1
        f4_val = self.params.home_advantage * 0.1
        raw_deltas["F4"] = {"home": f4_val, "draw": 0.0, "away": 0.0}

        # F5: 历史交锋 — delta = h2h_advantage × 0.08
        f5_h2h = self._compute_h2h_advantage(h2h_results)
        f5_val = f5_h2h * 0.08
        raw_deltas["F5"] = {"home": f5_val, "draw": 0.0, "away": -f5_val}

        # F6: 赛程密度 — delta = -(matches_7d - 1) × 0.03
        f6_val = -(matches_7d - 1) * 0.03
        raw_deltas["F6"] = {"home": f6_val, "draw": 0.0, "away": -f6_val}

        # F7: 联赛排名差 — delta = (rank_diff / 20) × 0.1
        # rank_diff = rank_away - rank_home (正值=主队排名更靠前)
        f7_val = (rank_diff / 20.0) * 0.1
        raw_deltas["F7"] = {"home": f7_val, "draw": 0.0, "away": -f7_val}

        # F8: 进球/失球差 — delta = (goal_diff / 10) × 0.06
        f8_val = (goal_diff / 10.0) * 0.06
        raw_deltas["F8"] = {"home": f8_val, "draw": 0.0, "away": -f8_val}

        # F9: xG差值 — delta = xG_diff × 0.12
        f9_val = xg_diff * 0.12
        raw_deltas["F9"] = {"home": f9_val, "draw": 0.0, "away": -f9_val}

        # F10: 赔率隐含概率 — delta = logit(market_prob) - logit(0.33)
        f10_home = market_probs.get("home", 0.33)
        f10_draw = market_probs.get("draw", 0.33)
        f10_away = market_probs.get("away", 0.33)
        raw_deltas["F10"] = {
            "home": self._logit(f10_home) - self._logit(0.33),
            "draw": self._logit(f10_draw) - self._logit(0.33),
            "away": self._logit(f10_away) - self._logit(0.33),
        }

        # F11: 市场赔率变动 — delta = (opening_prob - current_prob) × 0.5
        if opening_probs:
            f11_h = (opening_probs.get("home", f10_home) - f10_home) * 0.5
            f11_d = (opening_probs.get("draw", f10_draw) - f10_draw) * 0.5
            f11_a = (opening_probs.get("away", f10_away) - f10_away) * 0.5
        else:
            f11_h = f11_d = f11_a = 0.0
        raw_deltas["F11"] = {"home": f11_h, "draw": f11_d, "away": f11_a}

        # F12: 天气影响 — delta = weather_impact × 0.08
        f12_val = weather * 0.08
        raw_deltas["F12"] = {"home": 0.0, "draw": f12_val, "away": 0.0}

        # F13: 裁判风格 — delta = (ref_yellow_rate - league_avg) × 0.02
        f13_val = (ref_yellow_rate - self.params.yellow_card_rate) * 0.02
        raw_deltas["F13"] = {"home": f13_val, "draw": 0.0, "away": 0.0}

        # F14: 已废弃，跳过
        raw_deltas["F14"] = {"home": 0.0, "draw": 0.0, "away": 0.0}

        # F15: 教练更替 — delta = coach_change_effect × 0.07
        f15_val = coach_change_effect * 0.07
        raw_deltas["F15"] = {"home": f15_val, "draw": 0.0, "away": -f15_val}

        # F16: 欧战影响 — delta = fatigue_penalty × 0.04
        f16_val = fatigue_penalty * 0.04
        raw_deltas["F16"] = {"home": f16_val, "draw": 0.0, "away": -f16_val}

        # F17: 轮换预测 — delta = rotation_risk × 0.05
        f17_val = rotation_risk * 0.05
        raw_deltas["F17"] = {"home": f17_val, "draw": 0.0, "away": -f17_val}

        # F18: 德比战 — delta = derby_boost × 0.06
        f18_val = derby_boost * 0.06
        raw_deltas["F18"] = {"home": 0.0, "draw": f18_val, "away": 0.0}

        # ================================================
        # 第5章：通用增强因子 F19-F32
        # ================================================

        # F19: 攻击/防守风格 — delta = (style_matchup_score - 0.5) × 0.10
        f19_val = (style_matchup_score - 0.5) * 0.10
        raw_deltas["F19"] = {"home": f19_val, "draw": 0.0, "away": -f19_val}

        # F20: 连胜/连败动量 — delta = streak_momentum × 0.08
        f20_val = streak_momentum * 0.08
        raw_deltas["F20"] = {"home": f20_val, "draw": 0.0, "away": -f20_val}

        # F21: 核心球员状态 — delta = (player_form - 6.5) × 0.06
        f21_val = (player_form - 6.5) * 0.06
        raw_deltas["F21"] = {"home": f21_val, "draw": 0.0, "away": -f21_val}

        # F22: 市场情绪 — delta = (market_sentiment - 0) × 0.04
        f22_val = market_sentiment * 0.04
        raw_deltas["F22"] = {"home": f22_val, "draw": 0.0, "away": -f22_val}

        # F23: 赔率离散度 — delta = (odds_std - 0.05) × 0.5
        f23_val = (odds_std - 0.05) * 0.5
        raw_deltas["F23"] = {"home": 0.0, "draw": f23_val, "away": 0.0}

        # F24: 新闻NLP — delta = sentiment_score × 0.04
        f24_val = nlp_sentiment * 0.04
        raw_deltas["F24"] = {"home": f24_val, "draw": 0.0, "away": -f24_val}

        # F25: 时间衰减加权 — delta = time_decay_correction × 0.05
        f25_val = time_decay_factor * 0.05
        raw_deltas["F25"] = {"home": f25_val, "draw": 0.0, "away": -f25_val}

        # F26: 联赛强度调整 — delta = league_strength_bias × 0.05
        f26_val = league_strength_bias * 0.05
        raw_deltas["F26"] = {"home": f26_val, "draw": 0.0, "away": -f26_val}

        # F27: 进球分布修正 — delta = poisson_correction × 0.20
        f27_val = poisson_correction * 0.20
        raw_deltas["F27"] = {"home": f27_val, "draw": f27_val, "away": f27_val}

        # F28: 亚盘深度 — delta = handicap_depth × 0.06
        f28_val = handicap_depth * 0.06
        raw_deltas["F28"] = {"home": f28_val, "draw": 0.0, "away": -f28_val}

        # F29: 大小球趋势 — delta = totals_trend × 0.07
        f29_val = totals_trend * 0.07
        raw_deltas["F29"] = {"home": f29_val, "draw": 0.0, "away": f29_val}

        # F30: 赔率价值信号 — delta = value_signal × 0.10
        f30_val = value_signal * 0.10
        raw_deltas["F30"] = {"home": f30_val, "draw": f30_val, "away": f30_val}

        # F31: 反市场偏差 — delta = contrarian_signal × 0.05
        f31_val = contrarian_signal * 0.05
        raw_deltas["F31"] = {"home": f31_val, "draw": 0.0, "away": f31_val}

        # F32: 市场效率评分 — delta = market_efficiency × 0.03
        f32_val = market_efficiency * 0.03
        raw_deltas["F32"] = {"home": f32_val, "draw": f32_val, "away": f32_val}

        # ================================================
        # 第12章：联赛特定因子 F33-F41 (v5.5.1: F42已合并到F18)
        # ================================================

        # F33: 保级/争冠动力 — delta = motivation_boost / 100
        f33_val = motivation_boost / 100.0
        raw_deltas["F33"] = {"home": f33_val, "draw": 0.0, "away": 0.0}

        # F34: 财力差距 — delta = financial_gap_effect / 100
        f34_val = financial_gap_effect / 100.0
        raw_deltas["F34"] = {"home": f34_val, "draw": 0.0, "away": -f34_val}

        # F35: 冬歇期效应 — delta = winter_break_effect × 0.10
        f35_val = winter_break_effect * 0.10
        raw_deltas["F35"] = {"home": f35_val, "draw": f35_val, "away": -f35_val}

        # F36: 圣诞赛程 — delta = christmas_fatigue × 0.12
        f36_val = christmas_fatigue * 0.12
        raw_deltas["F36"] = {"home": -f36_val, "draw": f36_val, "away": 0.0}

        # F37: 中游无欲 — delta = complacency_effect × 0.05
        f37_val = complacency_effect * 0.05
        raw_deltas["F37"] = {"home": -f37_val, "draw": f37_val, "away": f37_val}

        # F38: 连胜/连败(联赛特定) — delta = streak_momentum × 0.08
        f38_val = streak_momentum_league * 0.08
        raw_deltas["F38"] = {"home": f38_val, "draw": 0.0, "away": -f38_val}

        # F39: 积分榜 — delta = (position_advantage / 20) × 0.12
        f39_val = (position_advantage / 20.0) * 0.12
        raw_deltas["F39"] = {"home": f39_val, "draw": 0.0, "away": -f39_val}

        # F40: 升班马数据 — delta = promoted_team_delta × 0.08
        f40_val = promoted_team_delta * 0.08
        raw_deltas["F40"] = {"home": f40_val, "draw": 0.0, "away": -f40_val}

        # F41: 赛程优势 — delta = schedule_advantage × 0.08
        f41_val = schedule_advantage * 0.08
        raw_deltas["F41"] = {"home": f41_val, "draw": 0.0, "away": -f41_val}

        # v5.5.1: F42 (德比战强度) 已合并到 F18 (德比战), 不再单独计算
        # F42 的 derby_intensity 参数保留兼容性，但不再生成独立因子

        # ================================================
        # v5.10.8: 第13章 — 比赛统计衍生因子 F42-F55
        # ================================================

        # F42: 半场动量 — delta = ht_momentum × 0.08
        f42_val = ht_momentum * 0.08
        raw_deltas["F42"] = {"home": f42_val, "draw": 0.0, "away": -f42_val}

        # F43: 射门效率差 — delta = shot_eff_diff × 0.06
        f43_val = shot_eff_diff * 0.06
        raw_deltas["F43"] = {"home": f43_val, "draw": 0.0, "away": -f43_val}

        # F44: 控场优势 — delta = territorial_dominance × 0.06
        f44_val = territorial_dominance * 0.06
        raw_deltas["F44"] = {"home": f44_val, "draw": 0.0, "away": -f44_val}

        # F45: 纪律指数 — delta = discipline_index × 0.05
        f45_val = discipline_index * 0.05
        raw_deltas["F45"] = {"home": f45_val, "draw": 0.0, "away": -f45_val}

        # F46: 赔率漂移信号 — delta = odds_drift × 0.10
        f46_val = odds_drift * 0.10
        raw_deltas["F46"] = {"home": f46_val, "draw": 0.0, "away": -f46_val}

        # F47: 市场分歧 — delta = market_disagreement × 0.06
        f47_val = market_disagreement * 0.06
        raw_deltas["F47"] = {"home": 0.0, "draw": f47_val, "away": 0.0}

        # F48: 裁判主场偏置 — delta = referee_home_bias × 0.08
        f48_val = referee_home_bias * 0.08
        raw_deltas["F48"] = {"home": f48_val, "draw": 0.0, "away": -f48_val}

        # F49: 逆转韧性 — delta = comeback_resilience × 0.06
        f49_val = comeback_resilience * 0.06
        raw_deltas["F49"] = {"home": f49_val, "draw": 0.0, "away": -f49_val}

        # F50: 连胜动量(增强) — delta = streak_momentum_enriched × 0.08
        f50_val = streak_momentum_enriched * 0.08
        raw_deltas["F50"] = {"home": f50_val, "draw": 0.0, "away": -f50_val}

        # F51: 进球波动率 — delta = goal_volatility × 0.05
        f51_val = goal_volatility * 0.05
        raw_deltas["F51"] = {"home": 0.0, "draw": f51_val, "away": 0.0}

        # F52: 角球优势 — delta = corner_dominance × 0.05
        f52_val = corner_dominance * 0.05
        raw_deltas["F52"] = {"home": f52_val, "draw": 0.0, "away": -f52_val}

        # F53: 射正率差 — delta = sot_rate_diff × 0.06
        f53_val = sot_rate_diff * 0.06
        raw_deltas["F53"] = {"home": f53_val, "draw": 0.0, "away": -f53_val}

        # F54: 亚盘赔率漂移 — delta = ah_odds_drift × 0.08
        f54_val = ah_odds_drift * 0.08
        raw_deltas["F54"] = {"home": f54_val, "draw": 0.0, "away": -f54_val}

        # F55: 大小球赔率漂移 — delta = totals_odds_drift × 0.06
        f55_val = totals_odds_drift * 0.06
        raw_deltas["F55"] = {"home": f55_val, "draw": 0.0, "away": f55_val}

        # ================================================
        # v5.11: 平局专属因子 F56-F58
        # ================================================

        # F56: 战术风格平局倾向 — delta = draw_tactical_matchup × 0.08
        f56_val = draw_tactical_matchup * 0.08
        raw_deltas["F56"] = {"home": 0.0, "draw": f56_val, "away": 0.0}

        # F57: 进球预期平局信号 — delta = draw_goal_expectancy × 0.10
        f57_val = draw_goal_expectancy * 0.10
        raw_deltas["F57"] = {"home": 0.0, "draw": f57_val, "away": 0.0}

        # F58: 球队平局历史倾向 — delta = draw_team_tendency × 0.06
        f58_val = draw_team_tendency * 0.06
        raw_deltas["F58"] = {"home": 0.0, "draw": f58_val, "away": 0.0}

        return self._apply_weights(raw_deltas)

    def _apply_weights(self, raw_deltas: Dict) -> Dict[str, Dict[str, float]]:
        """应用联赛特定权重、互斥逻辑和叠加上限"""
        weighted = {}
        # 处理 F20/F38 互斥: 优先使用 F38
        f38_w = get_factor_weight("F38", self.league_id)
        f20_w = get_factor_weight("F20", self.league_id)
        use_f38 = f38_w > 0 and f20_w > 0

        for fid, deltas in raw_deltas.items():
            # v6.0: 跳过禁用的因子 (如 F10/F11/F14)
            if is_factor_disabled(fid):
                continue
            if fid == "F20" and use_f38:
                continue  # F38 优先，禁用 F20

            weight = get_factor_weight(fid, self.league_id)
            if weight == 0.0:
                continue

            # v5.3b: 应用权重乘数 (用于校准)
            multiplier = self.weight_multipliers.get(fid, 1.0)
            effective_weight = weight * multiplier

            weighted[fid] = {
                "home": deltas["home"] * effective_weight,
                "draw": deltas["draw"] * effective_weight,
                "away": deltas["away"] * effective_weight,
            }

        # F6 + F36 叠加上限: 合计调整上限为 -8% (规范第12.1节)
        if "F6" in weighted and "F36" in weighted:
            for axis in ("home", "draw", "away"):
                # 规范: effective_delta = max(-0.08, F36_delta + clamp(F6_delta, -0.08, 0))
                f6_clamped = max(-0.08, min(0.0, weighted["F6"][axis]))
                combined = weighted["F36"][axis] + f6_clamped
                capped = max(-0.08, combined)
                # 按比例分配回 F6 和 F36
                if abs(combined) > 0.0001 and combined != capped:
                    ratio = capped / combined
                    weighted["F36"][axis] *= ratio
                    weighted["F6"][axis] = f6_clamped * ratio

        return weighted

    # ============================================================
    # 辅助函数
    # ============================================================

    @staticmethod
    def _compute_ewma(results: List[float], alpha: float) -> float:
        """指数加权移动平均 (胜=3, 平=1, 负=0)"""
        if not results:
            return 1.5
        ewma = 0.0
        weight_sum = 0.0
        for i, r in enumerate(results):
            w = (1 - alpha) ** i
            ewma += r * w
            weight_sum += w
        return ewma / weight_sum if weight_sum > 0 else 1.5

    @staticmethod
    def _compute_h2h_advantage(results: List[float]) -> float:
        """计算历史交锋优势 (近期权重更高)"""
        if not results:
            return 0.0
        advantage = 0.0
        for i, r in enumerate(results):
            w = 1.0 / (i + 1)  # 线性衰减
            # 胜=+1, 平=0, 负=-1
            if r == 3:
                advantage += w
            elif r == 0:
                advantage -= w
        return advantage / max(1, len(results))

    def compute_all_from_inputs(
        self,
        inputs: FactorInputs,
    ) -> Dict[str, Dict[str, float]]:
        """
        从 FactorInputs 数据类计算所有因子。

n        这是 compute_all() 的新接口，参数通过 FactorInputs 传入，
        避免58个参数的函数签名问题。

        增减因子只需修改 FactorInputs 和 registry.py，不会参数错位。
        """
        return self.compute_all(
            elo_diff=inputs.elo_diff,
            xi_rating=inputs.xi_rating,
            recent_results=inputs.recent_results,
            h2h_results=inputs.h2h_results,
            matches_7d=inputs.matches_7d,
            rank_diff=inputs.rank_diff,
            goal_diff=inputs.goal_diff,
            xg_diff=inputs.xg_diff,
            market_probs=inputs.market_probs,
            opening_probs=inputs.opening_probs,
            weather=inputs.weather,
            ref_yellow_rate=inputs.ref_yellow_rate,
            coach_change_effect=inputs.coach_change_effect,
            fatigue_penalty=inputs.fatigue_penalty,
            rotation_risk=inputs.rotation_risk,
            derby_boost=inputs.derby_boost,
            style_matchup_score=inputs.style_matchup_score,
            streak_momentum=inputs.streak_momentum,
            player_form=inputs.player_form,
            market_sentiment=inputs.market_sentiment,
            odds_std=inputs.odds_std,
            nlp_sentiment=inputs.nlp_sentiment,
            time_decay_factor=inputs.time_decay_factor,
            league_strength_bias=inputs.league_strength_bias,
            poisson_correction=inputs.poisson_correction,
            handicap_depth=inputs.handicap_depth,
            totals_trend=inputs.totals_trend,
            value_signal=inputs.value_signal,
            contrarian_signal=inputs.contrarian_signal,
            market_efficiency=inputs.market_efficiency,
            motivation_boost=inputs.motivation_boost,
            financial_gap_effect=inputs.financial_gap_effect,
            winter_break_effect=inputs.winter_break_effect,
            christmas_fatigue=inputs.christmas_fatigue,
            complacency_effect=inputs.complacency_effect,
            streak_momentum_league=inputs.streak_momentum_league,
            position_advantage=inputs.position_advantage,
            promoted_team_delta=inputs.promoted_team_delta,
            schedule_advantage=inputs.schedule_advantage,
            derby_intensity=inputs.derby_intensity,
            ht_momentum=inputs.ht_momentum,
            shot_eff_diff=inputs.shot_eff_diff,
            territorial_dominance=inputs.territorial_dominance,
            discipline_index=inputs.discipline_index,
            odds_drift=inputs.odds_drift,
            market_disagreement=inputs.market_disagreement,
            referee_home_bias=inputs.referee_home_bias,
            comeback_resilience=inputs.comeback_resilience,
            streak_momentum_enriched=inputs.streak_momentum_enriched,
            goal_volatility=inputs.goal_volatility,
            corner_dominance=inputs.corner_dominance,
            sot_rate_diff=inputs.sot_rate_diff,
            ah_odds_drift=inputs.ah_odds_drift,
            totals_odds_drift=inputs.totals_odds_drift,
            draw_tactical_matchup=inputs.draw_tactical_matchup,
            draw_goal_expectancy=inputs.draw_goal_expectancy,
            draw_team_tendency=inputs.draw_team_tendency,
        )

    @staticmethod
    def _logit(p: float) -> float:
        """logit 函数: log(p/(1-p))"""
        p = max(0.001, min(0.999, p))
        return math.log(p / (1.0 - p))


def compute_factors_from_context(
    league_id: str,
    match_data: Dict,
) -> Dict[str, Dict[str, float]]:
    """
    便捷函数：从 MatchContext 字典计算所有因子。
    match_data 应包含所有计算所需的键值。
    """
    engine = FactorComputationEngine(league_id)
    return engine.compute_all(
        elo_diff=match_data.get("elo_diff", 0.0),
        xi_rating=match_data.get("xi_rating", 6.0),
        recent_results=match_data.get("recent_results", [1.5, 1.5, 1.5, 1.5, 1.5]),
        h2h_results=match_data.get("h2h_results", [0, 0, 0, 0, 0]),
        matches_7d=match_data.get("matches_7d", 1),
        rank_diff=match_data.get("rank_diff", 0),
        goal_diff=match_data.get("goal_diff", 0.0),
        xg_diff=match_data.get("xg_diff", 0.0),
        market_probs=match_data.get("market_probs", {"home": 0.33, "draw": 0.33, "away": 0.33}),
        opening_probs=match_data.get("opening_probs", None),
        weather=match_data.get("weather", 0.0),
        ref_yellow_rate=match_data.get("ref_yellow_rate", 0.0),
        coach_change_effect=match_data.get("coach_change_effect", 0.0),
        fatigue_penalty=match_data.get("fatigue_penalty", 0.0),
        rotation_risk=match_data.get("rotation_risk", 0.0),
        derby_boost=match_data.get("derby_boost", 0.0),
        style_matchup_score=match_data.get("style_matchup_score", 0.5),
        streak_momentum=match_data.get("streak_momentum", 0.0),
        player_form=match_data.get("player_form", 6.5),
        market_sentiment=match_data.get("market_sentiment", 0.0),
        odds_std=match_data.get("odds_std", 0.05),
        nlp_sentiment=match_data.get("nlp_sentiment", 0.0),
        time_decay_factor=match_data.get("time_decay_factor", 1.0),
        league_strength_bias=match_data.get("league_strength_bias", 0.0),
        poisson_correction=match_data.get("poisson_correction", 0.0),
        handicap_depth=match_data.get("handicap_depth", 0.0),
        totals_trend=match_data.get("totals_trend", 0.0),
        value_signal=match_data.get("value_signal", 0.0),
        contrarian_signal=match_data.get("contrarian_signal", 0.0),
        market_efficiency=match_data.get("market_efficiency", 0.0),
        motivation_boost=match_data.get("motivation_boost", 0.0),
        financial_gap_effect=match_data.get("financial_gap_effect", 0.0),
        winter_break_effect=match_data.get("winter_break_effect", 0.0),
        christmas_fatigue=match_data.get("christmas_fatigue", 0.0),
        complacency_effect=match_data.get("complacency_effect", 0.0),
        streak_momentum_league=match_data.get("streak_momentum_league", 0.0),
        position_advantage=match_data.get("position_advantage", 0.0),
        promoted_team_delta=match_data.get("promoted_team_delta", 0.0),
        schedule_advantage=match_data.get("schedule_advantage", 0.0),
        derby_intensity=match_data.get("derby_intensity", 0.0),
        # v5.11: 平局专属因子
        draw_tactical_matchup=match_data.get("draw_tactical_matchup", 0.0),
        draw_goal_expectancy=match_data.get("draw_goal_expectancy", 0.0),
        draw_team_tendency=match_data.get("draw_team_tendency", 0.0),
    )