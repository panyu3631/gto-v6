"""
GTO-GameFlow v5.10.8 九阶段计算流水线编排器

v5.10.8 核心策略:
- 已验证的因子引擎 (55因子, 经过回测调优)
- 固定5pp裁剪
- 统一贝叶斯收缩
- 双域概率融合
"""
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    ProbabilityDistribution, BankrollState, CircuitBreakerState, ScoreMatrix,
)
from src.config.settings import config
from src.config.league_params import get_league_params
from src.factors.compute import FactorComputationEngine  # v5.10.8 已验证引擎
from src.factors.heterogeneous_groups import group_signal_cap  # v5.10.8: 异质化分组信号上限
from src.engine.probability import ProbabilityEngine
from src.engine.bankroll import BankrollManager, generate_bet_proposals, compute_confidence
from src.engine.risk_control import RiskController
from src.engine.unified_bayesian_shrinkage import UnifiedBayesianShrinkage, create_shrinkage_for_league



@dataclass
class PipelineResult:
    """流水线执行结果"""
    match_id: str
    league_id: str
    # Stage 1-5 输出
    factor_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_factor_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)  # v5.10.8: LASSO前的原始因子
    pre_shrinkage_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)  # v5.10.8: LASSO后、正交化前
    logits: Dict[str, float] = field(default_factory=dict)
    logit_probs: Optional[ProbabilityDistribution] = None
    poisson_probs: Optional[ProbabilityDistribution] = None
    poisson_score_matrix: Optional[ScoreMatrix] = None   # v5.5: 泊松比分矩阵，供多策略使用
    fused_probs: Optional[ProbabilityDistribution] = None
    value_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Stage 6-9 输出
    proposals: List[BetProposal] = field(default_factory=list)
    placements: List[BetPlacement] = field(default_factory=list)
    # 状态
    bankroll_state: Optional[BankrollState] = None
    circuit_broken: bool = False
    circuit_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class GameFlowPipeline:
    """
    九阶段计算流水线。

    Stage 1: factor_calculation     — 41因子delta计算
    Stage 2: logit_accumulation     — delta累加到logit空间
    Stage 3: sigmoid_normalization  — softmax转换回概率
    Stage 4: poisson_bridge         — 泊松模型独立计算 + Dual-Domain融合
    Stage 5: value_calculation      — 模型概率 vs 赔率隐含概率
    Stage 6: priority_sorting       — 价值排序 + 优先级评分
    Stage 7: bankroll_allocation    — Kelly资金分配
    Stage 8: risk_control           — 风控校验 (单注/日/周限额/相关性)
    Stage 9: circuit_breaker        — 熔断判断
    """

    def __init__(self, league_id: str, initial_bankroll: float = None,
                 weight_multipliers: Optional[Dict[str, float]] = None):
        self.league_id = league_id
        self.params = get_league_params(league_id)
        # v5.10.8: 使用已验证的因子引擎
        self.factor_engine = FactorComputationEngine(league_id, weight_multipliers)
        self.prob_engine = ProbabilityEngine(league_id)
        self.bankroll_mgr = BankrollManager(initial_bankroll, fixed_base=True)
        self.risk_ctrl = RiskController()
        self._shared_bankroll = False
        self.unified_shrinkage = create_shrinkage_for_league(league_id)
        self.lasso_weights: Optional[Dict[str, float]] = None

        # v5.10.8: 联赛独立核心参数 (从 league_params 读取)
        self.factor_scale = self.params.factor_scale
        self.fusion_weight = self.params.fusion_weight
        self.calibration_discount = self.params.calibration_discount
        self.confidence_threshold = self.params.confidence_threshold

    def set_bankroll_manager(self, mgr: "BankrollManager"):
        """注入共享资金管理器 (用于跨联赛全局回测)"""
        self.bankroll_mgr = mgr
        self._shared_bankroll = True

    def run_stages_1_5(
        self,
        match: MatchContext,
        extra_data: Optional[Dict] = None,
    ) -> PipelineResult:
        """执行 Stage 1-5: 因子计算 → 概率融合 → 价值评估"""
        result = PipelineResult(match_id=match.match_id, league_id=match.league_id)
        extra = extra_data or {}

        try:
            # === Stage 1: factor_calculation ===
            elo_diff = extra.get("elo_diff", match.home_elo - match.away_elo)

            # v5.10.8: 派生因子输入
            recent_results = extra.get("recent_results", [1.5, 1.5, 1.5, 1.5, 1.5])
            xi_rating = extra.get("xi_rating", 6.0 + (match.home_elo - 1500) / 200.0)
            avg_form = sum(recent_results) / max(len(recent_results), 1)
            player_form = extra.get("player_form", 5.0 + avg_form)
            style_matchup_score = extra.get("style_matchup_score", 0.5 + 0.2 * (1.0 - min(abs(elo_diff), 200) / 200.0))
            time_decay_factor = extra.get("time_decay_factor", 1.0)
            opening_probs = extra.get("opening_probs", None)
            if opening_probs:
                market_sentiment = extra.get("market_sentiment",
                    (opening_probs.get("home", 0.33) - 1.0/max(match.odds_home, 1.01)) * 0.5)
            else:
                market_sentiment = extra.get("market_sentiment", 0.0)
            # weather/derby 派生
            weather = extra.get("weather", 0.0)
            if abs(weather) < 0.001:
                seed = int(hashlib.md5(f"{match.home_team}{match.away_team}{match.kickoff_time}".encode()).hexdigest()[:8], 16)
                weather = ((seed % 100) / 100.0 - 0.5) * 0.1
            derby_boost = extra.get("derby_boost", 0.0)

            season_phase = extra.get("match_phase", 0.5)
            odds_std = extra.get("odds_std", 0.05)

            _mph = 1.0 / match.odds_home if match.odds_home > 0 else 0.33
            _mpd = 1.0 / match.odds_draw if match.odds_draw > 0 else 0.33
            _mpa = 1.0 / match.odds_away if match.odds_away > 0 else 0.33
            _mpt = _mph + _mpd + _mpa
            market_probs_norm = (_mph / _mpt, _mpd / _mpt, _mpa / _mpt)

            # v5.10.8: 调用已验证的因子引擎 (55因子)
            result.factor_deltas = self.factor_engine.compute_all(
                elo_diff=elo_diff,
                xi_rating=xi_rating,
                recent_results=recent_results,
                h2h_results=extra.get("h2h_results", [0, 0, 0, 0, 0]),
                matches_7d=extra.get("matches_7d", 1),
                rank_diff=extra.get("rank_diff", 0),
                goal_diff=extra.get("goal_diff", 0.0),
                xg_diff=extra.get("xg_diff", 0.0),
                player_form=player_form,
                style_matchup_score=style_matchup_score,
                time_decay_factor=time_decay_factor,
                market_sentiment=market_sentiment,
                weather=weather,
                derby_boost=derby_boost,
                market_probs={
                    "home": _mph / _mpt,
                    "draw": _mpd / _mpt,
                    "away": _mpa / _mpt,
                },
                opening_probs=opening_probs,
                # 传递所有v5.10兼容参数
                **{k: extra.get(k, 0.0) for k in [
                    "ref_yellow_rate", "coach_change_effect",
                    "fatigue_penalty", "rotation_risk", "streak_momentum",
                    "odds_std", "nlp_sentiment",
                    "league_strength_bias", "poisson_correction",
                    "handicap_depth", "totals_trend", "value_signal",
                    "contrarian_signal", "market_efficiency",
                    "motivation_boost", "financial_gap_effect",
                    "winter_break_effect", "christmas_fatigue",
                    "complacency_effect", "streak_momentum_league",
                    "position_advantage", "promoted_team_delta",
                    "schedule_advantage", "derby_intensity",
                    "ht_momentum", "shot_eff_diff", "territorial_dominance",
                    "discipline_index", "odds_drift", "market_disagreement",
                    "referee_home_bias", "comeback_resilience",
                    "streak_momentum_enriched", "goal_volatility",
                    "corner_dominance", "sot_rate_diff",
                    "ah_odds_drift", "totals_odds_drift",
                    # v5.11: 平局专属因子
                    "draw_tactical_matchup", "draw_goal_expectancy", "draw_team_tendency",
                ]},
            )
            result.raw_factor_deltas = dict(result.factor_deltas)

            # v5.10.8: 异质化分组信号上限 — 防止同质因子共振放大
            # 35/55因子测量"强弱"方向, 组内总信号限制在1.5倍单个因子强度
            capped_factors = group_signal_cap(
                result.factor_deltas, cap_factor=1.5
            )
            result.factor_deltas = capped_factors

            # v5.10.4: LASSO权重过滤
            if self.lasso_weights:
                result.factor_deltas = {k: v for k, v in result.factor_deltas.items() if self.lasso_weights.get(k, 0.0) != 0.0}

            result.pre_shrinkage_deltas = dict(result.factor_deltas)
            shrinkage_result = self.unified_shrinkage.process(
                factor_deltas=result.factor_deltas,
                market_probs=market_probs_norm,
                elo_diff=elo_diff,
            )
            result.factor_deltas = shrinkage_result.orthogonal_deltas
            result.unscaled_factor_deltas = dict(result.factor_deltas)

            if self.factor_scale != 1.0:
                result.factor_deltas = {
                    fid: {k: v * self.factor_scale for k, v in d.items()}
                    for fid, d in result.factor_deltas.items()
                }
            self._v510_alpha_used = shrinkage_result.alpha_used

            # v5.10.9: 赔率漂移信号 — 检测开盘→即时赔率变动
            # steam move: 赔率朝一个方向持续移动, 可能是"聪明钱"信号
            opening_probs = extra.get("opening_probs", None)
            if opening_probs and opening_probs.get("draw", 0) > 0:
                op_total = sum(1.0/max(opening_probs.get(d, 3.0), 1.01) for d in ["home", "draw", "away"])
                op_implied = {d: (1.0/max(opening_probs.get(d, 3.0), 1.01))/op_total for d in ["home", "draw", "away"]}
                drift = {}
                for i, direction in enumerate(["home", "draw", "away"]):
                    drift[direction] = market_probs_norm[i] - op_implied[direction]

                # 检测 steam move: 任一方向漂移 > 0.03 (3pp)
                steam_direction = max(drift, key=lambda d: abs(drift[d]))
                steam_magnitude = abs(drift[steam_direction])

                if steam_magnitude > 0.03:
                    # v5.11: 三方向归一化 — 漂移从其他方向扣除
                    # 避免总概率偏移
                    drift_amount = drift[steam_direction] * 0.5
                    drift_factor = {d: -drift_amount / 2.0 for d in ["home", "draw", "away"]}
                    drift_factor[steam_direction] = drift_amount
                    # 归一化: 总和应为 0
                    drift_total = sum(drift_factor.values())
                    if abs(drift_total) > 0.0001:
                        drift_factor[steam_direction] -= drift_total
                    result.factor_deltas["OD_DRIFT"] = drift_factor

            # === Stage 2: logit_accumulation ===
            market_probs = {
                "home": _mph / _mpt,
                "draw": _mpd / _mpt,
                "away": _mpa / _mpt,
            }
            result.logits = self.prob_engine.logit_accumulation(
                market_probs, result.factor_deltas, uniform_prior=False,
                factor_weights=None,
            )

            # === Stage 3: sigmoid_normalization ===
            temperature = 1.0 - 0.2 * (1.0 - season_phase)
            result.logit_probs = self.prob_engine.sigmoid_normalization(
                result.logits, temperature=temperature
            )

            # === Stage 4: poisson_bridge + Dual-Domain Fusion ===
            # v5.10.8: fusion_weight 从联赛参数读取
            fusion_weight = self.fusion_weight

            result.poisson_probs, result.poisson_score_matrix = self.prob_engine.poisson_bridge(
                home_elo=match.home_elo,
                away_elo=match.away_elo,
                factor_deltas=result.unscaled_factor_deltas,
            )
            result.fused_probs = self.prob_engine.dual_domain_fusion(
                result.logit_probs, result.poisson_probs, fusion_weight=fusion_weight,
                data_quality=getattr(match, 'data_quality', None),
                odds_std=getattr(match, 'odds_std', None),
            )

            # === v5.10.8: 平局校准 (在裁剪之前) ===
            raw_probs = [
                result.fused_probs.prob_home,
                result.fused_probs.prob_draw,
                result.fused_probs.prob_away,
            ]
            mkt_list = [market_probs_norm[0], market_probs_norm[1], market_probs_norm[2]]

            calibrated_probs = self._calibrate_draw(raw_probs, elo_diff, mkt_list)
            result.fused_probs = ProbabilityDistribution(
                prob_home=calibrated_probs[0],
                prob_draw=calibrated_probs[1],
                prob_away=calibrated_probs[2],
            )

            # === v5.10.8: 固定5pp裁剪 ===
            clipped = [max(mkt_list[i] - 0.05, min(mkt_list[i] + 0.05, calibrated_probs[i])) for i in range(3)]
            total = sum(clipped)
            if total > 0:
                clipped = [c / total for c in clipped]
            result.fused_probs = ProbabilityDistribution(
                prob_home=clipped[0],
                prob_draw=clipped[1],
                prob_away=clipped[2],
            )

            # v5.10.8: 平局增强
            draw_boost = self._maybe_boost_draw(clipped, mkt_list)
            if draw_boost is not None:
                result.fused_probs = ProbabilityDistribution(
                    prob_home=draw_boost[0],
                    prob_draw=draw_boost[1],
                    prob_away=draw_boost[2],
                )

            # v5.10.9: 联赛差异化渐进式校准 — 在score_matrix之前应用
            # 这样亚盘/大小球也能受益于校准 (之前仅1x2价值计算受益)
            # 法甲calibration_multiplier=1.5, 意甲=1.1, 其他=1.0
            calib_mult = self.params.calibration_multiplier
            if calib_mult > 1.0:
                post_clip = [
                    result.fused_probs.prob_home,
                    result.fused_probs.prob_draw,
                    result.fused_probs.prob_away,
                ]
                for i in range(3):
                    model_p = post_clip[i]
                    implied_p = mkt_list[i]
                    progressive = 1.0 + max(0.0, (model_p - 0.5) / 0.5) * 0.6
                    effective_discount = min(
                        self.calibration_discount * progressive * calib_mult, 0.40
                    )
                    model_weight = 1.0 - effective_discount
                    post_clip[i] = model_p * model_weight + implied_p * effective_discount
                total = sum(post_clip)
                if total > 0:
                    post_clip = [c / total for c in post_clip]
                result.fused_probs = ProbabilityDistribution(
                    prob_home=post_clip[0],
                    prob_draw=post_clip[1],
                    prob_away=post_clip[2],
                )

            # v5.10.10: 强队客场定向校准 — 直接降低客场概率, 而非拉向市场
            # 分析确认: 法甲强队客场系统性高估, 市场也有同样偏见
            # 策略: 直接惩罚客场概率 → 差额按比例重分配给主/平
            # elo_diff = home_elo - away_elo, 强队客场时 elo_diff < -100
            strong_away_penalty = getattr(self.params, 'strong_away_penalty', 0.0)
            if strong_away_penalty > 0 and elo_diff < -100:
                elo_scale = min((-elo_diff - 100) / 300, 1.0)
                penalty = strong_away_penalty * elo_scale
                post_clip = [
                    result.fused_probs.prob_home,
                    result.fused_probs.prob_draw,
                    result.fused_probs.prob_away,
                ]
                # 直接降低客场概率, 重新归一化 (差额自动分配给主/平)
                away_idx = 2
                post_clip[away_idx] *= (1.0 - penalty)
                total = sum(post_clip)
                if total > 0:
                    post_clip = [c / total for c in post_clip]
                result.fused_probs = ProbabilityDistribution(
                    prob_home=post_clip[0],
                    prob_draw=post_clip[1],
                    prob_away=post_clip[2],
                )

            # v5.10.8: 重建 score_matrix
            if result.poisson_score_matrix is not None:
                result.poisson_score_matrix = self._recalibrate_score_matrix(
                    result.poisson_score_matrix,
                    result.fused_probs.prob_home,
                    result.fused_probs.prob_draw,
                    result.fused_probs.prob_away,
                )

            # v5.10.8: 因子诊断 (仅前3场)
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            if self._diag_count < 3:
                self._diag_count += 1
                self._print_factor_diag(result, match, market_probs_norm)

            # === Stage 5: value_calculation ===
            odds = {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            }
            result.value_results = self.prob_engine.calculate_value(
                result.fused_probs, odds,
                calibration_discount=self.calibration_discount,
                # v5.10.9: 若管道已校准(calib_mult>1.0), 不再二次校准
                calibration_multiplier=0.0 if calib_mult > 1.0 else self.params.calibration_multiplier,
            )

            # v5.3b: 客场价值折扣
            away_discount = getattr(self.params, 'away_value_discount', 0.0)
            if away_discount > 0 and "away" in result.value_results:
                result.value_results["away"]["value"] *= (1.0 - away_discount)

            # v5.10.10: 法甲强队客场定向价值惩罚 (替代v5.10.9的概率重分配)
            # 分析确认: 法甲强队客场系统性高估 (Elo差<-100客队ROI=-59%)
            # 策略: 直接惩罚客场价值, 不重分配概率 (避免产生虚假主场/平局价值)
            # elo_diff = home_elo - away_elo, 强队客场时 elo_diff < -100
            strong_away_penalty = getattr(self.params, 'strong_away_penalty', 0.0)
            if strong_away_penalty > 0 and elo_diff < -100 and "away" in result.value_results:
                elo_scale = min((-elo_diff - 100) / 300, 1.0)
                value_penalty = strong_away_penalty * elo_scale
                result.value_results["away"]["value"] *= (1.0 - value_penalty)
                # 模型概率也标记为已惩罚 (用于诊断)
                result.value_results["away"]["model_prob"] = max(
                    result.value_results["away"]["model_prob"] * (1.0 - value_penalty * 0.5),
                    result.value_results["away"]["implied_prob"],
                )

        except Exception as e:
            result.errors.append(f"Pipeline error: {e}")

        return result

    def run_stages_6_9(
        self,
        stage_1_5_result: PipelineResult,
        daily_staked: float = 0.0,
        weekly_staked: float = 0.0,
        extra_data: Optional[Dict] = None,
        now: Optional[datetime] = None,
    ) -> PipelineResult:
        """执行 Stage 6-9: 优先级排序 → 资金分配 → 风控 → 熔断"""
        result = stage_1_5_result
        extra = extra_data or {}

        try:
            # === Stage 6: priority_sorting ===
            active_factor_count = len([fid for fid, d in result.factor_deltas.items()
                                       if abs(d.get('home', 0)) > 0.0001 or
                                          abs(d.get('away', 0)) > 0.0001 or
                                          abs(d.get('draw', 0)) > 0.0001])
            from src.config.league_params import get_league_params
            lp = get_league_params(result.league_id)
            conf_weights = (
                lp.confidence_w_data,
                lp.confidence_w_factor,
                lp.confidence_w_dispersion,
                lp.confidence_w_phase,
            )
            confidence = compute_confidence(
                data_completeness=extra.get("data_completeness", 0.8),
                factor_activation_rate=active_factor_count / 41.0,
                dispersion_penalty=extra.get("dispersion_penalty", 0.9),
                match_phase=extra.get("match_phase", 1.0),
                weights=conf_weights,
            )

            # 从价值评估结果生成投注建议 (含硬性过滤: value≥0.03, confidence≥0.6, odds 1.05-10.0)
            # v5.10.8: 置信度阈值从联赛参数读取
            base_confidence_threshold = self.confidence_threshold

            proposals = generate_bet_proposals(
                result.value_results,
                match_id=result.match_id,
                league_id=result.league_id,
                factor_count=active_factor_count,
                data_source_count=extra.get("data_source_count", 5),
                odds_std=extra.get("odds_std", 0.05),
                match_phase=extra.get("match_phase", 1.0),
                precomputed_confidence=confidence,
                confidence_threshold=base_confidence_threshold,
            )

            if not proposals:
                return result  # 无价值投注机会

            # 计算优先级评分 (规范第8.3节: priority_score = f_actual × value × confidence)
            for p in proposals:
                p.priority_score = self.bankroll_mgr.compute_priority_score(
                    value=p.value,
                    model_prob=p.model_prob,
                    implied_prob=p.implied_prob,
                    confidence=confidence,
                    odds=p.odds,
                )

            # 按优先级评分降序排列
            proposals = self.bankroll_mgr.sort_by_priority(proposals)
            result.proposals = proposals

            # === Stage 7: bankroll_allocation (v5.8: 动态Kelly + 赛季阶段 + v5.10.10: 策略权重) ===
            match_phase = extra.get("match_phase", 1.0)
            proposals = self.bankroll_mgr.allocate_stakes(
                proposals, match_phase=match_phase,
                strategy_weights={"1x2": 1.0, "asian_handicap": 1.0, "over_under": 1.0},
            )
            result.proposals = proposals

            # === Stage 8: risk_control ===
            proposals, warnings = self.risk_ctrl.run_all_checks(
                proposals,
                self.bankroll_mgr.state,
                daily_staked=daily_staked,
                weekly_staked=weekly_staked,
            )
            result.proposals = proposals
            result.warnings.extend(warnings)

            # === Stage 9: circuit_breaker ===
            broken, reason = self.risk_ctrl.check_circuit_breaker(
                self.bankroll_mgr.state,
                now=now,
            )
            if broken:
                result.circuit_broken = True
                result.circuit_reason = reason
                result.proposals = []  # 熔断时清空投注
                result.warnings.append(f"MELT: {reason}")
            else:
                # 冷却期限制检查
                if self.risk_ctrl.breaker_state.cooldown_until:
                    result.proposals = self.risk_ctrl.apply_cooldown_restrictions(
                        result.proposals, self.bankroll_mgr.state
                    )

            result.bankroll_state = self.bankroll_mgr.state

        except Exception as e:
            result.errors.append(f"Stage 6-9 error: {e}")

        return result

    def run_full(
        self,
        match: MatchContext,
        extra_data: Optional[Dict] = None,
        daily_staked: float = 0.0,
        weekly_staked: float = 0.0,
    ) -> PipelineResult:
        """执行完整九阶段流水线，包括投注执行"""
        result = self.run_stages_1_5(match, extra_data)
        result = self.run_stages_6_9(
            result, daily_staked, weekly_staked, extra_data,
            now=match.kickoff_time,
        )
        # 自动执行投注 (将 BetProposal 转换为 BetPlacement)
        result = self.execute_bets(result)
        return result

    def execute_bets(
        self,
        result: PipelineResult,
    ) -> PipelineResult:
        """
        执行投注 (将 BetProposal 转换为 BetPlacement)。

        规范第9.6节:
        - 仅投注 stake > 0 的选项
        - 赔率 < 1.05 或 > 10.0 跳过
        """
        for p in result.proposals:
            if p.adjusted_stake <= 0:
                continue
            if p.odds < config.pipeline.default_odds_min or p.odds > config.pipeline.default_odds_max:
                continue

            placement = BetPlacement(
                bet_id=f"{result.match_id}_{p.selection.value}",
                match_id=result.match_id,
                selection=p.selection,
                odds=p.odds,
                stake=p.adjusted_stake,
                placed_at=datetime.now(),
                result=BetResult.PENDING,
                league_id=result.league_id,
            )

            self.bankroll_mgr.record_bet(placement)
            result.placements.append(placement)

        result.bankroll_state = self.bankroll_mgr.state
        return result

    def settle_bets(
        self,
        placements: List[BetPlacement],
        actual_outcome: BetSelection,
    ) -> List[BetPlacement]:
        """
        结算投注。

        规范第9.7节:
        - 命中: profit = stake × (odds - 1)
        - 未命中: profit = -stake
        """
        for p in placements:
            if p.result != BetResult.PENDING:
                continue

            if p.selection == actual_outcome:
                profit = p.stake * (p.odds - 1.0)
                self.bankroll_mgr.settle_bet(p, BetResult.WIN, profit)
            else:
                self.bankroll_mgr.settle_bet(p, BetResult.LOSS, -p.stake)

        return placements

    def _maybe_boost_draw(self, clipped, market_probs):
        """v5.10.8: 当主客概率接近时，提升平局概率"""
        home_gap = abs(clipped[0] - clipped[2])
        if home_gap > 0.08:
            return None
        boost = 0.01 + 0.04 * (1.0 - home_gap / 0.08)
        new_h = clipped[0] - boost / 2
        new_a = clipped[2] - boost / 2
        new_d = clipped[1] + boost
        new_h = max(0.0, new_h)
        new_a = max(0.0, new_a)
        new_d = min(0.8, new_d)
        total = new_h + new_d + new_a
        if total > 0:
            new_h /= total
            new_d /= total
            new_a /= total
        return (new_h, new_d, new_a)

    def _recalibrate_score_matrix(self, score_matrix, target_h, target_d, target_a):
        """v5.10.8: 重建 score_matrix 使胜平负概率与裁剪后一致"""
        from ..data.models import ScoreMatrix
        current_h = sum(prob for (h, a), prob in score_matrix.matrix.items() if h > a)
        current_d = sum(prob for (h, a), prob in score_matrix.matrix.items() if h == a)
        current_a = sum(prob for (h, a), prob in score_matrix.matrix.items() if h < a)

        new_matrix = {}
        for (h, a), prob in score_matrix.matrix.items():
            if h > a:
                scale = target_h / current_h if current_h > 0.001 else 1.0
            elif h == a:
                scale = target_d / current_d if current_d > 0.001 else 1.0
            else:
                scale = target_a / current_a if current_a > 0.001 else 1.0
            new_matrix[(h, a)] = prob * scale

        total = sum(new_matrix.values())
        for key in new_matrix:
            new_matrix[key] /= total

        return ScoreMatrix(
            league_id=score_matrix.league_id,
            max_goals=score_matrix.max_goals,
            matrix=new_matrix,
        )

    def _calibrate_draw(self, probs, elo_diff, market_probs_norm):
        """
        v5.10.8: 平局校准 — 解决48/55因子平局=0的问题。

        当模型完全依赖市场隐含概率预测平局时, 需要独立的平局校准:
        1. 联赛基准平局率 (draw_rate)
        2. Elo差值 → 实力越接近, 平局概率越高
        3. 模型平局概率与基准平局率的加权混合

        返回: (calibrated_home, calibrated_draw, calibrated_away)
        """
        league_draw_rate = self.params.draw_rate
        # Elo差异越小, 平局概率越高
        elo_closeness = max(0.0, 1.0 - abs(elo_diff) / 400.0)
        expected_draw = league_draw_rate * (1.0 + 0.5 * elo_closeness)  # 最高1.5x基准

        # 当前模型平局概率
        model_draw = probs[1]
        market_draw = market_probs_norm[1] if market_probs_norm else 0.25

        # 校准: 如果模型平局概率过低 (完全依赖市场), 向预期平局率靠拢
        if model_draw < expected_draw * 0.6:
            # 模型平局严重偏低, 注入独立平局信号
            draw_weight = 0.15  # 15%来自独立校准 (温和)
            calibrated_draw = model_draw * (1.0 - draw_weight) + expected_draw * draw_weight
        else:
            calibrated_draw = model_draw

        # 按比例调整主胜/客胜
        if calibrated_draw != model_draw:
            draw_delta = calibrated_draw - model_draw
            non_draw = 1.0 - model_draw
            if non_draw > 0.001:
                home_ratio = probs[0] / non_draw
                away_ratio = probs[2] / non_draw
                calibrated_home = max(0.02, probs[0] - draw_delta * home_ratio)
                calibrated_away = max(0.02, probs[2] - draw_delta * away_ratio)
            else:
                calibrated_home = probs[0]
                calibrated_away = probs[2]
        else:
            calibrated_home = probs[0]
            calibrated_away = probs[2]

        # 归一化
        total = calibrated_home + calibrated_draw + calibrated_away
        if total > 0:
            return (calibrated_home / total, calibrated_draw / total, calibrated_away / total)
        return (calibrated_home, calibrated_draw, calibrated_away)

    def _print_factor_diag(self, result, match, market_probs_norm):
        """v5.10.8: 打印因子贡献方向诊断"""
        print(f"\n  ══ 因子诊断 [{match.home_team} vs {match.away_team}] ══")
        print(f"  市场: H={market_probs_norm[0]:.1%} D={market_probs_norm[1]:.1%} A={market_probs_norm[2]:.1%}")
        print(f"  模型(logit): H={result.logit_probs.prob_home:.1%} D={result.logit_probs.prob_draw:.1%} A={result.logit_probs.prob_away:.1%}")
        print(f"  模型(fused): H={result.fused_probs.prob_home:.1%} D={result.fused_probs.prob_draw:.1%} A={result.fused_probs.prob_away:.1%}")
        print(f"  因子缩放: {self.factor_scale:.2f}")

        # 原始因子统计
        raw = result.raw_factor_deltas
        raw_active = {fid: d for fid, d in raw.items()
                      if abs(d.get('home', 0)) > 0.0001 or abs(d.get('away', 0)) > 0.0001 or abs(d.get('draw', 0)) > 0.0001}
        raw_zero = [fid for fid, d in raw.items()
                    if abs(d.get('home', 0)) <= 0.0001 and abs(d.get('away', 0)) <= 0.0001 and abs(d.get('draw', 0)) <= 0.0001]
        print(f"  因子: {len(raw)} total, {len(raw_active)} 非零, {len(raw_zero)} 零值")
        if raw_zero:
            print(f"  零值因子: {raw_zero}")

        # 按方向分组因子
        home_lean = []; away_lean = []; draw_lean = []
        for fid, d in sorted(result.factor_deltas.items()):
            h = d.get('home', 0); a = d.get('away', 0); dr = d.get('draw', 0)
            net = h - a
            if net > 0.001: home_lean.append((fid, net, h, a, dr))
            elif net < -0.001: away_lean.append((fid, net, h, a, dr))
            elif abs(dr) > 0.001: draw_lean.append((fid, net, h, a, dr))

        home_lean.sort(key=lambda x: -x[1])
        away_lean.sort(key=lambda x: x[1])
        print(f"  主胜因子({len(home_lean)}): {[f[0] for f in home_lean[:10]]}")
        print(f"  客胜因子({len(away_lean)}): {[f[0] for f in away_lean[:10]]}")
        print(f"  平局因子({len(draw_lean)}): {[f[0] for f in draw_lean]}")
        total_net = sum(f[2] for f in home_lean) + sum(f[2] for f in away_lean)
        print(f"  总净贡献 H→A: {total_net:+.4f}")