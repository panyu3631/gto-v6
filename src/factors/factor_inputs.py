"""
GTO-GameFlow v6.0 — 因子输入数据类

将 compute_all() 的58个参数封装为单一数据类。
增减因子只需修改此文件和 registry.py，不会参数错位。

用法:
    inputs = FactorInputs(
        elo_diff=120,
        xi_rating=6.5,
        recent_results=[3, 3, 1, 0, 1],
        ...
    )
    deltas = engine.compute_all_from_inputs(inputs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FactorInputs:
    """因子计算所需的全部输入参数"""

    # ═══════════════════════════════════════════════════════
    # 基础因子输入 (F1-F18)
    # ═══════════════════════════════════════════════════════

    elo_diff: float = 0.0
    """ELO 差值 (主队 - 客队)"""

    xi_rating: float = 6.0
    """预期首发加权平均评分 (WhoScored 1-10分)"""

    recent_results: List[float] = field(default_factory=lambda: [1.5, 1.5, 1.5, 1.5, 1.5])
    """最近5场结果 [3,3,1,0,1] = 胜胜平负平"""

    h2h_results: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    """最近5场交锋 [1,3,0,3,1]"""

    matches_7d: int = 1
    """过去7天比赛数量"""

    rank_diff: int = 0
    """联赛排名差 (rank_away - rank_home, 正值=主队排名更靠前)"""

    goal_diff: float = 0.0
    """赛季累积净胜球差值"""

    xg_diff: float = 0.0
    """xG差值 = (xG_home-xGA_home) - (xG_away-xGA_away)"""

    market_probs: Dict[str, float] = field(default_factory=lambda: {"home": 0.33, "draw": 0.33, "away": 0.34})
    """市场隐含概率 {"home": 0.45, "draw": 0.28, "away": 0.27}"""

    opening_probs: Optional[Dict[str, float]] = None
    """开盘赔率隐含概率"""

    weather: float = 0.0
    """天气影响因子"""

    ref_yellow_rate: float = 0.0
    """裁判场均黄牌率"""

    coach_change_effect: float = 0.0
    """教练更替效应"""

    fatigue_penalty: float = 0.0
    """欧战体能惩罚"""

    rotation_risk: float = 0.0
    """轮换预测风险"""

    derby_boost: float = 0.0
    """德比战加成"""

    # ═══════════════════════════════════════════════════════
    # 增强因子输入 (F19-F32)
    # ═══════════════════════════════════════════════════════

    style_matchup_score: float = 0.5
    """攻击/防守风格匹配度"""

    streak_momentum: float = 0.0
    """连胜/连败动量"""

    player_form: float = 6.5
    """核心球员状态 (WhoScored评分)"""

    market_sentiment: float = 0.0
    """市场情绪 (NLP情感分析)"""

    odds_std: float = 0.05
    """赔率离散度 (跨公司标准差)"""

    nlp_sentiment: float = 0.0
    """新闻NLP情感"""

    time_decay_factor: float = 1.0
    """时间衰减因子"""

    league_strength_bias: float = 0.0
    """联赛强度偏差"""

    poisson_correction: float = 0.0
    """泊松分布修正"""

    handicap_depth: float = 0.0
    """亚盘深度"""

    totals_trend: float = 0.0
    """大小球趋势"""

    value_signal: float = 0.0
    """赔率价值信号"""

    contrarian_signal: float = 0.0
    """反市场偏差信号"""

    market_efficiency: float = 0.0
    """市场效率评分"""

    # ═══════════════════════════════════════════════════════
    # 联赛特定因子输入 (F33-F41)
    # ═══════════════════════════════════════════════════════

    motivation_boost: float = 0.0
    """保级/争冠动力"""

    financial_gap_effect: float = 0.0
    """财力差距效应"""

    winter_break_effect: float = 0.0
    """冬歇期效应"""

    christmas_fatigue: float = 0.0
    """圣诞赛程疲劳"""

    complacency_effect: float = 0.0
    """中游无欲效应"""

    streak_momentum_league: float = 0.0
    """连胜动量(联赛特定)"""

    position_advantage: float = 0.0
    """积分榜位置优势"""

    promoted_team_delta: float = 0.0
    """升班马数据偏差"""

    schedule_advantage: float = 0.0
    """赛程优势"""

    derby_intensity: float = 0.0
    """德比战强度"""

    # ═══════════════════════════════════════════════════════
    # 比赛统计衍生因子输入 (F42-F55, v5.10.8)
    # ═══════════════════════════════════════════════════════

    ht_momentum: float = 0.0
    """半场动量"""

    shot_eff_diff: float = 0.0
    """射门效率差"""

    territorial_dominance: float = 0.0
    """控场优势"""

    discipline_index: float = 0.0
    """纪律指数"""

    odds_drift: float = 0.0
    """赔率漂移信号"""

    market_disagreement: float = 0.0
    """市场分歧"""

    referee_home_bias: float = 0.0
    """裁判主场偏置"""

    comeback_resilience: float = 0.0
    """逆转韧性"""

    streak_momentum_enriched: float = 0.0
    """连胜动量(增强)"""

    goal_volatility: float = 0.0
    """进球波动率"""

    corner_dominance: float = 0.0
    """角球优势"""

    sot_rate_diff: float = 0.0
    """射正率差"""

    ah_odds_drift: float = 0.0
    """亚盘赔率漂移"""

    totals_odds_drift: float = 0.0
    """大小球赔率漂移"""

    # ═══════════════════════════════════════════════════════
    # 平局专属因子输入 (F56-F58, v5.11)
    # ═══════════════════════════════════════════════════════

    draw_tactical_matchup: float = 0.0
    """战术风格平局倾向"""

    draw_goal_expectancy: float = 0.0
    """进球预期平局信号"""

    draw_team_tendency: float = 0.0
    """球队平局历史倾向"""

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式（兼容旧接口）"""
        return {
            "elo_diff": self.elo_diff,
            "xi_rating": self.xi_rating,
            "matches_7d": self.matches_7d,
            "rank_diff": self.rank_diff,
            "goal_diff": self.goal_diff,
            "xg_diff": self.xg_diff,
            "weather": self.weather,
            "ref_yellow_rate": self.ref_yellow_rate,
            "coach_change_effect": self.coach_change_effect,
            "fatigue_penalty": self.fatigue_penalty,
            "rotation_risk": self.rotation_risk,
            "derby_boost": self.derby_boost,
            "style_matchup_score": self.style_matchup_score,
            "streak_momentum": self.streak_momentum,
            "player_form": self.player_form,
            "market_sentiment": self.market_sentiment,
            "odds_std": self.odds_std,
            "nlp_sentiment": self.nlp_sentiment,
            "time_decay_factor": self.time_decay_factor,
            "league_strength_bias": self.league_strength_bias,
            "poisson_correction": self.poisson_correction,
            "handicap_depth": self.handicap_depth,
            "totals_trend": self.totals_trend,
            "value_signal": self.value_signal,
            "contrarian_signal": self.contrarian_signal,
            "market_efficiency": self.market_efficiency,
            "motivation_boost": self.motivation_boost,
            "financial_gap_effect": self.financial_gap_effect,
            "winter_break_effect": self.winter_break_effect,
            "christmas_fatigue": self.christmas_fatigue,
            "complacency_effect": self.complacency_effect,
            "streak_momentum_league": self.streak_momentum_league,
            "position_advantage": self.position_advantage,
            "promoted_team_delta": self.promoted_team_delta,
            "schedule_advantage": self.schedule_advantage,
            "derby_intensity": self.derby_intensity,
            "ht_momentum": self.ht_momentum,
            "shot_eff_diff": self.shot_eff_diff,
            "territorial_dominance": self.territorial_dominance,
            "discipline_index": self.discipline_index,
            "odds_drift": self.odds_drift,
            "market_disagreement": self.market_disagreement,
            "referee_home_bias": self.referee_home_bias,
            "comeback_resilience": self.comeback_resilience,
            "streak_momentum_enriched": self.streak_momentum_enriched,
            "goal_volatility": self.goal_volatility,
            "corner_dominance": self.corner_dominance,
            "sot_rate_diff": self.sot_rate_diff,
            "ah_odds_drift": self.ah_odds_drift,
            "totals_odds_drift": self.totals_odds_drift,
            "draw_tactical_matchup": self.draw_tactical_matchup,
            "draw_goal_expectancy": self.draw_goal_expectancy,
            "draw_team_tendency": self.draw_team_tendency,
        }
