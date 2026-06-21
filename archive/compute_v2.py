"""
GTO-GameFlow v6.0 因子计算引擎 V2 — 根本性重构

v5.10 核心问题:
- 55个因子中28个是"谁强谁弱"的同质变体 → 产生虚假多因子共振
- 48个因子 draw=0 → 平局预测完全依赖市场
- 信号衰减链: factor_scale 0.25 × Elo分解 0-40% × 贝叶斯收缩 25-60% → 模型97.5%依赖市场

v6.0 根本性重构:
1. 因子去重合并 — 28个数值对比因子合并为5个独立维度
2. 平局第一公民 — 新增专用于平局预测的因子
3. 市场微观结构 — 聪明钱检测、赔率变动速度、亚盘跳变
4. 非线性动机 — 赛季阶段×积分榜位置交互项
5. 预期进球差 — 因子映射到预期进球差，再推导胜平负
"""
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.config.league_params import LeagueParams, get_league_params


# ================================================================
# v6.0 因子体系: 23个独立因子, 12个维度
# ================================================================

@dataclass
class FactorV2:
    """v6.0 因子定义"""
    factor_id: str
    name: str
    name_cn: str
    dimension: str          # 所属维度
    weight: float           # 默认权重
    draw_active: bool       # 是否在平局方向有非零delta
    description: str

# 维度定义
DIM_POWER = "power"              # 综合实力 (1个因子)
DIM_FORM = "form"                # 近期状态 (1个因子)
DIM_HOME = "home"                # 主场效应 (1个因子)
DIM_FATIGUE = "fatigue"          # 赛程疲劳 (1个因子)
DIM_H2H = "h2h"                  # 交锋心理 (1个因子)
DIM_MARKET = "market"            # 市场定价 (3个因子)
DIM_DRAW = "draw"                # 平局信号 (3个因子)
DIM_MATCH_STATS = "match_stats"  # 比赛统计 (1个因子)
DIM_MICRO = "market_micro"       # 市场微观结构 (3个因子)
DIM_MOTIVATION = "motivation"    # 动机 (2个因子)
DIM_SITUATIONAL = "situational"  # 情境 (4个因子)
DIM_EXPECTED_GOALS = "exp_goals" # 预期进球差 (2个因子)

V6_FACTOR_REGISTRY: Dict[str, FactorV2] = {
    # ============================================================
    # 维度1: POWER — 综合实力 (合并 F1, F2, F7, F8, F9, F34, F39)
    # ============================================================
    "P1": FactorV2(
        factor_id="P1", name="composite_strength", name_cn="综合实力",
        dimension=DIM_POWER, weight=0.7, draw_active=False,  # v6.0.1: 降低权重 1.0→0.7
        description="合并Elo(40%)+首发评分(15%)+排名(10%)+净胜球(10%)+xG差(15%)+财力(5%)+积分榜(5%)",
    ),
    # ============================================================
    # 维度2: FORM — 近期状态 (合并 F3, F20, F21, F38, F50)
    # ============================================================
    "P2": FactorV2(
        factor_id="P2", name="recent_form", name_cn="近期状态",
        dimension=DIM_FORM, weight=0.85, draw_active=False,
        description="合并EWMA状态(30%)+连胜动量(25%)+球员状态(25%)+联赛动量(10%)+增强动量(10%)",
    ),
    # ============================================================
    # 维度3: HOME — 主场效应 (合并 F4, F41, F48)
    # ============================================================
    "P3": FactorV2(
        factor_id="P3", name="home_advantage", name_cn="主场优势",
        dimension=DIM_HOME, weight=0.7, draw_active=False,
        description="合并主场系数(50%)+赛程优势(30%)+裁判主场偏置(20%)",
    ),
    # ============================================================
    # 维度4: FATIGUE — 赛程疲劳 (合并 F6, F16, F35, F36)
    # ============================================================
    "P4": FactorV2(
        factor_id="P4", name="fatigue_penalty", name_cn="疲劳惩罚",
        dimension=DIM_FATIGUE, weight=0.65, draw_active=True,
        description="合并赛程密度(30%)+欧战疲劳(30%)+冬歇期(20%)+圣诞赛程(20%)",
    ),
    # ============================================================
    # 维度5: H2H — 交锋心理 (合并 F5, F19, F40)
    # ============================================================
    "P5": FactorV2(
        factor_id="P5", name="h2h_psychology", name_cn="交锋心理",
        dimension=DIM_H2H, weight=0.6, draw_active=False,
        description="合并历史交锋(50%)+风格匹配(30%)+升班马调整(20%)",
    ),
    # ============================================================
    # 维度6: MARKET — 市场定价 (保留并增强 F10, F11, 新增赔率变动速度)
    # ============================================================
    "M1": FactorV2(
        factor_id="M1", name="market_implied", name_cn="市场隐含概率",
        dimension=DIM_MARKET, weight=0.6, draw_active=True,  # v6.0.1: 降低权重 0.9→0.6, 减少市场反馈循环
        description="去margin后的公平概率logit变换，三方向独立计算",
    ),
    "M2": FactorV2(
        factor_id="M2", name="odds_movement", name_cn="赔率变动方向",
        dimension=DIM_MARKET, weight=0.55, draw_active=True,
        description="开盘→当前赔率变动方向和幅度，三方向独立计算",
    ),
    "M3": FactorV2(
        factor_id="M3", name="odds_movement_velocity", name_cn="赔率变动速度",
        dimension=DIM_MARKET, weight=0.45, draw_active=True,
        description="NEW: 赔率变动速度(快=信息量大)，区分聪明钱和公众钱",
    ),
    # ============================================================
    # 维度7: DRAW — 平局信号 (合并 F12, F18, F23, F47, F51 + 新增)
    # ============================================================
    "D1": FactorV2(
        factor_id="D1", name="draw_uncertainty", name_cn="平局不确定性",
        dimension=DIM_DRAW, weight=0.55, draw_active=True,
        description="合并赔率离散度(40%)+市场分歧(30%)+进球波动率(30%)，高不确定性→推平局",
    ),
    "D2": FactorV2(
        factor_id="D2", name="draw_situational", name_cn="平局情境",
        dimension=DIM_DRAW, weight=0.5, draw_active=True,
        description="合并德比战(40%)+天气(30%)+圣诞赛程(30%)，特定情境→推平局",
    ),
    "D3": FactorV2(
        factor_id="D3", name="draw_strength_parity", name_cn="实力接近度",
        dimension=DIM_DRAW, weight=0.6, draw_active=True,
        description="NEW: 两队实力越接近，平局概率越高。非线性映射：|elo_diff|<50→强平局信号",
    ),
    # ============================================================
    # 维度8: MATCH_STATS — 比赛统计 (合并 F42-F45, F49, F52, F53)
    # ============================================================
    "S1": FactorV2(
        factor_id="S1", name="match_stats_composite", name_cn="比赛统计综合",
        dimension=DIM_MATCH_STATS, weight=0.4, draw_active=False,
        description="合并半场动量(20%)+射门效率(20%)+控场(15%)+纪律(10%)+逆转韧性(15%)+角球(10%)+射正率(10%)",
    ),
    # ============================================================
    # 维度9: MICRO — 市场微观结构 (全新)
    # ============================================================
    "U1": FactorV2(
        factor_id="U1", name="smart_money_flow", name_cn="聪明钱流向",
        dimension=DIM_MICRO, weight=0.5, draw_active=True,
        description="NEW: 赔率变动速度×变动方向，快速变动=聪明钱，慢速变动=公众钱",
    ),
    "U2": FactorV2(
        factor_id="U2", name="asian_handicap_jump", name_cn="亚盘跳变",
        dimension=DIM_MICRO, weight=0.4, draw_active=False,
        description="NEW: 亚盘盘口跳变检测(≥0.5盘=重大信息)，跳变方向=聪明钱方向",
    ),
    "U3": FactorV2(
        factor_id="U3", name="market_inefficiency", name_cn="市场低效度",
        dimension=DIM_MICRO, weight=0.35, draw_active=True,
        description="NEW: 1-Brier(市场概率, 实际结果)+赔率离散度+低关注度联赛，低效市场→模型权重应提高",
    ),
    # ============================================================
    # 维度10: MOTIVATION — 非线性动机 (合并 F33, F37, F15, F17)
    # ============================================================
    "V1": FactorV2(
        factor_id="V1", name="motivation_nonlinear", name_cn="非线性动机",
        dimension=DIM_MOTIVATION, weight=0.6, draw_active=True,
        description="NEW: 赛季阶段×积分榜位置交互项。保级队赛季末≠保级队赛季初，非线性",
    ),
    "V2": FactorV2(
        factor_id="V2", name="squad_disruption", name_cn="阵容扰动",
        dimension=DIM_MOTIVATION, weight=0.45, draw_active=True,
        description="合并教练更替(40%)+轮换风险(30%)+欧战影响(30%)",
    ),
    # ============================================================
    # 维度11: SITUATIONAL — 情境 (合并 F13, F22, F25, F26, F27, F28, F29, F30, F31, F32, F46, F54, F55)
    # ============================================================
    "C1": FactorV2(
        factor_id="C1", name="referee_influence", name_cn="裁判影响",
        dimension=DIM_SITUATIONAL, weight=0.35, draw_active=False,
        description="合并裁判风格(60%)+裁判主场偏置(40%)",
    ),
    "C2": FactorV2(
        factor_id="C2", name="league_context", name_cn="联赛情境",
        dimension=DIM_SITUATIONAL, weight=0.35, draw_active=False,
        description="合并联赛强度(30%)+时间衰减(25%)+亚盘深度(25%)+进球分布修正(20%)",
    ),
    "C3": FactorV2(
        factor_id="C3", name="value_contrarian", name_cn="价值逆向",
        dimension=DIM_SITUATIONAL, weight=0.45, draw_active=True,
        description="合并赔率价值信号(40%)+反市场偏差(30%)+市场效率(30%)",
    ),
    "C4": FactorV2(
        factor_id="C4", name="odds_drift_composite", name_cn="赔率漂移综合",
        dimension=DIM_SITUATIONAL, weight=0.4, draw_active=False,
        description="合并赔率漂移(40%)+亚盘漂移(35%)+大小球漂移(25%)",
    ),
    # ============================================================
    # 维度12: EXPECTED_GOALS — 预期进球差 (全新)
    # ============================================================
    "G1": FactorV2(
        factor_id="G1", name="expected_goal_diff", name_cn="预期进球差",
        dimension=DIM_EXPECTED_GOALS, weight=0.7, draw_active=True,
        description="NEW: 从实力+状态+交锋推导预期进球差，映射到泊松比分矩阵，平局自然推导",
    ),
    "G2": FactorV2(
        factor_id="G2", name="goal_trend", name_cn="进球趋势",
        dimension=DIM_EXPECTED_GOALS, weight=0.45, draw_active=True,
        description="合并大小球趋势(60%)+进球波动率(40%)，影响预期总进球数",
    ),
}


class FactorComputationEngineV2:
    """
    v6.0 因子计算引擎 — 根本性重构。

    核心变化:
    1. 28个同质因子 → 5个独立维度 (P1-P5)
    2. 平局从0个独立因子 → 3个专用因子 (D1-D3)
    3. 新增市场微观结构 (U1-U3)
    4. 新增非线性动机 (V1-V2)
    5. 新增预期进球差 (G1-G2)

    总计: 23个因子, 12个维度, 14个因子在draw方向有非零delta
    """

    def __init__(self, league_id: str):
        self.league_id = league_id
        self.params: LeagueParams = get_league_params(league_id)

    def compute_all(
        self,
        # === 基础数据 ===
        elo_diff: float,
        home_elo: float = 1500.0,
        away_elo: float = 1500.0,
        xi_rating: float = 6.0,
        recent_results: List[float] = None,
        h2h_results: List[float] = None,
        matches_7d: int = 1,
        rank_diff: int = 0,
        goal_diff: float = 0.0,
        xg_diff: float = 0.0,
        # === 市场数据 ===
        market_probs: Dict[str, float] = None,
        opening_probs: Optional[Dict[str, float]] = None,
        # === 情境数据 ===
        weather: float = 0.0,
        ref_yellow_rate: float = 0.0,
        derby_boost: float = 0.0,
        # === 增强数据 ===
        style_matchup_score: float = 0.5,
        player_form: float = 6.5,
        time_decay_factor: float = 1.0,
        league_strength_bias: float = 0.0,
        handicap_depth: float = 0.0,
        # === 动机数据 ===
        motivation_boost: float = 0.0,
        financial_gap_effect: float = 0.0,
        complacency_effect: float = 0.0,
        coach_change_effect: float = 0.0,
        rotation_risk: float = 0.0,
        fatigue_penalty: float = 0.0,
        winter_break_effect: float = 0.0,
        christmas_fatigue: float = 0.0,
        # === 比赛统计 ===
        ht_momentum: float = 0.0,
        shot_eff_diff: float = 0.0,
        territorial_dominance: float = 0.0,
        discipline_index: float = 0.0,
        comeback_resilience: float = 0.0,
        corner_dominance: float = 0.0,
        sot_rate_diff: float = 0.0,
        # === 市场微观结构 (NEW) ===
        odds_std: float = 0.05,
        market_disagreement: float = 0.0,
        goal_volatility: float = 0.0,
        odds_drift: float = 0.0,
        ah_odds_drift: float = 0.0,
        totals_odds_drift: float = 0.0,
        odds_movement_speed: float = 0.0,     # NEW: 赔率变动速度
        ah_jump_magnitude: float = 0.0,        # NEW: 亚盘跳变幅度
        ah_jump_direction: float = 0.0,        # NEW: 亚盘跳变方向
        market_efficiency: float = 0.0,
        value_signal: float = 0.0,
        contrarian_signal: float = 0.0,
        # === 动机增强 (NEW) ===
        season_phase: float = 0.5,             # NEW: 赛季阶段(0=开局, 0.5=中期, 1=末段)
        position_vs_par: float = 0.0,          # NEW: 排名vs预期(正值=超预期)
        next_match_importance: float = 0.0,    # NEW: 下场比赛重要性
        # === 其他 ===
        promoted_team_delta: float = 0.0,
        schedule_advantage: float = 0.0,
        referee_home_bias: float = 0.0,
        streak_momentum: float = 0.0,
        streak_momentum_league: float = 0.0,
        streak_momentum_enriched: float = 0.0,
        position_advantage: float = 0.0,
        totals_trend: float = 0.0,
        poisson_correction: float = 0.0,
        nlp_sentiment: float = 0.0,
        market_sentiment: float = 0.0,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        计算所有v6.0因子的delta值。
        返回: {factor_id: {"home": delta_home, "draw": delta_draw, "away": delta_away}}
        """
        if recent_results is None:
            recent_results = [1.5, 1.5, 1.5, 1.5, 1.5]
        if h2h_results is None:
            h2h_results = [0, 0, 0, 0, 0]
        if market_probs is None:
            market_probs = {"home": 0.33, "draw": 0.33, "away": 0.33}

        deltas = {}

        # ================================================================
        # P1: 综合实力 — 加权合并7个同质子因子
        # ================================================================
        p1_strength = (
            (elo_diff / 400.0) * 0.40 +              # F1: Elo差 (40%)
            (xi_rating - 6.0) * 0.08 * 1.875 * 0.15 + # F2: 首发评分 (15%)
            (rank_diff / 20.0) * 0.10 * 1.0 * 0.10 +  # F7: 排名差 (10%)
            (goal_diff / 10.0) * 0.06 * 1.0 * 0.10 +  # F8: 净胜球 (10%)
            xg_diff * 0.12 * 1.0 * 0.15 +             # F9: xG差 (15%)
            (financial_gap_effect / 100.0) * 0.05 +   # F34: 财力 (5%)
            (position_advantage / 20.0) * 0.06 * 0.05 # F39: 积分榜 (5%)
        )
        deltas["P1"] = {"home": p1_strength * 0.25, "draw": 0.0, "away": -p1_strength * 0.25}

        # ================================================================
        # P2: 近期状态 — 加权合并5个子因子
        # ================================================================
        f3_ewma = self._compute_ewma(recent_results, self.params.time_decay_alpha)
        p2_form = (
            (f3_ewma - 1.5) * 0.15 * 0.30 +          # F3: EWMA状态 (30%)
            streak_momentum * 0.08 * 0.25 +            # F20: 连胜动量 (25%)
            (player_form - 6.5) * 0.06 * 0.25 +       # F21: 球员状态 (25%)
            streak_momentum_league * 0.08 * 0.10 +     # F38: 联赛动量 (10%)
            streak_momentum_enriched * 0.08 * 0.10     # F50: 增强动量 (10%)
        )
        deltas["P2"] = {"home": p2_form * 0.25, "draw": 0.0, "away": -p2_form * 0.25}

        # ================================================================
        # P3: 主场优势 — 加权合并3个子因子
        # ================================================================
        p3_home = (
            self.params.home_advantage * 0.10 * 0.50 +  # F4: 主场系数 (50%)
            schedule_advantage * 0.08 * 0.30 +           # F41: 赛程优势 (30%)
            referee_home_bias * 0.08 * 0.20              # F48: 裁判偏置 (20%)
        )
        deltas["P3"] = {"home": p3_home * 0.25, "draw": 0.0, "away": 0.0}

        # ================================================================
        # P4: 疲劳惩罚 — 加权合并4个子因子 (draw_active=True)
        # ================================================================
        p4_fatigue = (
            -(matches_7d - 1) * 0.03 * 0.30 +     # F6: 赛程密度 (30%)
            fatigue_penalty * 0.04 * 0.30 +         # F16: 欧战疲劳 (30%)
            winter_break_effect * 0.10 * 0.20 +     # F35: 冬歇期 (20%)
            -(christmas_fatigue * 0.12) * 0.20      # F36: 圣诞 (20%)
        )
        # 疲劳惩罚: 推draw (两队都累→比赛质量下降→平局概率上升)
        draw_from_fatigue = abs(p4_fatigue) * 0.3
        deltas["P4"] = {
            "home": p4_fatigue * 0.25,
            "draw": draw_from_fatigue * 0.25,
            "away": -p4_fatigue * 0.25,
        }

        # ================================================================
        # P5: 交锋心理 — 加权合并3个子因子
        # ================================================================
        h2h_adv = self._compute_h2h_advantage(h2h_results)
        p5_h2h = (
            h2h_adv * 0.08 * 0.50 +                  # F5: 交锋 (50%)
            (style_matchup_score - 0.5) * 0.10 * 0.30 + # F19: 风格 (30%)
            promoted_team_delta * 0.08 * 0.20           # F40: 升班马 (20%)
        )
        deltas["P5"] = {"home": p5_h2h * 0.25, "draw": 0.0, "away": -p5_h2h * 0.25}

        # ================================================================
        # M1: 市场隐含概率 — 三方向独立 (保留F10核心逻辑)
        # ================================================================
        m1_home = market_probs.get("home", 0.33)
        m1_draw = market_probs.get("draw", 0.33)
        m1_away = market_probs.get("away", 0.33)
        deltas["M1"] = {
            "home": self._logit(m1_home) - self._logit(0.33),
            "draw": self._logit(m1_draw) - self._logit(0.33),
            "away": self._logit(m1_away) - self._logit(0.33),
        }

        # ================================================================
        # M2: 赔率变动方向 — 三方向独立 (保留F11核心逻辑)
        # ================================================================
        if opening_probs:
            m2_h = (opening_probs.get("home", m1_home) - m1_home) * 0.5
            m2_d = (opening_probs.get("draw", m1_draw) - m1_draw) * 0.5
            m2_a = (opening_probs.get("away", m1_away) - m1_away) * 0.5
        else:
            m2_h = m2_d = m2_a = 0.0
        deltas["M2"] = {"home": m2_h, "draw": m2_d, "away": m2_a}

        # ================================================================
        # M3: 赔率变动速度 — NEW (区分聪明钱和公众钱)
        # ================================================================
        # 快速变动 + 方向 = 聪明钱信号
        m3_smart = odds_movement_speed * 0.8  # 速度越快，信号越强
        # 慢速变动 = 公众钱，信号弱
        if odds_movement_speed < 0.3:
            m3_smart *= 0.3
        # 方向由开盘→收盘变动决定
        if opening_probs:
            m3_dir_h = opening_probs.get("home", 0.33) - market_probs.get("home", 0.33)
            m3_dir_d = opening_probs.get("draw", 0.33) - market_probs.get("draw", 0.33)
            m3_dir_a = opening_probs.get("away", 0.33) - market_probs.get("away", 0.33)
        else:
            m3_dir_h = m3_dir_d = m3_dir_a = 0.0
        deltas["M3"] = {
            "home": m3_dir_h * m3_smart * 0.5,
            "draw": m3_dir_d * m3_smart * 0.5,
            "away": m3_dir_a * m3_smart * 0.5,
        }

        # ================================================================
        # D1: 平局不确定性 — 合并赔率离散度+市场分歧+进球波动率
        # v6.0.1: 大幅增强 (×2.5) — 这是平局的核心市场信号
        # ================================================================
        d1_draw = (
            (odds_std - 0.05) * 0.5 * 0.40 +       # F23: 赔率离散度 (40%)
            market_disagreement * 0.06 * 0.30 +      # F47: 市场分歧 (30%)
            goal_volatility * 0.05 * 0.30            # F51: 进球波动率 (30%)
        )
        deltas["D1"] = {"home": 0.0, "draw": d1_draw * 2.5, "away": 0.0}

        # ================================================================
        # D2: 平局情境 — 合并德比+天气+圣诞
        # v6.0.1: 大幅增强 (×2.5)
        # ================================================================
        d2_draw = (
            derby_boost * 0.06 * 0.40 +              # F18: 德比 (40%)
            weather * 0.08 * 0.30 +                   # F12: 天气 (30%)
            christmas_fatigue * 0.12 * 0.30           # F36: 圣诞推draw (30%)
        )
        deltas["D2"] = {"home": 0.0, "draw": d2_draw * 2.5, "away": 0.0}

        # ================================================================
        # D3: 实力接近度 — 核心平局因子
        # v6.0.1: 大幅增强 (×3), 扩大有效范围
        # ================================================================
        abs_elo = abs(elo_diff)
        if abs_elo < 80:  # 扩大: 50→80
            d3_draw = 0.24 * (1.0 - abs_elo / 80.0)  # 增强: 0.08→0.24
        elif abs_elo < 200:  # 扩大: 150→200
            d3_draw = 0.12 * (1.0 - (abs_elo - 80) / 120.0)  # 增强: 0.04→0.12
        else:
            d3_draw = 0.0
        # 同时考虑联赛平局率
        league_draw_rate = getattr(self.params, 'draw_rate', 0.26)
        d3_draw *= (league_draw_rate / 0.26)  # 按联赛平局率缩放
        deltas["D3"] = {"home": 0.0, "draw": d3_draw, "away": 0.0}

        # ================================================================
        # S1: 比赛统计综合 — 合并7个统计因子
        # ================================================================
        s1_stats = (
            ht_momentum * 0.08 * 0.20 +               # F42: 半场动量 (20%)
            shot_eff_diff * 0.06 * 0.20 +              # F43: 射门效率 (20%)
            territorial_dominance * 0.06 * 0.15 +      # F44: 控场 (15%)
            discipline_index * 0.05 * 0.10 +           # F45: 纪律 (10%)
            comeback_resilience * 0.06 * 0.15 +        # F49: 逆转 (15%)
            corner_dominance * 0.05 * 0.10 +           # F52: 角球 (10%)
            sot_rate_diff * 0.06 * 0.10                # F53: 射正率 (10%)
        )
        deltas["S1"] = {"home": s1_stats * 0.25, "draw": 0.0, "away": -s1_stats * 0.25}

        # ================================================================
        # U1: 聪明钱流向 — NEW
        # ================================================================
        # 聪明钱 = 快速变动 × 明确方向
        u1_magnitude = min(odds_movement_speed * 2.0, 1.0)
        if opening_probs:
            u1_h = (opening_probs.get("home", 0.33) - market_probs.get("home", 0.33)) * u1_magnitude * 0.6
            u1_d = (opening_probs.get("draw", 0.33) - market_probs.get("draw", 0.33)) * u1_magnitude * 0.6
            u1_a = (opening_probs.get("away", 0.33) - market_probs.get("away", 0.33)) * u1_magnitude * 0.6
        else:
            u1_h = u1_d = u1_a = 0.0
        deltas["U1"] = {"home": u1_h, "draw": u1_d, "away": u1_a}

        # ================================================================
        # U2: 亚盘跳变 — NEW
        # ================================================================
        # 亚盘盘口跳变 ≥ 0.5 盘 → 重大信息
        u2_signal = 0.0
        if abs(ah_jump_magnitude) >= 0.5:
            u2_signal = ah_jump_direction * min(abs(ah_jump_magnitude) / 2.0, 1.0) * 0.15
        deltas["U2"] = {"home": u2_signal, "draw": 0.0, "away": -u2_signal}

        # ================================================================
        # U3: 市场低效度 — NEW
        # ================================================================
        # 市场效率越低 → 模型独立判断越有价值
        # 低效市场特征: 赔率离散度高、低关注度联赛、赛季初
        u3_inefficiency = (
            (1.0 - max(0.0, min(1.0, market_efficiency))) * 0.40 +  # 市场效率低 (40%)
            (odds_std / 0.15) * 0.30 +                               # 赔率离散度高 (30%)
            (1.0 - season_phase) * 0.30                               # 赛季初 (30%)
        )
        u3_inefficiency = max(0.0, min(1.0, u3_inefficiency))
        # 市场低效 → 提升模型权重 (通过三方向等值缩放)
        # 这个因子不推方向，而是放大模型整体信号
        deltas["U3"] = {
            "home": u3_inefficiency * 0.05,
            "draw": u3_inefficiency * 0.05,
            "away": u3_inefficiency * 0.05,
        }

        # ================================================================
        # V1: 非线性动机 — NEW (核心改进)
        # ================================================================
        # 赛季阶段 × 积分榜位置交互项
        # 保级队赛季末 ≠ 保级队赛季初
        v1_motivation = 0.0
        v1_draw_boost = 0.0

        # 基础动机: 保级/争冠动力
        base_motivation = motivation_boost / 100.0

        # 非线性调整: 赛季末段动机放大
        if season_phase > 0.7:  # 赛季末段
            base_motivation *= 1.5  # 动机放大50%

        # 中游无欲: 赛季末段 + 无目标 → 强烈抑制
        if complacency_effect > 0 and season_phase > 0.6:
            base_motivation -= complacency_effect * 0.08 * (1.0 + season_phase)

        # 下场比赛重要性: 如果下场比赛更重要，本轮可能轮换
        if next_match_importance > 0.5:
            base_motivation -= next_match_importance * 0.04

        v1_motivation = base_motivation

        # 动机场景 → 平局影响
        # 两队都有保级动机 → 平局可以接受 → 推平局
        if abs(motivation_boost) > 5 and season_phase > 0.6:
            v1_draw_boost = abs(motivation_boost) / 100.0 * 0.3

        deltas["V1"] = {
            "home": v1_motivation * 0.25,
            "draw": v1_draw_boost * 0.25,
            "away": 0.0,
        }

        # ================================================================
        # V2: 阵容扰动 — 合并教练更替+轮换+欧战
        # ================================================================
        v2_disruption = (
            coach_change_effect * 0.07 * 0.40 +    # F15: 教练更替 (40%)
            rotation_risk * 0.05 * 0.30 +            # F17: 轮换 (30%)
            fatigue_penalty * 0.04 * 0.30            # F16: 欧战 (30%)
        )
        # 阵容扰动 → 不确定性增加 → 平局概率上升
        draw_from_disruption = abs(v2_disruption) * 0.2
        deltas["V2"] = {
            "home": v2_disruption * 0.25,
            "draw": draw_from_disruption * 0.25,
            "away": -v2_disruption * 0.25,
        }

        # ================================================================
        # C1: 裁判影响 — 合并裁判风格+裁判主场偏置
        # ================================================================
        c1_ref = (
            (ref_yellow_rate - self.params.yellow_card_rate) * 0.02 * 0.60 +  # F13 (60%)
            referee_home_bias * 0.08 * 0.40                                     # F48 (40%)
        )
        deltas["C1"] = {"home": c1_ref * 0.25, "draw": 0.0, "away": 0.0}

        # ================================================================
        # C2: 联赛情境 — 合并联赛强度+时间衰减+亚盘深度+泊松修正
        # ================================================================
        c2_context = (
            league_strength_bias * 0.05 * 0.30 +     # F26 (30%)
            time_decay_factor * 0.05 * 0.25 +          # F25 (25%)
            handicap_depth * 0.06 * 0.25 +             # F28 (25%)
            poisson_correction * 0.20 * 0.20           # F27 (20%)
        )
        deltas["C2"] = {"home": c2_context * 0.25, "draw": 0.0, "away": -c2_context * 0.25}

        # ================================================================
        # C3: 价值逆向 — 合并价值信号+反市场+市场效率
        # ================================================================
        c3_value = (
            value_signal * 0.10 * 0.40 +              # F30 (40%)
            contrarian_signal * 0.05 * 0.30 +          # F31 (30%)
            market_efficiency * 0.03 * 0.30            # F32 (30%)
        )
        deltas["C3"] = {
            "home": c3_value * 0.25,
            "draw": c3_value * 0.25 * 0.5,  # 价值信号在draw方向打折
            "away": c3_value * 0.25,
        }

        # ================================================================
        # C4: 赔率漂移综合 — 合并三个漂移因子
        # ================================================================
        c4_drift = (
            odds_drift * 0.10 * 0.40 +               # F46 (40%)
            ah_odds_drift * 0.08 * 0.35 +             # F54 (35%)
            totals_odds_drift * 0.06 * 0.25           # F55 (25%)
        )
        deltas["C4"] = {"home": c4_drift * 0.25, "draw": 0.0, "away": -c4_drift * 0.25}

        # ================================================================
        # G1: 预期进球差 — NEW (核心改进)
        # ================================================================
        # 从实力+状态+交锋推导预期进球差
        # 这是从"预测胜平负"到"预测比分"的关键转变
        base_goal_diff = (
            elo_diff / 400.0 * 0.8 +                  # Elo → 进球差
            p2_form * 0.4 +                            # 状态 → 进球差
            p5_h2h * 0.3                               # 交锋 → 进球差
        )
        # 主场优势带来额外进球
        home_goal_bonus = self.params.home_advantage * 0.5
        expected_gd = base_goal_diff + home_goal_bonus

        # 从预期进球差推导胜平负概率 (通过简化的泊松映射)
        # |预期进球差| 越小 → 平局概率越高
        g1_home = max(0.0, expected_gd * 0.3)
        g1_away = max(0.0, -expected_gd * 0.3)
        g1_draw = max(0.0, 0.15 * math.exp(-abs(expected_gd) * 0.5))  # v6.0.1: 增强 0.08→0.15, 衰减 0.8→0.5

        deltas["G1"] = {
            "home": g1_home * 0.25,
            "draw": g1_draw * 0.25,
            "away": g1_away * 0.25,
        }

        # ================================================================
        # G2: 进球趋势 — 合并大小球趋势+进球波动率
        # ================================================================
        g2_trend = totals_trend * 0.07 * 0.60 + goal_volatility * 0.05 * 0.40
        # 大球趋势 → 推主场/客场 (更多进球 → 更可能分出胜负)
        # 小球趋势 → 推平局
        if g2_trend > 0:
            deltas["G2"] = {
                "home": g2_trend * 0.25,
                "draw": -g2_trend * 0.15 * 0.25,
                "away": g2_trend * 0.25,
            }
        else:
            deltas["G2"] = {
                "home": g2_trend * 0.25,
                "draw": -g2_trend * 0.3 * 0.25,
                "away": g2_trend * 0.25,
            }

        return deltas

    # ================================================================
    # 辅助函数
    # ================================================================

    @staticmethod
    def _compute_ewma(results: List[float], alpha: float) -> float:
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
        if not results:
            return 0.0
        advantage = 0.0
        for i, r in enumerate(results):
            w = 1.0 / (i + 1)
            if r == 3:
                advantage += w
            elif r == 0:
                advantage -= w
        return advantage / max(1, len(results))

    @staticmethod
    def _logit(p: float) -> float:
        p = max(0.001, min(0.999, p))
        return math.log(p / (1.0 - p))


def get_v6_factor_ids() -> List[str]:
    """获取所有v6.0因子ID"""
    return list(V6_FACTOR_REGISTRY.keys())


def get_v6_factor_weight(factor_id: str) -> float:
    """获取因子默认权重"""
    f = V6_FACTOR_REGISTRY.get(factor_id)
    return f.weight if f else 0.0


def get_v6_draw_factors() -> List[str]:
    """获取所有推平局的因子ID"""
    return [fid for fid, f in V6_FACTOR_REGISTRY.items() if f.draw_active]


def get_v6_factor_count() -> int:
    return len(V6_FACTOR_REGISTRY)