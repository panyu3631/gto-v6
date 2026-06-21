"""
GTO-GameFlow v5.0 因子注册中心

严格遵循规范文档第4章(F1-F18)、第5章(F19-F32)、第12章(F33-F41)。
共 41 个有效因子 (F14 已废弃)。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class FactorCategory(str, Enum):
    BASE = "base"
    ENHANCED = "enhanced"
    LEAGUE_SPECIFIC = "league"


class EloCategory(str, Enum):
    """因子与 Elo 的关系分类 — 用于正交分解"""
    DIRECT = "direct"           # 直接使用 Elo (F1)
    DERIVED = "derived"         # 从 Elo 派生 (F7, F8, F19, ...)
    CORRELATED = "correlated"   # 与 Elo 高度相关 (F3, F5, F20, F38)
    INDEPENDENT = "independent" # 独立于 Elo


# v6.0: 因子-Elo 分类映射 (从 signal_decomposer.py 迁移到此处)
ELO_CATEGORY_MAP: Dict[str, EloCategory] = {
    "F1": EloCategory.DIRECT,
    "F7": EloCategory.DERIVED, "F8": EloCategory.DERIVED,
    "F19": EloCategory.DERIVED, "F27": EloCategory.DERIVED,
    "F29": EloCategory.DERIVED, "F33": EloCategory.DERIVED,
    "F37": EloCategory.DERIVED, "F39": EloCategory.DERIVED, "F40": EloCategory.DERIVED,
    "F3": EloCategory.CORRELATED, "F5": EloCategory.CORRELATED,
    "F20": EloCategory.CORRELATED, "F38": EloCategory.CORRELATED,
}

# v6.0: 禁用的因子列表
DISABLED_FACTORS: set = {"F10", "F11", "F14"}


def get_elo_category(factor_id: str) -> EloCategory:
    """获取因子的 Elo 分类"""
    return ELO_CATEGORY_MAP.get(factor_id, EloCategory.INDEPENDENT)


def is_factor_disabled(factor_id: str) -> bool:
    """检查因子是否被禁用"""
    return factor_id in DISABLED_FACTORS


def get_factors_by_elo_category(category: EloCategory) -> set:
    """获取指定 Elo 分类的所有因子"""
    if category == EloCategory.INDEPENDENT:
        # 独立因子 = 所有不在其他分类中的因子
        all_categorized = set(ELO_CATEGORY_MAP.keys())
        return {fid for fid in FACTOR_REGISTRY.keys() if fid not in all_categorized}
    return {fid for fid, cat in ELO_CATEGORY_MAP.items() if cat == category}


class DegradationStrategy(str, Enum):
    USE_DEFAULT = "use_default"
    SKIP = "skip"


@dataclass
class FactorDefinition:
    factor_id: str
    name: str
    name_cn: str
    category: FactorCategory
    default_weight: float
    delta_signs: Dict[str, int]   # {"home": +1, "draw": 0, "away": -1}
    data_sources: List[str]
    formula: str = ""             # 公式文本
    degradation_strategy: DegradationStrategy = DegradationStrategy.USE_DEFAULT
    degradation_value: float = 0.0
    description: str = ""
    applicable_leagues: Optional[List[str]] = None


# ============================================================
# 第4章：基础因子 F1-F18 (F14 废弃)
# ============================================================
FACTOR_REGISTRY: Dict[str, FactorDefinition] = {
    "F1": FactorDefinition(
        factor_id="F1", name="elo_rating", name_cn="ELO评分",
        category=FactorCategory.BASE, default_weight=1.0,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["clubelo"],
        formula="delta = k × (elo_diff / 400), k=0.5",
        description="基于ClubElo数据的球队综合实力评分，ELO差值除以400的标准近似",
    ),
    "F2": FactorDefinition(
        factor_id="F2", name="core_injuries", name_cn="核心伤停",
        category=FactorCategory.BASE, default_weight=0.9,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["whoscored", "transfermarkt"],
        formula="delta = (xi_rating - 6.0) × 0.08",
        description="预期首发加权平均评分(WhoScored 1-10分)与联赛平均6.0的偏差",
    ),
    "F3": FactorDefinition(
        factor_id="F3", name="recent_form", name_cn="近期状态",
        category=FactorCategory.BASE, default_weight=0.85,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = (ewma_form - 1.5) × 0.15",
        description="近5场EWMA评分(胜=3/平=1/负=0)，1.5为中性基准线",
    ),
    "F4": FactorDefinition(
        factor_id="F4", name="home_advantage", name_cn="主客场",
        category=FactorCategory.BASE, default_weight=0.7,
        delta_signs={"home": +1, "draw": 0, "away": 0},
        data_sources=["league_params"],
        formula="delta_home = home_advantage × 0.1",
        description="联赛主场优势系数，基于历史主胜率与中立场地胜率差值",
    ),
    "F5": FactorDefinition(
        factor_id="F5", name="h2h_history", name_cn="历史交锋",
        category=FactorCategory.BASE, default_weight=0.6,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = h2h_advantage × 0.08",
        description="近3赛季历史交锋加权胜率差，近期权重更高，范围[-1,+1]",
    ),
    "F6": FactorDefinition(
        factor_id="F6", name="schedule_density", name_cn="赛程密度",
        category=FactorCategory.BASE, default_weight=0.7,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_schedule"],
        formula="delta = -(matches_7d - 1) × 0.03",
        description="过去7天比赛数量，超过1场产生体能惩罚",
    ),
    "F7": FactorDefinition(
        factor_id="F7", name="rank_diff", name_cn="联赛排名差",
        category=FactorCategory.BASE, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["league_table"],
        formula="delta = (rank_diff / 20) × 0.1",
        description="rank_diff = rank_away - rank_home，正值表示主队排名更靠前",
    ),
    "F8": FactorDefinition(
        factor_id="F8", name="goal_diff", name_cn="进球/失球差",
        category=FactorCategory.BASE, default_weight=0.55,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = (goal_diff / 10) × 0.06",
        description="赛季累积净胜球差值，除以10归一化",
    ),
    "F9": FactorDefinition(
        factor_id="F9", name="xg_diff", name_cn="xG差值",
        category=FactorCategory.BASE, default_weight=0.8,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["understat"],
        formula="delta = xG_diff × 0.12",
        description="xG差值=(xG_home-xGA_home)-(xG_away-xGA_away)，权重为F8的两倍",
    ),
    "F10": FactorDefinition(
        factor_id="F10", name="odds_implied", name_cn="赔率隐含概率",
        category=FactorCategory.BASE, default_weight=0.9,
        delta_signs={"home": +1, "draw": +1, "away": +1},
        data_sources=["api_football"],
        formula="delta = logit(market_prob) - logit(0.33)",
        description="去除庄家margin后的公平概率logit值与均匀分布先验的差值",
    ),
    "F11": FactorDefinition(
        factor_id="F11", name="odds_movement", name_cn="市场赔率变动",
        category=FactorCategory.BASE, default_weight=0.6,
        delta_signs={"home": +1, "draw": +1, "away": +1},
        data_sources=["api_football"],
        formula="delta = (opening_prob - current_prob) × 0.5",
        description="赔率从开盘到当前的变化幅度和方向，反映市场资金流向",
    ),
    "F12": FactorDefinition(
        factor_id="F12", name="weather_impact", name_cn="天气影响",
        category=FactorCategory.BASE, default_weight=0.3,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["weather_api"],
        formula="delta = weather_impact × 0.08",
        description="0.3×温度评分+0.4×降雨评分+0.3×风速评分，极端天气产生显著负向调整",
    ),
    "F13": FactorDefinition(
        factor_id="F13", name="referee_style", name_cn="裁判风格",
        category=FactorCategory.BASE, default_weight=0.4,
        delta_signs={"home": +1, "draw": 0, "away": 0},
        data_sources=["match_history"],
        formula="delta = (ref_yellow_rate - league_avg) × 0.02",
        description="裁判场均黄牌率与联赛平均的偏差，严格裁判抑制身体对抗优势",
    ),
    "F14": FactorDefinition(
        factor_id="F14", name="squad_value", name_cn="[已废弃] 球队身价",
        category=FactorCategory.BASE, default_weight=0.0,
        delta_signs={"home": 0, "draw": 0, "away": 0},
        data_sources=[],
        formula="已废弃",
        degradation_strategy=DegradationStrategy.SKIP,
        description="已废弃(VIF>8)，由F34联赛特定身价调整因子替代",
    ),
    "F15": FactorDefinition(
        factor_id="F15", name="coach_change", name_cn="教练更替",
        category=FactorCategory.BASE, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["transfermarkt"],
        formula="delta = coach_change_effect × 0.07",
        description="换帅后前3场+0.084，5场后+0.06，10场后衰减至0",
    ),
    "F16": FactorDefinition(
        factor_id="F16", name="european_fatigue", name_cn="欧战影响",
        category=FactorCategory.BASE, default_weight=0.6,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_schedule"],
        formula="delta = fatigue_penalty × 0.04",
        description="欧战后间隔3天=-0.8, 4天=-0.4, 5天=-0.1, 6+天=0",
    ),
    "F17": FactorDefinition(
        factor_id="F17", name="rotation_risk", name_cn="轮换预测",
        category=FactorCategory.BASE, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_schedule", "transfermarkt"],
        formula="delta = rotation_risk × 0.05",
        description="综合考虑下场比赛重要性、赛程密度和板凳深度，范围[0,1]",
    ),
    "F18": FactorDefinition(
        factor_id="F18", name="derby_match", name_cn="德比战",
        category=FactorCategory.BASE, default_weight=0.7,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["match_schedule"],
        formula="delta = derby_boost × 0.06",
        description="国家德比=0.8, 同城德比=0.6, 地区德比=0.4，平局概率上升。v5.5.1: 合并原F42(德比战强度), 统一为一个因子",
    ),

    # ============================================================
    # 第5章：通用增强因子 F19-F32
    # ============================================================
    "F19": FactorDefinition(
        factor_id="F19", name="attack_defense_style", name_cn="攻击/防守风格",
        category=FactorCategory.ENHANCED, default_weight=0.6,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["whoscored"],
        formula="delta = (style_matchup_score - 0.5) × 0.10",
        description="两队风格向量[T/D/C/P]的余弦相似度，风格相克产生正向调整",
    ),
    "F20": FactorDefinition(
        factor_id="F20", name="streak_momentum", name_cn="连胜/连败动量",
        category=FactorCategory.ENHANCED, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = streak_momentum × 0.08",
        description="3连胜=+0.4, 5连胜=+0.7, 8+连胜=+1.0(边际递减)。与F38互斥",
    ),
    "F21": FactorDefinition(
        factor_id="F21", name="key_player_form", name_cn="核心球员状态",
        category=FactorCategory.ENHANCED, default_weight=0.7,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["whoscored"],
        formula="delta = (player_form - 6.5) × 0.06",
        description="核心球员近3场WhoScored评分加权平均，6.5为基准线",
    ),
    "F22": FactorDefinition(
        factor_id="F22", name="market_sentiment", name_cn="市场情绪",
        category=FactorCategory.ENHANCED, default_weight=0.4,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["news_api"],
        formula="delta = (market_sentiment - 0) × 0.04",
        description="社交媒体/论坛NLP情感分析，范围[-1,+1]，捕捉非理性成分",
    ),
    "F23": FactorDefinition(
        factor_id="F23", name="odds_discrepancy", name_cn="赔率离散度",
        category=FactorCategory.ENHANCED, default_weight=0.5,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["api_football"],
        formula="delta = (odds_std - 0.05) × 0.5",
        description="跨公司赔率隐含概率标准差，0.05为正常阈值，高离散=高不确定性",
    ),
    "F24": FactorDefinition(
        factor_id="F24", name="news_nlp", name_cn="新闻NLP",
        category=FactorCategory.ENHANCED, default_weight=0.3,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["news_api"],
        formula="delta = sentiment_score × 0.04",
        description="BERT微调模型的新闻文本情感分类输出，范围[-1,+1]",
    ),
    "F25": FactorDefinition(
        factor_id="F25", name="time_decay_weighted", name_cn="时间衰减加权",
        category=FactorCategory.ENHANCED, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = time_decay_correction × 0.05",
        description="指数衰减e^(-λt)，λ=0.02，t为数据距今天数",
    ),
    "F26": FactorDefinition(
        factor_id="F26", name="league_strength", name_cn="联赛强度调整",
        category=FactorCategory.ENHANCED, default_weight=0.4,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["uefa_coefficients"],
        formula="delta = league_strength_bias × 0.05",
        description="基于UEFA联赛系数: 英超=1.0, 西甲=0.95, 德甲=0.88, 意甲=0.85, 法甲=0.78",
    ),
    "F27": FactorDefinition(
        factor_id="F27", name="goal_distribution_correction", name_cn="进球分布修正",
        category=FactorCategory.ENHANCED, default_weight=0.6,
        delta_signs={"home": +1, "draw": +1, "away": +1},
        data_sources=["match_history"],
        formula="delta = poisson_correction × 0.20",
        description="实际进球方差与泊松理论方差的偏差修正，调整幅度上限±0.20",
    ),
    "F28": FactorDefinition(
        factor_id="F28", name="asian_handicap_depth", name_cn="亚盘深度",
        category=FactorCategory.ENHANCED, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["api_football"],
        formula="delta = handicap_depth × 0.06",
        description="|handicap|/max_handicap，max_handicap=2.5，深盘反映市场信心",
    ),
    "F29": FactorDefinition(
        factor_id="F29", name="totals_trend", name_cn="大小球趋势",
        category=FactorCategory.ENHANCED, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": +1},
        data_sources=["match_history"],
        formula="delta = totals_trend × 0.07",
        description="近5场场均总进球与联赛均值的偏差，正值表示大球倾向",
    ),
    "F30": FactorDefinition(
        factor_id="F30", name="value_signal", name_cn="赔率价值信号",
        category=FactorCategory.ENHANCED, default_weight=0.7,
        delta_signs={"home": +1, "draw": +1, "away": +1},
        data_sources=["api_football"],
        formula="delta = value_signal × 0.10",
        description="P_system_excl_F30 - P_market，使用不含F30的40因子子集避免循环依赖",
    ),
    "F31": FactorDefinition(
        factor_id="F31", name="contrarian_signal", name_cn="反市场偏差",
        category=FactorCategory.ENHANCED, default_weight=0.4,
        delta_signs={"home": +1, "draw": 0, "away": +1},
        data_sources=["api_football"],
        formula="delta = contrarian_signal × 0.05",
        description="1-volume_on_favourite/total_volume，捕捉热门-弱势偏差",
    ),
    "F32": FactorDefinition(
        factor_id="F32", name="market_efficiency", name_cn="市场效率评分",
        category=FactorCategory.ENHANCED, default_weight=0.3,
        delta_signs={"home": +1, "draw": +1, "away": +1},
        data_sources=["api_football"],
        formula="delta = market_efficiency × 0.03",
        description="1-Brier(market_probs, outcomes)，高效市场>0.85，低效市场<0.65",
    ),

    # ============================================================
    # 第12章：联赛特定因子 F33-F41 (v5.5.1: F42已合并到F18)
    # ============================================================
    "F33": FactorDefinition(
        factor_id="F33", name="relegation_title_motivation", name_cn="保级/争冠动力",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.6,
        delta_signs={"home": +1, "draw": 0, "away": 0},
        data_sources=["league_table"],
        formula="delta = motivation_boost / 100",
        description="保级+8%, 争四/争冠+5%, 其他0。距边界≤3分触发",
        applicable_leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
    ),
    "F34": FactorDefinition(
        factor_id="F34", name="financial_disparity", name_cn="财力差距",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.4,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["transfermarkt"],
        formula="delta = financial_gap_effect / 100",
        description="身价比>5x +0.05, >10x +0.08。替代F14",
        applicable_leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
    ),
    "F35": FactorDefinition(
        factor_id="F35", name="winter_break_effect", name_cn="冬歇期效应",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.8,
        delta_signs={"home": +1, "draw": +1, "away": -1},
        data_sources=["match_schedule"],
        formula="delta = winter_break_effect × 0.10",
        description="仅德甲激活，冬歇期后首轮状态不确定性",
        applicable_leagues=["bundesliga"],
    ),
    "F36": FactorDefinition(
        factor_id="F36", name="christmas_fixtures", name_cn="圣诞赛程",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.7,
        delta_signs={"home": -1, "draw": +1, "away": 0},
        data_sources=["match_schedule"],
        formula="delta = christmas_fatigue × 0.12",
        description="仅英超12月-1月激活，密集赛程体能消耗",
        applicable_leagues=["premier_league"],
    ),
    "F37": FactorDefinition(
        factor_id="F37", name="midtable_complacency", name_cn="中游无欲",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.5,
        delta_signs={"home": -1, "draw": +1, "away": +1},
        data_sources=["league_table"],
        formula="delta = complacency_effect × 0.05",
        description="赛季末段中游球队缺乏动力，容易被有目标的球队击败",
        applicable_leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
    ),
    "F38": FactorDefinition(
        factor_id="F38", name="streak_momentum_league", name_cn="连胜/连败(联赛特定)",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = streak_momentum × 0.08 (联赛校准)",
        description="联赛特定动量参数校准版本，与F20互斥。优先使用F38",
        applicable_leagues=["premier_league", "bundesliga"],
    ),
    "F39": FactorDefinition(
        factor_id="F39", name="table_position", name_cn="积分榜",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.6,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["league_table"],
        formula="delta = (position_advantage / 20) × 0.12",
        description="积分榜排名差归一化，西甲/意甲权重最高",
        applicable_leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
    ),
    "F40": FactorDefinition(
        factor_id="F40", name="promoted_team_data", name_cn="升班马数据",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.5,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_history"],
        formula="delta = (secondary_league_data × mapping_coefficient) × 0.08",
        description="升班马使用次级联赛数据映射，冷启动10轮后使用本季数据",
        applicable_leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
    ),
    "F41": FactorDefinition(
        factor_id="F41", name="schedule_advantage", name_cn="赛程优势",
        category=FactorCategory.LEAGUE_SPECIFIC, default_weight=0.4,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_schedule"],
        formula="delta = schedule_advantage × 0.08",
        description="对手赛程密度 vs 本队赛程密度，休息优势产生正向调整",
        applicable_leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
    ),
    # v5.5.1: F42 已合并到 F18 (德比战)，不再单独存在
    
    # ============================================================
    # v5.10.8: 第13章 — 比赛统计衍生因子 F42-F55
    # 从 CSV 比赛统计(射门/犯规/角球/黄红牌/半场/收盘赔率)提取
    # ============================================================
    "F42": FactorDefinition(
        factor_id="F42", name="ht_momentum", name_cn="半场动量",
        category=FactorCategory.ENHANCED, default_weight=0.35,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = ht_momentum × 0.08",
        description="半场领先→全场获胜转化率，综合半场领先胜率和落后逆转率",
    ),
    "F43": FactorDefinition(
        factor_id="F43", name="shot_efficiency_diff", name_cn="射门效率差",
        category=FactorCategory.ENHANCED, default_weight=0.40,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = shot_eff_diff × 0.06",
        description="射正率×射门转化率的复合差值，反映进攻效率",
    ),
    "F44": FactorDefinition(
        factor_id="F44", name="territorial_dominance", name_cn="控场优势",
        category=FactorCategory.ENHANCED, default_weight=0.35,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = territorial_dominance × 0.06",
        description="射门比(0.5)+角球比(0.3)+犯规比倒数(0.2)的加权复合",
    ),
    "F45": FactorDefinition(
        factor_id="F45", name="discipline_index", name_cn="纪律指数",
        category=FactorCategory.ENHANCED, default_weight=0.25,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = discipline_index × 0.05",
        description="黄牌/犯规比+红牌风险的复合差值，正值=主队更纪律",
    ),
    "F46": FactorDefinition(
        factor_id="F46", name="odds_drift", name_cn="赔率漂移信号",
        category=FactorCategory.ENHANCED, default_weight=0.45,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["closing_odds"],
        formula="delta = odds_drift × 0.10",
        description="开盘→收盘赔率变动方向，正值=市场看好主队",
    ),
    "F47": FactorDefinition(
        factor_id="F47", name="market_disagreement", name_cn="市场分歧",
        category=FactorCategory.ENHANCED, default_weight=0.30,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["closing_odds"],
        formula="delta = market_disagreement × 0.06",
        description="6家博彩商赔率标准差，分歧越大平局概率越高",
    ),
    "F48": FactorDefinition(
        factor_id="F48", name="referee_home_bias", name_cn="裁判主场偏置",
        category=FactorCategory.ENHANCED, default_weight=0.30,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = referee_home_bias × 0.08",
        description="裁判历史主胜率 vs 联赛平均主胜率，正值=裁判偏主场",
    ),
    "F49": FactorDefinition(
        factor_id="F49", name="comeback_resilience", name_cn="逆转韧性",
        category=FactorCategory.ENHANCED, default_weight=0.25,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = comeback_resilience × 0.06",
        description="半场落后→全场不输的转化率，反映球队韧性",
    ),
    "F50": FactorDefinition(
        factor_id="F50", name="streak_momentum_enriched", name_cn="连胜动量(增强)",
        category=FactorCategory.ENHANCED, default_weight=0.40,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = streak_momentum × 0.08",
        description="近期连胜/连败场次，连续累计，胜+1/负-1",
    ),
    "F51": FactorDefinition(
        factor_id="F51", name="goal_volatility", name_cn="进球波动率",
        category=FactorCategory.ENHANCED, default_weight=0.20,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["match_stats"],
        formula="delta = goal_volatility × 0.05",
        description="近期进球数标准差，波动率差越大平局概率越高",
    ),
    "F52": FactorDefinition(
        factor_id="F52", name="corner_dominance", name_cn="角球优势",
        category=FactorCategory.ENHANCED, default_weight=0.20,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = corner_dominance × 0.05",
        description="角球比的差值，反映进攻端的控场能力",
    ),
    "F53": FactorDefinition(
        factor_id="F53", name="sot_rate_diff", name_cn="射正率差",
        category=FactorCategory.ENHANCED, default_weight=0.30,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["match_stats"],
        formula="delta = sot_rate_diff × 0.06",
        description="射正/射门比差值，衡量进攻精准度",
    ),
    "F54": FactorDefinition(
        factor_id="F54", name="ah_odds_drift", name_cn="亚盘赔率漂移",
        category=FactorCategory.ENHANCED, default_weight=0.35,
        delta_signs={"home": +1, "draw": 0, "away": -1},
        data_sources=["closing_odds"],
        formula="delta = ah_odds_drift × 0.08",
        description="亚盘开盘→收盘水位变动，正值=市场看好主队",
    ),
    "F55": FactorDefinition(
        factor_id="F55", name="totals_odds_drift", name_cn="大小球赔率漂移",
        category=FactorCategory.ENHANCED, default_weight=0.25,
        delta_signs={"home": +1, "draw": 0, "away": +1},
        data_sources=["closing_odds"],
        formula="delta = totals_odds_drift × 0.06",
        description="大小球开盘→收盘水位变动，正值=市场倾向大球",
    ),

    # ============================================================
    # v5.11: 平局专属因子 F56-F58
    # 解决54因子中仅5个影响平局的根本缺陷
    # ============================================================
    "F56": FactorDefinition(
        factor_id="F56", name="draw_tactical_matchup", name_cn="战术风格平局倾向",
        category=FactorCategory.ENHANCED, default_weight=0.50,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["match_stats"],
        formula="delta = draw_tactical_matchup × 0.08",
        description="防守型vs防守型对决→平局概率高(+1)，进攻型vs进攻型→分出胜负(-1)",
    ),
    "F57": FactorDefinition(
        factor_id="F57", name="draw_goal_expectancy", name_cn="进球预期平局信号",
        category=FactorCategory.ENHANCED, default_weight=0.45,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["match_stats"],
        formula="delta = draw_goal_expectancy × 0.10",
        description="低进球预期(射正少)→0:0/1:1概率高→平局+，高进球预期→平局-",
    ),
    "F58": FactorDefinition(
        factor_id="F58", name="draw_team_tendency", name_cn="球队平局历史倾向",
        category=FactorCategory.ENHANCED, default_weight=0.40,
        delta_signs={"home": 0, "draw": +1, "away": 0},
        data_sources=["match_stats"],
        formula="delta = draw_team_tendency × 0.06",
        description="两队近期平局率几何平均 vs 联赛平均，马竞/尤文等平局大师被捕捉",
    ),
}


# ============================================================
# 按联赛的因子权重配置 (第11.3节)
# 格式: { league_id: { factor_id: weight, ..., "kelly_discount": 0.25 } }
# ============================================================
LEAGUE_FACTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "premier_league": {
        "F1": 1.0, "F2": 0.9, "F3": 0.85, "F4": 0.7, "F5": 0.6,
        "F6": 0.7, "F7": 0.5, "F8": 0.55, "F9": 0.8, "F10": 0.9,
        "F11": 0.6, "F12": 0.3, "F13": 0.4, "F14": 0.0, "F15": 0.5,
        "F16": 0.6, "F17": 0.5, "F18": 0.7,
        "F19": 0.6, "F20": 0.5, "F21": 0.7, "F22": 0.4, "F23": 0.5,
        "F24": 0.3, "F25": 0.5, "F26": 0.4, "F27": 0.6, "F28": 0.5,
        "F29": 0.5, "F30": 0.7, "F31": 0.4, "F32": 0.3,
        "F33": 0.6, "F34": 0.4, "F35": 0.0, "F36": 0.7, "F37": 0.5,
        "F38": 0.5, "F39": 0.6, "F40": 0.5, "F41": 0.4,
        "F42": 0.35, "F43": 0.40, "F44": 0.35, "F45": 0.25, "F46": 0.45,
        "F47": 0.30, "F48": 0.30, "F49": 0.25, "F50": 0.40, "F51": 0.20,
        "F52": 0.20, "F53": 0.30, "F54": 0.35, "F55": 0.25,
        "F56": 0.50, "F57": 0.45, "F58": 0.40,
        "kelly_discount": 0.25,
    },
    "la_liga": {
        "F1": 1.0, "F2": 0.85, "F3": 0.8, "F4": 0.75, "F5": 0.65,
        "F6": 0.65, "F7": 0.55, "F8": 0.5, "F9": 0.75, "F10": 0.85,
        "F11": 0.55, "F12": 0.25, "F13": 0.5, "F14": 0.0, "F15": 0.5,
        "F16": 0.55, "F17": 0.5, "F18": 0.65,
        "F19": 0.7, "F20": 0.5, "F21": 0.7, "F22": 0.35, "F23": 0.45,
        "F24": 0.3, "F25": 0.45, "F26": 0.35, "F27": 0.55, "F28": 0.45,
        "F29": 0.45, "F30": 0.65, "F31": 0.4, "F32": 0.3,
        "F33": 0.7, "F34": 0.5, "F35": 0.0, "F36": 0.0, "F37": 0.5,
        "F38": 0.5, "F39": 0.65, "F40": 0.5, "F41": 0.35,
        "F42": 0.35, "F43": 0.40, "F44": 0.35, "F45": 0.25, "F46": 0.45,
        "F47": 0.30, "F48": 0.30, "F49": 0.25, "F50": 0.40, "F51": 0.20,
        "F52": 0.20, "F53": 0.30, "F54": 0.35, "F55": 0.25,
        "F56": 0.50, "F57": 0.45, "F58": 0.40,
        "kelly_discount": 0.25,
    },
    "bundesliga": {
        "F1": 1.0, "F2": 0.85, "F3": 0.85, "F4": 0.8, "F5": 0.6,
        "F6": 0.65, "F7": 0.5, "F8": 0.55, "F9": 0.8, "F10": 0.85,
        "F11": 0.55, "F12": 0.3, "F13": 0.45, "F14": 0.0, "F15": 0.5,
        "F16": 0.55, "F17": 0.5, "F18": 0.6,
        "F19": 0.55, "F20": 0.5, "F21": 0.65, "F22": 0.35, "F23": 0.45,
        "F24": 0.3, "F25": 0.5, "F26": 0.35, "F27": 0.55, "F28": 0.5,
        "F29": 0.5, "F30": 0.65, "F31": 0.4, "F32": 0.3,
        "F33": 0.55, "F34": 0.4, "F35": 0.8, "F36": 0.0, "F37": 0.45,
        "F38": 0.5, "F39": 0.55, "F40": 0.5, "F41": 0.4,
        "F42": 0.35, "F43": 0.40, "F44": 0.35, "F45": 0.25, "F46": 0.45,
        "F47": 0.30, "F48": 0.30, "F49": 0.25, "F50": 0.40, "F51": 0.20,
        "F52": 0.20, "F53": 0.30, "F54": 0.35, "F55": 0.25,
        "F56": 0.50, "F57": 0.45, "F58": 0.40,
        "kelly_discount": 0.25,
    },
    "serie_a": {
        "F1": 1.0, "F2": 0.8, "F3": 0.75, "F4": 0.7, "F5": 0.65,
        "F6": 0.6, "F7": 0.55, "F8": 0.5, "F9": 0.7, "F10": 0.8,
        "F11": 0.5, "F12": 0.2, "F13": 0.55, "F14": 0.0, "F15": 0.55,
        "F16": 0.5, "F17": 0.5, "F18": 0.6,
        "F19": 0.6, "F20": 0.5, "F21": 0.65, "F22": 0.3, "F23": 0.4,
        "F24": 0.25, "F25": 0.4, "F26": 0.3, "F27": 0.5, "F28": 0.4,
        "F29": 0.4, "F30": 0.6, "F31": 0.35, "F32": 0.25,
        "F33": 0.7, "F34": 0.45, "F35": 0.0, "F36": 0.0, "F37": 0.5,
        "F38": 0.5, "F39": 0.65, "F40": 0.45, "F41": 0.3,
        "F42": 0.35, "F43": 0.40, "F44": 0.35, "F45": 0.25, "F46": 0.45,
        "F47": 0.30, "F48": 0.30, "F49": 0.25, "F50": 0.40, "F51": 0.20,
        "F52": 0.20, "F53": 0.30, "F54": 0.35, "F55": 0.25,
        "F56": 0.50, "F57": 0.45, "F58": 0.40,
        "kelly_discount": 0.25,
    },
    "ligue_1": {
        "F1": 1.0, "F2": 0.8, "F3": 0.8, "F4": 0.65, "F5": 0.55,
        "F6": 0.6, "F7": 0.5, "F8": 0.5, "F9": 0.7, "F10": 0.8,
        "F11": 0.5, "F12": 0.2, "F13": 0.45, "F14": 0.0, "F15": 0.45,
        "F16": 0.45, "F17": 0.45, "F18": 0.55,
        "F19": 0.55, "F20": 0.45, "F21": 0.6, "F22": 0.3, "F23": 0.4,
        "F24": 0.25, "F25": 0.45, "F26": 0.3, "F27": 0.5, "F28": 0.4,
        "F29": 0.4, "F30": 0.6, "F31": 0.35, "F32": 0.25,
        "F33": 0.55, "F34": 0.4, "F35": 0.0, "F36": 0.0, "F37": 0.5,
        "F38": 0.45, "F39": 0.55, "F40": 0.4, "F41": 0.35,
        "F42": 0.35, "F43": 0.40, "F44": 0.35, "F45": 0.25, "F46": 0.45,
        "F47": 0.30, "F48": 0.30, "F49": 0.25, "F50": 0.40, "F51": 0.20,
        "F52": 0.20, "F53": 0.30, "F54": 0.35, "F55": 0.25,
        "F56": 0.50, "F57": 0.45, "F58": 0.40,
        "kelly_discount": 0.25,
    },
}


# ============================================================
# 工具函数
# ============================================================

def get_factor(factor_id: str) -> FactorDefinition:
    """获取因子定义"""
    if factor_id not in FACTOR_REGISTRY:
        raise KeyError(f"因子 {factor_id} 不存在于注册中心")
    return FACTOR_REGISTRY[factor_id]


# v5.10.9: 回测确认激活率<5%的废弃因子
ABANDONED_FACTORS = {
    "F1", "F14", "F15", "F17", "F18", "F19", "F20",
    "F24", "F28", "F34", "F35", "F36",
}


def get_active_factors(league_id: str = None) -> Dict[str, FactorDefinition]:
    """获取所有活跃因子 (排除废弃因子，排除该联赛权重为0的因子)"""
    active = {}
    weights = LEAGUE_FACTOR_WEIGHTS.get(league_id, {}) if league_id else {}
    for fid, factor in FACTOR_REGISTRY.items():
        if fid in ABANDONED_FACTORS:
            continue
        if factor.default_weight == 0.0:
            continue
        if league_id and weights:
            w = weights.get(fid, factor.default_weight)
            if w == 0.0:
                continue
        if factor.applicable_leagues is not None and league_id is not None:
            if league_id not in factor.applicable_leagues:
                continue
        active[fid] = factor
    return active


def get_factor_count(league_id: str = None) -> int:
    """获取活跃因子数量"""
    return len(get_active_factors(league_id))


def get_factor_weight(factor_id: str, league_id: str) -> float:
    """获取指定联赛中因子的权重"""
    if league_id not in LEAGUE_FACTOR_WEIGHTS:
        return FACTOR_REGISTRY[factor_id].default_weight
    return LEAGUE_FACTOR_WEIGHTS[league_id].get(factor_id, FACTOR_REGISTRY[factor_id].default_weight)


def get_factor_ids_by_category(category: FactorCategory) -> List[str]:
    """按类别获取因子 ID 列表"""
    return [fid for fid, f in FACTOR_REGISTRY.items()
            if f.category == category and fid != "F14" and f.default_weight > 0.0]


def validate_mutual_exclusion(league_id: str):
    """验证 F20/F38 互斥逻辑"""
    w20 = get_factor_weight("F20", league_id)
    w38 = get_factor_weight("F38", league_id)
    if w20 > 0 and w38 > 0:
        return "F38"  # 优先使用 F38，禁用 F20
    return None