"""
GTO-GameFlow v5.0 联赛参数配置

严格遵循规范文档第11.2节"联赛参数表"。
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LeagueParams:
    """联赛参数 (第11.2节)"""
    league_id: str
    league_name: str
    avg_goals: float            # 场均总进球数 (泊松λ基准)
    home_advantage: float       # 主场优势系数
    sigma: float                # 比分波动标准差
    draw_rate: float            # 平局概率基准
    draw_rate_decay: float      # EWMA平局衰减
    elo_diff_scale: float       # Elo差值缩放
    time_decay_alpha: float     # 时间衰减系数
    regression_mean_k: int      # 排名回归均值场次
    poisson_delta: float        # 泊松校正系数
    style_distribution: List[int]  # [T, D, C, P] 风格分布
    yellow_card_rate: float     # 黄牌率/场
    referee_threshold: float    # 裁判阈值
    upset_frequency: float      # 冷门频率
    relegation_gap: int         # 保级分差阈值
    europe_gap: int             # 欧战分差阈值
    # 冷启动默认值
    cold_start_elo: float = 1500.0
    # v5.2: 置信度公式权重 (4因子, 总和=1.0)
    confidence_w_data: float = 0.4       # 数据完整性权重
    confidence_w_factor: float = 0.3     # 因子激活率权重
    confidence_w_dispersion: float = 0.2 # 离散度惩罚权重
    confidence_w_phase: float = 0.1      # 赛季阶段权重
    # v5.3b: 客场价值折扣 (0.0=不折扣, 0.5=减半)
    away_value_discount: float = 0.0
    # v5.10.8: 联赛独立核心参数 (每个联赛独立调参)
    factor_scale: float = 0.25           # 因子缩放倍率
    fusion_weight: float = 0.60          # logit vs poisson 融合权重
    calibration_discount: float = 0.15   # 模型校准回撤 (模型×0.85 + 市场×0.15)
    confidence_threshold: float = 0.35   # 最低置信度阈值
    # v5.10.9: 渐进式校准强度乘数 (1.0=基准, >1.0=更强校准针对高概率区间)
    calibration_multiplier: float = 1.0  # 联赛差异化校准强度
    # v5.10.9: 强队客场惩罚 — 当Elo>100且投注客场时, model_prob降低的比例
    # 0.0=无惩罚, 0.15=降低15% (仅法甲使用)
    strong_away_penalty: float = 0.0


# 五大联赛参数 (规范文档第11.2节)
LEAGUE_PARAMS: Dict[str, LeagueParams] = {
    "premier_league": LeagueParams(
        league_id="premier_league",
        league_name="英超",
        avg_goals=2.98,  # v6.0: 2.85→2.98 匹配真实数据
        home_advantage=0.38,
        sigma=1.70,  # v6.0: 1.65→1.70 匹配真实方差
        draw_rate=0.225,  # v6.0: 0.24→0.225 匹配真实平局率
        draw_rate_decay=0.85,
        elo_diff_scale=1.0,
        time_decay_alpha=0.18,
        regression_mean_k=4,
        poisson_delta=0.06,
        style_distribution=[30, 25, 20, 25],
        yellow_card_rate=3.5,
        referee_threshold=4.0,
        upset_frequency=0.35,
        relegation_gap=6,
        europe_gap=8,
        confidence_w_data=0.40,       # 英超数据最丰富
        confidence_w_factor=0.30,
        confidence_w_dispersion=0.20,
        confidence_w_phase=0.10,
        factor_scale=0.23,             # 英超竞争激烈，因子信号略保守
        fusion_weight=0.60,            # 双域融合基准
        calibration_discount=0.15,     # 模型校准回撤基准
        confidence_threshold=0.36,     # 略高门槛过滤噪声
        calibration_multiplier=1.0,    # 英超基准校准
    ),
    "la_liga": LeagueParams(
        league_id="la_liga",
        league_name="西甲",
        avg_goals=2.55,
        home_advantage=0.42,
        sigma=1.50,
        draw_rate=0.26,
        draw_rate_decay=0.82,
        elo_diff_scale=0.95,
        time_decay_alpha=0.20,
        regression_mean_k=3,
        poisson_delta=0.08,
        style_distribution=[35, 15, 25, 25],
        yellow_card_rate=4.2,
        referee_threshold=5.0,
        upset_frequency=0.38,
        relegation_gap=6,
        europe_gap=8,
        confidence_w_data=0.35,       # 西甲战术性强，因子权重更高
        confidence_w_factor=0.35,
        confidence_w_dispersion=0.20,
        confidence_w_phase=0.10,
        factor_scale=0.25,             # 西甲模型表现最佳，保持基准
        fusion_weight=0.60,            # 双域融合基准
        calibration_discount=0.15,     # 模型校准回撤基准
        confidence_threshold=0.35,     # 基准门槛
        calibration_multiplier=1.0,    # 西甲模型表现最佳，基准校准
    ),
    "bundesliga": LeagueParams(
        league_id="bundesliga",
        league_name="德甲",
        avg_goals=3.17,  # v6.0: 3.05→3.17 匹配真实数据
        home_advantage=0.45,
        sigma=1.74,  # v6.0: 1.80→1.74 匹配真实方差
        draw_rate=0.249,  # v6.0: 0.22→0.249 匹配真实平局率
        draw_rate_decay=0.88,
        elo_diff_scale=1.05,
        time_decay_alpha=0.16,
        regression_mean_k=5,
        poisson_delta=0.06,
        style_distribution=[25, 30, 30, 15],
        yellow_card_rate=3.8,
        referee_threshold=3.5,
        upset_frequency=0.32,
        relegation_gap=6,
        europe_gap=8,
        confidence_w_data=0.35,       # 德甲高方差，离散度惩罚更重要
        confidence_w_factor=0.25,
        confidence_w_dispersion=0.30,
        confidence_w_phase=0.10,
        factor_scale=0.22,             # 德甲高比分方差，因子信号略保守
        fusion_weight=0.55,            # 偏向泊松模型 (高比分联赛)
        calibration_discount=0.15,     # 模型校准回撤基准
        confidence_threshold=0.37,     # 略高门槛过滤高方差噪声
        calibration_multiplier=1.0,    # 德甲高方差，基准校准
    ),
    "serie_a": LeagueParams(
        league_id="serie_a",
        league_name="意甲",
        avg_goals=2.65,
        home_advantage=0.40,
        sigma=1.55,
        draw_rate=0.28,
        draw_rate_decay=0.80,
        elo_diff_scale=0.95,       # v5.3b: 0.90→0.95 减少Elo压缩，避免高估弱队
        time_decay_alpha=0.22,
        regression_mean_k=3,
        poisson_delta=0.06,         # v5.3b: 0.10→0.06 降低泊松偏离度，抑制客场过度投注
        style_distribution=[25, 30, 25, 20],  # v5.3b: 防守40%→30% 平衡攻防风格
        yellow_card_rate=4.5,
        referee_threshold=5.5,
        upset_frequency=0.35,       # v5.3b: 0.40→0.35 降低冷门预期，与英超对齐
        relegation_gap=6,
        europe_gap=8,
        confidence_w_data=0.30,
        confidence_w_factor=0.30,
        confidence_w_dispersion=0.20,
        confidence_w_phase=0.10,    # v5.3b: 0.20→0.10 降低赛季阶段权重，缓解赛季末崩盘
        away_value_discount=0.35,  # v5.3b: 客场价值折扣35%，抑制客场过度投注
        factor_scale=0.20,             # 意甲战术防守强，因子信号更保守
        fusion_weight=0.65,            # 偏向logit模型 (战术性强)
        calibration_discount=0.18,     # 更信任市场 (意甲冷门多)
        confidence_threshold=0.33,     # 略低门槛增加投注机会
        calibration_multiplier=1.0,    # 意甲基准校准
    ),
    "ligue_1": LeagueParams(
        league_id="ligue_1",
        league_name="法甲",
        avg_goals=2.78,  # v6.0: 2.70→2.78 匹配真实数据
        home_advantage=0.42,            # 法甲主场优势 (分析确认)
        sigma=1.68,  # v6.0: 1.50→1.68 匹配真实方差
        draw_rate=0.258,  # v6.0: 0.30→0.258 匹配真实平局率 (关键修正!)
        draw_rate_decay=0.78,
        elo_diff_scale=0.85,
        time_decay_alpha=0.24,
        regression_mean_k=2,
        poisson_delta=0.09,
        style_distribution=[30, 30, 20, 20],
        yellow_card_rate=3.6,
        referee_threshold=4.0,
        upset_frequency=0.42,
        relegation_gap=6,
        europe_gap=8,
        confidence_w_data=0.30,       # 法甲冷门多，数据质量和因子权重并重
        confidence_w_factor=0.30,
        confidence_w_dispersion=0.25,
        confidence_w_phase=0.15,
        away_value_discount=0.50,      # v5.10.10: 强队客场折扣50% (分析: 强队客场大热倒灶)
        factor_scale=0.18,             # 法甲冷门频繁，因子信号最保守
        fusion_weight=0.60,            # 双域融合基准
        calibration_discount=0.20,     # 更信任市场 (法甲不确定性高)
        confidence_threshold=0.45,     # v5.10.10: 高门槛过滤冷门噪声 (0.40→0.45)
        calibration_multiplier=1.5,    # 法甲强队客场系统性高估 (分析确认)
        strong_away_penalty=0.20,      # v5.10.10: 直接降低客场概率+重归一化
    ),
}


def get_league_params(league_id: str) -> LeagueParams:
    """获取联赛参数"""
    if league_id not in LEAGUE_PARAMS:
        raise ValueError(f"不支持的联赛: {league_id}. 支持: {list(LEAGUE_PARAMS.keys())}")
    return LEAGUE_PARAMS[league_id]