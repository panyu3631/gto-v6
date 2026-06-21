"""
GTO-GameFlow v6.0 V2增强层 — 将V2新因子注入v5.10.8已验证引擎

v5.10.8引擎有55个因子，但缺失以下维度:
- 实力接近度 → 平局信号 (v5.10.8的48个因子draw=0)
- 预期进球差 → 从因子推导比分，再推导平局
- 市场低效度 → 动态调整模型自信度

此模块计算这些缺失因子，作为增强层注入pipeline。
"""
import math
from typing import Dict, List


def compute_v2_enhancement_factors(
    elo_diff: float,
    recent_results: List[float] = None,
    h2h_results: List[float] = None,
    odds_std: float = 0.05,
    season_phase: float = 0.5,
    market_efficiency: float = 0.5,
    motivation_boost: float = 0.0,
    complacency_effect: float = 0.0,
    draw_rate: float = 0.26,
    home_advantage: float = 0.35,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    计算V2增强因子，返回 {factor_id: {"home": delta, "draw": delta, "away": delta}}

    这些因子补充v5.10.8引擎缺失的3个维度:
    1. V2_D3: 实力接近度 → 平局
    2. V2_G1: 预期进球差 → 平局 (从进球差推导)
    3. V2_U3: 市场低效度 → 全局权重
    4. V2_V1: 非线性动机 → 赛季阶段×积分榜
    """
    if recent_results is None:
        recent_results = [1.5, 1.5, 1.5, 1.5, 1.5]
    if h2h_results is None:
        h2h_results = [0, 0, 0, 0, 0]

    deltas = {}

    # ================================================================
    # V2_D3: 实力接近度 — 核心平局因子
    # 逻辑: 两队实力越接近，平局概率越高
    # v5.10.8的48个因子draw=0，这是最关键的结构性补充
    # ================================================================
    abs_elo = abs(elo_diff)
    if abs_elo < 80:
        d3_draw = 0.20 * (1.0 - abs_elo / 80.0)
    elif abs_elo < 200:
        d3_draw = 0.10 * (1.0 - (abs_elo - 80) / 120.0)
    else:
        d3_draw = 0.0
    d3_draw *= (draw_rate / 0.26)
    # v6.0.1: 信号减半，避免破坏v5.10.8校准
    d3_draw *= 0.5
    deltas["V2_D3"] = {"home": 0.0, "draw": d3_draw, "away": 0.0}

    # ================================================================
    # V2_G1: 预期进球差 — 从实力+状态推导进球差，再推导平局
    # 逻辑: |预期进球差| 越小 → 平局概率越高
    # 这是从"预测胜平负"到"预测比分"的方法论转变
    # ================================================================
    # 计算近期状态 (EWMA)
    ewma = 0.0
    weight_sum = 0.0
    for i, r in enumerate(recent_results):
        w = (1 - 0.08) ** i
        ewma += r * w
        weight_sum += w
    avg_form = ewma / weight_sum if weight_sum > 0 else 1.5

    # 计算交锋优势
    h2h_adv = 0.0
    for i, r in enumerate(h2h_results):
        w = 1.0 / (i + 1)
        if r == 3:
            h2h_adv += w
        elif r == 0:
            h2h_adv -= w
    h2h_adv = h2h_adv / max(1, len(h2h_results))

    # 预期进球差
    base_gd = elo_diff / 400.0 * 0.8 + (avg_form - 1.5) * 0.4 + h2h_adv * 0.3
    home_bonus = home_advantage * 0.5
    expected_gd = base_gd + home_bonus

    g1_home = max(0.0, expected_gd * 0.15)  # v6.0.1: 减半 0.3→0.15
    g1_away = max(0.0, -expected_gd * 0.15)
    g1_draw = max(0.0, 0.06 * math.exp(-abs(expected_gd) * 0.5))  # v6.0.1: 减半 0.12→0.06

    deltas["V2_G1"] = {"home": g1_home, "draw": g1_draw, "away": g1_away}

    # ================================================================
    # V2_U3: 市场低效度 — 全局信号增强
    # 逻辑: 市场效率低 → 模型独立判断更有价值
    # 三方向等值不影响方向性，但放大模型整体信号
    # ================================================================
    inefficiency = (
        (1.0 - max(0.0, min(1.0, market_efficiency))) * 0.40 +
        (odds_std / 0.15) * 0.30 +
        (1.0 - season_phase) * 0.30
    )
    inefficiency = max(0.0, min(1.0, inefficiency))
    u3_signal = inefficiency * 0.03  # v6.0.1: 减半 0.06→0.03
    deltas["V2_U3"] = {"home": u3_signal, "draw": u3_signal, "away": u3_signal}

    # ================================================================
    # V2_V1: 非线性动机
    # 逻辑: 赛季末段+保级/争冠动力 → 非线性放大
    # 中游无欲 → 赛季末段强烈抑制
    # ================================================================
    v1_motivation = motivation_boost / 100.0 * 0.5  # v6.0.1: 减半
    v1_draw = 0.0

    if season_phase > 0.7:
        v1_motivation *= 1.5

    if complacency_effect > 0 and season_phase > 0.6:
        v1_motivation -= complacency_effect * 0.08 * (1.0 + season_phase)

    if abs(motivation_boost) > 5 and season_phase > 0.6:
        v1_draw = abs(motivation_boost) / 100.0 * 0.3

    deltas["V2_V1"] = {"home": v1_motivation, "draw": v1_draw, "away": 0.0}

    return deltas


def merge_enhancement_deltas(
    base_deltas: Dict[str, Dict[str, float]],
    enhancement_deltas: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    将增强因子合并到基础因子中。
    增强因子使用V2_前缀避免与v5.10.8因子冲突。
    """
    merged = dict(base_deltas)
    for fid, d in enhancement_deltas.items():
        merged[fid] = d
    return merged


def get_enhancement_factor_ids() -> List[str]:
    return ["V2_D3", "V2_G1", "V2_U3", "V2_V1"]