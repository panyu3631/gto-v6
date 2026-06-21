"""
GTO-GameFlow v5.10.8 因子异质化分组

解决35/55因子同质化问题 (均测量"home vs away"强弱对比):
- 组内使用 max-pooling: 只保留最强信号, 消除同质因子共振
- 组间独立加权: 不同语义组的信号独立贡献

分组策略:
  实力组 (Strength):  纯数值实力对比 - F1,F2,F5,F7,F8,F9,F26,F34,F39,F40
  状态组 (Form):      近期表现/动量   - F3,F20,F21,F25,F38,F49,F50
  市场组 (Market):    赔率/市场信号   - F10,F11,F22,F28,F30,F32,F46,F54
  情境组 (Context):   赛程/教练/动机  - F6,F15,F16,F17,F24,F33,F35,F36,F37,F41
  统计组 (Stats):     比赛统计指标   - F19,F42,F43,F44,F45,F48,F52,F53
  平局组 (Draw):      平局专属因子   - F12,F18,F23,F27,F47,F51
  其孔组 (Other):     其他方向因子   - F4,F13,F29,F31,F55
"""

from typing import Dict, List, Optional, Tuple
import math

# ============================================================
# 分组定义
# ============================================================
FACTOR_GROUPS: Dict[str, List[str]] = {
    "实力": ["F2", "F5", "F7", "F8", "F9", "F26", "F39", "F40"],  # -F1,-F34
    "状态": ["F3", "F21", "F25", "F38", "F49", "F50"],           # -F20
    "市场": ["F10", "F11", "F22", "F30", "F32", "F46", "F54"],   # -F28
    "情境": ["F6", "F16", "F33", "F37", "F41"],                  # -F15,-F17,-F24,-F35,-F36
    "统计": ["F42", "F43", "F44", "F45", "F48", "F52", "F53"],   # -F19
    "平局": ["F12", "F23", "F27", "F47", "F51", "F56", "F57", "F58"],                  # -F18
    "其他": ["F4", "F13", "F29", "F31", "F55"],
}

# 反向索引: factor_id → group_name
FACTOR_TO_GROUP: Dict[str, str] = {}
for group_name, factor_ids in FACTOR_GROUPS.items():
    for fid in factor_ids:
        FACTOR_TO_GROUP[fid] = group_name


def group_signal_cap(
    factor_deltas: Dict[str, Dict[str, float]],
    cap_factor: float = 1.5,
) -> Dict[str, Dict[str, float]]:
    """
    组内信号上限: 对每个异质组的总信号设置上限, 防止同质因子共振放大。

    原理:
    组内10个因子都指向"主胜"时, 简单加和会产生10倍信号。
    信号上限将组内总净值限制在 cap_factor × 单个因子平均强度。

    cap_factor: 组内信号倍数上限 (1.5 = 最多1.5个因子强度的信号)

    返回: 信号上限处理后的因子deltas (保留原始因子ID)
    """
    if not factor_deltas:
        return {}

    capped: Dict[str, Dict[str, float]] = {}

    for group_name, factor_ids in FACTOR_GROUPS.items():
        # 收集组内所有因子的delta
        group_deltas: Dict[str, Dict[str, float]] = {}
        for fid in factor_ids:
            if fid in factor_deltas:
                d = factor_deltas[fid]
                if isinstance(d, dict):
                    group_deltas[fid] = d

        if not group_deltas:
            continue

        # 计算组内每个方向的平均信号强度
        avg_strength = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for direction in ["home", "draw", "away"]:
            values = [abs(d.get(direction, 0.0)) for d in group_deltas.values()]
            nonzero = [v for v in values if v > 0.0001]
            if nonzero:
                avg_strength[direction] = sum(nonzero) / len(nonzero)

        # 计算组内每个方向的总净值
        net_signals = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for d in group_deltas.values():
            for direction in ["home", "draw", "away"]:
                net_signals[direction] += d.get(direction, 0.0)

        # 计算缩放比例: 如果总净值超过上限, 缩放到上限
        for direction in ["home", "draw", "away"]:
            cap = avg_strength[direction] * cap_factor
            net = net_signals[direction]
            if abs(net) > cap and cap > 0.0001:
                scale = cap / abs(net)
            else:
                scale = 1.0

            # 将缩放应用到每个因子
            for fid in group_deltas:
                if fid not in capped:
                    capped[fid] = {"home": 0.0, "draw": 0.0, "away": 0.0}
                capped[fid][direction] = group_deltas[fid].get(direction, 0.0) * scale

    return capped


def get_group_for_factor(factor_id: str) -> Optional[str]:
    """获取因子所属的异质组"""
    return FACTOR_TO_GROUP.get(factor_id)


def get_group_sizes() -> Dict[str, int]:
    """获取各组因子数量"""
    return {k: len(v) for k, v in FACTOR_GROUPS.items()}


def get_group_summary() -> str:
    """打印分组摘要"""
    lines = []
    for group_name, factor_ids in FACTOR_GROUPS.items():
        lines.append(f"  {group_name}({len(factor_ids)}): {', '.join(factor_ids)}")
    return "\n".join(lines)