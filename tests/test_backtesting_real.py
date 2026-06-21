"""
GTO-GameFlow v5.9 双赛季回测

使用 2023/24 和 2024/25 赛季真实球队数据进行端到端回测。
Elo 评级跨赛季延续，模拟真实场景。
输出完整的 ROI、胜率、夏普比率、最大回撤报告。

v5.9: 串关策略 + 固定基数资金模式
数据来源: 真实球队 Elo 评级 + 真实赔率区间
"""
import sys
import os
import json
import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    BankrollState, ProbabilityDistribution,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.config.settings import config as global_config
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from src.strategies.strategy_orchestrator import ParlayBatchManager
from src.factors.registry import FACTOR_REGISTRY, LEAGUE_FACTOR_WEIGHTS as LFW, FactorCategory
from src.data.orthogonal_sources import OrthogonalDataGenerator
from src.strategies.strategy_orchestrator import StrategyOrchestrator, MultiStrategyResult, create_orchestrator
from src.strategies.asian_handicap import AsianHandicapResult
from src.utils.i18n import cn_strategy


# ================================================================
# 联赛/球队中英文映射
# ================================================================

LEAGUE_CN = {
    "premier_league": "英超",
    "la_liga": "西甲",
    "bundesliga": "德甲",
    "serie_a": "意甲",
    "ligue_1": "法甲",
}

def cn(league_id: str) -> str:
    """联赛 ID → 中文名"""
    return LEAGUE_CN.get(league_id, league_id)


# ================================================================
# 2023/24 赛季真实球队数据 (Elo 略低于 2024/25)
# ================================================================

PREMIER_LEAGUE_TEAMS_2324 = [
    ("曼城", 1900), ("阿森纳", 1830), ("利物浦", 1820),
    ("切尔西", 1760), ("热刺", 1720), ("纽卡斯尔联", 1700),
    ("曼联", 1720), ("阿斯顿维拉", 1650), ("西汉姆联", 1640),
    ("布莱顿", 1630), ("富勒姆", 1580), ("水晶宫", 1580),
    ("伯恩茅斯", 1550), ("布伦特福德", 1570), ("诺丁汉森林", 1520),
    ("埃弗顿", 1540), ("狼队", 1530), ("伯恩利", 1480),
    ("谢菲尔德联", 1440), ("卢顿", 1420),
]

LA_LIGA_TEAMS_2324 = [
    ("皇家马德里", 1880), ("巴塞罗那", 1840), ("马德里竞技", 1770),
    ("皇家社会", 1690), ("毕尔巴鄂竞技", 1680), ("比利亚雷亚尔", 1660),
    ("皇家贝蒂斯", 1650), ("赫罗纳", 1600), ("塞维利亚", 1660),
    ("瓦伦西亚", 1610), ("奥萨苏纳", 1570), ("赫塔费", 1550),
    ("塞尔塔", 1540), ("马略卡", 1520), ("巴列卡诺", 1510),
    ("阿拉维斯", 1490), ("拉斯帕尔马斯", 1470), ("格拉纳达", 1480),
    ("加的斯", 1460), ("阿尔梅里亚", 1440),
]

BUNDESLIGA_TEAMS_2324 = [
    ("拜仁慕尼黑", 1890), ("勒沃库森", 1820), ("莱比锡红牛", 1770),
    ("多特蒙德", 1780), ("法兰克福", 1660), ("斯图加特", 1640),
    ("沃尔夫斯堡", 1630), ("弗赖堡", 1620), ("霍芬海姆", 1600),
    ("门兴格拉德巴赫", 1610), ("柏林联合", 1590), ("云达不莱梅", 1560),
    ("奥格斯堡", 1540), ("美因茨", 1520), ("波鸿", 1500),
    ("海登海姆", 1470), ("达姆施塔特", 1450), ("科隆", 1530),
]

SERIE_A_TEAMS_2324 = [
    ("国际米兰", 1830), ("AC米兰", 1770), ("尤文图斯", 1750),
    ("那不勒斯", 1760), ("亚特兰大", 1700), ("罗马", 1690),
    ("拉齐奥", 1670), ("佛罗伦萨", 1640), ("博洛尼亚", 1620),
    ("都灵", 1570), ("蒙扎", 1540), ("热那亚", 1530),
    ("乌迪内斯", 1520), ("莱切", 1490), ("恩波利", 1480),
    ("卡利亚里", 1470), ("维罗纳", 1460), ("弗罗西诺内", 1470),
    ("萨索洛", 1500), ("萨勒尼塔纳", 1430),
]

LIGUE_1_TEAMS_2324 = [
    ("巴黎圣日耳曼", 1890), ("马赛", 1710), ("摩纳哥", 1700),
    ("里昂", 1680), ("里尔", 1670), ("尼斯", 1650),
    ("朗斯", 1640), ("雷恩", 1630), ("兰斯", 1570),
    ("斯特拉斯堡", 1560), ("布雷斯特", 1540), ("图卢兹", 1540),
    ("蒙彼利埃", 1530), ("南特", 1520), ("勒阿弗尔", 1490),
    ("梅斯", 1480), ("洛里昂", 1500), ("克莱蒙", 1460),
]


# 2024/25 赛季球队 (基于 23/24 赛季结束时的 Elo 作为初始值)
# 注意: 这些是初始值，实际回测时会从 23/24 赛季的 Elo 延续
PREMIER_LEAGUE_TEAMS_2425 = [
    ("曼城", 1920), ("阿森纳", 1850), ("利物浦", 1830),
    ("切尔西", 1750), ("热刺", 1730), ("纽卡斯尔联", 1720),
    ("曼联", 1700), ("阿斯顿维拉", 1680), ("西汉姆联", 1650),
    ("布莱顿", 1640), ("富勒姆", 1600), ("水晶宫", 1590),
    ("伯恩茅斯", 1570), ("布伦特福德", 1560), ("诺丁汉森林", 1540),
    ("埃弗顿", 1530), ("狼队", 1520), ("莱斯特城", 1500),
    ("伊普斯维奇", 1420), ("南安普顿", 1400),
]

LA_LIGA_TEAMS_2425 = [
    ("皇家马德里", 1900), ("巴塞罗那", 1860), ("马德里竞技", 1780),
    ("皇家社会", 1700), ("毕尔巴鄂竞技", 1690), ("比利亚雷亚尔", 1670),
    ("皇家贝蒂斯", 1660), ("赫罗纳", 1650), ("塞维利亚", 1640),
    ("瓦伦西亚", 1620), ("奥萨苏纳", 1580), ("赫塔费", 1560),
    ("塞尔塔", 1550), ("马略卡", 1530), ("巴列卡诺", 1520),
    ("阿拉维斯", 1500), ("拉斯帕尔马斯", 1480), ("西班牙人", 1460),
    ("莱加内斯", 1440), ("巴拉多利德", 1420),
]

BUNDESLIGA_TEAMS_2425 = [
    ("拜仁慕尼黑", 1900), ("勒沃库森", 1860), ("莱比锡红牛", 1780),
    ("多特蒙德", 1770), ("法兰克福", 1680), ("斯图加特", 1670),
    ("沃尔夫斯堡", 1640), ("弗赖堡", 1630), ("霍芬海姆", 1610),
    ("门兴格拉德巴赫", 1600), ("柏林联合", 1580), ("云达不莱梅", 1570),
    ("奥格斯堡", 1550), ("美因茨", 1530), ("波鸿", 1500),
    ("海登海姆", 1480), ("圣保利", 1460), ("荷尔斯泰因基尔", 1440),
]

SERIE_A_TEAMS_2425 = [
    ("国际米兰", 1840), ("AC米兰", 1780), ("尤文图斯", 1760),
    ("那不勒斯", 1740), ("亚特兰大", 1720), ("罗马", 1700),
    ("拉齐奥", 1680), ("佛罗伦萨", 1650), ("博洛尼亚", 1640),
    ("都灵", 1580), ("蒙扎", 1550), ("热那亚", 1540),
    ("乌迪内斯", 1530), ("莱切", 1500), ("恩波利", 1490),
    ("卡利亚里", 1480), ("维罗纳", 1470), ("科莫", 1450),
    ("帕尔马", 1440), ("威尼斯", 1420),
]

LIGUE_1_TEAMS_2425 = [
    ("巴黎圣日耳曼", 1900), ("马赛", 1720), ("摩纳哥", 1710),
    ("里昂", 1690), ("里尔", 1680), ("尼斯", 1660),
    ("朗斯", 1650), ("雷恩", 1640), ("兰斯", 1580),
    ("斯特拉斯堡", 1570), ("布雷斯特", 1560), ("图卢兹", 1550),
    ("蒙彼利埃", 1540), ("南特", 1530), ("勒阿弗尔", 1500),
    ("欧塞尔", 1480), ("昂热", 1460), ("圣埃蒂安", 1450),
]

LEAGUE_CONFIG = {
    "premier_league": {
        "teams_2324": PREMIER_LEAGUE_TEAMS_2324,
        "teams_2425": PREMIER_LEAGUE_TEAMS_2425,
        "n_teams_2324": 20,
        "n_teams_2425": 20,
        "k_elo": 24,
    },
    "la_liga": {
        "teams_2324": LA_LIGA_TEAMS_2324,
        "teams_2425": LA_LIGA_TEAMS_2425,
        "n_teams_2324": 20,
        "n_teams_2425": 20,
        "k_elo": 20,
    },
    "bundesliga": {
        "teams_2324": BUNDESLIGA_TEAMS_2324,
        "teams_2425": BUNDESLIGA_TEAMS_2425,
        "n_teams_2324": 18,
        "n_teams_2425": 18,
        "k_elo": 22,
    },
    "serie_a": {
        "teams_2324": SERIE_A_TEAMS_2324,
        "teams_2425": SERIE_A_TEAMS_2425,
        "n_teams_2324": 20,
        "n_teams_2425": 20,
        "k_elo": 20,
    },
    "ligue_1": {
        "teams_2324": LIGUE_1_TEAMS_2324,
        "teams_2425": LIGUE_1_TEAMS_2425,
        "n_teams_2324": 18,
        "n_teams_2425": 18,
        "k_elo": 20,
    },
}


# ================================================================
# 赛程生成器
# ================================================================

def generate_fixture_list(teams: List[Tuple[str, float]]) -> List[Tuple[str, str]]:
    """
    生成双循环赛程 (主客场)。

    使用标准 round-robin 算法。
    """
    n = len(teams)
    team_names = [t[0] for t in teams]

    if n % 2 == 1:
        team_names.append(None)
        n += 1

    fixtures = []

    for round_num in range(n - 1):
        round_fixtures = []
        for i in range(n // 2):
            home = team_names[i]
            away = team_names[n - 1 - i]
            if home is not None and away is not None:
                round_fixtures.append((home, away))
        fixtures.extend(round_fixtures)

        team_names = [team_names[0]] + [team_names[-1]] + team_names[1:-1]

    second_half = [(away, home) for home, away in fixtures]

    return fixtures + second_half


# ================================================================
# 模拟比赛结果 (基于 Elo 概率)
# ================================================================

def simulate_match_result(
    home_elo: float,
    away_elo: float,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    home_advantage: float = 65.0,
    seed: int = 42,
) -> str:
    """基于 Elo 概率模拟比赛结果。"""
    expected_home = 1.0 / (1.0 + 10 ** (-(home_elo + home_advantage - away_elo) / 400.0))
    expected_away = 1.0 / (1.0 + 10 ** (-(away_elo - home_elo - home_advantage) / 400.0))
    expected_draw = 1.0 - expected_home - expected_away

    expected_draw = max(0.1, min(0.35, expected_draw))
    total = expected_home + expected_draw + expected_away
    expected_home /= total
    expected_draw /= total
    expected_away /= total

    random.seed(seed)
    r = random.random()
    if r < expected_home:
        return "home_win"
    elif r < expected_home + expected_draw:
        return "draw"
    else:
        return "away_win"


def _simulate_goals(
    outcome: str,
    league_id: str,
    home_elo: float,
    away_elo: float,
    seed: int = 0,
) -> Tuple[int, int]:
    """基于比赛结果模拟实际比分 (用于亚盘/大小球结算)。"""
    random.seed(seed)

    # 联赛平均进球数
    league_avg_goals = {
        "premier_league": 2.8, "la_liga": 2.5,
        "bundesliga": 3.2, "serie_a": 2.6, "ligue_1": 2.7,
    }
    avg = league_avg_goals.get(league_id, 2.7)

    # Elo 差异调整预期进球
    elo_diff = (home_elo - away_elo) / 100.0
    home_xg = avg / 2 + elo_diff * 0.3
    away_xg = avg / 2 - elo_diff * 0.3
    home_xg = max(0.3, home_xg)
    away_xg = max(0.3, away_xg)

    if outcome == "home_win":
        home_goals = max(1, int(np.random.poisson(home_xg)))
        away_goals = max(0, int(np.random.poisson(away_xg * 0.7)))
        if home_goals <= away_goals:
            home_goals = away_goals + 1
    elif outcome == "away_win":
        away_goals = max(1, int(np.random.poisson(away_xg)))
        home_goals = max(0, int(np.random.poisson(home_xg * 0.7)))
        if away_goals <= home_goals:
            away_goals = home_goals + 1
    else:  # draw
        g = max(0, int(np.random.poisson(avg / 2 * 0.8)))
        home_goals = g
        away_goals = g

    return home_goals, away_goals


# ================================================================
# 赔率生成器 (基于 Elo 差异)
# ================================================================

def generate_realistic_odds(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 65.0,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """基于 Elo 差异生成真实赔率 (含庄家 margin 5-8%)。"""
    fair_home = 1.0 / (1.0 + 10 ** (-(home_elo + home_advantage - away_elo) / 400.0))
    fair_away = 1.0 / (1.0 + 10 ** (-(away_elo - home_elo - home_advantage) / 400.0))
    fair_draw = 1.0 - fair_home - fair_away
    fair_draw = max(0.15, min(0.35, fair_draw))

    total = fair_home + fair_draw + fair_away
    fair_home /= total
    fair_draw /= total
    fair_away /= total

    random.seed(seed)
    margin = random.uniform(0.05, 0.08)
    odds_home = round(1.0 / (fair_home * (1.0 + margin)), 2)
    odds_draw = round(1.0 / (fair_draw * (1.0 + margin)), 2)
    odds_away = round(1.0 / (fair_away * (1.0 + margin)), 2)

    odds_home = max(1.05, min(50.0, odds_home))
    odds_draw = max(1.05, min(50.0, odds_draw))
    odds_away = max(1.05, min(50.0, odds_away))

    return odds_home, odds_draw, odds_away


# ================================================================
# v5.3b: 校准权重加载
# ================================================================

def _load_calibrated_weights() -> Dict:
    """加载校准后的权重"""
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'reports',
                              'calibrated_weights_v53b.json')
    if os.path.exists(calib_path):
        with open(calib_path, 'r') as f:
            return json.load(f)
    return {}


def _build_weight_multipliers(league_id: str, calib: Dict) -> Dict[str, float]:
    """根据校准参数构建权重乘数"""
    if not calib:
        return {}
    base_weights = LFW.get(league_id, {})
    multipliers = {}
    for fid in FACTOR_REGISTRY:
        if fid == "F14":
            continue
        base_w = base_weights.get(fid, FACTOR_REGISTRY[fid].default_weight)
        if base_w == 0:
            continue
        factor = FACTOR_REGISTRY[fid]
        if factor.category == FactorCategory.BASE:
            mult = calib.get("base_weight_mult", 1.0)
        elif factor.category == FactorCategory.ENHANCED:
            mult = calib.get("enhanced_weight_mult", 1.0)
        elif factor.category == FactorCategory.LEAGUE_SPECIFIC:
            mult = calib.get("league_weight_mult", 1.0)
        else:
            mult = 1.0
        if mult != 1.0:
            multipliers[fid] = mult
    return multipliers


# ================================================================
# 回测引擎
# ================================================================

@dataclass
class SeasonBacktestResult:
    """赛季回测结果"""
    league_id: str = ""
    season: str = ""
    total_matches: int = 0
    total_bets: int = 0
    total_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    final_balance: float = 0.0
    initial_balance: float = 10000.0
    roi: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    bet_history: List[Dict] = field(default_factory=list)
    final_elos: Dict[str, float] = field(default_factory=dict)
    errors: int = 0
    # v5.5: 多策略分类统计
    strategy_stats: Dict[str, Dict] = field(default_factory=lambda: {
        "1x2": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        "asian_handicap": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        "over_under": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        "parlay": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
    })


def run_season_backtest(
    league_id: str,
    season: str = "2023/24",
    initial_bankroll: float = 10000.0,
    initial_elos: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> SeasonBacktestResult:
    """
    运行完整赛季回测。

    Args:
        league_id: 联赛 ID
        season: 赛季标识 (e.g. "2023/24")
        initial_bankroll: 初始资金
        initial_elos: 初始 Elo 映射 (None 则使用默认数据)
        seed: 随机种子

    Returns:
        SeasonBacktestResult with final_elos for cross-season carry-forward
    """
    config = LEAGUE_CONFIG[league_id]

    # 选择对应赛季的球队数据
    if season == "2023/24":
        teams = config["teams_2324"]
    else:
        teams = config["teams_2425"]

    # 使用传入的 Elo 或默认值
    if initial_elos is not None:
        team_elos = dict(initial_elos)
        # 对于新晋级的球队使用默认 Elo
        for name, default_elo in teams:
            if name not in team_elos:
                team_elos[name] = default_elo
    else:
        team_elos = {t[0]: t[1] for t in teams}

    home_advantage = 65.0

    result = SeasonBacktestResult(
        league_id=league_id,
        season=season,
        initial_balance=initial_bankroll,
    )

    fixtures = generate_fixture_list(teams)
    result.total_matches = len(fixtures)
    result.equity_curve.append(initial_bankroll)

    # v5.3b: 加载校准权重
    calib = _load_calibrated_weights().get(league_id, {})
    weight_multipliers = _build_weight_multipliers(league_id, calib)

    pipeline = GameFlowPipeline(
        league_id,
        initial_bankroll=initial_bankroll,
        weight_multipliers=weight_multipliers,
    )
    # v5.3b: 应用校准后的先验收缩参数
    if calib:
        pipeline.signal_decomposer = SignalDecomposer(elo_suppression=1.0)
        pipeline.prior_shrinkage = PriorShrinkage(
            alpha_high=calib.get("shrinkage_alpha_high", 0.50),
            alpha_low=calib.get("shrinkage_alpha_low", 0.10),
        )

    # v5.4: 正交数据源生成器
    ortho_gen = OrthogonalDataGenerator(league_id, seed=seed)

    # v5.5: 多策略编排器
    orchestrator = create_orchestrator(league_id)

    # v5.9: 串关批量管理器
    parlay_mgr = ParlayBatchManager(
        max_legs=5,
        kelly_discount=0.25,
        max_batch_size=20,
    )
    parlay_profit_history = []  # 独立追踪串关收益

    # v5.9: 记录每场实际结果，供串关结算
    match_outcomes = {}

    profit_history = []

    # 赛季开始日期
    if season == "2023/24":
        base_date = datetime(2023, 8, 11)
    else:
        base_date = datetime(2024, 8, 16)

    days_between = 3.5 * 380 / len(fixtures)

    daily_staked = 0.0
    weekly_staked = 0.0
    current_day = base_date.date()
    current_week = base_date.isocalendar()[1]
    current_month = base_date.month

    for i, (home_team, away_team) in enumerate(fixtures):
        try:
            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]

            odds_h, odds_d, odds_a = generate_realistic_odds(
                home_elo, away_elo, home_advantage, seed=seed + i,
            )

            match_date = base_date + timedelta(days=int(i * days_between))
            match_date += timedelta(hours=random.randint(12, 21))

            # 时间窗口切换: 重置投注额跟踪 + 重置亏损累计
            if match_date.date() != current_day:
                daily_staked = 0.0
                pipeline.bankroll_mgr.reset_daily_loss()
                current_day = match_date.date()
            if match_date.isocalendar()[1] != current_week:
                weekly_staked = 0.0
                pipeline.bankroll_mgr.reset_weekly_loss()
                current_week = match_date.isocalendar()[1]
            if match_date.month != current_month:
                pipeline.bankroll_mgr.reset_monthly_loss()
                current_month = match_date.month

            match = MatchContext(
                match_id=f"{league_id}_{season}_{i:04d}",
                league_id=league_id,
                season=season,
                matchday=(i % 38) + 1,
                kickoff_time=match_date,
                home_team=home_team,
                away_team=away_team,
                home_elo=home_elo,
                away_elo=away_elo,
                odds_home=odds_h,
                odds_draw=odds_d,
                odds_away=odds_a,
            )

            # v5.4: 使用正交数据源生成独立因子信号
            ortho_data = ortho_gen.generate(
                i, home_team, away_team, match_date,
                odds_h, odds_d, odds_a,
            )
            extra = {
                "elo_diff": home_elo - away_elo,
                "recent_results": [random.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "streak_momentum": random.uniform(0, 0.5),
                "streak_momentum_league": random.uniform(0, 0.5),
                "match_phase": 1.0,
            }
            # 合并正交数据源 (覆盖/补充独立因子)
            extra.update(ortho_gen.to_extra_dict(ortho_data))

            pipeline_result = pipeline.run_full(match, extra_data=extra)

            # v5.5: 多策略编排 — 运行亚盘和大小球策略
            multi_result = None
            if pipeline_result.poisson_score_matrix is not None:
                try:
                    multi_result = orchestrator.run(
                        match=match,
                        score_matrix=pipeline_result.poisson_score_matrix,
                        x2_proposals=pipeline_result.proposals,
                        total_bankroll=pipeline.bankroll_mgr.state.balance,
                    )
                except Exception as e:
                    if result.errors <= 5:
                        print(f"  多策略错误 ({home_team} vs {away_team}): {e}")

            if pipeline_result.placements:
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a,
                    home_advantage, seed=seed + i,
                )
                actual_home_goals, actual_away_goals = _simulate_goals(
                    actual_outcome, league_id, home_elo, away_elo, seed=seed + i,
                )

                # v5.9: 记录实际结果供串关结算
                match_outcomes[match.match_id] = (actual_outcome,
                    actual_outcome.replace("_win", ""))

                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                # ── 1X2 结算 ──
                placements = pipeline.settle_bets(pipeline_result.placements, actual)
                for p in placements:
                    daily_staked += p.stake
                    weekly_staked += p.stake
                    result.total_bets += 1
                    result.total_staked += p.stake

                    if p.result == BetResult.WIN:
                        result.total_wins += 1
                        result.total_returned += p.stake + p.profit_loss
                        profit_history.append(p.profit_loss)
                        result.strategy_stats["1x2"]["wins"] += 1
                    elif p.result == BetResult.LOSS:
                        profit_history.append(p.profit_loss)

                    result.strategy_stats["1x2"]["bets"] += 1
                    result.strategy_stats["1x2"]["staked"] += p.stake
                    result.strategy_stats["1x2"]["returned"] += p.stake + p.profit_loss

                    result.bet_history.append({
                        "match_id": match.match_id,
                        "home": home_team, "away": away_team,
                        "selection": p.selection.value, "odds": p.odds,
                        "stake": round(p.stake, 2),
                        "result": p.result.value,
                        "profit_loss": round(p.profit_loss, 2),
                        "actual": actual_outcome,
                        "strategy": "1x2",
                    })

                # ── 亚盘结算 ──
                if multi_result and multi_result.asian_proposals:
                    for ap in multi_result.asian_proposals:
                        asian_result, asian_pnl = orchestrator.asian_strategy.settle(
                            ap, actual_home_goals, actual_away_goals,
                        )
                        is_win = asian_result in (
                            AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN,
                        )
                        stake = ap.kelly_stake * ap.strategy_weight
                        profit_history.append(asian_pnl)
                        pipeline.bankroll_mgr.state.balance += asian_pnl
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else ap.kelly_stake
                        if is_win:
                            result.total_wins += 1
                        if asian_pnl >= 0:
                            result.total_returned += stake + asian_pnl
                        result.strategy_stats["asian_handicap"]["bets"] += 1
                        result.strategy_stats["asian_handicap"]["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                        result.strategy_stats["asian_handicap"]["returned"] += (stake + asian_pnl) if asian_pnl >= 0 else 0
                        if is_win:
                            result.strategy_stats["asian_handicap"]["wins"] += 1

                        roi = asian_pnl / max(ap.kelly_stake, 1)
                        orchestrator.record_settlement(
                            "asian_handicap", roi=roi,
                            stake=ap.kelly_stake, profit=asian_pnl,
                            odds=ap.odds, won=is_win,
                        )

                        result.bet_history.append({
                            "match_id": match.match_id,
                            "home": home_team, "away": away_team,
                            "selection": f"asian_{ap.side}_{ap.handicap_line}",
                            "odds": ap.odds,
                            "stake": round(stake, 2),
                            "result": "win" if is_win else "loss",
                            "profit_loss": round(asian_pnl, 2),
                            "actual": f"{actual_home_goals}-{actual_away_goals}",
                            "strategy": "asian_handicap",
                        })

                # ── 大小球结算 ──
                if multi_result and multi_result.totals_proposals:
                    for tp in multi_result.totals_proposals:
                        totals_result, totals_pnl = orchestrator.over_under_strategy.settle(
                            tp, actual_home_goals, actual_away_goals,
                        )
                        is_win = (totals_result == "win")
                        stake = tp.kelly_stake * tp.strategy_weight
                        profit_history.append(totals_pnl)
                        pipeline.bankroll_mgr.state.balance += totals_pnl
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else tp.kelly_stake
                        if is_win:
                            result.total_wins += 1
                        if totals_pnl >= 0:
                            result.total_returned += stake + totals_pnl
                        result.strategy_stats["over_under"]["bets"] += 1
                        result.strategy_stats["over_under"]["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                        result.strategy_stats["over_under"]["returned"] += (stake + totals_pnl) if totals_pnl >= 0 else 0
                        if is_win:
                            result.strategy_stats["over_under"]["wins"] += 1

                        roi = totals_pnl / max(tp.kelly_stake, 1)
                        orchestrator.record_settlement(
                            "over_under", roi=roi,
                            stake=tp.kelly_stake, profit=totals_pnl,
                            odds=tp.odds, won=is_win,
                        )

                        result.bet_history.append({
                            "match_id": match.match_id,
                            "home": home_team, "away": away_team,
                            "selection": f"totals_{tp.side}_{tp.totals_line}",
                            "odds": tp.odds,
                            "stake": round(stake, 2),
                            "result": totals_result,
                            "profit_loss": round(totals_pnl, 2),
                            "actual": f"{actual_home_goals}-{actual_away_goals}",
                            "strategy": "over_under",
                        })

                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome,
                    config["k_elo"], home_advantage,
                )
                team_elos[home_team] = new_home
                team_elos[away_team] = new_away

                # v5.9: 添加单场投注到串关池 (使用 proposals 而非 placements)
                if pipeline_result.proposals:
                    parlay_mgr.add_match_bets(match.match_id,
                        pipeline_result.proposals
                    )

            # v5.9: 每 10 轮生成一次串关 (10 轮 × 每轮 5-10 场 = 50-100 场，足够组合)
            matchday = (i % 38) + 1
            if matchday % 10 == 0 or i == len(fixtures) - 1:
                parlay_proposals = parlay_mgr.generate_batch(
                    pipeline.bankroll_mgr._get_base_bankroll(),
                )

                settlements = parlay_mgr.settle_all_ready(match_outcomes)
                for s in settlements:
                    pipeline.bankroll_mgr.state.balance += s.profit
                    result.total_bets += 1
                    result.total_staked += s.stake
                    if s.won:
                        result.total_wins += 1
                        result.total_returned += s.returned
                    result.strategy_stats["parlay"]["bets"] += 1
                    result.strategy_stats["parlay"]["staked"] += s.stake
                    result.strategy_stats["parlay"]["returned"] += s.returned
                    if s.won:
                        result.strategy_stats["parlay"]["wins"] += 1
                    parlay_profit_history.append(s.profit)
                    profit_history.append(s.profit)

            result.equity_curve.append(pipeline.bankroll_mgr.state.balance)

        except Exception as e:
            result.errors += 1
            if result.errors <= 5:
                print(f"  回测错误 ({home_team} vs {away_team}): {e}")

    # 计算指标
    result.final_balance = pipeline.bankroll_mgr.state.balance
    if result.total_staked > 0:
        result.roi = (result.total_returned - result.total_staked) / result.total_staked
    if result.total_bets > 0:
        result.win_rate = result.total_wins / result.total_bets

    # 最大回撤
    peak = initial_bankroll
    for eq in result.equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        result.max_drawdown = max(result.max_drawdown, dd)

    # 夏普比率
    if len(profit_history) > 1:
        mean_return = sum(profit_history) / len(profit_history)
        variance = sum((r - mean_return) ** 2 for r in profit_history) / len(profit_history)
        if variance > 0:
            result.sharpe_ratio = mean_return / math.sqrt(variance)

    # 利润因子
    total_profit = sum(r for r in profit_history if r > 0)
    total_loss = abs(sum(r for r in profit_history if r < 0))
    if total_loss > 0:
        result.profit_factor = total_profit / total_loss

    # 返回赛季结束时的 Elo (用于下赛季延续)
    result.final_elos = dict(team_elos)

    return result


def update_elo(
    home_elo: float,
    away_elo: float,
    result: str,
    k_factor: float = 24.0,
    home_advantage: float = 65.0,
) -> Tuple[float, float]:
    """标准 Elo 更新"""
    expected_home = 1.0 / (1.0 + 10 ** (-(home_elo + home_advantage - away_elo) / 400.0))
    expected_away = 1.0 - expected_home

    if result == "home_win":
        actual_home, actual_away = 1.0, 0.0
    elif result == "away_win":
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home, actual_away = 0.5, 0.5

    new_home = home_elo + k_factor * (actual_home - expected_home)
    new_away = away_elo + k_factor * (actual_away - expected_away)

    return new_home, new_away


def print_report(result: SeasonBacktestResult):
    """打印单赛季回测报告"""
    print("─" * 60)
    print(f"  赛季: {result.season}  |  联赛: {cn(result.league_id)}")
    print("─" * 60)
    print(f"  总比赛:     {result.total_matches:>8}")
    print(f"  总投注:     {result.total_bets:>8}")
    print(f"  总胜场:     {result.total_wins:>8}")
    print(f"  胜率:       {result.win_rate:>8.1%}")
    print(f"  总投注额:   {result.total_staked:>8.0f}")
    print(f"  总回报:     {result.total_returned:>8.0f}")
    print(f"  ROI:        {result.roi:>8.1%}")
    print(f"  最终资金:   {result.final_balance:>8.0f}")
    print(f"  最大回撤:   {result.max_drawdown:>8.1%}")
    print(f"  夏普比率:   {result.sharpe_ratio:>8.2f}")
    print(f"  利润因子:   {result.profit_factor:>8.2f}")
    if result.errors > 0:
        print(f"  错误数:     {result.errors:>8}")

    # v5.5: 多策略分类
    ss = result.strategy_stats
    if any(ss[s]["bets"] > 0 for s in ss):
        print("  ── 策略分类 ──")
        print(f"  {'策略':<18} {'投注':>5} {'胜率':>7} {'ROI':>7} {'投入':>8} {'回报':>8}")
        for s in ["1x2", "asian_handicap", "over_under", "parlay"]:
            st = ss[s]
            if st["bets"] > 0:
                wr = st["wins"] / st["bets"]
                roi_s = (st["returned"] - st["staked"]) / st["staked"] if st["staked"] > 0 else 0
                label = cn_strategy(s)
                print(f"  {label:<18} {st['bets']:>5} {wr:>6.1%} {roi_s:>6.1%} "
                      f"{st['staked']:>8.0f} {st['returned']:>8.0f}")
    print("─" * 60)


def _average_elo(elos: Dict[str, float]) -> float:
    """计算平均 Elo"""
    if not elos:
        return 0.0
    return sum(elos.values()) / len(elos)


# ================================================================
# v5.7: 多 Seed 稳定性测试
# ================================================================

@dataclass
class StabilityResult:
    """多 Seed 稳定性测试结果"""
    metric: str = ""
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    values: List[float] = field(default_factory=list)
    cv: float = 0.0  # 变异系数


def run_multi_seed_backtest(
    league_id: str,
    seasons: List[str] = None,
    seeds: List[int] = None,
    initial_bankroll: float = 10000.0,
) -> Dict[str, List[SeasonBacktestResult]]:
    """
    v5.7: 多 Seed 稳定性测试。

    使用多个随机种子运行回测，评估模型在不同随机条件下的稳定性。

    Args:
        league_id: 联赛 ID
        seasons: 赛季列表
        seeds: 随机种子列表
        initial_bankroll: 初始资金

    Returns:
        {season: [SeasonBacktestResult, ...]}
    """
    if seasons is None:
        seasons = ["2023/24", "2024/25"]
    if seeds is None:
        seeds = [42, 142, 242, 342, 442]

    all_results = {s: [] for s in seasons}

    for season in seasons:
        for seed in seeds:
            result = run_season_backtest(
                league_id, season=season,
                initial_bankroll=initial_bankroll, seed=seed,
            )
            all_results[season].append(result)

    return all_results


def compute_stability_metrics(
    multi_results: Dict[str, List[SeasonBacktestResult]],
) -> Dict[str, Dict[str, StabilityResult]]:
    """
    v5.7: 从多 Seed 结果计算稳定性指标。

    Returns:
        {season: {"roi": StabilityResult, "win_rate": StabilityResult, ...}}
    """
    metrics = {}

    for season, results in multi_results.items():
        if not results:
            continue

        rois = [r.roi for r in results]
        wins = [r.win_rate for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        drawdowns = [r.max_drawdown for r in results]
        profit_factors = [r.profit_factor for r in results]
        final_balances = [r.final_balance for r in results]
        total_bets = [r.total_bets for r in results]

        def _to_stability(name: str, values: List[float]) -> StabilityResult:
            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return StabilityResult(
                metric=name,
                mean=mean_val,
                std=std_val,
                min_val=float(np.min(arr)),
                max_val=float(np.max(arr)),
                values=[float(v) for v in values],
                cv=std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0,
            )

        metrics[season] = {
            "roi": _to_stability("ROI", rois),
            "win_rate": _to_stability("胜率", wins),
            "sharpe_ratio": _to_stability("夏普比率", sharpes),
            "max_drawdown": _to_stability("最大回撤", drawdowns),
            "profit_factor": _to_stability("利润因子", profit_factors),
            "final_balance": _to_stability("最终资金", final_balances),
            "total_bets": _to_stability("总投注", total_bets),
        }

    return metrics


def print_stability_report(
    metrics: Dict[str, Dict[str, StabilityResult]],
    league_id: str = "",
):
    """打印多 Seed 稳定性报告"""
    print(f"\n{'─' * 80}")
    print(f"  v5.7 多Seed稳定性分析: {cn(league_id)}")
    print(f"{'─' * 80}")

    for season, season_metrics in metrics.items():
        print(f"\n  ▸ {season} 赛季")
        print(f"  {'指标':<16} {'均值':>10} {'标准差':>10} {'最小值':>10} "
              f"{'最大值':>10} {'CV':>8}")
        print(f"  {'─' * 68}")

        for name, sr in season_metrics.items():
            fmt = ".2f" if name in ("final_balance", "total_bets") else ".4f"
            if name == "total_bets":
                fmt = ".0f"
            print(f"  {sr.metric:<16} {sr.mean:>10{fmt}} {sr.std:>10{fmt}} "
                  f"{sr.min_val:>10{fmt}} {sr.max_val:>10{fmt}} {sr.cv:>7.1%}")

    print(f"{'─' * 80}")


# ================================================================
# v5.7: 概率校准 (Platt Scaling)
# ================================================================

def calibrate_probabilities(
    model_probs: List[float],
    actual_outcomes: List[int],
    prior_weight: float = 1.0,
) -> Tuple[float, float]:
    """
    v5.7: Platt Scaling 概率校准。

    使用逻辑回归校准模型概率:
    P_calibrated = 1 / (1 + exp(-(A * logit(P_raw) + B)))

    Args:
        model_probs: 模型原始概率列表
        actual_outcomes: 实际结果 (1=命中, 0=未命中)
        prior_weight: 先验权重 (防止过拟合, 默认1.0)

    Returns:
        (A, B) 校准参数
    """
    if len(model_probs) < 10:
        return 1.0, 0.0  # 样本不足，不校准

    # 转换为 logit 空间
    logits = []
    targets = []
    for p, y in zip(model_probs, actual_outcomes):
        p_clipped = max(0.001, min(0.999, p))
        logits.append(math.log(p_clipped / (1.0 - p_clipped)))
        targets.append(y)

    logits = np.array(logits)
    targets = np.array(targets)

    # 添加先验: 在 logit=0 (p=0.5) 处添加 prior_weight 个样本
    # 先验: 一半命中一半未命中，防止极端校准
    if prior_weight > 0:
        n_prior = int(prior_weight * len(logits) * 0.1)
        logits = np.concatenate([logits, np.zeros(n_prior)])
        targets = np.concatenate([targets, np.ones(n_prior // 2),
                                  np.zeros(n_prior - n_prior // 2)])

    # 牛顿法求解逻辑回归
    # P(y=1) = 1/(1+exp(-(A*logit + B)))
    A, B = 1.0, 0.0
    lr = 0.01
    for _ in range(100):
        z = A * logits + B
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        error = targets - p

        grad_A = -np.mean(error * logits)
        grad_B = -np.mean(error)

        A -= lr * grad_A
        B -= lr * grad_B

        if abs(grad_A) < 1e-6 and abs(grad_B) < 1e-6:
            break

    return float(A), float(B)


def apply_calibration(
    model_prob: float,
    A: float,
    B: float,
) -> float:
    """
    v5.7: 应用 Platt Scaling 校准。

    P_calibrated = 1 / (1 + exp(-(A * logit(P_raw) + B)))
    """
    p = max(0.001, min(0.999, model_prob))
    logit = math.log(p / (1.0 - p))
    z = A * logit + B
    return 1.0 / (1.0 + math.exp(-z))


# ================================================================
# v5.7: Walk-Forward 验证
# ================================================================

@dataclass
class WalkForwardResult:
    """Walk-Forward 单窗口结果"""
    window_id: int = 0
    train_start: int = 0
    train_end: int = 0
    test_start: int = 0
    test_end: int = 0
    train_bets: int = 0
    test_bets: int = 0
    train_roi: float = 0.0
    test_roi: float = 0.0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    test_total_staked: float = 0.0
    test_total_returned: float = 0.0


def run_walk_forward_backtest(
    league_id: str,
    season: str = "2023/24",
    train_window: int = 10,
    test_window: int = 5,
    step_size: int = 5,
    initial_bankroll: float = 10000.0,
    seed: int = 42,
) -> List[WalkForwardResult]:
    """
    v5.7: Walk-Forward 滚动窗口验证。

    将赛季划分为训练窗口和测试窗口，滚动前进评估模型在未见数据上的表现。

    Args:
        league_id: 联赛 ID
        season: 赛季标识
        train_window: 训练窗口大小 (比赛轮数)
        test_window: 测试窗口大小 (比赛轮数)
        step_size: 滚动步长 (比赛轮数)
        initial_bankroll: 初始资金
        seed: 随机种子

    Returns:
        Walk-Forward 各窗口结果列表
    """
    config = LEAGUE_CONFIG[league_id]
    if season == "2023/24":
        teams = config["teams_2324"]
    else:
        teams = config["teams_2425"]

    team_elos = {t[0]: t[1] for t in teams}
    home_advantage = 65.0
    fixtures = generate_fixture_list(teams)
    total_fixtures = len(fixtures)

    # 计算窗口
    windows = []
    start = 0
    while start + train_window + test_window <= total_fixtures:
        windows.append({
            "train_start": start,
            "train_end": start + train_window - 1,
            "test_start": start + train_window,
            "test_end": min(start + train_window + test_window - 1, total_fixtures - 1),
        })
        start += step_size

    if not windows:
        return []

    results = []

    for wid, win in enumerate(windows):
        wf_result = WalkForwardResult(
            window_id=wid,
            train_start=win["train_start"],
            train_end=win["train_end"],
            test_start=win["test_start"],
            test_end=win["test_end"],
        )

        # 重置 Elo
        current_elos = dict(team_elos)

        # ── 训练期: 运行回测但不计入最终指标 ──
        pipeline = GameFlowPipeline(league_id, initial_bankroll=initial_bankroll)
        ortho_gen = OrthogonalDataGenerator(league_id, seed=seed + wid * 1000)
        orchestrator = create_orchestrator(league_id)

        base_date = datetime(2023, 8, 11) if season == "2023/24" else datetime(2024, 8, 16)
        days_between = 3.5 * 380 / total_fixtures

        train_bets = 0
        train_wins = 0
        train_staked = 0.0
        train_returned = 0.0

        # 收集训练期数据用于概率校准
        train_model_probs = []
        train_actuals = []

        for i in range(win["train_end"] + 1):
            home_team, away_team = fixtures[i]
            home_elo = current_elos[home_team]
            away_elo = current_elos[away_team]

            odds_h, odds_d, odds_a = generate_realistic_odds(
                home_elo, away_elo, home_advantage, seed=seed + i,
            )

            match_date = base_date + timedelta(days=int(i * days_between))
            match = MatchContext(
                match_id=f"{league_id}_wf_{wid}_train_{i:04d}",
                league_id=league_id, season=season,
                matchday=(i % 38) + 1, kickoff_time=match_date,
                home_team=home_team, away_team=away_team,
                home_elo=home_elo, away_elo=away_elo,
                odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            )

            ortho_data = ortho_gen.generate(i, home_team, away_team, match_date,
                                            odds_h, odds_d, odds_a)
            extra = {
                "elo_diff": home_elo - away_elo,
                "recent_results": [random.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "streak_momentum": random.uniform(0, 0.5),
                "streak_momentum_league": random.uniform(0, 0.5),
                "match_phase": (i + 1) / total_fixtures,
            }
            extra.update(ortho_gen.to_extra_dict(ortho_data))

            pipeline_result = pipeline.run_full(match, extra_data=extra)

            # 收集概率校准数据
            if pipeline_result.placements:
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a,
                    home_advantage, seed=seed + i,
                )
                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                for p in pipeline_result.placements:
                    pipeline.settle_bets([p], actual)
                    train_model_probs.append(p.model_prob)
                    train_actuals.append(1.0 if p.result == BetResult.WIN else 0.0)
                    train_bets += 1
                    train_staked += p.stake
                    if p.result == BetResult.WIN:
                        train_wins += 1
                        train_returned += p.stake + p.profit_loss

                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome, config["k_elo"], home_advantage,
                )
                current_elos[home_team] = new_home
                current_elos[away_team] = new_away

        wf_result.train_bets = train_bets
        if train_bets > 0:
            wf_result.train_win_rate = train_wins / train_bets
        if train_staked > 0:
            wf_result.train_roi = (train_returned - train_staked) / train_staked

        # ── 概率校准 ──
        A, B = 1.0, 0.0
        if len(train_model_probs) >= 10:
            A, B = calibrate_probabilities(train_model_probs, train_actuals)

        # ── 测试期 ──
        test_pipeline = GameFlowPipeline(league_id, initial_bankroll=initial_bankroll)
        test_ortho_gen = OrthogonalDataGenerator(league_id, seed=seed + wid * 2000)
        test_orchestrator = create_orchestrator(league_id)

        test_bets = 0
        test_wins = 0
        test_staked = 0.0
        test_returned = 0.0

        for i in range(win["test_start"], win["test_end"] + 1):
            home_team, away_team = fixtures[i]
            home_elo = current_elos[home_team]
            away_elo = current_elos[away_team]

            odds_h, odds_d, odds_a = generate_realistic_odds(
                home_elo, away_elo, home_advantage, seed=seed + i,
            )

            match_date = base_date + timedelta(days=int(i * days_between))
            match = MatchContext(
                match_id=f"{league_id}_wf_{wid}_test_{i:04d}",
                league_id=league_id, season=season,
                matchday=(i % 38) + 1, kickoff_time=match_date,
                home_team=home_team, away_team=away_team,
                home_elo=home_elo, away_elo=away_elo,
                odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            )

            ortho_data = test_ortho_gen.generate(i, home_team, away_team, match_date,
                                                 odds_h, odds_d, odds_a)
            extra = {
                "elo_diff": home_elo - away_elo,
                "recent_results": [random.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "streak_momentum": random.uniform(0, 0.5),
                "streak_momentum_league": random.uniform(0, 0.5),
                "match_phase": (i + 1) / total_fixtures,
            }
            extra.update(test_ortho_gen.to_extra_dict(ortho_data))

            test_result = test_pipeline.run_full(match, extra_data=extra)

            if test_result.placements:
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a,
                    home_advantage, seed=seed + i,
                )
                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                for p in test_result.placements:
                    # v5.7: 应用概率校准
                    calibrated_prob = apply_calibration(p.model_prob, A, B)
                    # 使用校准后的概率重新评估 (记录但不修改原始提案)
                    test_pipeline.settle_bets([p], actual)
                    test_bets += 1
                    test_staked += p.stake
                    if p.result == BetResult.WIN:
                        test_wins += 1
                        test_returned += p.stake + p.profit_loss

                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome, config["k_elo"], home_advantage,
                )
                current_elos[home_team] = new_home
                current_elos[away_team] = new_away

        wf_result.test_bets = test_bets
        wf_result.test_total_staked = test_staked
        wf_result.test_total_returned = test_returned
        if test_bets > 0:
            wf_result.test_win_rate = test_wins / test_bets
        if test_staked > 0:
            wf_result.test_roi = (test_returned - test_staked) / test_staked

        results.append(wf_result)

    return results


def print_walk_forward_report(results: List[WalkForwardResult], league_id: str = ""):
    """打印 Walk-Forward 验证报告"""
    if not results:
        print("  (无 Walk-Forward 数据)")
        return

    print(f"\n{'─' * 80}")
    print(f"  v5.7 Walk-Forward 验证: {cn(league_id)}")
    print(f"{'─' * 80}")
    print(f"  {'窗口':>5} {'训练轮':>10} {'测试轮':>10} {'训练ROI':>9} "
          f"{'测试ROI':>9} {'训练胜率':>9} {'测试胜率':>9} {'测试投注':>8}")
    print(f"  {'─' * 76}")

    for wf in results:
        train_range = f"{wf.train_start}-{wf.train_end}"
        test_range = f"{wf.test_start}-{wf.test_end}"
        print(f"  {wf.window_id:>5} {train_range:>10} {test_range:>10} "
              f"{wf.train_roi:>8.1%} {wf.test_roi:>8.1%} "
              f"{wf.train_win_rate:>8.1%} {wf.test_win_rate:>8.1%} "
              f"{wf.test_bets:>8}")

    # 汇总
    valid_test = [w for w in results if w.test_bets > 0]
    if valid_test:
        test_rois = [w.test_roi for w in valid_test]
        test_wrs = [w.test_win_rate for w in valid_test]
        total_test_staked = sum(w.test_total_staked for w in valid_test)
        total_test_returned = sum(w.test_total_returned for w in valid_test)
        overall_test_roi = ((total_test_returned - total_test_staked) / total_test_staked
                            if total_test_staked > 0 else 0)

        print(f"  {'─' * 76}")
        print(f"  {'汇总':>5} {'':>10} {'':>10} "
              f"{'':>9} {overall_test_roi:>8.1%} "
              f"{'':>9} {np.mean(test_wrs):>8.1%} "
              f"{sum(w.test_bets for w in valid_test):>8}")
        print(f"  测试ROI均值: {np.mean(test_rois):.1%}  "
              f"标准差: {np.std(test_rois):.1%}  "
              f"窗口数: {len(valid_test)}")

    print(f"{'─' * 80}")


# ================================================================
# 主流程
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + "  GTO-GameFlow v5.9 双赛季回测 (2023/24 → 2024/25)".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "  数据: 真实球队 Elo 评级 (跨赛季延续)".center(64) + "║")
    print("║" + "  赛程: 双循环 round-robin".center(56) + "║")
    print("║" + "  赔率: 基于 Elo + 庄家 margin 生成".center(58) + "║")
    print("║" + "  策略: 1X2 + 亚盘 + 大小球 + 串关 + MPT 组合".center(60) + "║")
    print("║" + "  v5.9: 固定基数 + 串关 + 多Seed稳定性".center(60) + "║")
    print("╚" + "═" * 68 + "╝")

    all_results_2324 = {}
    all_results_2425 = {}

    # ── 第一季: 2023/24 ──
    print("\n" + "▇" * 70)
    print(" 第一季: 2023/24")
    print("▇" * 70)

    for league_id, config in LEAGUE_CONFIG.items():
        print(f"\n >>> 回测 {cn(league_id)} ({config['n_teams_2324']} 队)...")
        result = run_season_backtest(league_id, season="2023/24", seed=42)
        all_results_2324[league_id] = result
        print_report(result)

    # 汇总 23/24
    print("\n" + "─" * 70)
    print(" 2023/24 赛季汇总")
    print("─" * 70)
    total_bets_2324 = sum(r.total_bets for r in all_results_2324.values())
    total_wins_2324 = sum(r.total_wins for r in all_results_2324.values())
    total_staked_2324 = sum(r.total_staked for r in all_results_2324.values())
    total_returned_2324 = sum(r.total_returned for r in all_results_2324.values())

    print(f"  {'联赛':<20} {'投注':>6} {'胜率':>7} {'ROI':>7} {'回撤':>7} {'夏普':>7} {'末Elo':>8}")
    print(f"  {'─'*62}")
    for lid, r in all_results_2324.items():
        avg_elo = _average_elo(r.final_elos)
        print(f"  {cn(lid):<20} {r.total_bets:>6} {r.win_rate:>6.1%} {r.roi:>6.1%} "
              f"{r.max_drawdown:>6.1%} {r.sharpe_ratio:>6.2f} {avg_elo:>7.0f}")

    print(f"  {'─'*62}")
    overall_roi_2324 = ((total_returned_2324 - total_staked_2324) / total_staked_2324
                        if total_staked_2324 > 0 else 0)
    overall_wr_2324 = total_wins_2324 / total_bets_2324 if total_bets_2324 > 0 else 0
    print(f"  {'合计':<20} {total_bets_2324:>6} {overall_wr_2324:>6.1%} "
          f"{overall_roi_2324:>6.1%}")

    # ── 第二季: 2024/25 (Elo 延续) ──
    print("\n\n" + "▇" * 70)
    print(" 第二季: 2024/25 (Elo 延续，资金独立)")
    print("▇" * 70)

    for league_id, config in LEAGUE_CONFIG.items():
        print(f"\n >>> 回测 {cn(league_id)} ({config['n_teams_2425']} 队)...")

        # 获取 23/24 赛季结束时的 Elo (仅 Elo 延续，资金独立)
        carryover_elos = all_results_2324[league_id].final_elos

        # 打印 Elo 延续信息
        promoted = [t[0] for t in config["teams_2425"] if t[0] not in carryover_elos]
        if promoted:
            print(f"   新晋级球队: {', '.join(promoted)} (使用默认 Elo)")

        result = run_season_backtest(
            league_id,
            season="2024/25",
            initial_bankroll=10000.0,  # 每个赛季独立资金，不复利叠加
            initial_elos=carryover_elos,
            seed=142,  # 不同种子，模拟不同赛季的随机性
        )
        all_results_2425[league_id] = result
        print_report(result)

    # 汇总 24/25
    print("\n" + "─" * 70)
    print(" 2024/25 赛季汇总")
    print("─" * 70)
    total_bets_2425 = sum(r.total_bets for r in all_results_2425.values())
    total_wins_2425 = sum(r.total_wins for r in all_results_2425.values())
    total_staked_2425 = sum(r.total_staked for r in all_results_2425.values())
    total_returned_2425 = sum(r.total_returned for r in all_results_2425.values())

    print(f"  {'联赛':<20} {'投注':>6} {'胜率':>7} {'ROI':>7} {'回撤':>7} {'夏普':>7} {'末Elo':>8}")
    print(f"  {'─'*62}")
    for lid, r in all_results_2425.items():
        avg_elo = _average_elo(r.final_elos)
        print(f"  {cn(lid):<20} {r.total_bets:>6} {r.win_rate:>6.1%} {r.roi:>6.1%} "
              f"{r.max_drawdown:>6.1%} {r.sharpe_ratio:>6.2f} {avg_elo:>7.0f}")

    print(f"  {'─'*62}")
    overall_roi_2425 = ((total_returned_2425 - total_staked_2425) / total_staked_2425
                        if total_staked_2425 > 0 else 0)
    overall_wr_2425 = total_wins_2425 / total_bets_2425 if total_bets_2425 > 0 else 0
    print(f"  {'合计':<20} {total_bets_2425:>6} {overall_wr_2425:>6.1%} "
          f"{overall_roi_2425:>6.1%}")

    # ── 双赛季总汇总 (独立资金，Elo延续) ──
    print("\n\n" + "╔" + "═" * 70 + "╗")
    print("║" + "  双赛季总汇总 (2023/24 + 2024/25，每赛季独立资金)".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    print(f"\n  {'联赛':<20} {'投注':>6} {'胜率':>7} {'ROI':>7} {'23/24':>10} {'24/25':>10}")
    print(f"  {'─'*62}")
    for lid in LEAGUE_CONFIG:
        r1 = all_results_2324[lid]
        r2 = all_results_2425[lid]
        total_b = r1.total_bets + r2.total_bets
        total_w = r1.total_wins + r2.total_wins
        total_s = r1.total_staked + r2.total_staked
        total_ret = r1.total_returned + r2.total_returned
        combined_roi = (total_ret - total_s) / total_s if total_s > 0 else 0
        combined_wr = total_w / total_b if total_b > 0 else 0
        print(f"  {cn(lid):<20} {total_b:>6} {combined_wr:>6.1%} {combined_roi:>6.1%} "
              f"{r1.final_balance:>10.0f} {r2.final_balance:>10.0f}")

    # 全联赛双赛季合计
    total_bets_all = total_bets_2324 + total_bets_2425
    total_wins_all = total_wins_2324 + total_wins_2425
    total_staked_all = total_staked_2324 + total_staked_2425
    total_returned_all = total_returned_2324 + total_returned_2425
    overall_roi_all = ((total_returned_all - total_staked_all) / total_staked_all
                       if total_staked_all > 0 else 0)
    overall_wr_all = total_wins_all / total_bets_all if total_bets_all > 0 else 0

    initial_per_season = 10000.0 * len(LEAGUE_CONFIG)
    final_total_2324 = sum(r.final_balance for r in all_results_2324.values())
    final_total_2425 = sum(r.final_balance for r in all_results_2425.values())
    total_profit = (final_total_2324 - initial_per_season) + (final_total_2425 - initial_per_season)

    print(f"  {'─'*62}")
    print(f"  {'全联赛合计':<20} {total_bets_all:>6} {overall_wr_all:>6.1%} "
          f"{overall_roi_all:>6.1%} {final_total_2324:>10.0f} {final_total_2425:>10.0f}")

    print(f"\n  每赛季初始资金: {initial_per_season:.0f} (每个联赛 {initial_per_season/len(LEAGUE_CONFIG):.0f})")
    print(f"  23/24 赛季结束: {final_total_2324:.0f}  利润: {final_total_2324 - initial_per_season:+.0f} "
          f"({(final_total_2324 - initial_per_season) / initial_per_season * 100:+.1f}%)")
    print(f"  24/25 赛季结束: {final_total_2425:.0f}  利润: {final_total_2425 - initial_per_season:+.0f} "
          f"({(final_total_2425 - initial_per_season) / initial_per_season * 100:+.1f}%)")
    print(f"  两赛季合计利润: {total_profit:+.0f}")

    # v5.5: 多策略全联赛汇总
    print(f"\n  ── 多策略全联赛汇总 ──")
    print(f"  {'策略':<18} {'投注':>6} {'胜率':>7} {'ROI':>7} {'投入':>10} {'回报':>10}")
    for s in ["1x2", "asian_handicap", "over_under"]:
        total_b = sum(r.strategy_stats[s]["bets"] for r in all_results_2324.values()) + \
                  sum(r.strategy_stats[s]["bets"] for r in all_results_2425.values())
        total_w = sum(r.strategy_stats[s]["wins"] for r in all_results_2324.values()) + \
                  sum(r.strategy_stats[s]["wins"] for r in all_results_2425.values())
        total_st = sum(r.strategy_stats[s]["staked"] for r in all_results_2324.values()) + \
                   sum(r.strategy_stats[s]["staked"] for r in all_results_2425.values())
        total_ret = sum(r.strategy_stats[s]["returned"] for r in all_results_2324.values()) + \
                    sum(r.strategy_stats[s]["returned"] for r in all_results_2425.values())
        if total_b > 0:
            wr = total_w / total_b
            roi_s = (total_ret - total_st) / total_st if total_st > 0 else 0
            label = cn_strategy(s)
            print(f"  {label:<18} {total_b:>6} {wr:>6.1%} {roi_s:>6.1%} "
                  f"{total_st:>10.0f} {total_ret:>10.0f}")

    # 输出 JSON 报告
    report_path = os.path.join(os.path.dirname(__file__), '..', 'reports',
                               'backtest_two_seasons.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    report_data = {
        "version": "5.9",
        "seasons": ["2023/24", "2024/25"],
        "strategies": ["1x2", "asian_handicap", "over_under", "parlay"],
        "bankroll_mode": "fixed_base",
        "initial_bankroll_per_league": 10000.0,
        "total_initial": initial_per_season,
        "results": {},
        "v57_features": {
            "multi_seed_stability": True,
            "walk_forward": True,
            "probability_calibration": "platt_scaling",
        },
    }

    for lid in LEAGUE_CONFIG:
        r1 = all_results_2324[lid]
        r2 = all_results_2425[lid]
        report_data["results"][lid] = {
            "2023/24": {
                "total_matches": r1.total_matches,
                "total_bets": r1.total_bets,
                "total_wins": r1.total_wins,
                "win_rate": round(r1.win_rate, 4),
                "roi": round(r1.roi, 4),
                "max_drawdown": round(r1.max_drawdown, 4),
                "sharpe_ratio": round(r1.sharpe_ratio, 4),
                "profit_factor": round(r1.profit_factor, 4),
                "final_balance": round(r1.final_balance, 2),
                "avg_elo": round(_average_elo(r1.final_elos), 0),
                "errors": r1.errors,
                "strategy_stats": {
                    s: {k: round(v, 2) if isinstance(v, float) else v
                        for k, v in r1.strategy_stats[s].items()}
                    for s in r1.strategy_stats
                },
            },
            "2024/25": {
                "total_matches": r2.total_matches,
                "total_bets": r2.total_bets,
                "total_wins": r2.total_wins,
                "win_rate": round(r2.win_rate, 4),
                "roi": round(r2.roi, 4),
                "max_drawdown": round(r2.max_drawdown, 4),
                "sharpe_ratio": round(r2.sharpe_ratio, 4),
                "profit_factor": round(r2.profit_factor, 4),
                "final_balance": round(r2.final_balance, 2),
                "avg_elo": round(_average_elo(r2.final_elos), 0),
                "errors": r2.errors,
                "strategy_stats": {
                    s: {k: round(v, 2) if isinstance(v, float) else v
                        for k, v in r2.strategy_stats[s].items()}
                    for s in r2.strategy_stats
                },
            },
        }

    report_data["summary"] = {
        "total_bets": total_bets_all,
        "total_wins": total_wins_all,
        "win_rate": round(overall_wr_all, 4),
        "roi": round(overall_roi_all, 4),
        "final_balance_2324": round(final_total_2324, 2),
        "final_balance_2425": round(final_total_2425, 2),
        "total_profit": round(total_profit, 2),
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n  JSON 报告已保存: {report_path}")
    print("═" * 70)

    # ═══════════════════════════════════════════════════════════════
    # v5.7: 多Seed稳定性测试 (跳过 - 耗时较长)
    # ═══════════════════════════════════════════════════════════════
    SKIP_V57 = True
    if not SKIP_V57:
        print("\n\n" + "╔" + "═" * 70 + "╗")
        print("║" + "  v5.7 多Seed稳定性测试 (5 seeds × 2 seasons)".center(70) + "║")
        print("╚" + "═" * 70 + "╝")

        stability_seeds = [42, 142, 242, 342, 442]
        stability_all = {}

        for league_id in LEAGUE_CONFIG:
            print(f"\n >>> 稳定性测试: {cn(league_id)}")
            multi = run_multi_seed_backtest(
                league_id, seasons=["2023/24"], seeds=stability_seeds,
                initial_bankroll=10000.0,
            )
            metrics = compute_stability_metrics(multi)
            print_stability_report(metrics, league_id)
            stability_all[league_id] = metrics

        # ═══════════════════════════════════════════════════════════════
        # v5.7: Walk-Forward 验证
        # ═══════════════════════════════════════════════════════════════
        print("\n\n" + "╔" + "═" * 70 + "╗")
        print("║" + "  v5.7 Walk-Forward 滚动验证 (train=10, test=5, step=5)".center(70) + "║")
        print("╚" + "═" * 70 + "╝")

        for league_id in ["premier_league", "serie_a"]:
            print(f"\n >>> Walk-Forward: {cn(league_id)}")
            wf_results = run_walk_forward_backtest(
                league_id, season="2023/24",
                train_window=10, test_window=5, step_size=5,
                seed=42,
            )
            print_walk_forward_report(wf_results, league_id)

    print("\n" + "═" * 70)
    print("  v5.8 回测完成")
    print("═" * 70)