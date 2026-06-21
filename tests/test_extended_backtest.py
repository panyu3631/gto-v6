"""
GTO-GameFlow v5.9.1 扩展回测 — 10赛季五大联赛 + 6届世界杯

- 10赛季: 2014/15 → 2023/24
- 6届世界杯: 2006, 2010, 2014, 2018, 2022 (联赛后独立运行)
- 5大联赛共享全局资金池 10,000 (固定基数)
- Elo 跨赛季延续
- 世界杯由联赛 Elo 推算国家队 Elo 后独立回测
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json, random as _random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    AsianHandicapResult, AsianHandicapProposal, TotalsProposal,
    ScoreMatrix,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.strategies.strategy_orchestrator import ParlayBatchManager
from src.strategies import StrategyOrchestrator
from src.calibration.signal_decomposer import SignalDecomposer, PriorShrinkage
from src.utils.i18n import cn_strategy

# ── 从旧回测复用 ──
from tests.test_backtesting_real import (
    generate_fixture_list, generate_realistic_odds,
    simulate_match_result, update_elo, _simulate_goals,
    OrthogonalDataGenerator, _load_calibrated_weights, _build_weight_multipliers,
    create_orchestrator,
)

LEAGUE_CN = {
    "premier_league": "英超", "la_liga": "西甲",
    "bundesliga": "德甲", "serie_a": "意甲", "ligue_1": "法甲",
}
def cn(lid): return LEAGUE_CN.get(lid, lid)


# ═══════════════════════════════════════════════════════════════
# 联赛球队池 (跨10赛季)
# ═══════════════════════════════════════════════════════════════

LEAGUE_TEAM_POOLS = {
    "premier_league": {
        "n_teams": 20, "k_elo": 24,
        "base_teams": [
            ("曼城", 1920), ("阿森纳", 1850), ("利物浦", 1840),
            ("切尔西", 1770), ("热刺", 1730), ("曼联", 1720),
            ("纽卡斯尔联", 1650), ("阿斯顿维拉", 1620), ("西汉姆联", 1640),
            ("布莱顿", 1600), ("埃弗顿", 1580), ("狼队", 1560),
            ("水晶宫", 1570), ("富勒姆", 1550), ("伯恩茅斯", 1520),
            ("布伦特福德", 1530), ("诺丁汉森林", 1500), ("莱斯特城", 1560),
            ("南安普顿", 1480), ("利兹联", 1490), ("伯恩利", 1450),
            ("沃特福德", 1420), ("诺维奇", 1400), ("西布朗", 1400),
            ("谢菲尔德联", 1420), ("米德尔斯堡", 1380), ("斯旺西", 1380),
            ("斯托克城", 1370), ("哈德斯菲尔德", 1340), ("加的夫城", 1330),
        ],
    },
    "la_liga": {
        "n_teams": 20, "k_elo": 20,
        "base_teams": [
            ("皇家马德里", 1900), ("巴塞罗那", 1870), ("马德里竞技", 1780),
            ("塞维利亚", 1700), ("皇家社会", 1680), ("比利亚雷亚尔", 1670),
            ("毕尔巴鄂竞技", 1660), ("皇家贝蒂斯", 1650), ("瓦伦西亚", 1640),
            ("赫罗纳", 1600), ("奥萨苏纳", 1570), ("塞尔塔", 1560),
            ("赫塔费", 1550), ("马略卡", 1520), ("巴列卡诺", 1510),
            ("阿拉维斯", 1490), ("西班牙人", 1500), ("莱加内斯", 1460),
            ("巴拉多利德", 1440), ("格拉纳达", 1480), ("埃瓦尔", 1450),
            ("莱万特", 1440), ("拉科鲁尼亚", 1420), ("马拉加", 1400),
            ("拉斯帕尔马斯", 1470), ("加的斯", 1460), ("阿尔梅里亚", 1440),
            ("埃尔切", 1400), ("韦斯卡", 1380), ("希洪竞技", 1370),
        ],
    },
    "bundesliga": {
        "n_teams": 18, "k_elo": 22,
        "base_teams": [
            ("拜仁慕尼黑", 1910), ("多特蒙德", 1800), ("莱比锡红牛", 1770),
            ("勒沃库森", 1750), ("法兰克福", 1660), ("斯图加特", 1640),
            ("沃尔夫斯堡", 1650), ("弗赖堡", 1630), ("霍芬海姆", 1610),
            ("门兴格拉德巴赫", 1620), ("柏林联合", 1590), ("云达不莱梅", 1570),
            ("奥格斯堡", 1550), ("美因茨", 1530), ("波鸿", 1500),
            ("沙尔克04", 1580), ("汉堡", 1520), ("科隆", 1530),
            ("柏林赫塔", 1530), ("杜塞尔多夫", 1460), ("帕德博恩", 1430),
            ("纽伦堡", 1440), ("因戈尔施塔特", 1400), ("达姆施塔特", 1440),
            ("海登海姆", 1470), ("菲尔特", 1400), ("汉诺威96", 1460),
        ],
    },
    "serie_a": {
        "n_teams": 20, "k_elo": 20,
        "base_teams": [
            ("尤文图斯", 1830), ("国际米兰", 1800), ("AC米兰", 1780),
            ("那不勒斯", 1770), ("罗马", 1720), ("亚特兰大", 1700),
            ("拉齐奥", 1690), ("佛罗伦萨", 1650), ("博洛尼亚", 1580),
            ("都灵", 1570), ("乌迪内斯", 1550), ("热那亚", 1540),
            ("桑普多利亚", 1520), ("萨索洛", 1500), ("卡利亚里", 1480),
            ("维罗纳", 1470), ("恩波利", 1460), ("莱切", 1450),
            ("蒙扎", 1500), ("弗罗西诺内", 1440), ("贝内文托", 1400),
            ("斯佩齐亚", 1410), ("威尼斯", 1400), ("帕尔马", 1490),
            ("克罗托内", 1370), ("布雷西亚", 1400), ("萨勒尼塔纳", 1430),
        ],
    },
    "ligue_1": {
        "n_teams": 18, "k_elo": 20,  # 2023/24起改18队
        "base_teams": [
            ("巴黎圣日耳曼", 1900), ("马赛", 1720), ("摩纳哥", 1710),
            ("里昂", 1700), ("里尔", 1680), ("尼斯", 1660),
            ("朗斯", 1640), ("雷恩", 1630), ("兰斯", 1570),
            ("斯特拉斯堡", 1560), ("布雷斯特", 1540), ("图卢兹", 1540),
            ("蒙彼利埃", 1530), ("南特", 1520), ("勒阿弗尔", 1490),
            ("梅斯", 1480), ("洛里昂", 1500), ("克莱蒙", 1460),
            ("昂热", 1450), ("圣埃蒂安", 1500), ("波尔多", 1480),
            ("甘冈", 1420), ("卡昂", 1410), ("第戎", 1400),
            ("亚眠", 1390), ("特鲁瓦", 1380), ("尼姆", 1370),
        ],
    },
}


def get_season_teams(league_id: str, season_idx: int) -> List[Tuple[str, float]]:
    """
    获取指定赛季的球队列表。
    season_idx: 0=2014/15, 1=2015/16, ..., 9=2023/24
    通过轮换升降级球队来模拟真实赛季变化。
    """
    pool = LEAGUE_TEAM_POOLS[league_id]
    n = pool["n_teams"]
    base = pool["base_teams"]

    # 赛季属性: 法甲 2023/24 起改为 18 队
    if league_id == "ligue_1" and season_idx >= 9:
        n = 18

    # 每赛季轮换 3 支球队 (模拟升降级)
    rng = _random.Random(season_idx * 100 + hash(league_id) % 10000)
    rotate = 3
    core = base[:n]
    fringe = base[n:]

    if season_idx > 0:
        # 从 fringe 中选 rotate 支队替换 core 末尾
        swapped = rng.sample(range(n), min(rotate, n))
        fringe_pick = rng.sample(range(len(fringe)), min(rotate, len(fringe)))
        result = list(core)
        for i, s_idx in enumerate(swapped):
            if i < len(fringe_pick):
                result[s_idx] = fringe[fringe_pick[i]]
        core = result

    # 每个赛季微调 Elo (+/- 30)
    adjusted = []
    for name, elo in core:
        adj = rng.randint(-30, 30)
        adjusted.append((name, elo + adj))
    return adjusted


# ═══════════════════════════════════════════════════════════════
# 世界杯数据
# ═══════════════════════════════════════════════════════════════

WC_TEAMS = {
    2006: [
        ("德国", 1910), ("巴西", 1920), ("阿根廷", 1870), ("意大利", 1860),
        ("法国", 1850), ("英格兰", 1840), ("西班牙", 1830), ("荷兰", 1820),
        ("葡萄牙", 1800), ("捷克", 1760), ("瑞典", 1740), ("克罗地亚", 1720),
        ("墨西哥", 1700), ("巴拉圭", 1670), ("厄瓜多尔", 1650), ("美国", 1680),
        ("澳大利亚", 1660), ("日本", 1680), ("韩国", 1670), ("伊朗", 1630),
        ("沙特阿拉伯", 1600), ("突尼斯", 1620), ("科特迪瓦", 1690), ("加纳", 1680),
        ("安哥拉", 1550), ("多哥", 1540), ("特立尼达和多巴哥", 1530), ("哥斯达黎加", 1620),
        ("瑞士", 1720), ("乌克兰", 1700), ("波兰", 1680), ("塞尔维亚", 1660),
    ],
    2010: [
        ("西班牙", 1900), ("荷兰", 1860), ("德国", 1880), ("巴西", 1910),
        ("阿根廷", 1860), ("英格兰", 1830), ("意大利", 1820), ("法国", 1800),
        ("葡萄牙", 1790), ("乌拉圭", 1760), ("智利", 1740), ("巴拉圭", 1700),
        ("墨西哥", 1700), ("美国", 1690), ("加纳", 1720), ("日本", 1700),
        ("韩国", 1690), ("澳大利亚", 1660), ("南非", 1650), ("尼日利亚", 1680),
        ("喀麦隆", 1680), ("科特迪瓦", 1700), ("阿尔及利亚", 1650), ("斯洛伐克", 1660),
        ("斯洛文尼亚", 1640), ("塞尔维亚", 1670), ("丹麦", 1710), ("希腊", 1680),
        ("瑞士", 1700), ("洪都拉斯", 1600), ("新西兰", 1590), ("朝鲜", 1560),
    ],
    2014: [
        ("德国", 1920), ("阿根廷", 1890), ("荷兰", 1850), ("巴西", 1900),
        ("法国", 1840), ("西班牙", 1860), ("意大利", 1830), ("英格兰", 1820),
        ("葡萄牙", 1820), ("比利时", 1840), ("哥伦比亚", 1800), ("乌拉圭", 1780),
        ("智利", 1770), ("墨西哥", 1720), ("哥斯达黎加", 1680), ("美国", 1700),
        ("瑞士", 1730), ("希腊", 1700), ("尼日利亚", 1680), ("阿尔及利亚", 1700),
        ("科特迪瓦", 1700), ("加纳", 1690), ("喀麦隆", 1680), ("俄罗斯", 1680),
        ("克罗地亚", 1740), ("波黑", 1680), ("日本", 1700), ("韩国", 1700),
        ("澳大利亚", 1660), ("伊朗", 1640), ("洪都拉斯", 1600), ("厄瓜多尔", 1660),
    ],
    2018: [
        ("法国", 1920), ("克罗地亚", 1840), ("比利时", 1860), ("英格兰", 1840),
        ("巴西", 1910), ("德国", 1880), ("西班牙", 1860), ("阿根廷", 1840),
        ("葡萄牙", 1830), ("乌拉圭", 1800), ("哥伦比亚", 1800), ("瑞典", 1760),
        ("瑞士", 1750), ("墨西哥", 1730), ("丹麦", 1740), ("日本", 1720),
        ("俄罗斯", 1720), ("塞内加尔", 1700), ("韩国", 1710), ("沙特阿拉伯", 1640),
        ("伊朗", 1650), ("摩洛哥", 1680), ("秘鲁", 1700), ("埃及", 1690),
        ("尼日利亚", 1680), ("冰岛", 1680), ("塞尔维亚", 1700), ("哥斯达黎加", 1660),
        ("波兰", 1700), ("澳大利亚", 1670), ("突尼斯", 1640), ("巴拿马", 1560),
    ],
    2022: [
        ("阿根廷", 1900), ("法国", 1910), ("摩洛哥", 1760), ("克罗地亚", 1840),
        ("巴西", 1920), ("英格兰", 1850), ("荷兰", 1830), ("葡萄牙", 1840),
        ("西班牙", 1860), ("德国", 1840), ("比利时", 1820), ("日本", 1730),
        ("韩国", 1720), ("塞内加尔", 1720), ("美国", 1710), ("澳大利亚", 1680),
        ("瑞士", 1740), ("波兰", 1700), ("乌拉圭", 1780), ("墨西哥", 1720),
        ("丹麦", 1740), ("突尼斯", 1650), ("加纳", 1700), ("喀麦隆", 1680),
        ("厄瓜多尔", 1670), ("塞尔维亚", 1700), ("伊朗", 1660), ("威尔士", 1680),
        ("沙特阿拉伯", 1640), ("哥斯达黎加", 1660), ("加拿大", 1650), ("卡塔尔", 1620),
    ],
    # 额外一届: 2002
    2002: [
        ("巴西", 1930), ("德国", 1890), ("土耳其", 1760), ("韩国", 1750),
        ("西班牙", 1840), ("英格兰", 1830), ("阿根廷", 1880), ("意大利", 1850),
        ("法国", 1870), ("葡萄牙", 1800), ("塞内加尔", 1680), ("日本", 1700),
        ("美国", 1690), ("墨西哥", 1700), ("丹麦", 1720), ("瑞典", 1740),
        ("爱尔兰", 1700), ("巴拉圭", 1680), ("比利时", 1760), ("俄罗斯", 1670),
        ("乌拉圭", 1760), ("克罗地亚", 1720), ("南非", 1660), ("哥斯达黎加", 1640),
        ("喀麦隆", 1700), ("尼日利亚", 1680), ("突尼斯", 1630), ("沙特阿拉伯", 1610),
        ("厄瓜多尔", 1640), ("斯洛文尼亚", 1630), ("波兰", 1680), ("中国", 1580),
    ],
}

WC_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]


def generate_wc_fixtures(teams: List[Tuple[str, float]]) -> List[Tuple[str, str, str]]:
    """
    生成世界杯赛程: 小组赛。
    返回: [(home_team, away_team, stage), ...]
    """
    rng = _random.Random(42)
    team_names = [t[0] for t in teams]
    rng.shuffle(team_names)

    # 8 组 × 4 队
    groups = [team_names[i*4:(i+1)*4] for i in range(8)]
    fixtures = []

    # 小组赛: 每组单循环
    for g_idx, group in enumerate(groups):
        for i in range(4):
            for j in range(i + 1, 4):
                fixtures.append((group[i], group[j], "group"))

    return fixtures


# ═══════════════════════════════════════════════════════════════
# 回测结果
# ═══════════════════════════════════════════════════════════════

@dataclass
class LeagueSeasonResult:
    season: str
    initial_bankroll: float = 10000.0
    final_balance: float = 10000.0
    total_bets: int = 0
    total_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    profit_history: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    strategy_stats: Dict = field(default_factory=dict)
    all_elos: Dict = field(default_factory=dict)

    @property
    def roi(self):
        return (self.total_returned - self.total_staked) / self.total_staked if self.total_staked > 0 else 0.0
    @property
    def win_rate(self):
        return self.total_wins / self.total_bets if self.total_bets > 0 else 0.0
    @property
    def max_drawdown(self):
        if not self.equity_curve: return 0.0
        peak, max_dd = self.equity_curve[0], 0.0
        for v in self.equity_curve:
            if v > peak: peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
        return max_dd
    @property
    def sharpe(self):
        if len(self.profit_history) < 2: return 0.0
        arr = np.array(self.profit_history)
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        return mu / sigma * math.sqrt(252) if sigma > 1e-10 else 0.0
    @property
    def profit_factor(self):
        gp = sum(p for p in self.profit_history if p > 0)
        gl = abs(sum(p for p in self.profit_history if p < 0))
        return gp / gl if gl > 0 else 999.0


def run_league_season(
    season: str,
    season_idx: int,
    initial_bankroll: float = 10000.0,
    seed: int = 42,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> LeagueSeasonResult:
    """
    单赛季五大联赛全局回测。
    """
    _rng = _random.Random(seed + season_idx)

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(max_legs=4, kelly_discount=0.35, max_batch_size=20,
        min_single_value=0.01, min_combined_value=0.015)
    match_outcomes = {}

    result = LeagueSeasonResult(
        season=season, initial_bankroll=initial_bankroll,
        strategy_stats={
            "1x2": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "asian_handicap": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "over_under": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "parlay": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        },
    )
    result.equity_curve.append(initial_bankroll)

    league_order = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

    league_pipelines = {}
    league_ortho_gens = {}
    league_orchestrators = {}
    league_team_elos = {}
    league_fixtures = {}
    all_elos = {}

    for league_id in league_order:
        pool = LEAGUE_TEAM_POOLS[league_id]
        teams = get_season_teams(league_id, season_idx)

        if carryover_elos and league_id in carryover_elos:
            team_elos = dict(carryover_elos[league_id])
            for name, default_elo in teams:
                if name not in team_elos:
                    team_elos[name] = default_elo
        else:
            team_elos = {t[0]: t[1] for t in teams}

        league_team_elos[league_id] = team_elos
        all_elos[league_id] = dict(team_elos)

        calib = _load_calibrated_weights().get(league_id, {})
        weight_multipliers = _build_weight_multipliers(league_id, calib)

        pipeline = GameFlowPipeline(
            league_id, initial_bankroll=initial_bankroll,
            weight_multipliers=weight_multipliers,
        )
        pipeline.set_bankroll_manager(shared_bankroll)
        if calib:
            pipeline.signal_decomposer = SignalDecomposer(elo_suppression=1.0)
            pipeline.prior_shrinkage = PriorShrinkage(
                alpha_high=calib.get("shrinkage_alpha_high", 0.50),
                alpha_low=calib.get("shrinkage_alpha_low", 0.10),
            )
        league_pipelines[league_id] = pipeline
        league_ortho_gens[league_id] = OrthogonalDataGenerator(league_id, seed=seed + season_idx)
        league_orchestrators[league_id] = create_orchestrator(league_id)

        league_fixtures[league_id] = generate_fixture_list(teams)

    # 交叉排序
    max_rounds = max(len(league_fixtures[lid]) for lid in league_order)
    all_fixtures = []
    for rnd in range(max_rounds):
        for lid in league_order:
            if lid in league_fixtures and rnd < len(league_fixtures[lid]):
                h, a = league_fixtures[lid][rnd]
                all_fixtures.append((lid, rnd, h, a))

    year = int(season.split("/")[0])
    base_date = datetime(year, 8, 11)
    home_advantage = 65.0
    total_fixtures = len(all_fixtures)
    days_between = 3.5 * 380 / total_fixtures * 5

    for global_idx, (lid, round_idx, home_team, away_team) in enumerate(all_fixtures):
        try:
            pipeline = league_pipelines[lid]
            ortho_gen = league_ortho_gens[lid]
            team_elos = league_team_elos[lid]

            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]

            odds_h, odds_d, odds_a = generate_realistic_odds(
                home_elo, away_elo, home_advantage, seed=seed + season_idx * 1000 + global_idx,
            )

            match_date = base_date + timedelta(days=int(global_idx * days_between))
            match_date += timedelta(hours=_rng.randint(12, 21))

            match = MatchContext(
                match_id=f"{lid}_S{year}_M{global_idx:04d}",
                league_id=lid, season=season,
                matchday=round_idx + 1, kickoff_time=match_date,
                home_team=home_team, away_team=away_team,
                home_elo=home_elo, away_elo=away_elo,
                odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            )

            ortho_data = ortho_gen.generate(
                global_idx, home_team, away_team, match_date, odds_h, odds_d, odds_a,
            )
            extra = {
                "elo_diff": home_elo - away_elo,
                "recent_results": [_rng.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "streak_momentum": _rng.uniform(0, 0.5),
                "streak_momentum_league": _rng.uniform(0, 0.5),
                "match_phase": 1.0,
            }
            extra.update(ortho_gen.to_extra_dict(ortho_data))

            pipeline_result = pipeline.run_full(match, extra_data=extra)

            if pipeline_result.placements:
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a,
                    home_advantage, seed=seed + season_idx * 1000 + global_idx,
                )
                actual_home_goals, actual_away_goals = _simulate_goals(
                    actual_outcome, lid, home_elo, away_elo,
                    seed=seed + season_idx * 1000 + global_idx,
                )
                match_outcomes[match.match_id] = (actual_outcome, actual_outcome.replace("_win", ""))

                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                # 1X2 结算
                placements = pipeline.settle_bets(pipeline_result.placements, actual)
                for p in placements:
                    result.total_bets += 1
                    result.total_staked += p.stake
                    if p.result == BetResult.WIN:
                        result.total_wins += 1
                        result.total_returned += p.stake + p.profit_loss
                    result.profit_history.append(p.profit_loss)
                    result.strategy_stats["1x2"]["bets"] += 1
                    result.strategy_stats["1x2"]["staked"] += p.stake
                    result.strategy_stats["1x2"]["returned"] += p.stake + p.profit_loss
                    if p.result == BetResult.WIN:
                        result.strategy_stats["1x2"]["wins"] += 1

                # 串关池
                if pipeline_result.proposals:
                    parlay_mgr.add_match_bets(match.match_id, pipeline_result.proposals)

                # 亚盘 + 大小球
                orchestrator = league_orchestrators[lid]
                try:
                    synthetic_handicap = orchestrator.asian_strategy.generate_synthetic_odds(
                        pipeline_result.poisson_score_matrix)
                    synthetic_totals = orchestrator.over_under_strategy.generate_synthetic_odds(
                        pipeline_result.poisson_score_matrix)
                    multi_result = orchestrator.run(
                        match=match, score_matrix=pipeline_result.poisson_score_matrix,
                        handicap_odds=synthetic_handicap, totals_odds=synthetic_totals,
                        total_bankroll=shared_bankroll._get_base_bankroll(),
                    )
                    for ap in multi_result.asian_proposals:
                        asian_result_val, asian_pnl = orchestrator.asian_strategy.settle(
                            ap, actual_home_goals, actual_away_goals)
                        is_win = asian_result_val in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                        stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else ap.kelly_stake
                        if is_win: result.total_wins += 1
                        if asian_pnl >= 0: result.total_returned += stake + asian_pnl
                        result.profit_history.append(asian_pnl)
                        result.strategy_stats["asian_handicap"]["bets"] += 1
                        result.strategy_stats["asian_handicap"]["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                        result.strategy_stats["asian_handicap"]["returned"] += (stake + asian_pnl) if asian_pnl >= 0 else 0
                        if is_win: result.strategy_stats["asian_handicap"]["wins"] += 1

                    for tp in multi_result.totals_proposals:
                        totals_result_val, totals_pnl = orchestrator.over_under_strategy.settle(
                            tp, actual_home_goals, actual_away_goals)
                        is_win = (totals_result_val == "win")
                        stake = tp.kelly_stake * getattr(tp, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else tp.kelly_stake
                        if is_win: result.total_wins += 1
                        if totals_pnl >= 0: result.total_returned += stake + totals_pnl
                        result.profit_history.append(totals_pnl)
                        result.strategy_stats["over_under"]["bets"] += 1
                        result.strategy_stats["over_under"]["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                        result.strategy_stats["over_under"]["returned"] += (stake + totals_pnl) if totals_pnl >= 0 else 0
                        if is_win: result.strategy_stats["over_under"]["wins"] += 1
                except Exception:
                    pass

                # 更新 Elo
                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome,
                    LEAGUE_TEAM_POOLS[lid]["k_elo"], home_advantage,
                )
                team_elos[home_team] = new_home
                team_elos[away_team] = new_away

            # 每 20 场生成串关
            if (global_idx + 1) % 20 == 0 or global_idx == total_fixtures - 1:
                settlements = parlay_mgr.settle_all_ready(match_outcomes)
                for s in settlements:
                    shared_bankroll.state.balance += s.profit
                    result.total_bets += 1
                    result.total_staked += s.stake
                    if s.won: result.total_wins += 1; result.total_returned += s.returned
                    result.strategy_stats["parlay"]["bets"] += 1
                    result.strategy_stats["parlay"]["staked"] += s.stake
                    result.strategy_stats["parlay"]["returned"] += s.returned
                    if s.won: result.strategy_stats["parlay"]["wins"] += 1
                    result.profit_history.append(s.profit)

            result.equity_curve.append(shared_bankroll.state.balance)

        except Exception:
            pass

    result.final_balance = shared_bankroll.state.balance
    result.all_elos = {lid: dict(league_team_elos[lid]) for lid in league_order}
    return result


# ═══════════════════════════════════════════════════════════════
# 世界杯回测
# ═══════════════════════════════════════════════════════════════

def run_world_cup(
    year: int,
    initial_bankroll: float = 10000.0,
    seed: int = 42,
    carryover_elos: Optional[Dict[str, Dict[str, float]]] = None,
) -> LeagueSeasonResult:
    """
    单届世界杯回测。
    使用简化的 Club→National 映射: 国家队 Elo = 该队球员所在联赛的加权平均。
    如果没有联赛 Elo，使用 WC_TEAMS 中的默认值。
    """
    teams = WC_TEAMS.get(year, WC_TEAMS[2022])
    _rng = _random.Random(seed + year)

    # 推导国家队 Elo: 从联赛 Elo 平均
    # 简化: 直接使用 WC_TEAMS 中的初始 Elo，加联赛 Elo 的微调
    team_elos = {t[0]: t[1] for t in teams}
    if carryover_elos:
        # 计算联赛平均 Elo 作为参考
        all_league_elos = []
        for lid, elos in carryover_elos.items():
            all_league_elos.extend(elos.values())
        league_avg = sum(all_league_elos) / len(all_league_elos) if all_league_elos else 1650.0
        # 国家队 Elo 向联赛均值微调
        for name in team_elos:
            team_elos[name] = team_elos[name] * 0.7 + league_avg * 0.3

    shared_bankroll = BankrollManager(initial_bankroll, fixed_base=True)
    parlay_mgr = ParlayBatchManager(max_legs=4, kelly_discount=0.35, max_batch_size=10,
        min_single_value=0.01, min_combined_value=0.015)
    match_outcomes = {}

    result = LeagueSeasonResult(
        season=f"WC{year}", initial_bankroll=initial_bankroll,
        strategy_stats={
            "1x2": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "asian_handicap": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "over_under": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
            "parlay": {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0},
        },
    )
    result.equity_curve.append(initial_bankroll)

    fixtures = generate_wc_fixtures(teams)
    home_advantage = 0.0  # 世界杯中立场

    # 使用英超联赛 pipeline (世界杯用通用参数)
    calib = _load_calibrated_weights().get("premier_league", {})
    pipeline = GameFlowPipeline("premier_league", initial_bankroll=initial_bankroll,
        weight_multipliers=_build_weight_multipliers("premier_league", calib))
    pipeline.set_bankroll_manager(shared_bankroll)
    ortho_gen = OrthogonalDataGenerator("premier_league", seed=seed + year)
    orchestrator = create_orchestrator("premier_league")

    base_date = datetime(year, 6, 10)  # 世界杯通常在6月

    # 小组赛阶段: 追踪小组积分
    group_standings = defaultdict(lambda: defaultdict(int))
    group_matches_done = defaultdict(int)

    # 按 group 分组 fixtures
    group_fixtures = []
    knockout_fixtures = []
    for h, a, stage in fixtures:
        if stage == "group":
            group_fixtures.append((h, a, stage))
        else:
            knockout_fixtures.append((h, a, stage))

    # 先处理小组赛
    all_fixtures = group_fixtures + knockout_fixtures

    # 淘汰赛真实对阵需要动态确定
    # 简化: 按顺序处理所有比赛，淘汰赛用预定对阵
    for global_idx, (home_team, away_team, stage) in enumerate(all_fixtures):
        try:
            # 对于淘汰赛占位符，用实际出线球队替换
            h_actual = home_team
            a_actual = away_team

            home_elo = team_elos.get(h_actual, 1650)
            away_elo = team_elos.get(a_actual, 1650)

            odds_h, odds_d, odds_a = generate_realistic_odds(
                home_elo, away_elo, home_advantage, seed=seed + year * 1000 + global_idx,
            )

            match_date = base_date + timedelta(days=int(global_idx * 2.5))
            match_date += timedelta(hours=_rng.randint(12, 21))

            match = MatchContext(
                match_id=f"WC{year}_M{global_idx:04d}",
                league_id="premier_league", season=f"WC{year}",
                matchday=global_idx + 1, kickoff_time=match_date,
                home_team=h_actual, away_team=a_actual,
                home_elo=home_elo, away_elo=away_elo,
                odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            )

            ortho_data = ortho_gen.generate(
                global_idx, h_actual, a_actual, match_date, odds_h, odds_d, odds_a,
            )
            extra = {
                "elo_diff": home_elo - away_elo,
                "recent_results": [_rng.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)],
                "rank_diff": int((home_elo - away_elo) / 20),
                "goal_diff": (home_elo - away_elo) / 20,
                "xg_diff": (home_elo - away_elo) / 200,
                "streak_momentum": _rng.uniform(0, 0.5),
                "streak_momentum_league": _rng.uniform(0, 0.5),
                "match_phase": 1.0,
            }
            extra.update(ortho_gen.to_extra_dict(ortho_data))

            pipeline_result = pipeline.run_full(match, extra_data=extra)

            if pipeline_result.placements:
                actual_outcome = simulate_match_result(
                    home_elo, away_elo, odds_h, odds_d, odds_a,
                    home_advantage, seed=seed + year * 1000 + global_idx,
                )
                actual_home_goals, actual_away_goals = _simulate_goals(
                    actual_outcome, "premier_league", home_elo, away_elo,
                    seed=seed + year * 1000 + global_idx,
                )
                match_outcomes[match.match_id] = (actual_outcome, actual_outcome.replace("_win", ""))

                outcome_map = {
                    "home_win": BetSelection.HOME_WIN,
                    "draw": BetSelection.DRAW,
                    "away_win": BetSelection.AWAY_WIN,
                }
                actual = outcome_map[actual_outcome]

                placements = pipeline.settle_bets(pipeline_result.placements, actual)
                for p in placements:
                    result.total_bets += 1
                    result.total_staked += p.stake
                    if p.result == BetResult.WIN:
                        result.total_wins += 1
                        result.total_returned += p.stake + p.profit_loss
                    result.profit_history.append(p.profit_loss)
                    result.strategy_stats["1x2"]["bets"] += 1
                    result.strategy_stats["1x2"]["staked"] += p.stake
                    result.strategy_stats["1x2"]["returned"] += p.stake + p.profit_loss
                    if p.result == BetResult.WIN:
                        result.strategy_stats["1x2"]["wins"] += 1

                if pipeline_result.proposals:
                    parlay_mgr.add_match_bets(match.match_id, pipeline_result.proposals)

                try:
                    synthetic_handicap = orchestrator.asian_strategy.generate_synthetic_odds(
                        pipeline_result.poisson_score_matrix)
                    synthetic_totals = orchestrator.over_under_strategy.generate_synthetic_odds(
                        pipeline_result.poisson_score_matrix)
                    multi_result = orchestrator.run(
                        match=match, score_matrix=pipeline_result.poisson_score_matrix,
                        handicap_odds=synthetic_handicap, totals_odds=synthetic_totals,
                        total_bankroll=shared_bankroll._get_base_bankroll(),
                    )
                    for ap in multi_result.asian_proposals:
                        asian_result_val, asian_pnl = orchestrator.asian_strategy.settle(
                            ap, actual_home_goals, actual_away_goals)
                        is_win = asian_result_val in (AsianHandicapResult.FULL_WIN, AsianHandicapResult.HALF_WIN)
                        stake = ap.kelly_stake * getattr(ap, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else ap.kelly_stake
                        if is_win: result.total_wins += 1
                        if asian_pnl >= 0: result.total_returned += stake + asian_pnl
                        result.profit_history.append(asian_pnl)
                        result.strategy_stats["asian_handicap"]["bets"] += 1
                        result.strategy_stats["asian_handicap"]["staked"] += abs(stake) if stake > 0 else ap.kelly_stake
                        result.strategy_stats["asian_handicap"]["returned"] += (stake + asian_pnl) if asian_pnl >= 0 else 0
                        if is_win: result.strategy_stats["asian_handicap"]["wins"] += 1
                    for tp in multi_result.totals_proposals:
                        totals_result_val, totals_pnl = orchestrator.over_under_strategy.settle(
                            tp, actual_home_goals, actual_away_goals)
                        is_win = (totals_result_val == "win")
                        stake = tp.kelly_stake * getattr(tp, 'strategy_weight', 1.0)
                        result.total_bets += 1
                        result.total_staked += abs(stake) if stake > 0 else tp.kelly_stake
                        if is_win: result.total_wins += 1
                        if totals_pnl >= 0: result.total_returned += stake + totals_pnl
                        result.profit_history.append(totals_pnl)
                        result.strategy_stats["over_under"]["bets"] += 1
                        result.strategy_stats["over_under"]["staked"] += abs(stake) if stake > 0 else tp.kelly_stake
                        result.strategy_stats["over_under"]["returned"] += (stake + totals_pnl) if totals_pnl >= 0 else 0
                        if is_win: result.strategy_stats["over_under"]["wins"] += 1
                except Exception:
                    pass

                new_home, new_away = update_elo(
                    home_elo, away_elo, actual_outcome, 24, home_advantage,
                )
                team_elos[h_actual] = new_home
                team_elos[a_actual] = new_away

            if (global_idx + 1) % 10 == 0 or global_idx == len(all_fixtures) - 1:
                settlements = parlay_mgr.settle_all_ready(match_outcomes)
                for s in settlements:
                    shared_bankroll.state.balance += s.profit
                    result.total_bets += 1
                    result.total_staked += s.stake
                    if s.won: result.total_wins += 1; result.total_returned += s.returned
                    result.strategy_stats["parlay"]["bets"] += 1
                    result.strategy_stats["parlay"]["staked"] += s.stake
                    result.strategy_stats["parlay"]["returned"] += s.returned
                    if s.won: result.strategy_stats["parlay"]["wins"] += 1
                    result.profit_history.append(s.profit)

            result.equity_curve.append(shared_bankroll.state.balance)

        except Exception:
            pass

    result.final_balance = shared_bankroll.state.balance
    return result


# ═══════════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════════

def print_season_row(label: str, r: LeagueSeasonResult):
    ss = r.strategy_stats
    print(f"  {label:<12} {r.total_bets:>6} {r.win_rate:>6.1%} {r.roi:>7.1%} "
          f"{r.final_balance:>10,.0f} {r.max_drawdown:>6.1%} "
          f"{ss['1x2']['bets']:>5} {ss['asian_handicap']['bets']:>5} "
          f"{ss['over_under']['bets']:>5} {ss['parlay']['bets']:>5}")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + "  GTO-GameFlow v5.9.1 扩展回测 — 10赛季五大联赛 + 6届世界杯".center(74) + "║")
    print("╠" + "═" * 78 + "╣")
    print("║" + "  资金: 全局固定基数 10,000 (每赛季独立) | 策略: 1X2 + 亚盘 + 大小球 + 串关".center(70) + "║")
    print("╚" + "═" * 78 + "╝")

    # ── 10 赛季五大联赛 ──
    seasons = [f"{y}/{str(y+1)[-2:]}" for y in range(2014, 2024)]
    league_results: List[LeagueSeasonResult] = []
    carryover_elos = None

    print("\n" + "▇" * 78)
    print("  五大联赛 10 赛季回测 (2014/15 → 2023/24)")
    print("▇" * 78)

    print(f"\n  {'赛季':<12} {'投注':>6} {'胜率':>7} {'ROI':>7} {'最终资金':>10} "
          f"{'回撤':>6} {'1X2':>5} {'亚盘':>5} {'大小球':>5} {'串关':>5}")
    print(f"  {'─'*78}")

    total_league_bets = 0
    total_league_staked = 0.0
    total_league_returned = 0.0
    total_league_profit = 0.0

    for idx, season in enumerate(seasons):
        r = run_league_season(season, idx, seed=42 + idx * 10,
                              carryover_elos=carryover_elos)
        league_results.append(r)
        carryover_elos = r.all_elos

        total_league_bets += r.total_bets
        total_league_staked += r.total_staked
        total_league_returned += r.total_returned
        total_league_profit += (r.final_balance - 10000)

        print_season_row(season, r)

    # 联赛汇总
    combined_league_roi = ((total_league_returned - total_league_staked) / total_league_staked
                           if total_league_staked > 0 else 0)
    print(f"  {'─'*78}")
    print(f"  {'联赛合计':<12} {total_league_bets:>6} "
          f"{'':>7} {combined_league_roi:>7.1%} "
          f"{'':>10} {'':>6} {'':>5} {'':>5} {'':>5} {'':>5}")
    print(f"  10赛季联赛总利润: {total_league_profit:+,.0f}")

    # ── 6 届世界杯 ──
    print("\n\n" + "▇" * 78)
    print("  6 届世界杯回测 (2002 → 2022)")
    print("▇" * 78)

    print(f"\n  {'赛事':<12} {'投注':>6} {'胜率':>7} {'ROI':>7} {'最终资金':>10} "
          f"{'回撤':>6} {'1X2':>5} {'亚盘':>5} {'大小球':>5} {'串关':>5}")
    print(f"  {'─'*78}")

    total_wc_bets = 0
    total_wc_staked = 0.0
    total_wc_returned = 0.0
    total_wc_profit = 0.0

    for year in WC_YEARS:
        r = run_world_cup(year, seed=42 + year,
                          carryover_elos=carryover_elos)
        total_wc_bets += r.total_bets
        total_wc_staked += r.total_staked
        total_wc_returned += r.total_returned
        total_wc_profit += (r.final_balance - 10000)

        print_season_row(f"WC{year}", r)

    combined_wc_roi = ((total_wc_returned - total_wc_staked) / total_wc_staked
                       if total_wc_staked > 0 else 0)
    print(f"  {'─'*78}")
    print(f"  {'世界杯合计':<12} {total_wc_bets:>6} "
          f"{'':>7} {combined_wc_roi:>7.1%} "
          f"{'':>10} {'':>6} {'':>5} {'':>5} {'':>5} {'':>5}")
    print(f"  6届世界杯总利润: {total_wc_profit:+,.0f}")

    # ── 策略分类汇总 ──
    print("\n\n" + "╔" + "═" * 78 + "╗")
    print("║" + "  全时段策略分类汇总".center(74) + "║")
    print("╚" + "═" * 78 + "╝")

    all_strategies = {"1x2": [0, 0, 0.0, 0.0], "asian_handicap": [0, 0, 0.0, 0.0],
                      "over_under": [0, 0, 0.0, 0.0], "parlay": [0, 0, 0.0, 0.0]}
    for r in league_results:
        for s in all_strategies:
            ss = r.strategy_stats[s]
            all_strategies[s][0] += ss["bets"]
            all_strategies[s][1] += ss["wins"]
            all_strategies[s][2] += ss["staked"]
            all_strategies[s][3] += ss["returned"]

    print(f"\n  {'策略':<18} {'投注':>6} {'胜率':>7} {'ROI':>7} {'投入':>10} {'回报':>10}")
    print(f"  {'─'*62}")
    for s in ["1x2", "asian_handicap", "over_under", "parlay"]:
        bets, wins, staked, returned = all_strategies[s]
        if bets > 0:
            wr = wins / bets
            roi = (returned - staked) / staked if staked > 0 else 0
            label = cn_strategy(s)
            print(f"  {label:<18} {bets:>6} {wr:>6.1%} {roi:>6.1%} "
                  f"{staked:>10,.0f} {returned:>10,.0f}")

    # ── 总汇总 ──
    print("\n\n" + "╔" + "═" * 78 + "╗")
    print("║" + "  总汇总".center(74) + "║")
    print("╚" + "═" * 78 + "╝")

    grand_total_bets = total_league_bets + total_wc_bets
    grand_total_staked = total_league_staked + total_wc_staked
    grand_total_returned = total_league_returned + total_wc_returned
    grand_total_profit = total_league_profit + total_wc_profit
    grand_roi = ((grand_total_returned - grand_total_staked) / grand_total_staked
                 if grand_total_staked > 0 else 0)

    print(f"\n  联赛: 10赛季, {total_league_bets:,} 注, 利润 {total_league_profit:+,.0f}")
    print(f"  世界杯: 6届, {total_wc_bets:,} 注, 利润 {total_wc_profit:+,.0f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  总计: {grand_total_bets:,} 注, 利润 {grand_total_profit:+,.0f}, ROI {grand_roi:+.1%}")
    print(f"  总投入: {grand_total_staked:,.0f}, 总回报: {grand_total_returned:,.0f}")
    print(f"  总初始资金: {len(seasons) * 10000 + len(WC_YEARS) * 10000:,}")

    print("\n" + "═" * 78)
    print("  v5.9.1 扩展回测完成")
    print("═" * 78)