"""
GTO-GameFlow v5.11 真实CSV数据回测

使用 historical_odds/ 目录下的真实比赛数据进行端到端回测。
- Elo 从历史比赛逐场构建
- 因子从 CSV 真实数据计算 (射门/角球/赔率/排名等)
- 按时间顺序逐场预测，模拟真实投注场景

用法:
    python3 tests/test_csv_backtest.py
    python3 tests/test_csv_backtest.py --leagues premier_league --seasons 2023-24
"""
import sys
import os
import csv
import math
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.models import (
    MatchContext, BetProposal, BetSelection, BetPlacement, BetResult,
    ProbabilityDistribution, BankrollState,
)
from src.engine.elo_cold_start import EloColdStart
from src.engine.probability import ProbabilityEngine
from src.engine.bankroll import BankrollManager, generate_bet_proposals, compute_confidence
from src.engine.risk_control import RiskController
from src.engine.unified_bayesian_shrinkage import create_shrinkage_for_league
from src.engine.unified_probability_engine import UnifiedProbabilityEngine
from src.factors.compute import FactorComputationEngine
from src.factors.heterogeneous_groups import group_signal_cap
from src.config.settings import config
from src.config.league_params import get_league_params


# ================================================================
# 数据加载
# ================================================================

CSV_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'historical_odds')

LEAGUE_FILES = {
    "premier_league": "premier_league",
    "la_liga": "la_liga",
    "bundesliga": "bundesliga",
    "serie_a": "serie_a",
    "ligue_1": "ligue_1",
}

DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y"]


def parse_date(date_str: str) -> Optional[datetime]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val.strip()) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val.strip())) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


@dataclass
class RawMatch:
    """从CSV解析的原始比赛数据"""
    date: datetime
    home_team: str
    away_team: str
    fthg: int          # 全场主队进球
    ftag: int          # 全场客队进球
    ftr: str           # H/D/A
    hthg: int = 0      # 半场主队进球
    htag: int = 0      # 半场客队进球
    htr: str = ""      # H/D/A
    referee: str = ""
    hs: int = 0        # 主队射门
    as_: int = 0       # 客队射门 (避免关键字冲突)
    hst: int = 0       # 主队射正
    ast: int = 0       # 客队射正
    hf: int = 0        # 主队犯规
    af: int = 0        # 客队犯规
    hc: int = 0        # 主队角球
    ac: int = 0        # 客队角球
    hy: int = 0        # 主队黄牌
    ay: int = 0        # 客队黄牌
    hr: int = 0        # 主队红牌
    ar: int = 0        # 客队红牌
    # 赔率 (开盘)
    b365h: float = 0.0
    b365d: float = 0.0
    b365a: float = 0.0
    psh: float = 0.0
    psd: float = 0.0
    psa: float = 0.0
    maxh: float = 0.0
    maxd: float = 0.0
    maxa: float = 0.0
    avgh: float = 0.0
    avgd: float = 0.0
    avga: float = 0.0
    # 赔率 (收盘)
    b365ch: float = 0.0
    b365cd: float = 0.0
    b365ca: float = 0.0
    maxch: float = 0.0
    maxcd: float = 0.0
    maxca: float = 0.0
    # 亚盘
    ahh: float = 0.0   # 亚盘让球线
    # 大小球
    over25: float = 0.0
    under25: float = 0.0


def load_csv(filepath: str) -> List[RawMatch]:
    """加载单个CSV文件"""
    matches = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = parse_date(row.get('Date', ''))
            if not date:
                continue

            home = row.get('HomeTeam', '').strip()
            away = row.get('AwayTeam', '').strip()
            if not home or not away:
                continue

            fthg = safe_int(row.get('FTHG', '0'))
            ftag = safe_int(row.get('FTAG', '0'))
            ftr = row.get('FTR', '').strip().upper()
            if ftr not in ('H', 'D', 'A'):
                continue

            # 开盘赔率 (优先 B365, 备选 Pinnacle/Max/Avg)
            def get_odds(prefix, col):
                val = safe_float(row.get(f'{prefix}{col}', '0'))
                if val > 1.0:
                    return val
                return 0.0

            b365h = get_odds('B365', 'H')
            b365d = get_odds('B365', 'D')
            b365a = get_odds('B365', 'A')

            # 如果 B365 没有，用平均赔率
            if b365h <= 1.0:
                b365h = get_odds('Avg', 'H')
                b365d = get_odds('Avg', 'D')
                b365a = get_odds('Avg', 'A')
            if b365h <= 1.0:
                b365h = get_odds('Max', 'H')
                b365d = get_odds('Max', 'D')
                b365a = get_odds('Max', 'A')

            m = RawMatch(
                date=date,
                home_team=home,
                away_team=away,
                fthg=fthg,
                ftag=ftag,
                ftr=ftr,
                hthg=safe_int(row.get('HTHG', '0')),
                htag=safe_int(row.get('HTAG', '0')),
                htr=row.get('HTR', '').strip(),
                referee=row.get('Referee', '').strip(),
                hs=safe_int(row.get('HS', '0')),
                as_=safe_int(row.get('AS', '0')),
                hst=safe_int(row.get('HST', '0')),
                ast=safe_int(row.get('AST', '0')),
                hf=safe_int(row.get('HF', '0')),
                af=safe_int(row.get('AF', '0')),
                hc=safe_int(row.get('HC', '0')),
                ac=safe_int(row.get('AC', '0')),
                hy=safe_int(row.get('HY', '0')),
                ay=safe_int(row.get('AY', '0')),
                hr=safe_int(row.get('HR', '0')),
                ar=safe_int(row.get('AR', '0')),
                b365h=b365h,
                b365d=b365d,
                b365a=b365a,
                psh=get_odds('PS', 'H'),
                psd=get_odds('PS', 'D'),
                psa=get_odds('PS', 'A'),
                maxh=get_odds('Max', 'H'),
                maxd=get_odds('Max', 'D'),
                maxa=get_odds('Max', 'A'),
                avgh=get_odds('Avg', 'H'),
                avgd=get_odds('Avg', 'D'),
                avga=get_odds('Avg', 'A'),
                b365ch=get_odds('B365C', 'H'),
                b365cd=get_odds('B365C', 'D'),
                b365ca=get_odds('B365C', 'A'),
                maxch=get_odds('MaxC', 'H'),
                maxcd=get_odds('MaxC', 'D'),
                maxca=get_odds('MaxC', 'A'),
                ahh=safe_float(row.get('AHh', '0')),
                over25=get_odds('B365', '>2.5'),
                under25=get_odds('B365', '<2.5'),
            )
            matches.append(m)

    return matches


def load_all_matches(leagues: List[str], seasons: List[str]) -> Dict[str, List[RawMatch]]:
    """加载所有联赛-赛季的比赛，按日期排序"""
    all_matches: Dict[str, List[RawMatch]] = {}

    for league in leagues:
        league_matches = []
        prefix = LEAGUE_FILES.get(league, league)
        for season in seasons:
            filepath = os.path.join(CSV_DIR, f"{prefix}_{season}.csv")
            if not os.path.exists(filepath):
                # 尝试长格式 (2014-2015)
                parts = season.split('-')
                if len(parts) == 2:
                    alt_season = f"{parts[0]}-{parts[0][:2]}{parts[1]}"
                    filepath = os.path.join(CSV_DIR, f"{prefix}_{alt_season}.csv")
            if not os.path.exists(filepath):
                print(f"  ⚠ 未找到: {prefix}_{season}.csv")
                continue
            matches = load_csv(filepath)
            league_matches.extend(matches)
            print(f"  ✓ {league} {season}: {len(matches)} 场")

        league_matches.sort(key=lambda m: m.date)
        all_matches[league] = league_matches

    return all_matches


# ================================================================
# 回测引擎
# ================================================================

@dataclass
class BacktestBet:
    """单笔回测投注"""
    match_id: str
    date: datetime
    league: str
    home: str
    away: str
    selection: str        # home_win / draw / away_win
    odds: float
    model_prob: float
    implied_prob: float
    value: float
    stake: float
    actual_result: str    # H/D/A
    won: bool
    profit: float
    confidence: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    league: str
    season_range: str
    total_matches: int = 0
    total_bets: int = 0
    total_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    profit_history: List[float] = field(default_factory=list)
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    bets: List[BacktestBet] = field(default_factory=list)
    # 按类型分
    home_bets: int = 0
    home_wins: int = 0
    draw_bets: int = 0
    draw_wins: int = 0
    away_bets: int = 0
    away_wins: int = 0

    @property
    def roi(self) -> float:
        if self.total_staked <= 0:
            return 0.0
        return (self.total_returned - self.total_staked) / self.total_staked

    @property
    def win_rate(self) -> float:
        if self.total_bets <= 0:
            return 0.0
        return self.total_wins / self.total_bets

    @property
    def profit(self) -> float:
        return self.total_returned - self.total_staked


class CSVBacktestEngine:
    """
    基于真实CSV数据的回测引擎。

    流程:
    1. 前 N 场比赛用于 Elo 冷启动 (不投注)
    2. 之后每场比赛:
       a. 用历史数据计算所有因子
       b. 运行九阶段流水线
       c. 如果有价值投注，记录并结算
    """

    def __init__(
        self,
        league: str,
        initial_bankroll: float = 10000.0,
        min_history: int = 30,
        value_threshold: float = 0.03,
        confidence_threshold: float = 0.55,
        kelly_fraction: float = 0.25,
        factor_scale: float = None,  # None = 联赛自动选择
    ):
        self.league = league
        self.initial_bankroll = initial_bankroll
        self.min_history = min_history
        self.value_threshold = value_threshold
        self.confidence_threshold = confidence_threshold
        self.kelly_fraction = kelly_fraction

        # 联赛特化参数
        league_scales = {
            "premier_league": 0.22,
            "la_liga": 0.20,
            "bundesliga": 0.25,
            "serie_a": 0.18,
            "ligue_1": 0.18,
        }
        self.factor_scale = factor_scale if factor_scale is not None else league_scales.get(league, 0.25)

        # 组件
        self.elo_engine = EloColdStart(default_elo=1500.0, k=20, home_advantage=65)
        self.prob_engine = ProbabilityEngine(league)
        self.risk_ctrl = RiskController()
        self.params = get_league_params(league)

        # 状态
        self.bankroll = initial_bankroll
        self.bet_history: List[BacktestBet] = []
        self.balance_history: List[float] = []

        # 历史统计缓存
        self.team_stats: Dict[str, Dict] = defaultdict(lambda: {
            'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0,
            'shots': 0, 'shots_on_target': 0, 'corners': 0,
            'fouls': 0, 'yellow_cards': 0, 'red_cards': 0,
            'recent_results': [],  # 最近10场: 3=胜, 1=平, 0=负
            'home_draws': 0, 'home_matches': 0,
            'away_draws': 0, 'away_matches': 0,
            'ht_leads': 0,      # 半场领先次数
            'ht_trails': 0,     # 半场落后次数
            'ht_draws': 0,      # 半场平局次数
            'comebacks': 0,     # 半场落后但最终不败
            'collapses': 0,     # 半场领先但最终不胜
            'goals_by_match': [],  # 每场总进球 (用于波动性)
            'match_dates': [],    # 比赛日期 (用于赛程密度)
        })
        self.h2h: Dict[str, List] = defaultdict(list)  # "teamA_vs_teamB" -> [results]
        # 联赛积分表
        self.league_table: Dict[str, Dict] = defaultdict(lambda: {'pts': 0, 'gf': 0, 'ga': 0, 'played': 0})
        # 裁判统计
        self.referee_stats: Dict[str, Dict] = defaultdict(lambda: {'yellows': 0, 'reds': 0, 'matches': 0, 'home_wins': 0})
        # 德比配对 (简化版)
        self.derby_pairs = {
            frozenset({'Man United', 'Man City'}), frozenset({'Arsenal', 'Tottenham'}),
            frozenset({'Liverpool', 'Everton'}), frozenset({'Chelsea', 'Tottenham'}),
            frozenset({'Arsenal', 'Chelsea'}), frozenset({'West Ham', 'Tottenham'}),
            frozenset({'Real Madrid', 'Barcelona'}), frozenset({'Real Madrid', 'Atletico Madrid'}),
            frozenset({'Barcelona', 'Espanyol'}), frozenset({'Sevilla', 'Real Betis'}),
            frozenset({'Bayern Munich', 'Dortmund'}), frozenset({'Schalke', 'Dortmund'}),
            frozenset({'Juventus', 'Inter'}), frozenset({'Juventus', 'Torino'}),
            frozenset({'AC Milan', 'Inter'}), frozenset({'Roma', 'Lazio'}),
            frozenset({'PSG', 'Marseille'}), frozenset({'Lyon', 'Saint-Etienne'}),
        }

    def run(self, matches: List[RawMatch]) -> BacktestResult:
        """运行回测"""
        result = BacktestResult(
            league=self.league,
            season_range=f"{matches[0].date.strftime('%Y-%m-%d')} ~ {matches[-1].date.strftime('%Y-%m-%d')}",
            peak_balance=self.initial_bankroll,
        )
        self.balance_history = [self.initial_bankroll]

        print(f"\n  ▶ 开始回测: {self.league} ({len(matches)} 场)")
        print(f"    资金: {self.initial_bankroll:.0f} | 最少历史: {self.min_history} | 价值阈值: {self.value_threshold}")

        for i, match in enumerate(matches):
            match_id = f"{self.league}_{match.date.strftime('%Y%m%d')}_{match.home_team}_{match.away_team}"

            # 阶段1: 用已有数据预测 (如果历史足够)
            if i >= self.min_history and match.b365h > 1.0:
                bet = self._predict_and_bet(match, match_id)
                if bet:
                    self.bet_history.append(bet)
                    self.bankroll += bet.profit
                    result.bets.append(bet)
                    result.total_bets += 1
                    result.total_staked += bet.stake
                    if bet.won:
                        result.total_wins += 1
                        result.total_returned += bet.stake * bet.odds
                    else:
                        result.total_returned += 0

                    # 按类型统计
                    if bet.selection == "home_win":
                        result.home_bets += 1
                        if bet.won: result.home_wins += 1
                    elif bet.selection == "draw":
                        result.draw_bets += 1
                        if bet.won: result.draw_wins += 1
                    else:
                        result.away_bets += 1
                        if bet.won: result.away_wins += 1

            # 阶段2: 更新历史数据 (无论是否投注)
            self._update_history(match)
            result.total_matches += 1
            self.balance_history.append(self.bankroll)

            # 更新最大回撤
            if self.bankroll > result.peak_balance:
                result.peak_balance = self.bankroll
            dd = (result.peak_balance - self.bankroll) / result.peak_balance if result.peak_balance > 0 else 0
            if dd > result.max_drawdown:
                result.max_drawdown = dd

        return result

    def _predict_and_bet(self, match: RawMatch, match_id: str) -> Optional[BacktestBet]:
        """预测单场比赛并决定是否投注"""
        try:
            # 1. 获取 Elo
            elo_home = self.elo_engine.get_elo(self.league, match.home_team)
            elo_away = self.elo_engine.get_elo(self.league, match.away_team)
            elo_diff = elo_home - elo_away

            # 2. 从历史数据计算因子
            factors = self._compute_factors(match, elo_diff)

            # 3. 市场概率 (从赔率反推)
            if match.b365h <= 1.0 or match.b365d <= 1.0 or match.b365a <= 1.0:
                return None  # 没有有效赔率

            margin = 1.0/match.b365h + 1.0/match.b365d + 1.0/match.b365a
            market_home = (1.0/match.b365h) / margin
            market_draw = (1.0/match.b365d) / margin
            market_away = (1.0/match.b365a) / margin

            # 4. 计算因子 delta
            factor_engine = FactorComputationEngine(self.league)
            factor_deltas = factor_engine.compute_all(
                elo_diff=elo_diff,
                xi_rating=factors.get('xi_rating', 6.0),
                recent_results=factors.get('recent_results', [1.5]*5),
                h2h_results=factors.get('h2h_results', [0]*5),
                matches_7d=factors.get('matches_7d', 1),
                rank_diff=factors.get('rank_diff', 0),
                goal_diff=factors.get('goal_diff', 0.0),
                xg_diff=factors.get('xg_diff', 0.0),
                player_form=factors.get('player_form', 5.0),
                style_matchup_score=factors.get('style_matchup', 0.5),
                time_decay_factor=factors.get('time_decay', 1.0),
                market_sentiment=factors.get('market_sentiment', 0.0),
                weather=factors.get('weather', 0.0),
                derby_boost=factors.get('derby_boost', 0.0),
                market_probs={"home": market_home, "draw": market_draw, "away": market_away},
                ref_yellow_rate=factors.get('ref_yellow_rate', 0.0),
                fatigue_penalty=factors.get('fatigue_penalty', 0.0),
                rotation_risk=factors.get('rotation_risk', 0.0),
                streak_momentum=factors.get('streak_momentum', 0.0),
                odds_std=factors.get('odds_std', 0.05),
                poisson_correction=factors.get('poisson_correction', 0.0),
                handicap_depth=factors.get('handicap_depth', 0.0),
                totals_trend=factors.get('totals_trend', 0.0),
                value_signal=factors.get('value_signal', 0.0),
                contrarian_signal=factors.get('contrarian_signal', 0.0),
                market_efficiency=factors.get('market_efficiency', 0.0),
                motivation_boost=factors.get('motivation_boost', 0.0),
                # v5.11: 平局因子
                draw_tactical_matchup=factors.get('draw_tactical_matchup', 0.0),
                draw_goal_expectancy=factors.get('draw_goal_expectancy', 0.0),
                draw_team_tendency=factors.get('draw_team_tendency', 0.0),
                # 比赛统计因子
                ht_momentum=factors.get('ht_momentum', 0.0),
                shot_eff_diff=factors.get('shot_eff_diff', 0.0),
                territorial_dominance=factors.get('territorial_dominance', 0.0),
                discipline_index=factors.get('discipline_index', 0.0),
                odds_drift=factors.get('odds_drift', 0.0),
                market_disagreement=factors.get('market_disagreement', 0.0),
                referee_home_bias=factors.get('referee_home_bias', 0.0),
                comeback_resilience=factors.get('comeback_resilience', 0.0),
                streak_momentum_enriched=factors.get('streak_momentum_enriched', 0.0),
                goal_volatility=factors.get('goal_volatility', 0.0),
                corner_dominance=factors.get('corner_dominance', 0.0),
                sot_rate_diff=factors.get('sot_rate_diff', 0.0),
                ah_odds_drift=factors.get('ah_odds_drift', 0.0),
                totals_odds_drift=factors.get('totals_odds_drift', 0.0),
            )

            # 5. 异质化分组信号上限
            factor_deltas = group_signal_cap(factor_deltas, cap_factor=1.5)

            # 6. 因子缩放
            if self.factor_scale != 1.0:
                factor_deltas = {
                    fid: {k: v * self.factor_scale for k, v in d.items()}
                    for fid, d in factor_deltas.items()
                }

            # 7. Logit 累加
            market_probs = {"home": market_home, "draw": market_draw, "away": market_away}
            logits = self.prob_engine.logit_accumulation(
                market_probs, factor_deltas, uniform_prior=False,
            )

            # 8. Sigmoid 归一化
            logit_probs = self.prob_engine.sigmoid_normalization(logits, temperature=1.0)

            # 9. 泊松桥接 (Dixon-Coles)
            poisson_probs, score_matrix = self.prob_engine.poisson_bridge(
                home_elo=elo_home, away_elo=elo_away,
                factor_deltas=factor_deltas,
            )

            # 10. 双域融合
            fusion_weight = self.params.fusion_weight
            fused_probs = self.prob_engine.dual_domain_fusion(
                logit_probs, poisson_probs, fusion_weight=fusion_weight,
            )

            # 11. 自适应裁剪 — 根据因子激活数量决定偏离幅度
            # 因子信号越强，越信任模型；信号越弱，越贴近市场
            active_count = len([f for f, d in factor_deltas.items()
                                if abs(d.get('home', 0)) > 0.0001 or
                                   abs(d.get('away', 0)) > 0.0001 or
                                   abs(d.get('draw', 0)) > 0.0001])
            # 裁剪幅度: 5pp (最少因子) → 10pp (最多因子)
            clip_range = 0.05 + min(0.05, active_count * 0.002)

            mkt = [market_home, market_draw, market_away]
            raw = [fused_probs.prob_home, fused_probs.prob_draw, fused_probs.prob_away]
            clipped = [
                max(mkt[i] - clip_range, min(mkt[i] + clip_range, raw[i]))
                for i in range(3)
            ]
            total = sum(clipped)
            if total > 0:
                clipped = [c/total for c in clipped]

            model_home, model_draw, model_away = clipped

            # 12. 价值计算
            odds = {"home": match.b365h, "draw": match.b365d, "away": match.b365a}
            model_probs = {"home": model_home, "draw": model_draw, "away": model_away}

            best_bet = None
            best_value = 0

            for direction in ["home", "draw", "away"]:
                model_p = model_probs[direction]
                implied_p = 1.0 / odds[direction]
                value = model_p - implied_p

                # 平局/客胜需要更高阈值 (方差大、命中率低)
                direction_threshold = self.value_threshold
                if direction == "draw":
                    direction_threshold = self.value_threshold * 2.5  # 平局 7.5%
                elif direction == "away":
                    direction_threshold = self.value_threshold * 1.5  # 客胜 4.5%

                if value > direction_threshold and value > best_value:
                    # Kelly 公式
                    b = odds[direction] - 1
                    p = model_p
                    q = 1 - p
                    f_kelly = (b * p - q) / b if b > 0 else 0
                    f_actual = f_kelly * self.kelly_fraction

                    stake = self.bankroll * f_actual
                    # 单注上限 5%
                    stake = min(stake, self.bankroll * 0.05)
                    # 最小投注
                    if stake < 10:
                        continue

                    # 置信度
                    active_factors = len([f for f, d in factor_deltas.items()
                                         if abs(d.get('home', 0)) > 0.0001 or
                                            abs(d.get('away', 0)) > 0.0001 or
                                            abs(d.get('draw', 0)) > 0.0001])
                    confidence = compute_confidence(
                        data_completeness=factors.get('data_completeness', 0.7),
                        factor_activation_rate=active_factors / 41.0,
                        dispersion_penalty=0.9,
                        match_phase=0.5,
                    )

                    if confidence < self.confidence_threshold:
                        continue

                    # 赔率区间过滤: 避免极端赔率
                    if odds[direction] < 1.50 or odds[direction] > 15.0:
                        continue

                    best_value = value
                    selection_map = {"home": "home_win", "draw": "draw", "away": "away_win"}
                    best_bet = BacktestBet(
                        match_id=match_id,
                        date=match.date,
                        league=self.league,
                        home=match.home_team,
                        away=match.away_team,
                        selection=selection_map[direction],
                        odds=odds[direction],
                        model_prob=model_p,
                        implied_prob=implied_p,
                        value=value,
                        stake=round(stake, 2),
                        actual_result=match.ftr,
                        won=False,
                        profit=0.0,
                        confidence=confidence,
                    )

            if best_bet:
                # 结算
                result_map = {"H": "home_win", "D": "draw", "A": "away_win"}
                actual = result_map.get(match.ftr, "")
                if best_bet.selection == actual:
                    best_bet.won = True
                    best_bet.profit = best_bet.stake * (best_bet.odds - 1)
                else:
                    best_bet.won = False
                    best_bet.profit = -best_bet.stake

                return best_bet

        except Exception as e:
            pass  # 静默降级

        return None

    def _compute_factors(self, match: RawMatch, elo_diff: float) -> Dict:
        """从历史数据计算因子值 — 完整版 (P0: 25个零值因子全部激活)"""
        home = match.home_team
        away = match.away_team
        hs = self.team_stats[home]
        as_ = self.team_stats[away]

        factors = {}

        # ── 基础因子 ──
        factors['recent_results'] = hs['recent_results'][-5:] if hs['recent_results'] else [1.5]*5
        avg_form = sum(factors['recent_results']) / max(len(factors['recent_results']), 1)
        factors['xi_rating'] = 5.0 + avg_form * 0.5
        factors['player_form'] = 5.0 + avg_form
        factors['time_decay'] = 1.0
        factors['matches_7d'] = 1

        # ── F7: 排名差 (从联赛积分表) ──
        ht = self.league_table[home]
        at = self.league_table[away]
        # 按积分排序计算排名
        sorted_teams = sorted(self.league_table.items(), key=lambda x: (-x[1]['pts'], -(x[1]['gf']-x[1]['ga'])))
        rank_map = {t: i+1 for i, (t, _) in enumerate(sorted_teams)}
        factors['rank_diff'] = rank_map.get(away, 10) - rank_map.get(home, 10)

        # ── 进球差 ──
        home_gd = (hs['goals_for'] - hs['goals_against']) / max(hs['matches'], 1)
        away_gd = (as_['goals_for'] - as_['goals_against']) / max(as_['matches'], 1)
        factors['goal_diff'] = home_gd - away_gd

        # ── F9: xG 代理 (射门+角球) ──
        if hs['matches'] > 3:
            home_xg = (0.02*(hs['shots']-hs['shots_on_target']) + 0.30*hs['shots_on_target'] + 0.03*hs['corners']) / hs['matches']
        else: home_xg = 0
        if as_['matches'] > 3:
            away_xg = (0.02*(as_['shots']-as_['shots_on_target']) + 0.30*as_['shots_on_target'] + 0.03*as_['corners']) / as_['matches']
        else: away_xg = 0
        factors['xg_diff'] = home_xg - away_xg

        # ── F19: 风格匹配 (射正率差异 — 修正为有区分度) ──
        if hs['matches'] > 5 and as_['matches'] > 5:
            home_sot_rate = hs['shots_on_target'] / max(hs['shots'], 1)
            away_sot_rate = as_['shots_on_target'] / max(as_['shots'], 1)
            # 修正: 不是加0.5，而是直接用差异
            factors['style_matchup'] = home_sot_rate - away_sot_rate  # [-1, +1]
        else:
            factors['style_matchup'] = 0.0

        # ── F12: 天气 (无法从CSV获取) ──
        factors['weather'] = 0.0

        # ── F13: 裁判黄牌率 ──
        if match.referee and self.referee_stats[match.referee]['matches'] > 5:
            ref = self.referee_stats[match.referee]
            avg_yellows = ref['yellows'] / ref['matches']
            factors['ref_yellow_rate'] = (avg_yellows - 4.0) / 2.0  # 归一化: 4张=0, 6张=1
        else:
            factors['ref_yellow_rate'] = 0.0

        # ── F15: 教练更替 (无法从CSV获取) ──
        factors['coach_change_effect'] = 0.0

        # ── F16: 欧战疲劳 (简化: 检测近7天是否有多场比赛) ──
        if hs['match_dates'] and len(hs['match_dates']) >= 2:
            recent_dates = hs['match_dates'][-3:]
            if len(recent_dates) >= 2:
                days_span = (recent_dates[-1] - recent_dates[0]).days
                if days_span <= 4 and len(recent_dates) >= 2:
                    factors['fatigue_penalty'] = 0.5  # 4天内多场 = 疲劳
                elif days_span <= 7 and len(recent_dates) >= 3:
                    factors['fatigue_penalty'] = 0.3
                else:
                    factors['fatigue_penalty'] = 0.0
            else:
                factors['fatigue_penalty'] = 0.0
        else:
            factors['fatigue_penalty'] = 0.0

        # ── F18: 德比战 ──
        pair = frozenset({home, away})
        factors['derby_boost'] = 0.3 if pair in self.derby_pairs else 0.0

        # ── F20/F38: 连胜动量 ──
        if hs['recent_results']:
            streak = 0
            for r in reversed(hs['recent_results'][-5:]):
                if r == 3: streak += 1
                elif r == 0: streak -= 1
                else: break
            factors['streak_momentum'] = max(-1.0, min(1.0, streak * 0.2))
        else:
            factors['streak_momentum'] = 0.0

        # ── 赔率相关 (P1: 移除F10/F11的循环论证) ──
        # 不再使用赔率作为因子输入，避免循环论证
        factors['market_sentiment'] = 0.0
        factors['odds_std'] = 0.0
        factors['odds_drift'] = 0.0
        factors['market_efficiency'] = 0.0
        factors['value_signal'] = 0.0
        factors['contrarian_signal'] = 0.0
        factors['handicap_depth'] = abs(match.ahh) / 2.5 if match.ahh else 0.0
        factors['totals_trend'] = 0.0
        factors['poisson_correction'] = 0.0
        factors['market_disagreement'] = 0.0
        factors['ah_odds_drift'] = 0.0
        factors['totals_odds_drift'] = 0.0

        # ── F42: 半场动量 ──
        if hs['matches'] > 5:
            total_ht = hs['ht_leads'] + hs['ht_trails'] + hs['ht_draws']
            if total_ht > 0:
                factors['ht_momentum'] = (hs['ht_leads'] - hs['ht_trails']) / total_ht
            else:
                factors['ht_momentum'] = 0.0
        else:
            factors['ht_momentum'] = 0.0

        # ── F43: 射门效率差 ──
        if hs['matches'] > 5 and as_['matches'] > 5:
            home_eff = hs['shots_on_target'] / max(hs['shots'], 1)
            away_eff = as_['shots_on_target'] / max(as_['shots'], 1)
            factors['shot_eff_diff'] = home_eff - away_eff
        else:
            factors['shot_eff_diff'] = 0.0

        # ── F44: 领地优势 (射门比) ──
        if hs['matches'] > 5 and as_['matches'] > 5:
            home_shots_pg = hs['shots'] / hs['matches']
            away_shots_pg = as_['shots'] / as_['matches']
            total = home_shots_pg + away_shots_pg
            if total > 0:
                factors['territorial_dominance'] = (home_shots_pg - away_shots_pg) / total
            else:
                factors['territorial_dominance'] = 0.0
        else:
            factors['territorial_dominance'] = 0.0

        # ── F45: 纪律指数 (犯规/黄牌比) ──
        if hs['matches'] > 5 and as_['matches'] > 5:
            home_disc = hs['yellow_cards'] / max(hs['fouls'], 1)
            away_disc = as_['yellow_cards'] / max(as_['fouls'], 1)
            factors['discipline_index'] = away_disc - home_disc  # 正值=客队更粗暴
        else:
            factors['discipline_index'] = 0.0

        # ── F46: 裁判主场偏向 ──
        if match.referee and self.referee_stats[match.referee]['matches'] > 10:
            ref = self.referee_stats[match.referee]
            hw_rate = ref['home_wins'] / ref['matches']
            factors['referee_home_bias'] = hw_rate - 0.45  # 45%为基准
        else:
            factors['referee_home_bias'] = 0.0

        # ── F47: 逆转韧性 ──
        if hs['matches'] > 10 and hs['ht_trails'] > 0:
            factors['comeback_resilience'] = hs['comebacks'] / hs['ht_trails']
        else:
            factors['comeback_resilience'] = 0.0

        # ── F49: 进球波动性 ──
        if len(hs['goals_by_match']) > 5:
            import statistics
            factors['goal_volatility'] = statistics.stdev(hs['goals_by_match'][-15:]) / 3.0
        else:
            factors['goal_volatility'] = 0.0

        # ── F50: 角球优势 ──
        if hs['matches'] > 5 and as_['matches'] > 5:
            home_cpg = hs['corners'] / hs['matches']
            away_cpg = as_['corners'] / as_['matches']
            factors['corner_dominance'] = (home_cpg - away_cpg) / 10.0  # 归一化
        else:
            factors['corner_dominance'] = 0.0

        # ── F52: 射正率差 ──
        factors['sot_rate_diff'] = factors.get('shot_eff_diff', 0.0)

        # ── F33: 保级/争冠动力 ──
        if ht['played'] > 10:
            pts_per_game = ht['pts'] / ht['played']
            if pts_per_game < 1.0:  # 保级区
                factors['motivation_boost'] = 0.3
            elif pts_per_game > 2.2:  # 争冠
                factors['motivation_boost'] = 0.2
            else:
                factors['motivation_boost'] = 0.0
        else:
            factors['motivation_boost'] = 0.0

        # ── 轮换风险 ──
        factors['rotation_risk'] = factors.get('fatigue_penalty', 0.0) * 0.5

        # ── 连胜动量 enriched ──
        factors['streak_momentum_enriched'] = factors.get('streak_momentum', 0.0)

        # ── v5.11: 平局因子 ──
        # F56: 战术风格平局倾向
        if hs['matches'] > 5 and as_['matches'] > 5:
            home_activity = (hs['goals_for'] + hs['goals_against']) / hs['matches']
            away_activity = (as_['goals_for'] + as_['goals_against']) / as_['matches']
            league_avg = 2.7
            if home_activity < league_avg and away_activity < league_avg:
                factors['draw_tactical_matchup'] = min(1.0, (league_avg - min(home_activity, away_activity)) / league_avg)
            elif home_activity > league_avg and away_activity > league_avg:
                factors['draw_tactical_matchup'] = -min(1.0, (max(home_activity, away_activity) - league_avg) / league_avg)
            else:
                factors['draw_tactical_matchup'] = 0.1
        else:
            factors['draw_tactical_matchup'] = 0.0

        # F57: 进球预期平局信号
        if hs['matches'] > 5 and as_['matches'] > 5:
            avg_sot = (hs['shots_on_target'] + as_['shots_on_target']) / (hs['matches'] + as_['matches'])
            expected_goals = avg_sot / 3.0
            factors['draw_goal_expectancy'] = -(expected_goals - 2.7) / 2.7
        else:
            factors['draw_goal_expectancy'] = 0.0

        # F58: 球队平局历史倾向
        if hs['home_matches'] > 5 and as_['away_matches'] > 5:
            home_draw_rate = hs['home_draws'] / hs['home_matches']
            away_draw_rate = as_['away_draws'] / as_['away_matches']
            combined = (home_draw_rate * away_draw_rate) ** 0.5
            league_draw = 0.25
            factors['draw_team_tendency'] = max(-1.0, min(1.0, (combined - league_draw) / league_draw))
        else:
            factors['draw_team_tendency'] = 0.0

        # data_completeness
        non_zero = sum(1 for v in factors.values() if isinstance(v, (int, float)) and abs(v) > 0.001)
        factors['data_completeness'] = min(1.0, non_zero / 25.0)

        return factors
        factors['data_completeness'] = min(1.0, non_zero / 20.0)

        return factors

    def _update_history(self, match: RawMatch):
        """更新历史统计数据 — 完整版"""
        home = match.home_team
        away = match.away_team

        # 半场结果
        htr = match.htr.upper() if match.htr else ''

        # 更新球队统计
        for team, goals_for, goals_against, is_home in [
            (home, match.fthg, match.ftag, True),
            (away, match.ftag, match.fthg, False),
        ]:
            s = self.team_stats[team]
            s['matches'] += 1
            s['goals_for'] += goals_for
            s['goals_against'] += goals_against
            s['shots'] += match.hs if is_home else match.as_
            s['shots_on_target'] += match.hst if is_home else match.ast
            s['corners'] += match.hc if is_home else match.ac
            s['fouls'] += match.hf if is_home else match.af
            s['yellow_cards'] += match.hy if is_home else match.ay
            s['red_cards'] += match.hr if is_home else match.ar
            s['goals_by_match'].append(goals_for + goals_against)
            s['goals_by_match'] = s['goals_by_match'][-30:]
            s['match_dates'].append(match.date)
            s['match_dates'] = s['match_dates'][-30:]

            if goals_for > goals_against:
                s['wins'] += 1
                s['recent_results'].append(3)
            elif goals_for == goals_against:
                s['draws'] += 1
                s['recent_results'].append(1)
            else:
                s['losses'] += 1
                s['recent_results'].append(0)

            s['recent_results'] = s['recent_results'][-20:]

            # 主客场平局统计
            if is_home:
                s['home_matches'] += 1
                if goals_for == goals_against:
                    s['home_draws'] += 1
            else:
                s['away_matches'] += 1
                if goals_for == goals_against:
                    s['away_draws'] += 1

            # 半场动量 & 逆转韧性
            if htr == 'H':
                if is_home:
                    s['ht_leads'] += 1
                else:
                    s['ht_trails'] += 1
            elif htr == 'A':
                if is_home:
                    s['ht_trails'] += 1
                else:
                    s['ht_leads'] += 1
            else:
                s['ht_draws'] += 1

            # 逆转统计
            if is_home:
                if htr == 'A' and goals_for >= goals_against:  # 半场落后，最终不败
                    s['comebacks'] += 1
                if htr == 'H' and goals_for <= goals_against:  # 半场领先，最终不胜
                    s['collapses'] += 1
            else:
                if htr == 'H' and goals_against >= goals_for:
                    s['comebacks'] += 1
                if htr == 'A' and goals_against <= goals_for:
                    s['collapses'] += 1

        # 更新 H2H
        h2h_key = f"{home}_vs_{away}"
        self.h2h[h2h_key].append(match.ftr)
        self.h2h[h2h_key] = self.h2h[h2h_key][-10:]

        # 更新联赛积分表
        ht = self.league_table[home]
        at = self.league_table[away]
        ht['played'] += 1
        at['played'] += 1
        ht['gf'] += match.fthg
        ht['ga'] += match.ftag
        at['gf'] += match.ftag
        at['ga'] += match.fthg
        if match.ftr == 'H':
            ht['pts'] += 3
        elif match.ftr == 'D':
            ht['pts'] += 1
            at['pts'] += 1
        else:
            at['pts'] += 3

        # 更新裁判统计
        if match.referee:
            ref = self.referee_stats[match.referee]
            ref['matches'] += 1
            ref['yellows'] += match.hy + match.ay
            ref['reds'] += match.hr + match.ar
            if match.ftr == 'H':
                ref['home_wins'] += 1

        # 更新 Elo
        self.elo_engine._process_single_match({
            'league_id': self.league,
            'home_team': home,
            'away_team': away,
            'fthg': match.fthg,
            'ftag': match.ftag,
            'ftr': match.ftr,
            'date': match.date,
        })


# ================================================================
# 报告生成
# ================================================================

def print_result(result: BacktestResult, league_name: str):
    """打印回测结果"""
    print(f"\n{'='*70}")
    print(f"  {league_name} 回测结果")
    print(f"{'='*70}")
    print(f"  时间范围:     {result.season_range}")
    print(f"  总比赛:       {result.total_matches}")
    print(f"  总投注:       {result.total_bets}")
    print(f"  总胜场:       {result.total_wins}")
    print(f"  胜率:         {result.win_rate:.1%}")
    print(f"  总投注额:     {result.total_staked:.0f}")
    print(f"  总回报:       {result.total_returned:.0f}")
    print(f"  利润:         {result.profit:+.0f}")
    print(f"  ROI:          {result.roi:+.1%}")
    print(f"  最大回撤:     {result.max_drawdown:.1%}")

    if result.total_bets > 0:
        print(f"\n  ── 按投注类型 ──")
        if result.home_bets > 0:
            print(f"  主胜: {result.home_bets} 注, 胜率 {result.home_wins/result.home_bets:.1%}")
        if result.draw_bets > 0:
            print(f"  平局: {result.draw_bets} 注, 胜率 {result.draw_wins/result.draw_bets:.1%}")
        if result.away_bets > 0:
            print(f"  客胜: {result.away_bets} 注, 胜率 {result.away_wins/result.away_bets:.1%}")

        # 最佳和最差投注
        if result.bets:
            best = max(result.bets, key=lambda b: b.profit)
            worst = min(result.bets, key=lambda b: b.profit)
            print(f"\n  ── 最佳投注 ──")
            print(f"  {best.home} vs {best.away}: {best.selection} @{best.odds:.2f}, "
                  f"stake={best.stake:.0f}, profit={best.profit:+.0f}")
            print(f"  ── 最差投注 ──")
            print(f"  {worst.home} vs {worst.away}: {worst.selection} @{worst.odds:.2f}, "
                  f"stake={worst.stake:.0f}, profit={worst.profit:+.0f}")


def run_multi_league_backtest(
    leagues: List[str],
    seasons: List[str],
    initial_bankroll: float = 10000.0,
):
    """多联联回测"""
    print(f"\n{'#'*70}")
    print(f"  GTO-GameFlow v5.11 真实CSV数据回测")
    print(f"  联赛: {', '.join(leagues)}")
    print(f"  赛季: {', '.join(seasons)}")
    print(f"{'#'*70}")

    # 加载数据
    print(f"\n📂 加载比赛数据...")
    all_matches = load_all_matches(leagues, seasons)

    total_matches = sum(len(m) for m in all_matches.values())
    print(f"\n  共加载 {total_matches} 场比赛")

    if total_matches == 0:
        print("  ❌ 没有找到任何比赛数据！")
        return

    # 逐联回测
    results: Dict[str, BacktestResult] = {}
    total_profit = 0.0
    total_staked = 0.0
    total_bets = 0
    total_wins = 0

    for league in leagues:
        matches = all_matches.get(league, [])
        if not matches:
            continue

        engine = CSVBacktestEngine(
            league=league,
            initial_bankroll=initial_bankroll,
            min_history=30,
            value_threshold=0.03,
            confidence_threshold=0.55,
            kelly_fraction=0.25,
            factor_scale=None,  # 联赛自动选择
        )
        result = engine.run(matches)
        results[league] = result

        league_name = {
            "premier_league": "英超", "la_liga": "西甲",
            "bundesliga": "德甲", "serie_a": "意甲", "ligue_1": "法甲",
        }.get(league, league)

        print_result(result, league_name)

        total_profit += result.profit
        total_staked += result.total_staked
        total_bets += result.total_bets
        total_wins += result.total_wins

    # 汇总
    print(f"\n{'#'*70}")
    print(f"  全联回测汇总")
    print(f"{'#'*70}")
    print(f"  总比赛:       {sum(r.total_matches for r in results.values())}")
    print(f"  总投注:       {total_bets}")
    print(f"  总胜场:       {total_wins}")
    print(f"  总胜率:       {total_wins/total_bets:.1%}" if total_bets > 0 else "  总胜率: N/A")
    print(f"  总投注额:     {total_staked:.0f}")
    print(f"  总利润:       {total_profit:+.0f}")
    print(f"  总ROI:        {total_profit/total_staked:+.1%}" if total_staked > 0 else "  总ROI: N/A")

    print(f"\n  ── 分联赛 ──")
    for league, result in results.items():
        league_name = {
            "premier_league": "英超", "la_liga": "西甲",
            "bundesliga": "德甲", "serie_a": "意甲", "ligue_1": "法甲",
        }.get(league, league)
        print(f"  {league_name:8s}  投注={result.total_bets:3d}  胜率={result.win_rate:.1%}  "
              f"ROI={result.roi:+.1%}  回撤={result.max_drawdown:.1%}")


# ================================================================
# 主入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTO CSV Backtest")
    parser.add_argument("--leagues", nargs="+", default=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"])
    parser.add_argument("--seasons", nargs="+", default=["2021-22", "2022-23", "2023-24"])
    parser.add_argument("--bankroll", type=float, default=10000.0)
    args = parser.parse_args()

    run_multi_league_backtest(
        leagues=args.leagues,
        seasons=args.seasons,
        initial_bankroll=args.bankroll,
    )
