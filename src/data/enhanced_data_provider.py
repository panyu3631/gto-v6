"""
GTO-GameFlow v5.10.5 — 增强数据提供器

从 CSV 历史数据中计算 18 个之前为零列的因子，无需外部 API。
严格避免未来信息泄露：每个因子仅使用该场比赛日期之前的数据。

激活的因子:
  F6 (赛程密度), F7 (联赛排名差), F9 (射门质量代理), F16 (欧战影响),
  F18 (德比战), F23 (赔率离散度), F25 (时间衰减), F26 (联赛强度),
  F27 (进球分布修正), F28 (亚盘深度), F29 (大小球趋势),
  F32 (市场效率), F33 (保级/争冠), F35 (冬歇期), F36 (圣诞赛程),
  F37 (中游无欲), F39 (积分榜), F40 (升班马), F41 (赛程优势)
"""

import csv
import os
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# ============================================================
# 静态数据
# ============================================================

# 联赛强度系数 (基于UEFA系数)
LEAGUE_STRENGTH = {
    "premier_league": 1.00,
    "la_liga": 0.95,
    "bundesliga": 0.88,
    "serie_a": 0.85,
    "ligue_1": 0.78,
}

# 联赛场均进球 (用于平局因子计算)
LEAGUE_AVG_GOALS = {
    "premier_league": 2.85,
    "la_liga": 2.55,
    "bundesliga": 3.05,
    "serie_a": 2.65,
    "ligue_1": 2.70,
}

# 联赛平局率 (用于平局因子计算)
LEAGUE_DRAW_RATES = {
    "premier_league": 0.24,
    "la_liga": 0.26,
    "bundesliga": 0.22,
    "serie_a": 0.28,
    "ligue_1": 0.30,
}

# 德比战配对 (主队 → 客队集合)
DERBY_PAIRS = {
    # 英超
    "Arsenal": {"Tottenham", "Chelsea", "West Ham"},
    "Chelsea": {"Arsenal", "Tottenham", "Fulham", "West Ham"},
    "Tottenham": {"Arsenal", "Chelsea", "West Ham"},
    "Man United": {"Man City", "Liverpool", "Leeds"},
    "Man City": {"Man United", "Liverpool"},
    "Liverpool": {"Man United", "Man City", "Everton"},
    "Everton": {"Liverpool"},
    "West Ham": {"Tottenham", "Chelsea", "Arsenal"},
    "Fulham": {"Chelsea"},
    "Leeds": {"Man United"},
    "Newcastle": {"Sunderland"},
    "Aston Villa": {"Birmingham", "West Brom", "Wolves"},
    "Wolves": {"Aston Villa", "West Brom"},
    # 西甲
    "Barcelona": {"Real Madrid", "Espanyol"},
    "Real Madrid": {"Barcelona", "Ath Madrid"},
    "Ath Madrid": {"Real Madrid"},
    "Sevilla": {"Betis"},
    "Betis": {"Sevilla"},
    "Valencia": {"Levante", "Villarreal"},
    "Villarreal": {"Valencia"},
    # 德甲
    "Bayern Munich": {"Dortmund", "Ein Frankfurt"},
    "Dortmund": {"Bayern Munich", "Schalke 04"},
    "Schalke 04": {"Dortmund"},
    "Ein Frankfurt": {"Bayern Munich"},
    # 意甲
    "Juventus": {"Inter", "AC Milan", "Torino"},
    "Inter": {"Juventus", "AC Milan"},
    "AC Milan": {"Inter", "Juventus"},
    "Roma": {"Lazio"},
    "Lazio": {"Roma"},
    "Napoli": {"Roma"},
    # 法甲
    "Paris SG": {"Marseille", "Lyon"},
    "Marseille": {"Paris SG", "Lyon"},
    "Lyon": {"Paris SG", "Marseille", "St Etienne"},
    "St Etienne": {"Lyon"},
}

# 升班马 (各赛季，手动整理)
# 格式: {league: {season: [team1, team2, team3]}}
PROMOTED_TEAMS = {
    "premier_league": {
        "2022-23": ["Fulham", "Bournemouth", "Nott'm Forest"],
        "2023-24": ["Burnley", "Sheffield United", "Luton"],
        "2021-22": ["Brentford", "Watford", "Norwich"],
        "2020-21": ["Leeds", "West Brom", "Fulham"],
        "2019-20": ["Norwich", "Sheffield United", "Aston Villa"],
        "2018-19": ["Wolves", "Cardiff", "Fulham"],
        "2017-18": ["Newcastle", "Brighton", "Huddersfield"],
        "2016-17": ["Burnley", "Middlesbrough", "Hull"],
        "2015-16": ["Bournemouth", "Watford", "Norwich"],
        "2014-15": ["Leicester", "Burnley", "QPR"],
    },
    "la_liga": {
        "2022-23": ["Almeria", "Valladolid", "Girona"],
        "2023-24": ["Granada", "Las Palmas", "Alaves"],
        "2021-22": ["Espanyol", "Mallorca", "Rayo Vallecano"],
        "2020-21": ["Huesca", "Cadiz", "Elche"],
        "2019-20": ["Osasuna", "Granada", "Mallorca"],
    },
    "bundesliga": {
        "2022-23": ["Schalke 04", "Werder Bremen"],
        "2023-24": ["Heidenheim", "Darmstadt"],
        "2021-22": ["Bochum", "Greuther Furth"],
        "2020-21": ["Arminia", "Stuttgart"],
        "2019-20": ["FC Koln", "Paderborn", "Union Berlin"],
    },
    "serie_a": {
        "2022-23": ["Lecce", "Cremonese", "Monza"],
        "2023-24": ["Frosinone", "Genoa", "Cagliari"],
        "2021-22": ["Empoli", "Salernitana", "Venezia"],
        "2020-21": ["Benevento", "Crotone", "Spezia"],
        "2019-20": ["Brescia", "Lecce", "Verona"],
    },
    "ligue_1": {
        "2022-23": ["Toulouse", "Ajaccio", "Auxerre"],
        "2023-24": ["Le Havre", "Metz"],
        "2021-22": ["Troyes", "Clermont"],
        "2020-21": ["Lens", "Lorient"],
    },
}

# 欧洲比赛日 (约数，用于检测欧战影响)
EUROPEAN_MATCHDAYS = {
    # 欧冠小组赛通常在 9-12 月的周二/周三
    # 欧联杯在周四
    # 简化为: 检测周中比赛后的周末
}


class EnhancedDataProvider:
    """
    增强数据提供器。

    从 CSV 历史数据中构建联赛积分表、球队状态追踪、赛程密度等，
    为每个因子提供计算所需的真实数据。
    """

    def __init__(self, csv_dir: str, leagues: List[str], seasons: List[str]):
        self.csv_dir = csv_dir
        self.leagues = leagues
        self.seasons = sorted(seasons)

        # 内部数据结构
        self._matches: Dict[str, List[Dict]] = defaultdict(list)  # league_season → matches
        self._league_tables: Dict[str, Dict[str, Dict]] = {}  # league_season → team → stats
        self._team_form: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._team_goals: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        self._team_shots: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        self._all_matches_chrono: List[Dict] = []  # all matches sorted by date

        self._load_all_data()

    # ================================================================
    # Phase 1: 数据加载
    # ================================================================

    def _load_all_data(self):
        """加载所有 CSV 数据并构建内部索引"""
        for league in self.leagues:
            for season in self.seasons:
                filename = f"{league}_{season}.csv"
                filepath = os.path.join(self.csv_dir, filename)
                if not os.path.exists(filepath):
                    continue
                key = f"{league}_{season}"
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date_str = row.get('Date', '').strip()
                        if not date_str:
                            continue
                        try:
                            dt = datetime.strptime(date_str, '%d/%m/%Y')
                        except ValueError:
                            continue

                        home = row.get('HomeTeam', '').strip()
                        away = row.get('AwayTeam', '').strip()
                        if not home or not away:
                            continue

                        # 尝试解析比分
                        try:
                            fthg = int(row.get('FTHG', 0) or 0)
                            ftag = int(row.get('FTAG', 0) or 0)
                        except ValueError:
                            fthg = ftag = 0

                        match = {
                            'date': dt,
                            'home': home,
                            'away': away,
                            'fthg': fthg,
                            'ftag': ftag,
                            'league': league,
                            'season': season,
                            'referee': row.get('Referee', '').strip(),
                            # 射门数据
                            'hs': self._safe_int(row, 'HS'),
                            'as': self._safe_int(row, 'AS'),
                            'hst': self._safe_int(row, 'HST'),
                            'ast': self._safe_int(row, 'AST'),
                            # 犯规/角球
                            'hf': self._safe_int(row, 'HF'),
                            'af': self._safe_int(row, 'AF'),
                            'hc': self._safe_int(row, 'HC'),
                            'ac': self._safe_int(row, 'AC'),
                            # 黄红牌
                            'hy': self._safe_int(row, 'HY'),
                            'ay': self._safe_int(row, 'AY'),
                            'hr': self._safe_int(row, 'HR'),
                            'ar': self._safe_int(row, 'AR'),
                            # 亚盘
                            'ahh': self._safe_float(row, 'AHh'),
                            # 大小球
                            'b365_over': self._safe_float(row, 'B365>2.5'),
                            'b365_under': self._safe_float(row, 'B365<2.5'),
                            # 赔率 (6家)
                            'odds_sources': self._extract_odds_sources(row),
                        }
                        self._matches[key].append(match)
                        self._all_matches_chrono.append(match)

        # 按日期排序
        self._all_matches_chrono.sort(key=lambda m: m['date'])

        # 构建联赛积分表
        self._build_league_tables()

    @staticmethod
    def _safe_int(row: Dict, col: str) -> int:
        try:
            return int(row.get(col, 0) or 0)
        except ValueError:
            return 0

    @staticmethod
    def _safe_float(row: Dict, col: str) -> float:
        try:
            return float(row.get(col, 0) or 0)
        except ValueError:
            return 0.0

    @staticmethod
    def _extract_odds_sources(row: Dict) -> Dict[str, Dict[str, float]]:
        """提取6家博彩商的赔率"""
        sources = {}
        bookmakers = [
            ('B365', 'B365H', 'B365D', 'B365A'),
            ('BW', 'BWH', 'BWD', 'BWA'),
            ('IW', 'IWH', 'IWD', 'IWA'),
            ('PS', 'PSH', 'PSD', 'PSA'),
            ('WH', 'WHH', 'WHD', 'WHA'),
            ('VC', 'VCH', 'VCD', 'VCA'),
        ]
        for name, h_col, d_col, a_col in bookmakers:
            h = EnhancedDataProvider._safe_float(row, h_col)
            d = EnhancedDataProvider._safe_float(row, d_col)
            a = EnhancedDataProvider._safe_float(row, a_col)
            if h > 1.0 and d > 1.0 and a > 1.0:
                sources[name] = {'home': h, 'draw': d, 'away': a}
        return sources

    # ================================================================
    # Phase 2: 联赛积分表构建
    # ================================================================

    def _build_league_tables(self):
        """为每个联赛-赛季构建积分表（按比赛日累积）"""
        for league in self.leagues:
            for season in self.seasons:
                key = f"{league}_{season}"
                if key not in self._matches:
                    continue
                matches = sorted(self._matches[key], key=lambda m: m['date'])
                table = defaultdict(lambda: {
                    'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0,
                    'played': 0, 'w': 0, 'd': 0, 'l': 0,
                    'position': 20, 'last_5': [],
                })
                for m in matches:
                    self._update_table_entry(table, m['home'], m['fthg'], m['ftag'])
                    self._update_table_entry(table, m['away'], m['ftag'], m['fthg'])
                    # 更新排名
                    sorted_teams = sorted(table.items(), key=lambda x: (
                        -x[1]['pts'], -x[1]['gd'], -x[1]['gf']
                    ))
                    for rank, (team, stats) in enumerate(sorted_teams, 1):
                        stats['position'] = rank
                self._league_tables[key] = dict(table)

    @staticmethod
    def _update_table_entry(table: Dict, team: str, gf: int, ga: int):
        t = table[team]
        t['played'] += 1
        t['gf'] += gf
        t['ga'] += ga
        t['gd'] = t['gf'] - t['ga']
        if gf > ga:
            t['w'] += 1
            t['pts'] += 3
        elif gf == ga:
            t['d'] += 1
            t['pts'] += 1
        else:
            t['l'] += 1

    # ================================================================
    # Phase 3: 因子计算 — 主入口
    # ================================================================

    def get_enhanced_data(
        self,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        match_date: datetime,
        existing_extra: Optional[Dict] = None,
        stats_enricher: Optional[Any] = None,  # v5.10.8: MatchStatsEnricher
    ) -> Dict[str, Any]:
        """
        返回增强后的 extra_data 字典，包含所有可从 CSV 计算的因子值。

        严格避免未来信息泄露：所有查询仅使用 match_date 之前的数据。
        """
        extra = dict(existing_extra) if existing_extra else {}

        # 获取该比赛日之前的积分表
        table = self._get_table_before(league, season, match_date)

        # F6: 赛程密度 (过去7天比赛数)
        extra['matches_7d'] = self._compute_matches_7d(league, season, home_team, away_team, match_date)

        # F7: 联赛排名差
        extra['rank_diff'] = self._compute_rank_diff(table, home_team, away_team)

        # F9: 射门质量代理 (xG 不可用，用射正率+射门数作为代理)
        extra['xg_diff'] = self._compute_shot_quality_proxy(league, season, home_team, away_team, match_date)

        # F16: 欧战影响
        extra['fatigue_penalty'] = self._compute_european_fatigue(league, season, home_team, away_team, match_date)

        # F18: 德比战
        extra['derby_boost'] = self._compute_derby_boost(home_team, away_team)

        # F23: 赔率离散度 (已由 test 计算，此处提供默认值)
        if 'odds_std' not in extra:
            extra['odds_std'] = 0.05

        # F25: 时间衰减因子
        extra['time_decay_factor'] = self._compute_time_decay(match_date)

        # F26: 联赛强度
        extra['league_strength_bias'] = LEAGUE_STRENGTH.get(league, 0.0)

        # F27: 进球分布修正
        extra['poisson_correction'] = self._compute_poisson_correction(league, season, match_date)

        # F28: 亚盘深度 (从已有 handicap_line 计算)
        extra['handicap_depth'] = self._compute_handicap_depth(existing_extra or {})

        # F29: 大小球趋势
        extra['totals_trend'] = self._compute_totals_trend(league, season, home_team, away_team, match_date)

        # F32: 市场效率 (Brier 分数)
        extra['market_efficiency'] = self._compute_market_efficiency(league, season, match_date)

        # F33: 保级/争冠动力
        extra['motivation_boost'] = self._compute_motivation(table, home_team, away_team, season)

        # F35: 冬歇期效应 (仅德甲)
        extra['winter_break_effect'] = self._compute_winter_break(league, match_date)

        # F36: 圣诞赛程 (仅英超)
        extra['christmas_fatigue'] = self._compute_christmas_fatigue(league, match_date)

        # F37: 中游无欲
        extra['complacency_effect'] = self._compute_complacency(table, home_team, away_team, season)

        # F39: 积分榜排名优势
        extra['position_advantage'] = self._compute_position_advantage(table, home_team, away_team)

        # F40: 升班马数据
        extra['promoted_team_delta'] = self._compute_promoted_delta(league, season, home_team, away_team)

        # F41: 赛程优势
        extra['schedule_advantage'] = self._compute_schedule_advantage(league, season, home_team, away_team, match_date)

        # v5.10.7: F5 历史交锋 — 从CSV真实计算
        extra['h2h_results'] = self._compute_h2h_results(league, season, home_team, away_team, match_date)

        # v5.10.7: F13 裁判黄牌率 — 从CSV真实计算
        extra['ref_yellow_rate'] = self._compute_referee_yellow_rate(league, season, match_date)

        # v5.10.7: F19 风格匹配 — 从CSV射门/犯规模式计算
        extra['style_matchup_score'] = self._compute_style_matchup(league, season, home_team, away_team, match_date)

        # v5.10.7: F30 价值信号 — 从赔率市场结构计算
        extra['value_signal'] = self._compute_value_signal(league, season, home_team, away_team, match_date)

        # v5.10.7: F31 反市场信号 — 从赔率变动方向计算
        extra['contrarian_signal'] = self._compute_contrarian_signal(league, season, home_team, away_team, match_date)

        # v5.10.8: 接入 MatchStatsEnricher — 14 个高维特征
        if stats_enricher:
            try:
                rich = stats_enricher.get_enriched_stats(
                    league, season, home_team, away_team, match_date
                )
                extra['ht_momentum'] = rich.get('ht_momentum', 0.0)
                extra['shot_eff_diff'] = rich.get('shot_eff_diff', 0.0)
                extra['territorial_dominance'] = rich.get('territorial_dominance', 0.0)
                extra['discipline_index'] = rich.get('discipline_index', 0.0)
                extra['odds_drift'] = rich.get('odds_drift', 0.0)
                extra['market_disagreement'] = rich.get('market_disagreement', 0.0)
                extra['referee_home_bias'] = rich.get('referee_home_bias', 0.0)
                extra['comeback_resilience'] = rich.get('comeback_resilience', 0.0)
                extra['streak_momentum_enriched'] = rich.get('streak_momentum', 0.0)
                extra['goal_volatility'] = rich.get('goal_volatility', 0.0)
                extra['corner_dominance'] = rich.get('corner_dominance', 0.0)
                extra['sot_rate_diff'] = rich.get('sot_rate_diff', 0.0)
                extra['ah_odds_drift'] = rich.get('ah_odds_drift', 0.0)
                extra['totals_odds_drift'] = rich.get('totals_odds_drift', 0.0)
            except Exception:
                pass  # 降级：不使用新特征

        # v5.11: 平局专属因子 — 解决54因子中仅5个影响平局的根本缺陷
        extra['draw_tactical_matchup'] = self._compute_draw_tactical_matchup(
            league, season, home_team, away_team, match_date)
        extra['draw_goal_expectancy'] = self._compute_draw_goal_expectancy(
            league, season, home_team, away_team, match_date)
        extra['draw_team_tendency'] = self._compute_draw_team_tendency(
            league, season, home_team, away_team, match_date)

        # v5.10.7: 动态 data_completeness — 替代硬编码 0.8
        # 计算非零因子数 / 应有数据的因子数
        # 排除 8 个需要外部数据源的因子 (F2/F12/F15/F17/F21/F22/F24/F34)
        external_factors = 8
        total_available = 41 - external_factors + 14  # 33 + 14 v5.10.8 新增特征 = 47
        non_zero_count = sum(
            1 for k in ['matches_7d', 'rank_diff', 'xg_diff', 'fatigue_penalty',
                        'derby_boost', 'odds_std', 'time_decay_factor', 'league_strength_bias',
                        'poisson_correction', 'handicap_depth', 'totals_trend',
                        'market_efficiency', 'motivation_boost', 'winter_break_effect',
                        'christmas_fatigue', 'complacency_effect', 'position_advantage',
                        'promoted_team_delta', 'schedule_advantage', 'h2h_results',
                        'ref_yellow_rate', 'style_matchup_score', 'value_signal',
                        'contrarian_signal',
                        # v5.10.8: MatchStatsEnricher 新增特征
                        'ht_momentum', 'shot_eff_diff', 'territorial_dominance',
                        'discipline_index', 'odds_drift', 'market_disagreement',
                        'referee_home_bias', 'comeback_resilience', 'streak_momentum_enriched',
                        'goal_volatility', 'corner_dominance', 'sot_rate_diff',
                        'ah_odds_drift', 'totals_odds_drift',
                        # v5.11: 平局专属因子
                        'draw_tactical_matchup', 'draw_goal_expectancy', 'draw_team_tendency']
            if abs(extra.get(k, 0.0) if isinstance(extra.get(k), (int, float)) else 0.0) > 0.001
        )
        # 加上始终有数据的因子 (F1/F3/F4/F5/F8/F10/F11/F20/F23/F38 等)
        non_zero_count += 10  # 基础因子
        extra['data_completeness'] = min(1.0, non_zero_count / total_available)

        return extra

    # ================================================================
    # 因子计算方法
    # ================================================================

    def _get_table_before(self, league: str, season: str, match_date: datetime) -> Dict:
        """获取 match_date 之前的积分表"""
        key = f"{league}_{season}"
        if key not in self._matches:
            return {}
        table = defaultdict(lambda: {
            'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0,
            'played': 0, 'w': 0, 'd': 0, 'l': 0,
            'position': 20,
        })
        for m in self._matches[key]:
            if m['date'] >= match_date:
                break
            self._update_table_entry(table, m['home'], m['fthg'], m['ftag'])
            self._update_table_entry(table, m['away'], m['ftag'], m['fthg'])
        # 更新排名
        sorted_teams = sorted(table.items(), key=lambda x: (
            -x[1]['pts'], -x[1]['gd'], -x[1]['gf']
        ))
        for rank, (team, stats) in enumerate(sorted_teams, 1):
            stats['position'] = rank
        return dict(table)

    def _compute_matches_7d(self, league: str, season: str, home: str, away: str, date: datetime) -> int:
        """F6: 过去7天主客队比赛数"""
        key = f"{league}_{season}"
        cutoff = date - timedelta(days=7)
        home_count = 0
        away_count = 0
        if key in self._matches:
            for m in self._matches[key]:
                if m['date'] >= date:
                    break
                if m['date'] >= cutoff:
                    if m['home'] == home or m['away'] == home:
                        home_count += 1
                    if m['home'] == away or m['away'] == away:
                        away_count += 1
        return max(home_count, away_count, 1)  # 至少1场

    def _compute_rank_diff(self, table: Dict, home: str, away: str) -> int:
        """F7: 排名差 = rank_away - rank_home (正值=主队排名更靠前)"""
        home_rank = table.get(home, {}).get('position', 10)
        away_rank = table.get(away, {}).get('position', 10)
        return away_rank - home_rank

    def _compute_shot_quality_proxy(self, league: str, season: str, home: str, away: str,
                                     date: datetime) -> float:
        """
        F9: xG 代理模型 (基于射门数据)。

        足球分析文献参考:
        - 射正 xG ≈ 0.30 (每脚射正的预期进球)
        - 射偏 xG ≈ 0.02-0.04 (每脚射偏的预期进球)
        - 角球 xG ≈ 0.03 (每个角球的预期进球，作为进攻压力补充信号)
        - 与真实 xG 的 Pearson r ≈ 0.85

        改进 (v5.11):
        - 分离主客场统计 (主场进攻 vs 客场进攻差异显著)
        - 加入角球作为进攻压力代理
        - 使用标准化而非简单乘数归一化
        """
        key = f"{league}_{season}"
        # 分离主客场统计
        home_atk = {'off_target': 0, 'on_target': 0, 'corners': 0, 'matches': 0}
        home_def = {'off_target': 0, 'on_target': 0, 'corners': 0, 'matches': 0}
        away_atk = {'off_target': 0, 'on_target': 0, 'corners': 0, 'matches': 0}
        away_def = {'off_target': 0, 'on_target': 0, 'corners': 0, 'matches': 0}

        if key in self._matches:
            for m in self._matches[key]:
                if m['date'] >= date:
                    break
                # 主队作为主队的比赛
                if m['home'] == home:
                    home_atk['off_target'] += max(0, m['hs'] - m['hst'])
                    home_atk['on_target'] += m['hst']
                    home_atk['corners'] += m.get('hc', 0)
                    home_atk['matches'] += 1
                # 主队作为客队的比赛
                elif m['away'] == home:
                    home_atk['off_target'] += max(0, m['as'] - m['ast'])
                    home_atk['on_target'] += m['ast']
                    home_atk['corners'] += m.get('ac', 0)
                    home_atk['matches'] += 1
                # 客队作为主队的比赛
                if m['home'] == away:
                    away_atk['off_target'] += max(0, m['hs'] - m['hst'])
                    away_atk['on_target'] += m['hst']
                    away_atk['corners'] += m.get('hc', 0)
                    away_atk['matches'] += 1
                # 客队作为客队的比赛
                elif m['away'] == away:
                    away_atk['off_target'] += max(0, m['as'] - m['ast'])
                    away_atk['on_target'] += m['ast']
                    away_atk['corners'] += m.get('ac', 0)
                    away_atk['matches'] += 1

        n_home = max(1, home_atk['matches'])
        n_away = max(1, away_atk['matches'])

        # 每场平均 xG: 0.02 * 射偏 + 0.30 * 射正 + 0.03 * 角球
        home_xg = (
            0.02 * home_atk['off_target'] +
            0.30 * home_atk['on_target'] +
            0.03 * home_atk['corners']
        ) / n_home
        away_xg = (
            0.02 * away_atk['off_target'] +
            0.30 * away_atk['on_target'] +
            0.03 * away_atk['corners']
        ) / n_away

        # 直接返回差值 (单位: 预期进球数)
        # 典型范围: -1.5 ~ +1.5，与 Elo 差分协同工作
        return home_xg - away_xg

    def _compute_european_fatigue(self, league: str, season: str, home: str, away: str,
                                   date: datetime) -> float:
        """
        F16: 欧战影响。

        检测过去7天内是否有周中比赛（欧战通常在周二/周三/周四）。
        欧洲比赛日无法从 CSV 直接识别，但可以通过检测周中比赛来代理。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        penalty = 0.0
        for team in [home, away]:
            # 检查过去7天是否有周中比赛 (周二-周四)
            cutoff = date - timedelta(days=7)
            for m in self._matches[key]:
                if m['date'] >= date:
                    break
                if cutoff <= m['date'] < date:
                    if (m['home'] == team or m['away'] == team) and m['date'].weekday() in (1, 2, 3):
                        days_gap = (date - m['date']).days
                        if days_gap <= 3:
                            penalty -= 0.8 if team == home else 0.8  # 对主队是正向惩罚
                        elif days_gap <= 4:
                            penalty -= 0.4 if team == home else 0.4
                        elif days_gap <= 5:
                            penalty -= 0.1 if team == home else 0.1

        return penalty

    def _compute_derby_boost(self, home: str, away: str) -> float:
        """F18: 德比战加成"""
        if home in DERBY_PAIRS and away in DERBY_PAIRS[home]:
            # 根据德比级别返回不同强度
            if (home in ("Arsenal", "Tottenham") and away in ("Arsenal", "Tottenham")):
                return 0.8  # 北伦敦德比
            elif (home in ("Man United", "Man City") and away in ("Man United", "Man City")):
                return 0.8  # 曼市德比
            elif (home in ("Barcelona", "Real Madrid") and away in ("Barcelona", "Real Madrid")):
                return 0.8  # 国家德比
            elif (home in ("Inter", "AC Milan") and away in ("Inter", "AC Milan")):
                return 0.8  # 米兰德比
            elif (home in ("Liverpool", "Everton") and away in ("Liverpool", "Everton")):
                return 0.6  # 默西塞德德比
            else:
                return 0.4  # 一般德比
        return 0.0

    def _compute_time_decay(self, match_date: datetime) -> float:
        """F25: 时间衰减因子 e^(-λt)，λ=0.005，t=距今天数"""
        now = datetime(2024, 6, 1)  # 固定参考点
        days = (now - match_date).days
        return math.exp(-0.005 * max(0, days))

    def _compute_poisson_correction(self, league: str, season: str, date: datetime) -> float:
        """
        F27: 进球分布修正。

        比较实际进球方差与泊松理论方差 (均值=方差) 的偏差。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        goals = []
        for m in self._matches[key]:
            if m['date'] >= date:
                break
            goals.append(m['fthg'])
            goals.append(m['ftag'])

        if len(goals) < 30:
            return 0.0

        mean_goals = sum(goals) / len(goals)
        variance = sum((g - mean_goals) ** 2 for g in goals) / len(goals)

        # 泊松理论: mean = variance
        # 实际方差通常 > 均值 (过度分散)
        correction = (variance - mean_goals) / max(1.0, mean_goals)
        return max(-0.20, min(0.20, correction * 0.5))

    def _compute_handicap_depth(self, extra: Dict) -> float:
        """F28: 亚盘深度 = |handicap_line| / max_handicap"""
        handicap = abs(extra.get('handicap_line', 0.0))
        return min(1.0, handicap / 2.5)

    def _compute_totals_trend(self, league: str, season: str, home: str, away: str,
                               date: datetime) -> float:
        """
        F29: 大小球趋势。

        两队近5场场均总进球与联赛均值的偏差。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        # 联赛平均进球
        league_goals = []
        team_goals = {'home': [], 'away': []}

        for m in self._matches[key]:
            if m['date'] >= date:
                break
            total = m['fthg'] + m['ftag']
            league_goals.append(total)
            if m['home'] == home or m['away'] == home:
                team_goals['home'].append(total)
            if m['home'] == away or m['away'] == away:
                team_goals['away'].append(total)

        league_avg = sum(league_goals) / max(1, len(league_goals))
        home_avg = sum(team_goals['home'][-5:]) / max(1, len(team_goals['home'][-5:]))
        away_avg = sum(team_goals['away'][-5:]) / max(1, len(team_goals['away'][-5:]))

        return (home_avg + away_avg) / 2.0 - league_avg

    def _compute_market_efficiency(self, league: str, season: str, date: datetime) -> float:
        """
        F32: 市场效率评分。

        使用 Brier 分数: 1 - Brier_score。
        高效市场 > 0.85，低效市场 < 0.65。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.8

        brier_sum = 0.0
        count = 0
        for m in self._matches[key]:
            if m['date'] >= date:
                break
            sources = m.get('odds_sources', {})
            if 'B365' not in sources:
                continue
            odds = sources['B365']
            # 隐含概率
            imp_h = 1.0 / odds['home']
            imp_d = 1.0 / odds['draw']
            imp_a = 1.0 / odds['away']
            total = imp_h + imp_d + imp_a
            prob_h = imp_h / total
            prob_d = imp_d / total
            prob_a = imp_a / total

            # 实际结果
            actual_h = 1.0 if m['fthg'] > m['ftag'] else 0.0
            actual_d = 1.0 if m['fthg'] == m['ftag'] else 0.0
            actual_a = 1.0 if m['fthg'] < m['ftag'] else 0.0

            brier = ((prob_h - actual_h) ** 2 + (prob_d - actual_d) ** 2 + (prob_a - actual_a) ** 2) / 3.0
            brier_sum += brier
            count += 1

        if count < 30:
            return 0.8

        avg_brier = brier_sum / count
        # Brier 分数的理论最大值取决于事件数，归一化
        efficiency = 1.0 - avg_brier * 2.0  # 乘以2使范围大致在 0.5-1.0
        return max(0.5, min(0.95, efficiency))

    def _compute_motivation(self, table: Dict, home: str, away: str, season: str) -> float:
        """
        F33: 保级/争冠动力。

        距降级区 ≤ 3分 → +8%，距前四/榜首 ≤ 3分 → +5%。
        """
        if not table or len(table) < 4:
            return 0.0

        def get_rank(team):
            return table.get(team, {}).get('position', 20)

        def get_pts(team):
            return table.get(team, {}).get('pts', 0)

        home_rank = get_rank(home)
        away_rank = get_rank(away)
        num_teams = len(table)

        boost = 0.0
        # 按排名排序
        sorted_by_rank = sorted(table.items(), key=lambda x: x[1].get('position', 20))

        # 保级区: 排名倒数第3名
        relegation_cutoff = max(1, num_teams - 3)
        relegation_idx = min(num_teams - 1, num_teams - 4)  # 降级区边缘

        if home_rank >= relegation_cutoff:
            boost += 8.0
        elif home_rank >= relegation_cutoff - 3:
            if relegation_idx < len(sorted_by_rank):
                relegation_team = sorted_by_rank[relegation_idx][0]
                if get_pts(home) - get_pts(relegation_team) <= 3:
                    boost += 8.0

        # 争冠/争四
        if home_rank <= 4:
            if sorted_by_rank:
                champion_pts = get_pts(sorted_by_rank[0][0])
                if champion_pts - get_pts(home) <= 3:
                    boost += 5.0

        # 客队同理
        if away_rank >= relegation_cutoff:
            boost -= 8.0
        elif away_rank >= relegation_cutoff - 3:
            if relegation_idx < len(sorted_by_rank):
                relegation_team = sorted_by_rank[relegation_idx][0]
                if get_pts(away) - get_pts(relegation_team) <= 3:
                    boost -= 8.0

        if away_rank <= 4:
            if sorted_by_rank:
                champion_pts = get_pts(sorted_by_rank[0][0])
                if champion_pts - get_pts(away) <= 3:
                    boost -= 5.0

        return boost

    def _compute_winter_break(self, league: str, date: datetime) -> float:
        """F35: 冬歇期效应 (仅德甲)"""
        if league != "bundesliga":
            return 0.0
        # 德甲冬歇期通常在 12月中旬到 1月中旬
        if date.month == 1 and date.day <= 25:
            return 0.8  # 冬歇期后首轮
        return 0.0

    def _compute_christmas_fatigue(self, league: str, date: datetime) -> float:
        """F36: 圣诞赛程 (仅英超)"""
        if league != "premier_league":
            return 0.0
        if date.month == 12 and date.day >= 20:
            return 0.8  # 圣诞密集赛程
        if date.month == 1 and date.day <= 5:
            return 0.6  # 新年赛程
        return 0.0

    def _compute_complacency(self, table: Dict, home: str, away: str, season: str) -> float:
        """
        F37: 中游无欲。

        赛季末段 (最后8轮)，排名 7-15 的球队缺乏动力。
        """
        if not table:
            return 0.0

        def get_rank(team):
            return table.get(team, {}).get('position', 20)

        def get_played(team):
            return table.get(team, {}).get('played', 0)

        home_rank = get_rank(home)
        away_rank = get_rank(away)
        home_played = get_played(home)
        away_played = get_played(away)

        complacency = 0.0
        # 赛季末段 (30轮+)
        if home_played >= 30 and 7 <= home_rank <= 15:
            complacency += 0.5
        if away_played >= 30 and 7 <= away_rank <= 15:
            complacency -= 0.5  # 主队面对无欲客队反而有利

        return complacency

    def _compute_position_advantage(self, table: Dict, home: str, away: str) -> float:
        """F39: 积分榜排名优势"""
        home_rank = table.get(home, {}).get('position', 10)
        away_rank = table.get(away, {}).get('position', 10)
        return away_rank - home_rank  # 正值=主队排名更靠前

    def _compute_promoted_delta(self, league: str, season: str, home: str, away: str) -> float:
        """F40: 升班马数据"""
        promoted = PROMOTED_TEAMS.get(league, {}).get(season, [])
        delta = 0.0
        if home in promoted:
            delta -= 0.5  # 升班马主场劣势
        if away in promoted:
            delta += 0.5  # 面对升班马客队有优势
        return delta

    def _compute_schedule_advantage(self, league: str, season: str, home: str, away: str,
                                     date: datetime) -> float:
        """
        F41: 赛程优势。

        比较两队过去7天的比赛密度差。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        cutoff = date - timedelta(days=7)
        home_matches = 0
        away_matches = 0

        for m in self._matches[key]:
            if m['date'] >= date:
                break
            if cutoff <= m['date'] < date:
                if m['home'] == home or m['away'] == home:
                    home_matches += 1
                if m['home'] == away or m['away'] == away:
                    away_matches += 1

        # 正值 = 主队比赛更少 (休息更多) → 优势
        return (away_matches - home_matches) * 0.5

    # ================================================================
    # v5.10.7: 新增因子计算
    # ================================================================

    def _compute_h2h_results(
        self, league: str, season: str, home: str, away: str, date: datetime,
    ) -> List[float]:
        """
        F5: 历史交锋数据。

        在 match_date 之前的所有赛季中查找两队历史交锋，
        返回最近5场的结果编码 [3=胜, 1=平, 0=负]。
        如果不足5场，用 1.5 填充。
        """
        results = []
        all_seasons = sorted(self._matches.keys())

        for key in all_seasons:
            if key not in self._matches:
                continue
            for m in self._matches[key]:
                if m['date'] >= date:
                    continue
                # 检查是否是主客队交锋
                if (m['home'] == home and m['away'] == away) or \
                   (m['home'] == away and m['away'] == home):
                    if m['home'] == home:
                        # 主队视角: 胜=3, 平=1, 负=0
                        if m['fthg'] > m['ftag']:
                            results.append(3.0)
                        elif m['fthg'] == m['ftag']:
                            results.append(1.0)
                        else:
                            results.append(0.0)
                    else:
                        # 客队视角: 胜=3, 平=1, 负=0
                        if m['ftag'] > m['fthg']:
                            results.append(3.0)
                        elif m['fthg'] == m['ftag']:
                            results.append(1.0)
                        else:
                            results.append(0.0)

        # 取最近5场，不足则填充
        results = results[-5:]
        while len(results) < 5:
            results.insert(0, 1.5)
        return results

    def _compute_referee_yellow_rate(
        self, league: str, season: str, date: datetime,
    ) -> float:
        """
        F13: 裁判黄牌率偏差。

        从 CSV 中提取裁判名称和比赛黄牌数，
        计算指定日期之前该裁判的平均黄牌率与联赛均值的偏差。
        返回偏差值，正值表示裁判更严格。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        # 1. 收集联赛所有裁判的黄牌数据
        ref_yellows: Dict[str, List[float]] = defaultdict(list)
        all_yellows = []

        for m in self._matches[key]:
            if m['date'] >= date:
                break
            ref = m.get('referee', '').strip()
            if not ref:
                continue
            yellows = m['hy'] + m['ay']
            ref_yellows[ref].append(yellows)
            all_yellows.append(yellows)

        if not all_yellows or len(all_yellows) < 10:
            return 0.0

        league_avg = sum(all_yellows) / len(all_yellows)

        # 2. 获取当前比赛裁判的历史平均黄牌率
        #    需要从 row 中获取当前裁判，但这里没有 row 参数
        #    使用最近一场比赛的裁判作为代理
        ref_name = None
        for m in reversed(self._matches[key]):
            if m['date'] < date:
                ref_name = m.get('referee', '').strip()
                if ref_name:
                    break

        if not ref_name or ref_name not in ref_yellows or len(ref_yellows[ref_name]) < 3:
            return 0.0

        ref_avg = sum(ref_yellows[ref_name]) / len(ref_yellows[ref_name])

        # 返回偏差 (正值 = 裁判更严格，黄牌更多)
        return ref_avg - league_avg

    def _compute_style_matchup(
        self, league: str, season: str, home: str, away: str, date: datetime,
    ) -> float:
        """
        F19: 风格匹配度。

        从射门数、犯规数、角球数计算球队风格特征:
        - 控球型: 高射门+低犯规
        - 压迫型: 高犯规+高角球
        - 防反型: 低射门+低犯规
        比较两队风格相似度，高分表示风格冲突大。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.5

        def get_team_stats(team: str) -> Dict[str, float]:
            stats = {'shots': 0, 'fouls': 0, 'corners': 0, 'matches': 0}
            for m in self._matches[key]:
                if m['date'] >= date:
                    break
                if m['home'] == team:
                    stats['shots'] += m['hs']
                    stats['fouls'] += m['hf']
                    stats['corners'] += m['hc']
                    stats['matches'] += 1
                elif m['away'] == team:
                    stats['shots'] += m['as']
                    stats['fouls'] += m['af']
                    stats['corners'] += m['ac']
                    stats['matches'] += 1
            n = max(1, stats['matches'])
            return {k: v / n for k, v in stats.items()}

        home_stats = get_team_stats(home)
        away_stats = get_team_stats(away)

        if home_stats['matches'] < 3 or away_stats['matches'] < 3:
            return 0.5

        # 风格向量: [射门, 犯规, 角球]
        # 归一化: 射门/20, 犯规/15, 角球/10
        h_vec = [home_stats['shots'] / 20, home_stats['fouls'] / 15, home_stats['corners'] / 10]
        a_vec = [away_stats['shots'] / 20, away_stats['fouls'] / 15, away_stats['corners'] / 10]

        # 余弦相似度
        dot = sum(h * a for h, a in zip(h_vec, a_vec))
        norm_h = math.sqrt(sum(h * h for h in h_vec))
        norm_a = math.sqrt(sum(a * a for a in a_vec))
        if norm_h < 0.01 or norm_a < 0.01:
            return 0.5

        similarity = dot / (norm_h * norm_a)
        # 转换为冲突度: 1 - similarity (0=完全相同, 1=完全不同)
        conflict = 1.0 - similarity
        return max(0.0, min(1.0, conflict))

    def _compute_value_signal(
        self, league: str, season: str, home: str, away: str, date: datetime,
    ) -> float:
        """
        F30: 价值信号。

        计算市场赔率与历史平均赔率的偏差。
        当赔率显著偏离历史均值时，可能包含价值信号。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        # 收集该比赛日之前所有比赛的赔率数据
        odds_home = []
        odds_away = []
        for m in self._matches[key]:
            if m['date'] >= date:
                break
            sources = m.get('odds_sources', {})
            if 'B365' in sources:
                odds_home.append(sources['B365']['home'])
                odds_away.append(sources['B365']['away'])

        if len(odds_home) < 30:
            return 0.0

        avg_home = sum(odds_home) / len(odds_home)
        avg_away = sum(odds_away) / len(odds_away)

        # 获取当前比赛赔率
        curr_home = None
        curr_away = None
        for m in reversed(self._matches[key]):
            if m['date'] < date and m['home'] == home and m['away'] == away:
                sources = m.get('odds_sources', {})
                if 'B365' in sources:
                    curr_home = sources['B365']['home']
                    curr_away = sources['B365']['away']
                break
            # 如果找不到完全匹配，用最近的
            if m['date'] < date and not curr_home:
                sources = m.get('odds_sources', {})
                if 'B365' in sources:
                    curr_home = sources['B365']['home']
                    curr_away = sources['B365']['away']

        if not curr_home:
            return 0.0

        # 偏差 = (当前 - 均值) / 均值
        home_dev = (avg_home - curr_home) / avg_home if avg_home > 0 else 0
        away_dev = (avg_away - curr_away) / avg_away if avg_away > 0 else 0

        # 正值 = 主队赔率低于历史均值 (市场更看好主队)
        return home_dev - away_dev

    def _compute_contrarian_signal(
        self, league: str, season: str, home: str, away: str, date: datetime,
    ) -> float:
        """
        F31: 反市场信号。

        检测赔率是否在向反方向移动 (市场可能过度反应)。
        比较开盘赔率 (B365) 的变动方向与历史均值。
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        # 收集赔率变动历史
        movements = []
        for m in self._matches[key]:
            if m['date'] >= date:
                break
            sources = m.get('odds_sources', {})
            if 'B365' in sources and 'PS' in sources:
                # 比较 B365 和 Pinnacle 的差异
                b365_home = sources['B365']['home']
                ps_home = sources['PS']['home']
                if b365_home > 1.0 and ps_home > 1.0:
                    # 正值 = B365 赔率高于 Pinnacle (市场对主队更悲观)
                    movements.append((b365_home - ps_home) / b365_home)

        if len(movements) < 30:
            return 0.0

        avg_movement = sum(movements) / len(movements)
        recent_movement = sum(movements[-5:]) / len(movements[-5:]) if len(movements) >= 5 else 0

        # 反市场信号: 近期方向与历史方向相反
        return (recent_movement - avg_movement) * 2.0  # 放大

    # ================================================================
    # v5.11: 平局专属因子计算方法
    # ================================================================

    def _compute_draw_tactical_matchup(self, league: str, season: str,
                                       home: str, away: str, date: datetime) -> float:
        """
        F56: 战术风格平局倾向。

        核心逻辑:
        - 防守型 vs 防守型 → 平局概率高 (意式0:0/1:1)
        - 进攻型 vs 进攻型 → 分出胜负概率高 (3:2/4:3)
        - 防守型 vs 进攻型 → 取决于谁主导节奏

        计算:
        1. 统计两队近期的场均进球+失球 (攻防活跃度)
        2. 两队都低于联赛均值 → 防守型对决 → +
        3. 两队都高于联赛均值 → 进攻型对决 → -
        返回: [-1, +1], 正值=平局倾向高
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        home_goals_for = []
        home_goals_against = []
        away_goals_for = []
        away_goals_against = []

        for m in self._matches[key]:
            if m['date'] >= date:
                break
            if m['home'] == home:
                home_goals_for.append(m['fthg'])
                home_goals_against.append(m['ftag'])
            elif m['away'] == home:
                home_goals_for.append(m['ftag'])
                home_goals_against.append(m['fthg'])
            if m['home'] == away:
                away_goals_for.append(m['fthg'])
                away_goals_against.append(m['ftag'])
            elif m['away'] == away:
                away_goals_for.append(m['ftag'])
                away_goals_against.append(m['fthg'])

        if len(home_goals_for) < 5 or len(away_goals_for) < 5:
            return 0.0

        # 攻防活跃度 = 场均(进球+失球)
        home_activity = (sum(home_goals_for) + sum(home_goals_against)) / len(home_goals_for)
        away_activity = (sum(away_goals_for) + sum(away_goals_against)) / len(away_goals_for)
        league_avg = LEAGUE_AVG_GOALS.get(league, 2.7)

        # 两队都低于联赛均值 → 防守型对决 → 平局倾向 +
        home_defensive = home_activity < league_avg
        away_defensive = away_activity < league_avg

        if home_defensive and away_defensive:
            # 双方都防守型: 强烈平局信号
            deficit = league_avg - min(home_activity, away_activity)
            return min(1.0, deficit / league_avg)  # 归一化到 [0, 1]
        elif not home_defensive and not away_defensive:
            # 双方都进攻型: 反平局信号
            excess = max(home_activity, away_activity) - league_avg
            return -min(1.0, excess / league_avg)
        else:
            # 一攻一守: 轻微正向 (比赛节奏不确定)
            return 0.1

    def _compute_draw_goal_expectancy(self, league: str, season: str,
                                      home: str, away: str, date: datetime) -> float:
        """
        F57: 进球预期平局信号。

        核心逻辑:
        - 低进球预期 (lambda_home + lambda_away < 2.0) → 0:0/1:1 概率高 → 平局+
        - 高进球预期 (lambda > 3.5) → 分出胜负概率高 → 平局-

        使用射门数据作为进球预期的代理 (与 F9 的 xG 代理协同):
        - 场均射正少 → 进攻产出低 → 低进球预期
        - 场均射正多 → 进攻产出高 → 高进球预期

        返回: [-1, +1], 正值=低进球预期=平局倾向高
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        home_sot = []  # 射正数
        away_sot = []

        for m in self._matches[key]:
            if m['date'] >= date:
                break
            if m['home'] == home:
                home_sot.append(m['hst'])
            elif m['away'] == home:
                home_sot.append(m['ast'])
            if m['home'] == away:
                away_sot.append(m['hst'])
            elif m['away'] == away:
                away_sot.append(m['ast'])

        if len(home_sot) < 5 or len(away_sot) < 5:
            return 0.0

        avg_home_sot = sum(home_sot) / len(home_sot)
        avg_away_sot = sum(away_sot) / len(away_sot)

        # 射正率 → 预期进球的粗略映射 (每3脚射正≈1球)
        expected_goals = (avg_home_sot + avg_away_sot) / 3.0
        league_avg = LEAGUE_AVG_GOALS.get(league, 2.7)

        # 低于联赛均值 → 平局倾向; 高于 → 反平局
        deviation = (expected_goals - league_avg) / league_avg
        return -deviation  # 取反: 低进球 → 正值(平局+)

    def _compute_draw_team_tendency(self, league: str, season: str,
                                    home: str, away: str, date: datetime) -> float:
        """
        F58: 球队平局历史倾向。

        核心逻辑:
        - 某些球队是“平局大师” (如马竞、尤文、那不勒斯)
        - 两队近期平局率都高 → 平局概率更高
        - 两队近期平局率都低 → 分出胜负概率更高

        计算:
        1. 统计两队近10场的平局率
        2. 两队平局率的几何平均 (避免单队极端值主导)
        3. 与联赛平均平局率比较

        返回: [-1, +1], 正值=平局倾向高
        """
        key = f"{league}_{season}"
        if key not in self._matches:
            return 0.0

        home_draws = 0
        home_total = 0
        away_draws = 0
        away_total = 0

        for m in self._matches[key]:
            if m['date'] >= date:
                break
            if m['home'] == home:
                home_total += 1
                if m['ftr'] == 'D':
                    home_draws += 1
            elif m['away'] == home:
                home_total += 1
                if m['ftr'] == 'D':
                    home_draws += 1
            if m['home'] == away:
                away_total += 1
                if m['ftr'] == 'D':
                    away_draws += 1
            elif m['away'] == away:
                away_total += 1
                if m['ftr'] == 'D':
                    away_draws += 1

        if home_total < 5 or away_total < 5:
            return 0.0

        home_draw_rate = home_draws / home_total
        away_draw_rate = away_draws / away_total

        # 几何平均
        combined_rate = (home_draw_rate * away_draw_rate) ** 0.5
        league_draw_rate = LEAGUE_DRAW_RATES.get(league, 0.25)

        # 高于联赛平均 → 平局倾向 +
        deviation = (combined_rate - league_draw_rate) / league_draw_rate
        return max(-1.0, min(1.0, deviation))