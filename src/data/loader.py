"""
GTO-GameFlow v5.0 数据加载与预处理

将原始 API 数据转换为流水线可用的 MatchContext 和 extra_data 字典。
处理数据降级、默认值填充、异常检测 (3σ 原则)。
"""
import math
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from src.data.models import MatchContext, BetSelection
from src.config.league_params import get_league_params
from src.config.settings import config as global_config

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器 — 将原始数据转换为流水线输入格式。

    处理:
    - 数据降级 (使用默认值)
    - 异常检测 (3σ 原则)
    - 缺失值填充
    - 格式转换
    """

    def __init__(self, league_id: str):
        self.league_id = league_id
        self.params = get_league_params(league_id)

    def build_match_context(
        self,
        raw_match: Dict,
        odds_data: Optional[Dict] = None,
        elo_data: Optional[Dict] = None,
    ) -> MatchContext:
        """
        从原始比赛数据构建 MatchContext。

        raw_match 字段:
        - match_id, league_id, season, matchday, kickoff_time
        - home_team, away_team
        - home_odds, draw_odds, away_odds (可选)
        """
        # 赔率: 优先使用 raw_match 中的，其次 odds_data
        odds_home = raw_match.get("home_odds", 0.0)
        odds_draw = raw_match.get("draw_odds", 0.0)
        odds_away = raw_match.get("away_odds", 0.0)

        if odds_data and not (odds_home > 0 and odds_draw > 0 and odds_away > 0):
            odds_home = odds_data.get("home", odds_home)
            odds_draw = odds_data.get("draw", odds_draw)
            odds_away = odds_data.get("away", odds_away)

        # 默认赔率 (防御)
        if odds_home <= 0:
            odds_home = 2.0
        if odds_draw <= 0:
            odds_draw = 3.5
        if odds_away <= 0:
            odds_away = 3.5

        # Elo
        home_elo = elo_data.get("home_elo", self.params.cold_start_elo) if elo_data else self.params.cold_start_elo
        away_elo = elo_data.get("away_elo", self.params.cold_start_elo) if elo_data else self.params.cold_start_elo

        return MatchContext(
            match_id=raw_match.get("match_id", ""),
            league_id=raw_match.get("league_id", self.league_id),
            season=raw_match.get("season", ""),
            matchday=raw_match.get("matchday", 0),
            kickoff_time=raw_match.get("kickoff_time", datetime.now()),
            home_team=raw_match.get("home_team", ""),
            away_team=raw_match.get("away_team", ""),
            home_elo=home_elo,
            away_elo=away_elo,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            home_xg=raw_match.get("home_xg"),
            away_xg=raw_match.get("away_xg"),
            home_possession=raw_match.get("home_possession"),
            away_possession=raw_match.get("away_possession"),
        )

    def build_extra_data(
        self,
        recent_form: Optional[List[float]] = None,
        h2h_results: Optional[List[float]] = None,
        standings: Optional[List[Dict]] = None,
        team_stats: Optional[Dict] = None,
        odds_history: Optional[Dict] = None,
        weather: Optional[Dict] = None,
        nlp_sentiment: Optional[float] = None,
        match_phase: Optional[float] = None,
    ) -> Dict:
        """
        构建 extra_data 字典，提供流水线所需的全部上下文数据。

        所有缺失值均使用规范定义的默认值填充。
        """
        extra = {}

        # --- 近期状态 (胜=3, 平=1, 负=0) ---
        extra["recent_results"] = recent_form or [1.5, 1.5, 1.5, 1.5, 1.5]

        # --- 历史交锋 ---
        extra["h2h_results"] = h2h_results or [0, 0, 0, 0, 0]

        # --- 联赛排名 ---
        if standings:
            extra["rank_diff"] = self._compute_rank_diff(standings)
        else:
            extra["rank_diff"] = 0

        # --- 赛程密度 (7天内比赛数) ---
        extra["matches_7d"] = extra.get("matches_7d", 1)

        # --- 球队统计 ---
        if team_stats:
            extra["goal_diff"] = self._extract_goal_diff(team_stats)
            extra["xg_diff"] = self._extract_xg_diff(team_stats)
        else:
            extra["goal_diff"] = 0.0
            extra["xg_diff"] = 0.0

        # --- 赔率历史 ---
        if odds_history:
            extra["opening_probs"] = odds_history.get("opening")
        else:
            extra["opening_probs"] = None

        # --- 天气 ---
        if weather:
            extra["weather"] = self._compute_weather_score(weather)
        else:
            extra["weather"] = 0.0

        # --- 比赛阶段 ---
        extra["match_phase"] = match_phase if match_phase is not None else 1.0

        # --- 默认填充 (规范第2.2节) ---
        defaults = {
            "elo_diff": 0.0,
            "xi_rating": 6.0,
            "ref_yellow_rate": self.params.yellow_card_rate,
            "coach_change_effect": 0.0,
            "fatigue_penalty": 0.0,
            "rotation_risk": 0.0,
            "derby_boost": 0.0,
            "style_matchup_score": 0.5,
            "streak_momentum": 0.0,
            "player_form": 6.5,
            "market_sentiment": 0.0,
            "odds_std": 0.05,
            "nlp_sentiment": nlp_sentiment or 0.0,
            "time_decay_factor": 1.0,
            "league_strength_bias": 0.0,
            "poisson_correction": 0.0,
            "handicap_depth": 0.0,
            "totals_trend": 0.0,
            "value_signal": 0.0,
            "contrarian_signal": 0.0,
            "market_efficiency": 0.0,
            "motivation_boost": 0.0,
            "financial_gap_effect": 0.0,
            "winter_break_effect": 0.0,
            "christmas_fatigue": 0.0,
            "complacency_effect": 0.0,
            "streak_momentum_league": 0.0,
            "position_advantage": 0.0,
            "promoted_team_delta": 0.0,
            "schedule_advantage": 0.0,
            "derby_intensity": 0.0,
            "data_source_count": 5,
            "data_completeness": 0.8,
            "dispersion_penalty": 0.9,
        }
        for key, value in defaults.items():
            extra.setdefault(key, value)

        return extra

    # ================================================================
    # 辅助函数
    # ================================================================

    def _compute_rank_diff(self, standings: List[Dict]) -> int:
        """
        计算联赛排名差。

        rank_diff = rank_away - rank_home (正值=主队排名更靠前)
        """
        # 简化实现: 需要传入 home_rank 和 away_rank
        if isinstance(standings, dict):
            return standings.get("away_rank", 0) - standings.get("home_rank", 0)
        return 0

    def _extract_goal_diff(self, team_stats: Dict) -> float:
        """提取赛季净胜球差"""
        home_gf = team_stats.get("home_goals_for", 0)
        home_ga = team_stats.get("home_goals_against", 0)
        away_gf = team_stats.get("away_goals_for", 0)
        away_ga = team_stats.get("away_goals_against", 0)
        return (home_gf - home_ga) - (away_gf - away_ga)

    def _extract_xg_diff(self, team_stats: Dict) -> float:
        """提取 xG 差值"""
        home_xg = team_stats.get("home_xg", 0.0)
        home_xga = team_stats.get("home_xga", 0.0)
        away_xg = team_stats.get("away_xg", 0.0)
        away_xga = team_stats.get("away_xga", 0.0)
        return (home_xg - home_xga) - (away_xg - away_xga)

    def _compute_weather_score(self, weather: Dict) -> float:
        """
        计算天气影响评分。

        规范F12: 0.3×温度 + 0.4×降雨 + 0.3×风速
        """
        temp_score = self._normalize_weather(weather.get("temperature", 15), 0, 35)
        rain_score = weather.get("rain_mm", 0.0) / 10.0  # 10mm = 1.0
        wind_score = self._normalize_weather(weather.get("wind_speed", 10), 0, 50)

        return 0.3 * temp_score + 0.4 * min(1.0, rain_score) + 0.3 * wind_score

    @staticmethod
    def _normalize_weather(value: float, lo: float, hi: float) -> float:
        """归一化天气值到 [0, 1]"""
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    @staticmethod
    def detect_anomalies(
        values: List[float],
        window: int = 100,
        sigma: float = 3.0,
    ) -> List[bool]:
        """
        异常检测 — 3σ 原则 (规范第2.4节)。

        检测窗口: 100 场或最近 90 天 (取较大者)
        超出 μ ± 3σ 标记为异常
        """
        if len(values) < 10:
            return [False] * len(values)

        arr = np.array(values[-window:])
        mu = np.mean(arr)
        std = np.std(arr)

        if std < 1e-10:
            return [False] * len(values)

        anomalies = []
        for v in values:
            anomalies.append(abs(v - mu) > sigma * std)

        return anomalies

    @staticmethod
    def compute_elos(
        home_elo: float,
        away_elo: float,
        result: str,  # "home_win" / "draw" / "away_win"
        k_factor: float = 20.0,
        home_advantage: float = 65.0,
    ) -> Tuple[float, float]:
        """
        Elo 评分更新 (标准 Elo 公式)。

        规范: 使用联赛特定 K 因子和主场加分。
        """
        # 预期胜率
        expected_home = 1.0 / (1.0 + 10 ** (-(home_elo + home_advantage - away_elo) / 400.0))
        expected_away = 1.0 - expected_home

        # 实际结果
        if result == "home_win":
            actual_home, actual_away = 1.0, 0.0
        elif result == "away_win":
            actual_home, actual_away = 0.0, 1.0
        else:  # draw
            actual_home, actual_away = 0.5, 0.5

        # 更新
        new_home = home_elo + k_factor * (actual_home - expected_home)
        new_away = away_elo + k_factor * (actual_away - expected_away)

        return new_home, new_away

    @staticmethod
    def compute_recent_form(
        results: List[str],
        last_n: int = 5,
        alpha: float = 0.18,
    ) -> List[float]:
        """
        计算近期状态 EWMA。

        result: "W" (3), "D" (1), "L" (0)
        """
        points = {"W": 3.0, "D": 1.0, "L": 0.0}
        values = [points.get(r, 1.5) for r in results[-last_n:]]

        while len(values) < last_n:
            values.insert(0, 1.5)

        # EWMA
        ewma = 0.0
        weight_sum = 0.0
        for i, v in enumerate(values):
            w = (1 - alpha) ** i
            ewma += v * w
            weight_sum += w

        return values

    @staticmethod
    def compute_market_probs(
        odds: Dict[str, float],
    ) -> Dict[str, float]:
        """
        从赔率计算市场隐含概率 (去除 overround)。

        规范第7.4节: 公平概率 = (1/odds) / overround
        """
        overround = sum(1.0 / o for o in odds.values() if o > 0)
        if overround <= 0:
            return {"home": 0.33, "draw": 0.34, "away": 0.33}

        return {
            outcome: (1.0 / odds[outcome]) / overround
            for outcome in odds
        }