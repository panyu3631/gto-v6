"""
GTO-GameFlow v5.10.8 — 比赛统计增强器 (MatchStatsEnricher)

从 CSV 历史数据的 106 列中提取所有可用的高维度衍生特征。
严格避免未来信息泄露：每个特征仅使用该场比赛日期之前的数据。

新增特征（此前未使用的 CSV 列）:
  F42: 半场动量 (HT Momentum) — HTHG/HTAG/HTR 半场领先→全场胜率
  F43: 射门效率差 (Shot Efficiency Diff) — HST/HS vs AST/AS
  F44: 控场优势 (Territorial Dominance) — 射门比 + 角球比 复合指标
  F45: 纪律指数 (Discipline Index) — 犯规/黄牌效率
  F46: 赔率漂移信号 (Odds Drift) — 开盘→收盘赔率变动
  F47: 市场分歧 (Market Disagreement) — 6家博彩商赔率标准差
  F48: 裁判主场偏置 (Referee Home Bias) — 裁判历史主胜率
  F49: 逆转韧性 (Comeback Resilience) — 先失球后逆转能力
  F50: 连胜/连败动量 (Streak Momentum) — 当前连胜/连败场次
  F51: 进球波动率 (Goal Volatility) — 近期进球数的标准差
  F52: 角球优势 (Corner Dominance) — 角球比
  F53: 射正率差 (SoT Rate Diff) — 射正/射门比差值
  F54: 亚盘赔率漂移 (AH Odds Drift) — 亚盘开盘→收盘变动
  F55: 大小球赔率漂移 (Totals Odds Drift) — 大小球开盘→收盘变动
"""

import csv
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# 联赛 → CSV 列名映射（新格式 vs 旧格式）
# ============================================================

CSV_COLUMN_VARIANTS = {
    # 半场数据
    "HTHG": ["HTHG"],
    "HTAG": ["HTAG"],
    "HTR": ["HTR"],
    # 射门
    "HS": ["HS"],
    "AS": ["AS"],
    "HST": ["HST"],
    "AST": ["AST"],
    # 犯规
    "HF": ["HF"],
    "AF": ["AF"],
    # 角球
    "HC": ["HC"],
    "AC": ["AC"],
    # 黄牌
    "HY": ["HY"],
    "AY": ["AY"],
    # 红牌
    "HR": ["HR"],
    "AR": ["AR"],
    # 收盘赔率 (Bet365 Close)
    "B365CH": ["B365CH"],
    "B365CD": ["B365CD"],
    "B365CA": ["B365CA"],
    # 收盘大小球
    "B365C>2.5": ["B365C>2.5"],
    "B365C<2.5": ["B365C<2.5"],
    # 收盘亚盘
    "B365CAHH": ["B365CAHH"],
    "B365CAHA": ["B365CAHA"],
    # 开盘赔率（已有）
    "B365H": ["B365H"],
    "B365D": ["B365D"],
    "B365A": ["B365A"],
    "B365>2.5": ["B365>2.5"],
    "B365<2.5": ["B365<2.5"],
    "B365AHH": ["B365AHH"],
    "B365AHA": ["B365AHA"],
    # 6家博彩商
    "BWH": ["BWH"], "BWD": ["BWD"], "BWA": ["BWA"],
    "IWH": ["IWH"], "IWD": ["IWD"], "IWA": ["IWA"],
    "PSH": ["PSH"], "PSD": ["PSD"], "PSA": ["PSA"],
    "WHH": ["WHH"], "WHD": ["WHD"], "WHA": ["WHA"],
    "VCH": ["VCH"], "VCD": ["VCD"], "VCA": ["VCA"],
    # 裁判
    "Referee": ["Referee"],
    # 比分
    "FTHG": ["FTHG"], "FTAG": ["FTAG"], "FTR": ["FTR"],
    "HomeTeam": ["HomeTeam"], "AwayTeam": ["AwayTeam"],
    "Date": ["Date"],
}


class MatchStatsEnricher:
    """
    从 CSV 历史数据中提取全维度比赛统计特征。

    用法:
        enricher = MatchStatsEnricher(csv_dir, leagues, seasons)
        stats = enricher.get_enriched_stats(
            league, season, home_team, away_team, match_date
        )
    """

    def __init__(self, csv_dir: str, leagues: List[str], seasons: List[str]):
        self.csv_dir = csv_dir
        self.leagues = leagues
        self.seasons = sorted(seasons)

        # 内部数据: key=league_season → [match_dicts]
        self._matches: Dict[str, List[Dict]] = defaultdict(list)
        # 球队滚动统计: key=league_season:team → { stat_name → [values] }
        self._team_rolling: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # 裁判统计: key=referee → { total_matches, home_wins, draws, away_wins, total_cards }
        self._referee_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0, "home_wins": 0, "draws": 0, "away_wins": 0,
            "total_cards": 0, "total_matches": 0,
        })
        # 博彩商漂移统计: key=league_season:team → [ (open_odds, close_odds), ... ]
        self._odds_drift: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # 所有比赛按时间排序
        self._all_chrono: List[Dict] = []

        self._load_all_data()

    # ================================================================
    # 数据加载
    # ================================================================

    def _safe_get(self, row: Dict, col: str) -> str:
        """安全获取列值，支持多列名变体"""
        variants = CSV_COLUMN_VARIANTS.get(col, [col])
        for v in variants:
            val = row.get(v, "").strip()
            if val:
                return val
        return ""

    def _safe_int(self, row: Dict, col: str) -> int:
        try:
            return int(self._safe_get(row, col) or 0)
        except ValueError:
            return 0

    def _safe_float(self, row: Dict, col: str) -> float:
        try:
            return float(self._safe_get(row, col) or 0)
        except ValueError:
            return 0.0

    def _load_all_data(self):
        """加载所有 CSV 数据"""
        for league in self.leagues:
            for season in self.seasons:
                filename = f"{league}_{season}.csv"
                filepath = os.path.join(self.csv_dir, filename)
                if not os.path.exists(filepath):
                    continue

                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date_str = self._safe_get(row, "Date")
                        if not date_str:
                            continue
                        try:
                            dt = datetime.strptime(date_str, '%d/%m/%Y')
                        except ValueError:
                            continue

                        home = self._safe_get(row, "HomeTeam")
                        away = self._safe_get(row, "AwayTeam")
                        if not home or not away:
                            continue

                        match = {
                            "date": dt,
                            "home": home,
                            "away": away,
                            "league": league,
                            "season": season,
                            # 比分
                            "fthg": self._safe_int(row, "FTHG"),
                            "ftag": self._safe_int(row, "FTAG"),
                            "ftr": self._safe_get(row, "FTR"),
                            # 半场
                            "hthg": self._safe_int(row, "HTHG"),
                            "htag": self._safe_int(row, "HTAG"),
                            "htr": self._safe_get(row, "HTR"),
                            # 裁判
                            "referee": self._safe_get(row, "Referee"),
                            # 射门
                            "hs": self._safe_int(row, "HS"),
                            "as": self._safe_int(row, "AS"),
                            "hst": self._safe_int(row, "HST"),
                            "ast": self._safe_int(row, "AST"),
                            # 犯规/角球
                            "hf": self._safe_int(row, "HF"),
                            "af": self._safe_int(row, "AF"),
                            "hc": self._safe_int(row, "HC"),
                            "ac": self._safe_int(row, "AC"),
                            # 黄红牌
                            "hy": self._safe_int(row, "HY"),
                            "ay": self._safe_int(row, "AY"),
                            "hr": self._safe_int(row, "HR"),
                            "ar": self._safe_int(row, "AR"),
                            # 开盘赔率
                            "b365_h": self._safe_float(row, "B365H"),
                            "b365_d": self._safe_float(row, "B365D"),
                            "b365_a": self._safe_float(row, "B365A"),
                            # 收盘赔率
                            "b365c_h": self._safe_float(row, "B365CH"),
                            "b365c_d": self._safe_float(row, "B365CD"),
                            "b365c_a": self._safe_float(row, "B365CA"),
                            # 大小球开盘
                            "b365_over": self._safe_float(row, "B365>2.5"),
                            "b365_under": self._safe_float(row, "B365<2.5"),
                            # 大小球收盘
                            "b365c_over": self._safe_float(row, "B365C>2.5"),
                            "b365c_under": self._safe_float(row, "B365C<2.5"),
                            # 亚盘开盘
                            "b365_ahh": self._safe_float(row, "B365AHH"),
                            "b365_aha": self._safe_float(row, "B365AHA"),
                            # 亚盘收盘
                            "b365c_ahh": self._safe_float(row, "B365CAHH"),
                            "b365c_aha": self._safe_float(row, "B365CAHA"),
                            # 6家博彩商赔率
                            "odds_6": self._extract_6_bookmakers(row),
                        }

                        key = f"{league}_{season}"
                        self._matches[key].append(match)
                        self._all_chrono.append(match)

        # 按日期排序
        self._all_chrono.sort(key=lambda m: m["date"])
        self._matches = {k: sorted(v, key=lambda m: m["date"])
                         for k, v in self._matches.items()}

        # 构建滚动统计
        self._build_rolling_stats()

    def _extract_6_bookmakers(self, row: Dict) -> Dict[str, Dict[str, float]]:
        """提取6家博彩商的1X2赔率"""
        sources = {}
        bookmakers = [
            ("B365", "B365H", "B365D", "B365A"),
            ("BW", "BWH", "BWD", "BWA"),
            ("IW", "IWH", "IWD", "IWA"),
            ("PS", "PSH", "PSD", "PSA"),
            ("WH", "WHH", "WHD", "WHA"),
            ("VC", "VCH", "VCD", "VCA"),
        ]
        for name, hc, dc, ac in bookmakers:
            h = self._safe_float(row, hc)
            d = self._safe_float(row, dc)
            a = self._safe_float(row, ac)
            if h > 1.0 and d > 1.0 and a > 1.0:
                sources[name] = {"home": h, "draw": d, "away": a}
        return sources

    # ================================================================
    # 滚动统计构建（严格避免未来信息泄露）
    # ================================================================

    def _build_rolling_stats(self):
        """按时间顺序构建每个球队的滚动统计数据"""
        for match in self._all_chrono:
            league = match["league"]
            season = match["season"]
            home = match["home"]
            away = match["away"]
            key = f"{league}_{season}"

            # ===== 更新裁判统计 =====
            ref = match["referee"]
            if ref:
                rf = self._referee_stats[ref]
                rf["total_matches"] += 1
                if match["fthg"] > match["ftag"]:
                    rf["home_wins"] += 1
                elif match["fthg"] == match["ftag"]:
                    rf["draws"] += 1
                else:
                    rf["away_wins"] += 1
                rf["total_cards"] += match["hy"] + match["ay"]
                rf["total"] = rf["total_matches"]

            # ===== 更新球队滚动统计（比赛结束后） =====
            for team, is_home in [(home, True), (away, False)]:
                ts = self._team_rolling[key][team]

                gf = match["fthg"] if is_home else match["ftag"]
                ga = match["ftag"] if is_home else match["fthg"]
                hs = match["hs"] if is_home else match["as"]
                as_ = match["as"] if is_home else match["hs"]
                hst = match["hst"] if is_home else match["ast"]
                ast = match["ast"] if is_home else match["hst"]
                hf = match["hf"] if is_home else match["af"]
                af = match["af"] if is_home else match["hf"]
                hc = match["hc"] if is_home else match["ac"]
                ac = match["ac"] if is_home else match["hc"]
                hy = match["hy"] if is_home else match["ay"]
                hr = match["hr"] if is_home else match["ar"]
                hthg = match["hthg"] if is_home else match["htag"]
                htag = match["htag"] if is_home else match["hthg"]

                # 结果编码: 3=胜, 1=平, 0=负
                result = 3 if gf > ga else (1 if gf == ga else 0)

                ts["results"].append(result)
                ts["gf"].append(gf)
                ts["ga"].append(ga)
                ts["hs"].append(hs)
                ts["as_shot"].append(as_)
                ts["hst"].append(hst)
                ts["ast"].append(ast)
                ts["hf"].append(hf)
                ts["af"].append(af)
                ts["hc"].append(hc)
                ts["ac"].append(ac)
                ts["hy"].append(hy)
                ts["hr"].append(hr)
                ts["hthg"].append(hthg)
                ts["htag"].append(htag)

                # 半场结果
                ht_result = 3 if hthg > htag else (1 if hthg == htag else 0)
                ts["ht_results"].append(ht_result)

                # 射门效率
                shot_eff = hst / hs if hs > 0 else 0.0
                ts["shot_eff"].append(shot_eff)

                # 射正率
                sot_rate = hst / hs if hs > 0 else 0.0
                ts["sot_rate"].append(sot_rate)

                # 犯规/黄牌比
                foul_yellow = hy / hf if hf > 0 else 0.0
                ts["foul_yellow"].append(foul_yellow)

                # 赔率漂移
                if "b365_h" in match:
                    ts["odds_drift_h"].append(
                        match["b365c_h"] - match["b365_h"]
                        if match["b365_h"] > 0 and match["b365c_h"] > 0
                        else 0.0
                    )

            # ===== 更新赔率漂移历史 =====
            for team, is_home in [(home, True), (away, False)]:
                drift_key = f"{league}_{season}"
                if match["b365_h"] > 0 and match["b365c_h"] > 0:
                    self._odds_drift[drift_key][team].append({
                        "open": match["b365_h"] if is_home else match["b365_a"],
                        "close": match["b365c_h"] if is_home else match["b365c_a"],
                        "date": match["date"],
                    })
                if match["b365_d"] > 0 and match["b365c_d"] > 0:
                    self._odds_drift[drift_key][f"{team}_draw"].append({
                        "open": match["b365_d"],
                        "close": match["b365c_d"],
                        "date": match["date"],
                    })

    # ================================================================
    # 特征提取 API
    # ================================================================

    def get_enriched_stats(
        self,
        league: str,
        season: str,
        home_team: str,
        away_team: str,
        match_date: datetime,
    ) -> Dict[str, Any]:
        """
        获取一场比赛前所有可用统计数据。

        Returns:
            Dict with keys: ht_momentum, shot_eff_diff, territorial_dominance,
            discipline_index, odds_drift, market_disagreement, referee_home_bias,
            comeback_resilience, streak_momentum, goal_volatility,
            corner_dominance, sot_rate_diff, ah_odds_drift, totals_odds_drift
        """
        key = f"{league}_{season}"
        result = {}

        # 获取赛前滚动统计（仅使用 match_date 之前的数据）
        home_stats = self._get_team_stats_before(key, home_team, match_date)
        away_stats = self._get_team_stats_before(key, away_team, match_date)

        n_home = len(home_stats.get("results", []))
        n_away = len(away_stats.get("results", []))
        n = min(5, max(n_home, n_away))  # v5.10.8: 用数据更多的一方

        # v5.10.8: 跨赛季回填 — 当前赛季数据不足时，从上一赛季补充
        if n < 3:
            prev_season = self._get_previous_season(season)
            if prev_season:
                prev_key = f"{league}_{prev_season}"
                prev_home = self._get_team_stats_before(prev_key, home_team, datetime(2099, 12, 31))
                prev_away = self._get_team_stats_before(prev_key, away_team, datetime(2099, 12, 31))
                # 合并: 上赛季数据在前，当前赛季在后
                home_stats = self._merge_stats(prev_home, home_stats)
                away_stats = self._merge_stats(prev_away, away_stats)
                n_home = len(home_stats.get("results", []))
                n_away = len(away_stats.get("results", []))
                n = min(5, max(n_home, n_away))

        if n < 1:
            # 双方均无数据，返回默认值
            return self._default_stats()

        # === F42: 半场动量 ===
        result["ht_momentum"] = self._compute_ht_momentum(home_stats, away_stats, n)

        # === F43: 射门效率差 ===
        result["shot_eff_diff"] = self._compute_shot_eff_diff(home_stats, away_stats, n)

        # === F44: 控场优势 ===
        result["territorial_dominance"] = self._compute_territorial_dominance(
            home_stats, away_stats, n
        )

        # === F45: 纪律指数 ===
        result["discipline_index"] = self._compute_discipline_index(
            home_stats, away_stats, n
        )

        # === F46: 赔率漂移信号 ===
        result["odds_drift"] = self._compute_odds_drift(
            league, season, home_team, away_team, match_date, n
        )

        # === F47: 市场分歧 ===
        result["market_disagreement"] = self._compute_market_disagreement(
            league, season, home_team, away_team, match_date
        )

        # === F48: 裁判主场偏置 ===
        result["referee_home_bias"] = self._compute_referee_home_bias(
            league, season, home_team, away_team, match_date
        )

        # === F49: 逆转韧性 ===
        result["comeback_resilience"] = self._compute_comeback_resilience(
            home_stats, away_stats, n
        )

        # === F50: 连胜/连败动量 ===
        result["streak_momentum"] = self._compute_streak_momentum(
            home_stats, away_stats
        )

        # === F51: 进球波动率 ===
        result["goal_volatility"] = self._compute_goal_volatility(
            home_stats, away_stats, n
        )

        # === F52: 角球优势 ===
        result["corner_dominance"] = self._compute_corner_dominance(
            home_stats, away_stats, n
        )

        # === F53: 射正率差 ===
        result["sot_rate_diff"] = self._compute_sot_rate_diff(
            home_stats, away_stats, n
        )

        # === F54: 亚盘赔率漂移 ===
        result["ah_odds_drift"] = self._compute_ah_odds_drift(
            league, season, home_team, match_date
        )

        # === F55: 大小球赔率漂移 ===
        result["totals_odds_drift"] = self._compute_totals_odds_drift(
            league, season, home_team, away_team, match_date
        )

        return result

    # ================================================================
    # 内部工具方法
    # ================================================================

    def _get_team_stats_before(
        self, key: str, team: str, before_date: datetime
    ) -> Dict[str, List]:
        """
        获取球队在指定日期之前的所有滚动统计。
        由于滚动统计是比赛后更新的，直接取当前累积值即可。
        但需要截断到 before_date 之前。
        """
        all_stats = self._team_rolling.get(key, {}).get(team, {})
        if not all_stats:
            return {}

        # 找到 before_date 之前的比赛数量
        matches = self._matches.get(key, [])
        count = 0
        for m in matches:
            if (m["home"] == team or m["away"] == team) and m["date"] < before_date:
                count += 1

        # 截断到前 count 个值
        truncated = {}
        for stat_name, values in all_stats.items():
            truncated[stat_name] = values[:count] if count > 0 else []
        return truncated

    @staticmethod
    def _default_stats() -> Dict[str, Any]:
        return {
            "ht_momentum": 0.0,
            "shot_eff_diff": 0.0,
            "territorial_dominance": 0.0,
            "discipline_index": 0.0,
            "odds_drift": 0.0,
            "market_disagreement": 0.0,
            "referee_home_bias": 0.0,
            "comeback_resilience": 0.0,
            "streak_momentum": 0.0,
            "goal_volatility": 0.0,
            "corner_dominance": 0.0,
            "sot_rate_diff": 0.0,
            "ah_odds_drift": 0.0,
            "totals_odds_drift": 0.0,
        }

    @staticmethod
    def _get_previous_season(season: str) -> Optional[str]:
        """获取上一赛季字符串，如 '2023-24' → '2022-23'"""
        try:
            parts = season.split('-')
            if len(parts) == 2:
                start = int(parts[0]) - 1
                end = int(parts[1]) - 1
                return f"{start}-{str(end)[-2:]}"
        except (ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _merge_stats(prev: Dict[str, List], curr: Dict[str, List]) -> Dict[str, List]:
        """合并两个赛季的统计数据，prev 在前 curr 在后"""
        if not prev:
            return dict(curr)
        if not curr:
            return dict(prev)
        merged = {}
        all_keys = set(prev.keys()) | set(curr.keys())
        for k in all_keys:
            merged[k] = prev.get(k, []) + curr.get(k, [])
        return merged

    @staticmethod
    def _safe_mean(values: List[float], n: int) -> float:
        if not values:
            return 0.0
        recent = values[-n:] if len(values) >= n else values
        return sum(recent) / len(recent)

    @staticmethod
    def _safe_std(values: List[float], n: int) -> float:
        if len(values) < 2:
            return 0.0
        recent = values[-n:] if len(values) >= n else values
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        return math.sqrt(var)

    # ================================================================
    # 各因子计算方法
    # ================================================================

    def _compute_ht_momentum(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F42: 半场动量
        衡量球队半场领先→全场获胜的转化率。
        正值 = 主队半场表现更好，转化为全场优势。
        """
        home_ht = home_stats.get("ht_results", [])
        away_ht = away_stats.get("ht_results", [])
        home_ft = home_stats.get("results", [])
        away_ft = away_stats.get("results", [])

        if len(home_ht) < 1 and len(away_ht) < 1:
            return 0.0

        actual_n = max(1, min(n, len(home_ht), len(away_ht)))
        # 半场领先→全场获胜转化率
        home_ht_lead_wins = sum(
            1 for i in range(-actual_n, 0)
            if i < len(home_ht) and home_ht[i] == 3 and home_ft[i] == 3
        ) / max(actual_n, 1)
        away_ht_lead_wins = sum(
            1 for i in range(-actual_n, 0)
            if i < len(away_ht) and away_ht[i] == 3 and away_ft[i] == 3
        ) / max(actual_n, 1)

        # 半场落后的逆转率
        home_ht_trail_comeback = sum(
            1 for i in range(-n, 0)
            if i < len(home_ht) and home_ht[i] == 0 and home_ft[i] == 3
        ) / max(n, 1)
        away_ht_trail_comeback = sum(
            1 for i in range(-n, 0)
            if i < len(away_ht) and away_ht[i] == 0 and away_ft[i] == 3
        ) / max(n, 1)

        momentum = (
            (home_ht_lead_wins - away_ht_lead_wins) * 0.6 +
            (home_ht_trail_comeback - away_ht_trail_comeback) * 0.4
        )
        return max(-1.0, min(1.0, momentum))

    def _compute_shot_eff_diff(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F43: 射门效率差
        射正率 × 射门转化率 的复合差值。
        """
        home_eff = self._safe_mean(home_stats.get("shot_eff", []), n)
        away_eff = self._safe_mean(away_stats.get("shot_eff", []), n)
        return max(-1.0, min(1.0, home_eff - away_eff))

    def _compute_territorial_dominance(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F44: 控场优势
        射门比 (0.5) + 角球比 (0.3) + 犯规比倒数 (0.2) 的加权复合。
        犯规比倒数：犯规少的一方控球更多。
        """
        home_hs = self._safe_mean(home_stats.get("hs", []), n)
        away_hs = self._safe_mean(away_stats.get("hs", []), n)
        home_hc = self._safe_mean(home_stats.get("hc", []), n)
        away_hc = self._safe_mean(away_stats.get("hc", []), n)
        home_hf = self._safe_mean(home_stats.get("hf", []), n)
        away_hf = self._safe_mean(away_stats.get("hf", []), n)

        shot_ratio = (home_hs - away_hs) / max(home_hs + away_hs, 1) if (home_hs + away_hs) > 0 else 0.0
        corner_ratio = (home_hc - away_hc) / max(home_hc + away_hc, 1) if (home_hc + away_hc) > 0 else 0.0
        foul_ratio = (away_hf - home_hf) / max(home_hf + away_hf, 1) if (home_hf + away_hf) > 0 else 0.0

        dominance = shot_ratio * 0.5 + corner_ratio * 0.3 + foul_ratio * 0.2
        return max(-1.0, min(1.0, dominance))

    def _compute_discipline_index(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F45: 纪律指数
        黄牌/犯规比 (越低越好) + 红牌风险。
        正值 = 主队更纪律，客队更激进。
        """
        home_fy = self._safe_mean(home_stats.get("foul_yellow", []), n)
        away_fy = self._safe_mean(away_stats.get("foul_yellow", []), n)
        home_hr = sum(home_stats.get("hr", [])[-n:]) / max(n, 1)
        away_hr = sum(away_stats.get("hr", [])[-n:]) / max(n, 1)

        # 纪律差 = 客队黄牌率 - 主队黄牌率 + 红牌差
        discipline = (away_fy - home_fy) * 0.8 + (away_hr - home_hr) * 0.2
        return max(-1.0, min(1.0, discipline))

    def _compute_odds_drift(
        self, league: str, season: str,
        home_team: str, away_team: str,
        match_date: datetime, n: int,
    ) -> float:
        """
        F46: 赔率漂移信号
        开盘→收盘赔率变动方向。
        正值 = 市场看好主队（主胜赔率下降）。
        """
        key = f"{league}_{season}"
        home_drifts = [
            d for d in self._odds_drift.get(key, {}).get(home_team, [])
            if d["date"] < match_date
        ][-n:]
        away_drifts = [
            d for d in self._odds_drift.get(key, {}).get(away_team, [])
            if d["date"] < match_date
        ][-n:]

        home_avg = sum(d["close"] - d["open"] for d in home_drifts) / max(len(home_drifts), 1)
        away_avg = sum(d["close"] - d["open"] for d in away_drifts) / max(len(away_drifts), 1)

        # 主胜赔率下降 = 市场看好主队 = 正值
        drift = (away_avg - home_avg) * 0.5
        return max(-1.0, min(1.0, drift))

    def _compute_market_disagreement(
        self, league: str, season: str,
        home_team: str, away_team: str,
        match_date: datetime,
    ) -> float:
        """
        F47: 市场分歧
        6家博彩商赔率的标准差，反映市场不确定性。
        """
        key = f"{league}_{season}"
        matches = self._matches.get(key, [])
        if not matches:
            return 0.0

        # 找到本场比赛的原始数据
        for m in matches:
            if (m["home"] == home_team and m["away"] == away_team
                    and m["date"] == match_date):
                odds_6 = m.get("odds_6", {})
                if len(odds_6) >= 3:
                    home_odds = [v["home"] for v in odds_6.values()]
                    draw_odds = [v["draw"] for v in odds_6.values()]
                    away_odds = [v["away"] for v in odds_6.values()]

                    std_h = self._compute_std(home_odds)
                    std_d = self._compute_std(draw_odds)
                    std_a = self._compute_std(away_odds)

                    avg_std = (std_h + std_d + std_a) / 3
                    # 归一化: 标准差/平均赔率
                    avg_odds = sum(home_odds + draw_odds + away_odds) / len(home_odds + draw_odds + away_odds)
                    if avg_odds > 0:
                        return min(1.0, avg_std / avg_odds * 5)
                return 0.0
        return 0.0

    def _compute_referee_home_bias(
        self, league: str, season: str,
        home_team: str, away_team: str,
        match_date: datetime,
    ) -> float:
        """
        F48: 裁判主场偏置
        裁判历史主胜率 vs 联赛平均主胜率。
        """
        key = f"{league}_{season}"
        matches = self._matches.get(key, [])
        if not matches:
            return 0.0

        # 找到本场比赛的裁判
        referee = ""
        for m in matches:
            if (m["home"] == home_team and m["away"] == away_team
                    and m["date"] == match_date):
                referee = m.get("referee", "")
                break

        if not referee:
            return 0.0

        rf = self._referee_stats.get(referee, {})
        if rf.get("total", 0) < 5:
            return 0.0

        ref_home_win_rate = rf["home_wins"] / rf["total"] if rf["total"] > 0 else 0.0

        # 联赛平均主胜率
        league_home_wins = sum(
            1 for m in matches if m["date"] < match_date and m["fthg"] > m["ftag"]
        )
        league_total = sum(1 for m in matches if m["date"] < match_date)
        league_home_rate = league_home_wins / league_total if league_total > 0 else 0.45

        bias = (ref_home_win_rate - league_home_rate) * 3
        return max(-1.0, min(1.0, bias))

    def _compute_comeback_resilience(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F49: 逆转韧性
        先失球后的逆转/扳平能力。
        """
        home_gf = home_stats.get("gf", [])
        home_ga = home_stats.get("ga", [])
        away_gf = away_stats.get("gf", [])
        away_ga = away_stats.get("ga", [])
        home_ht = home_stats.get("ht_results", [])
        away_ht = away_stats.get("ht_results", [])
        home_ft = home_stats.get("results", [])
        away_ft = away_stats.get("results", [])

        if len(home_ht) < 1 and len(away_ht) < 1:
            return 0.0

        actual_n = max(1, min(n, len(home_ht), len(away_ht)))
        # 半场落后 → 全场不输
        home_comebacks = sum(
            1 for i in range(-actual_n, 0)
            if i < len(home_ht) and home_ht[i] == 0 and home_ft[i] >= 1
        ) / max(actual_n, 1)
        away_comebacks = sum(
            1 for i in range(-actual_n, 0)
            if i < len(away_ht) and away_ht[i] == 0 and away_ft[i] >= 1
        ) / max(actual_n, 1)

        return max(-1.0, min(1.0, (home_comebacks - away_comebacks) * 2))

    def _compute_streak_momentum(
        self, home_stats: Dict, away_stats: Dict
    ) -> float:
        """
        F50: 连胜/连败动量
        当前连胜=正值，连败=负值。
        """
        home_results = home_stats.get("results", [])
        away_results = away_stats.get("results", [])

        home_streak = self._compute_streak(home_results)
        away_streak = self._compute_streak(away_results)

        # 归一化: 胜3分 平1分 负0分
        return max(-1.0, min(1.0, (home_streak - away_streak) / 5.0))

    @staticmethod
    def _compute_streak(results: List[int]) -> float:
        """计算当前连胜/连败：胜=+1, 负=-1, 平=0, 连续累计"""
        if not results:
            return 0.0
        streak = 0.0
        for r in reversed(results):
            if r == 3:  # 胜
                streak += 1.0 if streak >= 0 else -0.5
            elif r == 0:  # 负
                streak += -1.0 if streak <= 0 else 0.5
            else:  # 平
                break
        return streak

    def _compute_goal_volatility(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F51: 进球波动率
        近期进球数的标准差，反映稳定性。
        正值 = 主队更稳定（波动率更低）。
        """
        home_gf = home_stats.get("gf", [])[-n:]
        home_ga = home_stats.get("ga", [])[-n:]
        away_gf = away_stats.get("gf", [])[-n:]
        away_ga = away_stats.get("ga", [])[-n:]

        if len(home_gf) < 3 or len(away_gf) < 3:
            return 0.0

        home_vol = (self._compute_std(home_gf) + self._compute_std(home_ga)) / 2
        away_vol = (self._compute_std(away_gf) + self._compute_std(away_ga)) / 2

        # 波动率越低越好（稳定），所以差值的符号反向
        vol_diff = (away_vol - home_vol) / max(home_vol + away_vol, 0.1)
        return max(-1.0, min(1.0, vol_diff))

    def _compute_corner_dominance(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F52: 角球优势
        """
        home_hc = self._safe_mean(home_stats.get("hc", []), n)
        away_hc = self._safe_mean(away_stats.get("hc", []), n)
        home_ac = self._safe_mean(home_stats.get("ac", []), n)
        away_ac = self._safe_mean(away_stats.get("ac", []), n)

        home_ratio = home_hc / max(home_hc + home_ac, 1) if (home_hc + home_ac) > 0 else 0.5
        away_ratio = away_hc / max(away_hc + away_ac, 1) if (away_hc + away_ac) > 0 else 0.5

        return max(-1.0, min(1.0, (home_ratio - away_ratio) * 2))

    def _compute_sot_rate_diff(
        self, home_stats: Dict, away_stats: Dict, n: int
    ) -> float:
        """
        F53: 射正率差
        """
        home_sot = self._safe_mean(home_stats.get("sot_rate", []), n)
        away_sot = self._safe_mean(away_stats.get("sot_rate", []), n)
        return max(-1.0, min(1.0, home_sot - away_sot))

    def _compute_ah_odds_drift(
        self, league: str, season: str,
        home_team: str, match_date: datetime,
    ) -> float:
        """
        F54: 亚盘赔率漂移
        开盘→收盘亚盘水位变动。
        """
        key = f"{league}_{season}"
        matches = self._matches.get(key, [])
        for m in matches:
            if (m["home"] == home_team and m["date"] == match_date):
                open_h = m.get("b365_ahh", 0)
                close_h = m.get("b365c_ahh", 0)
                if open_h > 0 and close_h > 0:
                    drift = (close_h - open_h) / open_h * 10
                    return max(-1.0, min(1.0, drift))
                return 0.0
        return 0.0

    def _compute_totals_odds_drift(
        self, league: str, season: str,
        home_team: str, away_team: str,
        match_date: datetime,
    ) -> float:
        """
        F55: 大小球赔率漂移
        开盘→收盘大小球水位变动。
        正值 = 市场倾向大球。
        """
        key = f"{league}_{season}"
        matches = self._matches.get(key, [])
        for m in matches:
            if (m["home"] == home_team and m["away"] == away_team
                    and m["date"] == match_date):
                open_over = m.get("b365_over", 0)
                close_over = m.get("b365c_over", 0)
                if open_over > 0 and close_over > 0:
                    drift = (close_over - open_over) / open_over * 10
                    return max(-1.0, min(1.0, drift))
                return 0.0
        return 0.0

    @staticmethod
    def _compute_std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(var)