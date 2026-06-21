"""
GTO v6.0 — 回测适配器 (完整版 + 全部网页抓取数据)

因子激活策略:
- P0: F6/F7/F33/F37/F39/F40/F41 从历史数据+抓取积分榜计算
- P1: F27/F29/F26 从历史数据/静态常量计算
- P2: F9(xG) 从抓取数据, F12(天气) 从API, F15(教练) 从抓取数据
- P3: F2(伤病) 用默认值, F22/F24(新闻NLP) 跳过
- CSV: F42-F58 从CSV比赛统计计算
"""

from __future__ import annotations
import json, os, logging, math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

UEFA_COEFFICIENTS = {"premier_league": 1.0, "la_liga": 0.95, "bundesliga": 0.88, "serie_a": 0.85, "ligue_1": 0.78}
SCRAPED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'scraped')


class LeagueTable:
    def __init__(self):
        self.standings: Dict[str, Dict] = defaultdict(lambda: {"points": 0, "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0})
    def update(self, h, a, fthg, ftag, ftr):
        hd, ad = self.standings[h], self.standings[a]
        hd["played"] += 1; ad["played"] += 1
        hd["gf"] += fthg; hd["ga"] += ftag; hd["gd"] = hd["gf"] - hd["ga"]
        ad["gf"] += ftag; ad["ga"] += fthg; ad["gd"] = ad["gf"] - ad["ga"]
        if ftr == "H": hd["wins"] += 1; hd["points"] += 3; ad["losses"] += 1
        elif ftr == "D": hd["draws"] += 1; hd["points"] += 1; ad["draws"] += 1; ad["points"] += 1
        else: ad["wins"] += 1; ad["points"] += 3; hd["losses"] += 1
    def get_rank(self, t):
        for i, (n, _) in enumerate(sorted(self.standings.items(), key=lambda x: (-x[1]["points"], -x[1]["gd"]))):
            if n == t: return i + 1
        return 10
    def get_points(self, t): return self.standings[t]["points"]
    def get_gd(self, t): return self.standings[t]["gd"]


class BacktestAdapter:
    def __init__(self, league_id, use_production=True, initial_elo=1500.0, elo_k=20.0, home_adv=65.0):
        self.league_id = league_id
        self.use_production = use_production
        self.initial_elo = initial_elo
        self.elo_k = elo_k
        self.home_adv = home_adv
        self.elo: Dict[str, float] = defaultdict(lambda: initial_elo)
        self.match_history: Dict[str, List[Dict]] = defaultdict(list)
        self.h2h_history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.match_dates: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.league_table = LeagueTable()
        self._pipeline = None
        self._match_count = 0
        self._scraped_standings = None
        self._scraped_xg = None
        self._scraped_managers = None
        self._scraped_promoted = None
        self._team_mapping = None
        from src.config.league_params import get_league_params
        self.params = get_league_params(league_id)

    def _load_json(self, filename):
        path = os.path.join(SCRAPED_DIR, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _load_team_mapping(self):
        if self._team_mapping is not None:
            return self._team_mapping
        self._team_mapping = self._load_json("team_name_mapping.json") or {}
        return self._team_mapping

    def _map_team_name(self, name):
        mapping = self._load_team_mapping()
        # 反向映射: CSV名 -> 抓取名
        reverse = {v: k for k, v in mapping.items()}
        return reverse.get(name, name)

    def _load_scraped_standings(self):
        if self._scraped_standings is not None:
            return self._scraped_standings
        raw = self._load_json(f"standings_{self.league_id}_2023-24.json")
        if raw:
            mapping = self._load_team_mapping()
            self._scraped_standings = {}
            for scraped_name, data in raw.items():
                csv_name = mapping.get(scraped_name, scraped_name)
                self._scraped_standings[csv_name] = data
            return self._scraped_standings
        return None

    def _load_scraped_xg(self):
        if self._scraped_xg is not None:
            return self._scraped_xg
        raw = self._load_json("xg_data.json")
        if raw and self.league_id in raw:
            mapping = self._load_team_mapping()
            self._scraped_xg = {}
            for scraped_name, data in raw[self.league_id].items():
                csv_name = mapping.get(scraped_name, scraped_name)
                self._scraped_xg[csv_name] = data
            return self._scraped_xg
        return None

    def _load_scraped_managers(self):
        if self._scraped_managers is not None:
            return self._scraped_managers
        raw = self._load_json("manager_changes.json")
        if raw and self.league_id in raw:
            mapping = self._load_team_mapping()
            self._scraped_managers = {}
            for scraped_name, data in raw[self.league_id].items():
                csv_name = mapping.get(scraped_name, scraped_name)
                self._scraped_managers[csv_name] = data
            return self._scraped_managers
        return None

    def _load_scraped_promoted(self):
        if self._scraped_promoted is not None:
            return self._scraped_promoted
        raw = self._load_json("promoted_teams.json")
        if raw and self.league_id in raw:
            mapping = self._load_team_mapping()
            self._scraped_promoted = [mapping.get(t, t) for t in raw[self.league_id]]
            return self._scraped_promoted
        return None

    def _get_pipeline(self):
        if self._pipeline is None:
            from src.pipeline.orchestrator import GameFlowPipeline
            self._pipeline = GameFlowPipeline(self.league_id)
        return self._pipeline

    def predict(self, match):
        if self.use_production:
            return self._predict_production(match)
        return self._predict_simplified(match)

    def _predict_production(self, match):
        try:
            pipeline = self._get_pipeline()
            context = self._build_match_context(match)
            extra_data = self._build_extra_data(match)
            result = pipeline.run_stages_1_5(context, extra_data)
            if result.fused_probs:
                hp, dp, ap = result.fused_probs.prob_home, result.fused_probs.prob_draw, result.fused_probs.prob_away
            else:
                return self._predict_simplified(match)
            sm = result.poisson_score_matrix.matrix if result.poisson_score_matrix else {}
            hl = self.params.avg_goals / 2.0 * (1 + self.params.home_advantage)
            al = self.params.avg_goals / 2.0 * (1 - self.params.home_advantage * 0.3)
            return {"home_prob": hp, "draw_prob": dp, "away_prob": ap, "home_lambda": hl, "away_lambda": al, "score_matrix": sm, "source": "production"}
        except Exception as e:
            logger.warning(f"生产引擎失败: {e}")
            return self._predict_simplified(match)

    def _predict_simplified(self, match):
        eh, ea = self.elo[match.home_team], self.elo[match.away_team]
        base = self.params.avg_goals / 2.0
        ef = (eh - ea) / 400.0
        lh = base * (1.0 + ef * 0.5 + self.home_adv / 400.0)
        la = base * (1.0 - ef * 0.5)
        s = self.params.avg_goals * 0.93 / max(lh + la, 0.1)
        lh, la = max(0.3, min(4.0, lh * s)), max(0.3, min(4.0, la * s))
        rho = {"premier_league": -0.08, "la_liga": -0.10, "bundesliga": -0.06, "serie_a": -0.13, "ligue_1": -0.12}.get(self.league_id, -0.10)
        sm = {}
        for h in range(9):
            for a in range(9):
                sm[(h, a)] = self._poisson_pmf(h, lh) * self._poisson_pmf(a, la) * self._dc_tau(h, a, lh, la, rho)
        t = sum(sm.values())
        if t > 0:
            for k in sm: sm[k] /= t
        return {"home_prob": sum(v for (h, a), v in sm.items() if h > a), "draw_prob": sum(v for (h, a), v in sm.items() if h == a),
                "away_prob": sum(v for (h, a), v in sm.items() if h < a), "home_lambda": lh, "away_lambda": la, "score_matrix": sm, "source": "simplified"}

    def update(self, match):
        eh, ea = self.elo[match.home_team], self.elo[match.away_team]
        expected = 1.0 / (1.0 + 10 ** (-(eh + self.home_adv - ea) / 400.0))
        actual = 1.0 if match.ftr == 'H' else (0.5 if match.ftr == 'D' else 0.0)
        delta = self.elo_k * (1.0 + min(abs(match.fthg - match.ftag), 3) * 0.33) * (actual - expected)
        self.elo[match.home_team] += delta; self.elo[match.away_team] -= delta
        self.match_history[match.home_team].append({"gf": match.fthg, "ga": match.ftag, "result": match.ftr, "is_home": True, "hs": match.hs, "as_": match.as_, "hst": match.hst, "ast": match.ast, "hc": match.hc, "ac": match.ac, "hy": match.hy, "ay": match.ay, "hthg": match.hthg, "htag": match.htag})
        self.match_history[match.away_team].append({"gf": match.ftag, "ga": match.fthg, "result": match.ftr, "is_home": False, "hs": match.as_, "as_": match.hs, "hst": match.ast, "ast": match.hst, "hc": match.ac, "ac": match.hc, "hy": match.ay, "ay": match.hy, "hthg": match.htag, "htag": match.hthg})
        self.h2h_history[(match.home_team, match.away_team)].append(3 if match.ftr == 'H' else (1 if match.ftr == 'D' else 0))
        self.match_dates[match.home_team].append((match.date, 1))
        self.match_dates[match.away_team].append((match.date, 0))
        self.league_table.update(match.home_team, match.away_team, match.fthg, match.ftag, match.ftr)
        self._match_count += 1
        for t in [match.home_team, match.away_team]:
            self.match_history[t] = self.match_history[t][-30:]
            self.match_dates[t] = self.match_dates[t][-30:]
        self.h2h_history[(match.home_team, match.away_team)] = self.h2h_history[(match.home_team, match.away_team)][-10:]

    def _build_match_context(self, match):
        from src.data.models import MatchContext
        oh, od, oa = self._best(match, 'home'), self._best(match, 'draw'), self._best(match, 'away')
        return MatchContext(match_id=f"{match.league}_{match.date}_{match.home_team}_{match.away_team}",
            league_id=match.league, season="2023-24", matchday=0, kickoff_time=datetime.now(),
            home_team=match.home_team, away_team=match.away_team,
            home_elo=self.elo[match.home_team], away_elo=self.elo[match.away_team], odds_home=oh, odds_draw=od, odds_away=oa)

    def _best(self, match, d):
        if d == "home":
            for o in [match.psh, match.b365h, match.avgh, match.maxh]:
                if o > 1: return o
        elif d == "draw":
            for o in [match.psd, match.b365d, match.avgd, match.maxd]:
                if o > 1: return o
        elif d == "away":
            for o in [match.psa, match.b365a, match.avga, match.maxa]:
                if o > 1: return o
        return 2.5

    def _build_extra_data(self, match) -> Dict[str, Any]:
        hh = self.match_history.get(match.home_team, [])
        ah = self.match_history.get(match.away_team, [])
        rr = []
        for m in hh[-5:]:
            if m['result'] == 'H' or (m['result'] == 'A' and not m['is_home']): rr.append(3)
            elif m['result'] == 'D': rr.append(1)
            else: rr.append(0)
        h2h = self.h2h_history.get((match.home_team, match.away_team), [])
        oh, od, oa = self._best(match, 'home'), self._best(match, 'draw'), self._best(match, 'away')
        mg = 1.0/oh + 1.0/od + 1.0/oa
        mp = {"home": (1.0/oh)/mg, "draw": (1.0/od)/mg, "away": (1.0/oa)/mg}
        op = None
        if match.b365h > 1 and match.b365d > 1 and match.b365a > 1:
            omg = 1.0/match.b365h + 1.0/match.b365d + 1.0/match.b365a
            op = {"home": (1.0/match.b365h)/omg, "draw": (1.0/match.b365d)/omg, "away": (1.0/match.b365a)/omg}

        # 积分榜
        st = self._load_scraped_standings()
        if st:
            hr = st.get(match.home_team, {}).get("position", 10)
            ar = st.get(match.away_team, {}).get("position", 10)
            hp = st.get(match.home_team, {}).get("points", 0)
            ap = st.get(match.away_team, {}).get("points", 0)
        else:
            hr, ar = self.league_table.get_rank(match.home_team), self.league_table.get_rank(match.away_team)
            hp, ap = self.league_table.get_points(match.home_team), self.league_table.get_points(match.away_team)
        rd = ar - hr

        # F9: xG
        xg = self._load_scraped_xg()
        xg_diff = 0.0
        if xg:
            hxg = xg.get(match.home_team, {}).get("xg_for", 0)
            axg = xg.get(match.away_team, {}).get("xg_for", 0)
            hxga = xg.get(match.home_team, {}).get("xg_against", 0)
            axga = xg.get(match.away_team, {}).get("xg_against", 0)
            xg_diff = (hxg - hxga) - (axg - axga)

        # F15: 教练更替
        mc = self._load_scraped_managers()
        coach = 0.0
        if mc:
            if mc.get(match.home_team, {}).get("changed"): coach += 0.08
            if mc.get(match.away_team, {}).get("changed"): coach -= 0.05

        # F33/F37/F39
        mb = 0.0
        if hr <= 3: mb += 0.05
        if ar <= 3: mb -= 0.03
        if hr >= 18: mb += 0.08
        if ar >= 18: mb -= 0.05
        comp = 0.03 if 8 <= hr <= 14 else 0.0
        pa = (ap - hp) / 100.0

        # F6 赛程密度
        m7d = min(3, max(1, sum(1 for d, h in self.match_dates[match.home_team][-5:] if h)))

        # F40 升班马
        prom = self._load_scraped_promoted() or []
        pd = 0.0
        if match.home_team in prom and len(hh) < 15: pd = -0.05
        if match.away_team in prom and len(ah) < 15: pd = 0.03

        # F41 赛程优势
        sa = (len(ah[-5:]) - len(hh[-5:])) * 0.02 if len(hh) >= 2 and len(ah) >= 2 else 0.0

        # F27 进球分布修正
        pc = 0.0
        if len(hh) >= 5:
            goals = [m['gf'] for m in hh[-10:]]
            avg_g = sum(goals) / len(goals)
            var_g = sum((g - avg_g)**2 for g in goals) / len(goals)
            pc = (var_g - avg_g) / max(avg_g, 1) * 0.5

        # F29 大小球趋势
        tt = 0.0
        if hh:
            avg_t = sum(m['gf'] + m['ga'] for m in hh[-5:]) / max(len(hh[-5:]), 1)
            tt = (avg_t - self.params.avg_goals) * 0.5

        # F26 联赛强度
        ls = UEFA_COEFFICIENTS.get(self.league_id, 0.85) - 0.9

        # CSV比赛统计
        htm = 0.0
        if match.hthg > 0 or match.htag > 0:
            if match.ftr == 'H' and match.hthg > match.htag: htm = 0.5
            elif match.ftr == 'A' and match.htag > match.hthg: htm = -0.5
            elif match.hthg == match.htag: htm = 0.1 if match.ftr == 'D' else 0.0

        sed = 0.0
        if match.hs > 0 and match.as_ > 0: sed = (match.hst/max(match.hs,1) - match.ast/max(match.as_,1)) * 2.0

        terr = 0.0
        if match.hs > 0 and match.as_ > 0:
            sr = (match.hs - match.as_) / max(match.hs + match.as_, 1)
            cr = (match.hc - match.ac) / max(match.hc + match.ac, 1) if (match.hc + match.ac) > 0 else 0
            terr = sr * 0.5 + cr * 0.3

        disc = (match.ay - match.hy) / max(match.hy + match.ay, 1) if (match.hy + match.ay) > 0 else 0.0
        rb = (match.ay - match.hy) / max(match.hy + match.ay, 1) * 0.5 if (match.hy + match.ay) > 0 else 0.0

        cb = 0.0
        if match.hthg < match.htag and match.ftr == 'H': cb = 1.0
        elif match.htag < match.hthg and match.ftr == 'A': cb = -1.0

        st_val = 0.0
        for m in hh[-5:]:
            if m['is_home']:
                if m['result'] == 'H': st_val += 0.2
                elif m['result'] == 'A': st_val -= 0.2

        gv = 0.0
        if len(hh) >= 3:
            goals = [m['gf'] for m in hh[-5:]]
            avg_g = sum(goals) / len(goals)
            gv = (sum((g - avg_g)**2 for g in goals) / len(goals)) ** 0.5

        cd = (match.hc - match.ac) / max(match.hc + match.ac, 1) if (match.hc + match.ac) > 0 else 0.0
        sd = (match.hst/max(match.hs,1) - match.ast/max(match.as_,1)) if match.hs > 0 and match.as_ > 0 else 0.0

        od_val = 0.0
        os_val = 0.05
        if match.psch > 0 and match.b365h > 1:
            op2 = 1.0/match.b365h / (1.0/match.b365h + 1.0/match.b365d + 1.0/match.b365a)
            cp2 = 1.0/match.psch / (1.0/match.psch + 1.0/match.pscd + 1.0/match.psca)
            od_val = (cp2 - op2) * 5.0
        bko = [o for o in [match.bwh, match.iwh, match.whh, match.vch] if o > 1]
        if len(bko) >= 3:
            probs = [1.0/o for o in bko]
            avg_p = sum(probs) / len(probs)
            os_val = (sum((p - avg_p)**2 for p in probs) / len(probs)) ** 0.5

        ah_d = (match.b365cahh - match.b365ahh) * 0.5 if match.b365ahh > 0 and hasattr(match, 'b365cahh') and match.b365cahh > 0 else 0.0
        td = (1.0/match.b365c_over25 - 1.0/match.b365_over25) * 5.0 if match.b365_over25 > 0 and hasattr(match, 'b365c_over25') and match.b365c_over25 > 0 else 0.0

        dt = 0.0
        if match.hs > 0 and match.as_ > 0:
            if match.hs < 10 and match.as_ < 10: dt = 0.3
            elif match.hs > 15 and match.as_ > 15: dt = -0.2
        avg_t = (sum(m['gf']+m['ga'] for m in hh[-5:]) / max(len(hh[-5:]),1)) if hh else 2.5
        dge = 0.2 if 2.0 < avg_t < 2.8 else 0.0
        draws = sum(1 for m in hh[-10:] if m['result'] == 'D')
        dtd = (draws / len(hh[-10:]) - 0.25) * 2.0 if len(hh) >= 5 else 0.0

        return {
            "elo_diff": self.elo[match.home_team] - self.elo[match.away_team],
            "xi_rating": 6.0, "recent_results": rr if rr else [1.5]*5,
            "h2h_results": h2h[-5:] if h2h else [0]*5, "matches_7d": m7d, "rank_diff": rd,
            "goal_diff": self.league_table.get_gd(match.home_team) / 10.0, "xg_diff": xg_diff,
            "market_probs": mp, "opening_probs": op,
            "ref_yellow_rate": (match.hy + match.ay) / 2.0 if (match.hy + match.ay) > 0 else 3.5,
            "coach_change_effect": coach, "fatigue_penalty": 0.0, "rotation_risk": 0.0, "derby_boost": 0.0,
            "style_matchup_score": 0.5 + sed * 0.5, "streak_momentum": st_val, "player_form": 6.5,
            "market_sentiment": 0.0, "odds_std": os_val, "nlp_sentiment": 0.0, "time_decay_factor": 1.0,
            "league_strength_bias": ls, "poisson_correction": pc,
            "handicap_depth": abs(match.ahh) / 2.5 if match.ahh != 0 else 0.0,
            "totals_trend": tt, "value_signal": 0.0, "contrarian_signal": 0.0, "market_efficiency": 0.0,
            "motivation_boost": mb, "financial_gap_effect": 0.0, "winter_break_effect": 0.0,
            "christmas_fatigue": 0.0, "complacency_effect": comp, "streak_momentum_league": st_val,
            "position_advantage": pa, "promoted_team_delta": pd, "schedule_advantage": sa,
            "derby_intensity": 0.0, "ht_momentum": htm, "shot_eff_diff": sed,
            "territorial_dominance": terr, "discipline_index": disc, "odds_drift": od_val,
            "market_disagreement": os_val, "referee_home_bias": rb, "comeback_resilience": cb,
            "streak_momentum_enriched": st_val, "goal_volatility": gv, "corner_dominance": cd,
            "sot_rate_diff": sd, "ah_odds_drift": ah_d, "totals_odds_drift": td,
            "draw_tactical_matchup": dt, "draw_goal_expectancy": dge, "draw_team_tendency": dtd,
        }

    @staticmethod
    def _poisson_pmf(k, lam):
        if lam <= 0: return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    @staticmethod
    def _dc_tau(x, y, lh, la, rho):
        if x >= 2 or y >= 2: return 1.0
        if x == 0 and y == 0: return max(0.0, 1.0 - lh * la * rho)
        elif x == 0 and y == 1: return max(0.0, 1.0 + lh * rho)
        elif x == 1 and y == 0: return max(0.0, 1.0 + la * rho)
        elif x == 1 and y == 1: return max(0.0, 1.0 - rho)
        return 1.0
