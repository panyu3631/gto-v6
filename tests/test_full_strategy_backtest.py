"""
GTO-GameFlow v5.11 全策略集成回测

覆盖:
1. 胜平负 (1X2)
2. 亚洲让球盘 (Asian Handicap)
3. 大小球 (Over/Under)
4. 串关 (2串1 Parlay)

每场比赛:
- 4模型集成 → 比分概率矩阵
- 从矩阵推导 1X2 / AH / O/U 概率
- 找到各市场有价值的投注
- 最有价值的投注作为串关腿

用法:
    python3 tests/test_full_strategy_backtest.py
"""
import sys, os, math, argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine.elo_cold_start import EloColdStart
from src.engine.unified_probability_engine import poisson_pmf
from src.config.league_params import get_league_params
from tests.test_csv_backtest import RawMatch, load_all_matches


# ================================================================
# 数据模型
# ================================================================
@dataclass
class BetRecord:
    strategy: str       # 1x2 / ah / ou / parlay
    selection: str      # home_win / draw / away_win / home_-0.5 / over_2.5 等
    odds: float
    model_prob: float
    implied_prob: float
    value: float
    stake: float
    actual_result: str
    won: bool
    profit: float
    match_id: str = ""
    league: str = ""
    date: datetime = field(default_factory=datetime.now)
    # 串关专用
    parlay_legs: List[Dict] = field(default_factory=list)


# ================================================================
# 模型 (复用 ensemble)
# ================================================================
def _dc_tau(x, y, lh, la, rho):
    if x >= 2 or y >= 2: return 1.0
    if x == 0 and y == 0: return max(0.0, 1.0 - lh * la * rho)
    elif x == 0 and y == 1: return max(0.0, 1.0 + lh * rho)
    elif x == 1 and y == 0: return max(0.0, 1.0 + la * rho)
    elif x == 1 and y == 1: return max(0.0, 1.0 - rho)
    return 1.0


class EloDCModel:
    def __init__(self, league):
        self.elo = EloColdStart(default_elo=1500.0, k=20, home_advantage=65)
        self.params = get_league_params(league)
        self.league = league
        self.rho = {"premier_league":-0.08,"la_liga":-0.10,"bundesliga":-0.06,"serie_a":-0.13,"ligue_1":-0.12}.get(league,-0.10)

    def predict_matrix(self, m):
        """返回比分概率矩阵 {(h,a): prob}"""
        eh = self.elo.get_elo(self.league, m.home_team)
        ea = self.elo.get_elo(self.league, m.away_team)
        base = self.params.avg_goals / 2.0
        ef = (eh - ea) / 400.0
        lh = base * (1.0 + ef*0.5 + self.params.home_advantage*0.3)
        la = base * (1.0 - ef*0.5)
        s = self.params.avg_goals * 0.93 / max(lh+la, 0.1)
        lh, la = max(0.3, min(4.0, lh*s)), max(0.3, min(4.0, la*s))
        mat = {}
        for h in range(9):
            for a in range(9):
                mat[(h,a)] = poisson_pmf(h,lh)*poisson_pmf(a,la)*_dc_tau(h,a,lh,la,self.rho)
        t = sum(mat.values())
        if t > 0:
            for k in mat: mat[k] /= t
        return mat

    def predict_1x2(self, m):
        mat = self.predict_matrix(m)
        return {
            "home": sum(v for (h,a),v in mat.items() if h>a),
            "draw": sum(v for (h,a),v in mat.items() if h==a),
            "away": sum(v for (h,a),v in mat.items() if h<a),
        }

    def update(self, m):
        self.elo._process_single_match({"league_id":self.league,"home_team":m.home_team,"away_team":m.away_team,"fthg":m.fthg,"ftag":m.ftag,"ftr":m.ftr,"date":m.date})


class PoissonModel:
    def __init__(self, league):
        self.params = get_league_params(league)
        self.ts = defaultdict(lambda: {"hgf":[],"hga":[],"agf":[],"aga":[]})

    def predict_1x2(self, m):
        hs, as_ = self.ts[m.home_team], self.ts[m.away_team]
        if len(hs["hgf"]) < 5 or len(as_["agf"]) < 5: return None
        lg = self.params.avg_goals / 2.0
        lh = (sum(hs["hgf"][-10:])/len(hs["hgf"][-10:])) * (sum(as_["aga"][-10:])/len(as_["aga"][-10:])) / max(lg,0.5)
        la = (sum(as_["agf"][-10:])/len(as_["agf"][-10:])) * (sum(hs["hga"][-10:])/len(hs["hga"][-10:])) / max(lg,0.5)
        lh, la = max(0.3,min(4.0,lh)), max(0.3,min(4.0,la))
        ph=pd=pa=0.0
        for h in range(9):
            for a in range(9):
                p = poisson_pmf(h,lh)*poisson_pmf(a,la)
                if h>a: ph+=p
                elif h==a: pd+=p
                else: pa+=p
        t=ph+pd+pa
        return {"home":ph/t,"draw":pd/t,"away":pa/t} if t>0 else None

    def update(self, m):
        hs, as_ = self.ts[m.home_team], self.ts[m.away_team]
        hs["hgf"].append(m.fthg); hs["hga"].append(m.ftag)
        as_["agf"].append(m.ftag); as_["aga"].append(m.fthg)
        for d in [hs,as_]:
            for k in d: d[k]=d[k][-30:]


class MarketModel:
    def predict_1x2(self, m):
        if m.b365h<=1 or m.b365d<=1 or m.b365a<=1: return None
        mg=1.0/m.b365h+1.0/m.b365d+1.0/m.b365a
        return {"home":(1.0/m.b365h)/mg,"draw":(1.0/m.b365d)/mg,"away":(1.0/m.b365a)/mg}
    def update(self, m): pass


class FormModel:
    def __init__(self):
        self.ts = defaultdict(lambda: {"r":[],"gf":[],"ga":[],"hr":[],"ar":[]})
    def predict_1x2(self, m):
        hs, as_ = self.ts[m.home_team], self.ts[m.away_team]
        if len(hs["r"])<5 or len(as_["r"])<5: return None
        hf = sum(hs["hr"][-5:])/max(len(hs["hr"][-5:]),1) if hs["hr"] else sum(hs["r"][-5:])/5
        af = sum(as_["ar"][-5:])/max(len(as_["ar"][-5:]),1) if as_["ar"] else sum(as_["r"][-5:])/5
        hg = sum(hs["gf"][-5:])/5; ag = sum(as_["gf"][-5:])/5
        score_h = hf*0.4 + sum(hs["r"][-5:])/5*0.3 + (hg-sum(hs["ga"][-5:])/5)*0.3 + 0.3
        score_a = af*0.4 + sum(as_["r"][-5:])/5*0.3 + (ag-sum(as_["ga"][-5:])/5)*0.3
        d = score_h - score_a
        ph = 1.0/(1.0+math.exp(-d*1.5))
        pa = 1.0/(1.0+math.exp(d*1.5))
        pd = max(0.1, 1.0-ph-pa)
        t=ph+pd+pa
        return {"home":ph/t,"draw":pd/t,"away":pa/t}
    def update(self, m):
        for team,gf,ga,ih in [(m.home_team,m.fthg,m.ftag,True),(m.away_team,m.ftag,m.fthg,False)]:
            s=self.ts[team]; r=3 if gf>ga else (1 if gf==ga else 0)
            s["r"].append(r); s["r"]=s["r"][-30:]
            s["gf"].append(gf); s["gf"]=s["gf"][-30:]
            s["ga"].append(ga); s["ga"]=s["ga"][-30:]
            if ih: s["hr"].append(r); s["hr"]=s["hr"][-20:]
            else: s["ar"].append(r); s["ar"]=s["ar"][-20:]


# ================================================================
# 比分矩阵 → 各市场概率
# ================================================================
def matrix_to_1x2(mat):
    return {
        "home": sum(v for (h,a),v in mat.items() if h>a),
        "draw": sum(v for (h,a),v in mat.items() if h==a),
        "away": sum(v for (h,a),v in mat.items() if h<a),
    }


def matrix_to_ah(mat, line):
    """从比分矩阵推导亚盘概率"""
    home_cov = 0.0
    is_quarter = abs(line*4 - round(line*4)) < 0.01 and (round(line*4) % 2 == 1)
    if is_quarter:
        lower = math.floor(line*2)/2.0
        upper = math.ceil(line*2)/2.0
        home_cov = 0.5 * _ah_cover(mat, lower) + 0.5 * _ah_cover(mat, upper)
    else:
        home_cov = _ah_cover(mat, line)
    return {"home": home_cov, "away": 1.0 - home_cov}


def _ah_cover(mat, line):
    prob = 0.0
    is_int = abs(line - round(line)) < 0.01
    for (h,a), p in mat.items():
        diff = h - a
        if diff > line:
            prob += p
        elif is_int and diff == line:
            prob += p * 0.5
    return prob


def matrix_to_ou(mat, line):
    """从比分矩阵推导大小球概率"""
    over = under = exact = 0.0
    is_int = abs(line - round(line)) < 0.01
    for (h,a), p in mat.items():
        total = h + a
        if is_int and total == int(line):
            exact += p
        elif total > line:
            over += p
        elif total < line:
            under += p
    t = over + under + exact
    if t > 0:
        return {"over": over/t, "under": under/t, "exact": exact/t}
    return {"over": 0.33, "under": 0.33, "exact": 0.33}


# ================================================================
# 全策略回测引擎
# ================================================================
class FullStrategyBacktest:
    def __init__(self, league, bankroll=10000.0, min_history=30,
                 consensus=3, value_thresh=0.05, ah_lines=None, ou_lines=None):
        self.league = league
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.min_history = min_history
        self.consensus = consensus
        self.value_thresh = value_thresh
        self.ah_lines = ah_lines or [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        self.ou_lines = ou_lines or [1.5, 2.0, 2.5, 3.0, 3.5]

        self.m1 = EloDCModel(league)
        self.m2 = PoissonModel(league)
        self.m3 = MarketModel()
        self.m4 = FormModel()
        self.models = [self.m1, self.m2, self.m3, self.m4]

        self.bets: List[BetRecord] = []
        self.parlay_pool: List[Dict] = []
        self.pending_parlay_legs: List[Dict] = []  # 未结算的串关腿

    def run(self, matches):
        result = defaultdict(lambda: {"bets":0,"wins":0,"staked":0,"returned":0})
        self.bankroll = self.initial_bankroll
        peak = self.bankroll
        mdd = 0

        ln = {"premier_league":"英超","la_liga":"西甲","bundesliga":"德甲","serie_a":"意甲","ligue_1":"法甲"}.get(self.league,self.league)
        print(f"\n  ▶ {ln} ({len(matches)} 场)")

        for i, m in enumerate(matches):
            if i < self.min_history:
                self._update_all(m)
                continue
            if m.b365h <= 1:
                self._update_all(m)
                continue

            # 4模型获取1X2共识
            preds = {}
            for mod in self.models:
                p = mod.predict_1x2(m)
                if p: preds[id(mod)] = p

            if len(preds) < 2:
                self._update_all(m)
                continue

            dirs = ["home","draw","away"]
            votes = {d: 0 for d in dirs}
            probs_1x2 = {d: [] for d in dirs}
            for pred in preds.values():
                best = max(dirs, key=lambda d: pred[d])
                votes[best] += 1
                for d in dirs: probs_1x2[d].append(pred[d])

            best_dir = max(dirs, key=lambda d: votes[d])
            if votes[best_dir] < self.consensus:
                self._update_all(m)
                continue

            # 共识概率
            consensus_1x2 = {d: sum(probs_1x2[d])/len(probs_1x2[d]) for d in dirs}

            # 用 EloDC 的比分矩阵推导 AH 和 O/U
            mat = self.m1.predict_matrix(m)

            # 市场赔率
            mg = 1.0/m.b365h + 1.0/m.b365d + 1.0/m.b365a
            market_1x2 = {"home":(1.0/m.b365h)/mg, "draw":(1.0/m.b365d)/mg, "away":(1.0/m.b365a)/mg}
            odds_1x2 = {"home":m.b365h, "draw":m.b365d, "away":m.b365a}

            # === 策略1: 胜平负 (1X2) ===
            for d in dirs:
                value = consensus_1x2[d] - market_1x2[d]
                if value > self.value_thresh and votes[d] >= self.consensus:
                    stake = self._calc_stake(consensus_1x2[d], odds_1x2[d])
                    if stake >= 10:
                        rmap = {"H":"home","D":"draw","A":"away"}
                        won = d == rmap.get(m.ftr, "")
                        profit = stake*(odds_1x2[d]-1) if won else -stake
                        self._record_bet("1x2", d, odds_1x2[d], consensus_1x2[d],
                                        market_1x2[d], value, stake, m.ftr, won, profit, m, result)
                        # 加入串关池
                        self.parlay_pool.append({
                            "selection": d, "odds": odds_1x2[d],
                            "prob": consensus_1x2[d], "value": value,
                            "match": f"{m.home_team}_vs_{m.away_team}",
                            "league": self.league, "date": m.date,
                            "strategy": "1x2",
                            "actual_ftr": m.ftr,
                        })

            # === 策略2: 亚洲让球盘 ===
            # 收盘亚盘赔率 (简化: 用 B365 的 AH 数据)
            ah_line = m.ahh if m.ahh else 0
            if ah_line != 0 and abs(ah_line) <= 2.5:
                ah_probs = matrix_to_ah(mat, ah_line)
                # 亚盘赔率通常 ~1.9 / ~1.9
                ah_odds_home = 1.90  # 简化
                ah_odds_away = 1.90
                ah_market_home = 0.526  # 1/1.9

                ah_value_home = ah_probs["home"] - ah_market_home
                ah_value_away = ah_probs["away"] - ah_market_home

                if ah_value_home > self.value_thresh:
                    stake = self._calc_stake(ah_probs["home"], ah_odds_home)
                    if stake >= 10:
                        # 结算: 根据实际净胜球
                        actual_diff = m.fthg - m.ftag
                        if ah_line < 0:  # 主让
                            won = actual_diff > abs(ah_line)
                        else:  # 客让
                            won = actual_diff > -ah_line
                        profit = stake*(ah_odds_home-1) if won else -stake
                        self._record_bet("ah", f"home_{ah_line:+.2f}", ah_odds_home,
                                        ah_probs["home"], ah_market_home, ah_value_home,
                                        stake, m.ftr, won, profit, m, result)
                        self.parlay_pool.append({
                            "selection": f"ah_home_{ah_line:+.2f}", "odds": ah_odds_home,
                            "prob": ah_probs["home"], "value": ah_value_home,
                            "match": f"{m.home_team}_vs_{m.away_team}",
                            "league": self.league, "date": m.date, "strategy": "ah",
                            "actual_ftr": m.ftr,
                        })

            # === 策略3: 大小球 ===
            for line in self.ou_lines:
                ou_probs = matrix_to_ou(mat, line)
                ou_odds_over = 1.90
                ou_odds_under = 1.90
                ou_market = 0.526

                ou_value_over = ou_probs["over"] - ou_market
                ou_value_under = ou_probs["under"] - ou_market

                if ou_value_over > self.value_thresh:
                    stake = self._calc_stake(ou_probs["over"], ou_odds_over)
                    if stake >= 10:
                        total = m.fthg + m.ftag
                        won = total > line
                        profit = stake*(ou_odds_over-1) if won else -stake
                        self._record_bet("ou", f"over_{line}", ou_odds_over,
                                        ou_probs["over"], ou_market, ou_value_over,
                                        stake, m.ftr, won, profit, m, result)
                        self.parlay_pool.append({
                            "selection": f"over_{line}", "odds": ou_odds_over,
                            "prob": ou_probs["over"], "value": ou_value_over,
                            "match": f"{m.home_team}_vs_{m.away_team}",
                            "league": self.league, "date": m.date, "strategy": "ou",
                            "actual_ftr": m.ftr,
                        })

                if ou_value_under > self.value_thresh:
                    stake = self._calc_stake(ou_probs["under"], ou_odds_under)
                    if stake >= 10:
                        total = m.fthg + m.ftag
                        won = total < line
                        profit = stake*(ou_odds_under-1) if won else -stake
                        self._record_bet("ou", f"under_{line}", ou_odds_under,
                                        ou_probs["under"], ou_market, ou_value_under,
                                        stake, m.ftr, won, profit, m, result)

            # === 策略4: 串关 ===
            # 当前比赛有价值 → 尝试与之前的未结算腿配对
            current_legs = []  # 当前比赛产生的有价值投注
            for d in dirs:
                value = consensus_1x2[d] - market_1x2[d]
                if value > self.value_thresh and votes[d] >= self.consensus:
                    current_legs.append({
                        "selection": d, "odds": odds_1x2[d],
                        "prob": consensus_1x2[d], "value": value,
                        "match": f"{m.home_team}_vs_{m.away_team}",
                        "league": self.league, "strategy": "1x2",
                        "actual_ftr": m.ftr,
                    })

            # 用当前比赛结果结算之前的待处理串关
            self._settle_pending_parlays(m, result)

            # 将当前比赛的有价值投注加入待处理池
            self.pending_parlay_legs.extend(current_legs)
            # 只保留最近的腿
            self.pending_parlay_legs = self.pending_parlay_legs[-30:]

            # 尝试用当前腿 + 之前的腿构建新串关
            if current_legs and len(self.pending_parlay_legs) >= 2:
                self._try_parlay(m, result)

            # 更新回撤
            if self.bankroll > peak: peak = self.bankroll
            dd = (peak - self.bankroll) / peak if peak > 0 else 0
            if dd > mdd: mdd = dd

            self._update_all(m)

        return dict(result), mdd

    def _calc_stake(self, prob, odds):
        b = odds - 1
        q = 1 - prob
        fk = (b*prob - q) / b if b > 0 else 0
        stake = self.bankroll * fk * 0.25
        return min(stake, self.bankroll * 0.03)

    def _record_bet(self, strategy, selection, odds, model_prob, implied_prob,
                    value, stake, actual, won, profit, m, result):
        self.bankroll += profit
        bet = BetRecord(
            strategy=strategy, selection=selection, odds=odds,
            model_prob=model_prob, implied_prob=implied_prob,
            value=value, stake=round(stake,2), actual_result=actual,
            won=won, profit=round(profit,2),
            match_id=f"{m.home_team}_vs_{m.away_team}",
            league=self.league, date=m.date,
        )
        self.bets.append(bet)
        result[strategy]["bets"] += 1
        result[strategy]["staked"] += stake
        if won:
            result[strategy]["wins"] += 1
            result[strategy]["returned"] += stake * odds

    def _settle_pending_parlays(self, current_match, result):
        """
        用当前比赛的真实结果结算之前的待处理串关。

        关键: 串关的腿来自之前已结束的比赛，当前比赛的真实结果
        用于结算包含当前比赛腿的串关。

        串关规则:
        - 串关由2条腿组成，来自2场不同的比赛
        - 当一场比赛结束后，检查是否有包含该比赛腿的待处理串关
        - 如果两条腿都已结算，计算串关结果
        """
        current_match_id = f"{current_match.home_team}_vs_{current_match.away_team}"

        # 检查待处理池中是否有当前比赛的腿
        legs_for_current = [
            l for l in self.pending_parlay_legs
            if l["match"] == current_match_id
        ]

        if not legs_for_current:
            return

        # 用当前比赛的真实结果检查该腿是否命中
        for leg in legs_for_current:
            sel = leg["selection"]
            actual = current_match.ftr

            if sel == "home":
                leg["won"] = (actual == "H")
            elif sel == "draw":
                leg["won"] = (actual == "D")
            elif sel == "away":
                leg["won"] = (actual == "A")
            else:
                # AH/O/U 暂用概率估计
                leg["won"] = leg["prob"] >= 0.55
            leg["settled"] = True

        # 查找已结算的腿，尝试配对成串关
        settled_legs = [l for l in self.pending_parlay_legs if l.get("settled")]

        if len(settled_legs) >= 2:
            # 按价值排序，取最高的2个不同比赛的腿
            sorted_legs = sorted(settled_legs, key=lambda x: x["value"], reverse=True)
            parlay_legs = []
            seen = set()
            for leg in sorted_legs:
                if leg["match"] not in seen:
                    parlay_legs.append(leg)
                    seen.add(leg["match"])
                if len(parlay_legs) == 2:
                    break

            if len(parlay_legs) == 2:
                # 计算串关
                combined_odds = parlay_legs[0]["odds"] * parlay_legs[1]["odds"]
                combined_prob = parlay_legs[0]["prob"] * parlay_legs[1]["prob"] * 0.9
                value = combined_prob - (1.0 / combined_odds)

                if value > 0.02:
                    b = combined_odds - 1
                    fk = (b * combined_prob - (1-combined_prob)) / b if b > 0 else 0
                    stake = self.bankroll * fk * 0.15
                    stake = min(stake, self.bankroll * 0.005)

                    if stake >= 5:
                        # 结算: 两条腿都必须命中
                        parlay_won = all(l.get("won", False) for l in parlay_legs)
                        profit = stake * (combined_odds - 1) if parlay_won else -stake

                        self.bankroll += profit
                        bet = BetRecord(
                            strategy="parlay", selection="2-leg",
                            odds=round(combined_odds, 2),
                            model_prob=combined_prob,
                            implied_prob=1.0/combined_odds,
                            value=value, stake=round(stake, 2),
                            actual_result="W" if parlay_won else "L",
                            won=parlay_won, profit=round(profit, 2),
                            match_id=f"parlay_{len(self.bets)}",
                            league=self.league,
                            parlay_legs=[{"match":l["match"],"sel":l["selection"],"odds":l["odds"],"won":l.get("won")} for l in parlay_legs],
                        )
                        self.bets.append(bet)
                        result["parlay"]["bets"] += 1
                        result["parlay"]["staked"] += stake
                        if parlay_won:
                            result["parlay"]["wins"] += 1
                            result["parlay"]["returned"] += stake * combined_odds

                        # 从池中移除已用的腿
                        for l in parlay_legs:
                            if l in self.pending_parlay_legs:
                                self.pending_parlay_legs.remove(l)

    def _try_parlay(self, m, result):
        """不再直接构建串关 — 由 _settle_pending_parlays 处理"""
        pass

    def _update_all(self, m):
        for mod in self.models: mod.update(m)


# ================================================================
# 主入口
# ================================================================
def run_full_backtest(leagues, seasons, bankroll=10000.0):
    sep = "="*70
    print(f"\n{'#'*70}")
    print(f"  GTO v5.11 全策略集成回测")
    print(f"  策略: 1X2 + 亚盘 + 大小球 + 串关(2串1)")
    print(f"{'#'*70}")

    all_m = load_all_matches(leagues, seasons)
    total = sum(len(m) for m in all_m.values())
    print(f"  共 {total} 场")

    grand = defaultdict(lambda: {"bets":0,"wins":0,"staked":0,"returned":0})

    for league in leagues:
        matches = all_m.get(league, [])
        if not matches: continue

        engine = FullStrategyBacktest(league, bankroll, consensus=4, value_thresh=0.05)
        result, mdd = engine.run(matches)

        ln = {"premier_league":"英超","la_liga":"西甲","bundesliga":"德甲","serie_a":"意甲","ligue_1":"法甲"}.get(league,league)
        print(f"\n{sep}")
        print(f"  {ln} 汇总")
        print(f"{sep}")

        for strat in ["1x2","ah","ou","parlay"]:
            r = result.get(strat, {"bets":0,"wins":0,"staked":0,"returned":0})
            if r["bets"] > 0:
                roi = (r["returned"]-r["staked"])/r["staked"] if r["staked"]>0 else 0
                wr = r["wins"]/r["bets"]
                sn = {"1x2":"胜平负","ah":"亚盘","ou":"大小球","parlay":"串关"}[strat]
                print(f"  {sn:8s}  投注={r['bets']:3d}  胜={r['wins']:3d}  胜率={wr:.1%}  ROI={roi:+.1%}")
                grand[strat]["bets"] += r["bets"]
                grand[strat]["wins"] += r["wins"]
                grand[strat]["staked"] += r["staked"]
                grand[strat]["returned"] += r["returned"]

        # 汇总
        total_bets = sum(r["bets"] for r in result.values())
        total_staked = sum(r["staked"] for r in result.values())
        total_returned = sum(r["returned"] for r in result.values())
        total_wins = sum(r["wins"] for r in result.values())
        if total_staked > 0:
            print(f"  {'合计':8s}  投注={total_bets:3d}  胜={total_wins:3d}  胜率={total_wins/total_bets:.1%}  ROI={(total_returned-total_staked)/total_staked:+.1%}  回撤={mdd:.1%}")

    # 全联汇总
    print(f"\n{'#'*70}")
    print(f"  全联汇总")
    print(f"{'#'*70}")
    for strat in ["1x2","ah","ou","parlay"]:
        r = grand[strat]
        if r["bets"] > 0:
            roi = (r["returned"]-r["staked"])/r["staked"] if r["staked"]>0 else 0
            sn = {"1x2":"胜平负","ah":"亚盘","ou":"大小球","parlay":"串关"}[strat]
            print(f"  {sn:8s}  投注={r['bets']:4d}  胜={r['wins']:4d}  胜率={r['wins']/r['bets']:.1%}  ROI={roi:+.1%}")

    g_bets = sum(r["bets"] for r in grand.values())
    g_staked = sum(r["staked"] for r in grand.values())
    g_returned = sum(r["returned"] for r in grand.values())
    g_wins = sum(r["wins"] for r in grand.values())
    if g_staked > 0:
        print(f"  {'总计':8s}  投注={g_bets:4d}  胜={g_wins:4d}  胜率={g_wins/g_bets:.1%}  ROI={(g_returned-g_staked)/g_staked:+.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--leagues",nargs="+",default=["premier_league","la_liga","bundesliga","serie_a","ligue_1"])
    p.add_argument("--seasons",nargs="+",default=["2021-22","2022-23","2023-24"])
    p.add_argument("--bankroll",type=float,default=10000.0)
    a = p.parse_args()
    run_full_backtest(a.leagues, a.seasons, a.bankroll)
