"""GTO v5.11 多模型集成回测 (路径4)"""
import sys, os, math, argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine.elo_cold_start import EloColdStart
from src.engine.unified_probability_engine import poisson_pmf
from src.config.league_params import get_league_params
from tests.test_csv_backtest import RawMatch, load_all_matches

def _dc_tau(x, y, lh, la, rho):
    if x >= 2 or y >= 2: return 1.0
    if x == 0 and y == 0: return max(0.0, 1.0 - lh * la * rho)
    elif x == 0 and y == 1: return max(0.0, 1.0 + lh * rho)
    elif x == 1 and y == 0: return max(0.0, 1.0 + la * rho)
    elif x == 1 and y == 1: return max(0.0, 1.0 - rho)
    return 1.0

class EloDCModel:
    def __init__(self, league):
        self.league = league
        self.elo = EloColdStart(default_elo=1500.0, k=20, home_advantage=65)
        self.params = get_league_params(league)
        self.rho = {"premier_league":-0.08,"la_liga":-0.10,"bundesliga":-0.06,"serie_a":-0.13,"ligue_1":-0.12}.get(league,-0.10)
    def predict(self, m):
        eh = self.elo.get_elo(self.league, m.home_team)
        ea = self.elo.get_elo(self.league, m.away_team)
        base = self.params.avg_goals / 2.0
        ef = (eh - ea) / 400.0
        lh = base * (1.0 + ef*0.5 + self.params.home_advantage*0.3)
        la = base * (1.0 - ef*0.5)
        s = self.params.avg_goals * 0.93 / max(lh+la, 0.1)
        lh, la = max(0.3,min(4.0,lh*s)), max(0.3,min(4.0,la*s))
        mat = {}
        for h in range(9):
            for a in range(9):
                mat[(h,a)] = poisson_pmf(h,lh)*poisson_pmf(a,la)*_dc_tau(h,a,lh,la,self.rho)
        t = sum(mat.values())
        if t > 0:
            for k in mat: mat[k] /= t
        return {"home": sum(v for (h,a),v in mat.items() if h>a),
                "draw": sum(v for (h,a),v in mat.items() if h==a),
                "away": sum(v for (h,a),v in mat.items() if h<a)}
    def update(self, m):
        self.elo._process_single_match({"league_id":self.league,"home_team":m.home_team,"away_team":m.away_team,"fthg":m.fthg,"ftag":m.ftag,"ftr":m.ftr,"date":m.date})

class PoissonModel:
    def __init__(self, league):
        self.league = league
        self.params = get_league_params(league)
        self.ts = defaultdict(lambda: {"hgf":[],"hga":[],"agf":[],"aga":[]})
    def predict(self, m):
        hs, as_ = self.ts[m.home_team], self.ts[m.away_team]
        if len(hs["hgf"]) < 8 or len(as_["agf"]) < 8: return None
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
    def predict(self, m):
        if m.b365h<=1 or m.b365d<=1 or m.b365a<=1: return None
        mg=1.0/m.b365h+1.0/m.b365d+1.0/m.b365a
        return {"home":(1.0/m.b365h)/mg,"draw":(1.0/m.b365d)/mg,"away":(1.0/m.b365a)/mg}
    def update(self, m): pass

class FormModel:
    def __init__(self):
        self.ts = defaultdict(lambda: {"r":[],"gf":[],"ga":[],"hr":[],"ar":[]})
    def predict(self, m):
        hs, as_ = self.ts[m.home_team], self.ts[m.away_team]
        if len(hs["r"])<10 or len(as_["r"])<10: return None
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
            s=self.ts[team]
            r=3 if gf>ga else (1 if gf==ga else 0)
            s["r"].append(r); s["r"]=s["r"][-30:]
            s["gf"].append(gf); s["gf"]=s["gf"][-30:]
            s["ga"].append(ga); s["ga"]=s["ga"][-30:]
            if ih: s["hr"].append(r); s["hr"]=s["hr"][-20:]
            else: s["ar"].append(r); s["ar"]=s["ar"][-20:]

def run_ensemble(leagues, seasons, bankroll=10000.0, consensus=3, value_thresh=0.05):
    sep = '#'*70
    print(f"\n{sep}\n  GTO v5.11 多模型集成回测\n{sep}")
    all_m = load_all_matches(leagues, seasons)
    total = sum(len(m) for m in all_m.values())
    print(f"  共 {total} 场")

    all_r = {}
    tb=tw=ts=tr=0

    for league in leagues:
        matches = all_m.get(league, [])
        if not matches: continue
        ln = {"premier_league":"英超","la_liga":"西甲","bundesliga":"德甲","serie_a":"意甲","ligue_1":"法甲"}.get(league,league)

        m1 = EloDCModel(league)
        m2 = PoissonModel(league)
        m3 = MarketModel()
        m4 = FormModel()
        models = [m1, m2, m3, m4]

        # Warm up
        for m in matches[:50]:
            for mod in models: mod.update(m)

        bets = []
        bal = bankroll
        peak = bankroll
        mdd = 0

        print(f"  {ln}: {len(matches)}场 共识={consensus}/4 阈值={value_thresh:.0%}")

        for i, m in enumerate(matches[50:], 50):
            if m.b365h <= 1:
                for mod in models: mod.update(m)
                continue

            preds = {}
            for mod in models:
                p = mod.predict(m)
                if p: preds[id(mod)] = p

            if len(preds) < 2:
                for mod in models: mod.update(m)
                continue

            dirs = ["home","draw","away"]
            votes = {d: 0 for d in dirs}
            probs = {d: [] for d in dirs}

            for pred in preds.values():
                best = max(dirs, key=lambda d: pred[d])
                votes[best] += 1
                for d in dirs: probs[d].append(pred[d])

            best_dir = max(dirs, key=lambda d: votes[d])
            if votes[best_dir] < consensus:
                for mod in models: mod.update(m)
                continue

            cp = sum(probs[best_dir]) / len(probs[best_dir])
            odds_map = {"home": m.b365h, "draw": m.b365d, "away": m.b365a}
            mg = 1.0/m.b365h + 1.0/m.b365d + 1.0/m.b365a
            ip = (1.0/odds_map[best_dir]) / mg

            if cp - ip < value_thresh:
                for mod in models: mod.update(m)
                continue

            odds = odds_map[best_dir]
            b = odds - 1
            fk = (b*cp - (1-cp)) / b if b > 0 else 0
            stake = min(bal * fk * 0.25, bal * 0.03)
            if stake < 10:
                for mod in models: mod.update(m)
                continue

            rmap = {"H":"home","D":"draw","A":"away"}
            won = best_dir == rmap.get(m.ftr, "")
            profit = stake*(odds-1) if won else -stake
            bal += profit
            bets.append({"won":won,"profit":profit,"stake":stake,"votes":votes[best_dir],"dir":best_dir})

            if bal > peak: peak = bal
            dd = (peak-bal)/peak if peak>0 else 0
            if dd > mdd: mdd = dd

            for mod in models: mod.update(m)

        if bets:
            ws = sum(1 for b in bets if b["won"])
            ss = sum(b["stake"] for b in bets)
            rs = sum(b["stake"]*(b["profit"]/b["stake"]+1) for b in bets if b["won"])
            roi = (rs-ss)/ss if ss>0 else 0
            print(f"  投注={len(bets)} 胜={ws} 胜率={ws/len(bets):.1%} ROI={roi:+.1%} 回撤={mdd:.1%}")

            # 按共识级别
            by_v = defaultdict(lambda: {"b":0,"w":0})
            for b in bets:
                by_v[b["votes"]]["b"] += 1
                if b["won"]: by_v[b["votes"]]["w"] += 1
            for v in sorted(by_v):
                d = by_v[v]
                print(f"    {v}票共识: {d["b"]}注 胜率={d["w"]/d["b"]:.1%}" if d["b"]>0 else "")

            all_r[league] = {"bets":len(bets),"wins":ws,"staked":ss,"returned":rs,"mdd":mdd}
            tb += len(bets); tw += ws; ts += ss; tr += rs
        else:
            print(f"  无投注")
            all_r[league] = {"bets":0,"wins":0,"staked":0,"returned":0,"mdd":0}

    roi = (tr-ts)/ts if ts>0 else 0
    if tb > 0:
        print(f"\n{sep}")
        print(f"  全联汇总: 投注={tb} 胜={tw} 胜率={tw/tb:.1%} ROI={roi:+.1%}")
    for lg, r in all_r.items():
        ln = {"premier_league":"英超","la_liga":"西甲","bundesliga":"德甲","serie_a":"意甲","ligue_1":"法甲"}.get(lg,lg)
        roi_l = (r["returned"]-r["staked"])/r["staked"] if r["staked"]>0 else 0
        if r["bets"] > 0:
            print(f"  {ln:8s}  投注={r['bets']:3d}  胜率={r['wins']/r['bets']:.1%}  ROI={roi_l:+.1%}")
        else:
            print(f"  {ln:8s}  无投注")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--leagues",nargs="+",default=["premier_league","la_liga","bundesliga","serie_a","ligue_1"])
    p.add_argument("--seasons",nargs="+",default=["2021-22","2022-23","2023-24"])
    p.add_argument("--bankroll",type=float,default=10000.0)
    a = p.parse_args()
    run_ensemble(a.leagues, a.seasons, a.bankroll)
