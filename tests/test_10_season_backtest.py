"""
GTO v5.11 — 10赛季逐赛季回测 + 复盘总结

每个赛季单独结算，生成完整报告。
"""
import sys, os, math, argparse, json
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine.elo_cold_start import EloColdStart
from src.engine.unified_probability_engine import poisson_pmf
from src.config.league_params import get_league_params
from tests.test_csv_backtest import RawMatch, load_all_matches


# ================================================================
# 模型 (与 test_full_strategy_backtest 相同)
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
        return {"home": sum(v for (h,a),v in mat.items() if h>a),
                "draw": sum(v for (h,a),v in mat.items() if h==a),
                "away": sum(v for (h,a),v in mat.items() if h<a)}
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
# 比分矩阵推导
# ================================================================
def matrix_to_ah(mat, line):
    is_quarter = abs(line*4 - round(line*4)) < 0.01 and (round(line*4) % 2 == 1)
    if is_quarter:
        lower = math.floor(line*2)/2.0
        upper = math.ceil(line*2)/2.0
        return 0.5 * _ah_cover(mat, lower) + 0.5 * _ah_cover(mat, upper)
    return _ah_cover(mat, line)

def _ah_cover(mat, line):
    prob = 0.0
    is_int = abs(line - round(line)) < 0.01
    for (h,a), p in mat.items():
        diff = h - a
        if diff > line: prob += p
        elif is_int and diff == line: prob += p * 0.5
    return prob

def matrix_to_ou(mat, line):
    over = under = exact = 0.0
    is_int = abs(line - round(line)) < 0.01
    for (h,a), p in mat.items():
        total = h + a
        if is_int and total == int(line): exact += p
        elif total > line: over += p
        elif total < line: under += p
    t = over + under + exact
    if t > 0: return {"over": over/t, "under": under/t}
    return {"over": 0.5, "under": 0.5}


# ================================================================
# 逐赛季回测
# ================================================================
def run_single_season(league, season, warmup_matches=30, bankroll=10000.0):
    """单赛季回测，返回各策略结果"""
    all_m = load_all_matches([league], [season])
    matches = all_m.get(league, [])
    if not matches:
        return None

    m1 = EloDCModel(league)
    m2 = PoissonModel(league)
    m3 = MarketModel()
    m4 = FormModel()
    models = [m1, m2, m3, m4]

    stats = {
        "1x2": {"bets":0,"wins":0,"staked":0,"returned":0,"selections":{"home":0,"draw":0,"away":0,"home_w":0,"draw_w":0,"away_w":0}},
        "ou":   {"bets":0,"wins":0,"staked":0,"returned":0},
        "ah":   {"bets":0,"wins":0,"staked":0,"returned":0},
        "parlay":{"bets":0,"wins":0,"staked":0,"returned":0},
    }
    bal = bankroll
    peak = bal
    mdd = 0
    value_thresh = 0.05
    consensus = 3

    # Warm up
    for m in matches[:warmup_matches]:
        for mod in models: mod.update(m)

    parlay_legs_pending = []

    for i, m in enumerate(matches[warmup_matches:], warmup_matches):
        if m.b365h <= 1:
            for mod in models: mod.update(m)
            continue

        # 4模型预测
        preds = {}
        for mod in models:
            p = mod.predict_1x2(m)
            if p: preds[id(mod)] = p

        if len(preds) < 2:
            for mod in models: mod.update(m)
            continue

        dirs = ["home","draw","away"]
        votes = {d: 0 for d in dirs}
        probs_1x2 = {d: [] for d in dirs}
        for pred in preds.values():
            best = max(dirs, key=lambda d: pred[d])
            votes[best] += 1
            for d in dirs: probs_1x2[d].append(pred[d])

        consensus_1x2 = {d: sum(probs_1x2[d])/len(probs_1x2[d]) for d in dirs}
        mg = 1.0/m.b365h + 1.0/m.b365d + 1.0/m.b365a
        market_1x2 = {"home":(1.0/m.b365h)/mg, "draw":(1.0/m.b365d)/mg, "away":(1.0/m.b365a)/mg}
        odds_1x2 = {"home":m.b365h, "draw":m.b365d, "away":m.b365a}

        # === 1X2 ===
        for d in dirs:
            value = consensus_1x2[d] - market_1x2[d]
            if value > value_thresh and votes[d] >= consensus:
                b = odds_1x2[d] - 1
                fk = (b*consensus_1x2[d] - (1-consensus_1x2[d])) / b if b > 0 else 0
                stake = min(bal * fk * 0.25, bal * 0.03)
                if stake >= 10:
                    rmap = {"H":"home","D":"draw","A":"away"}
                    won = d == rmap.get(m.ftr, "")
                    profit = stake*(odds_1x2[d]-1) if won else -stake
                    bal += profit
                    stats["1x2"]["bets"] += 1
                    stats["1x2"]["staked"] += stake
                    stats["1x2"]["selections"][d] += 1
                    if won:
                        stats["1x2"]["wins"] += 1
                        stats["1x2"]["returned"] += stake * odds_1x2[d]
                        stats["1x2"]["selections"][d+"_w"] += 1

                    # 串关池
                    parlay_legs_pending.append({
                        "selection": d, "odds": odds_1x2[d],
                        "prob": consensus_1x2[d], "value": value,
                        "match": f"{m.home_team}_vs_{m.away_team}",
                        "actual_ftr": m.ftr, "won": None,
                    })

        # === 大小球 ===
        mat = m1.predict_matrix(m)
        for line in [1.5, 2.5, 3.5]:
            ou_probs = matrix_to_ou(mat, line)
            ou_odds = 1.90
            ou_market = 0.526

            for side in ["over","under"]:
                value = ou_probs[side] - ou_market
                if value > value_thresh:
                    b = ou_odds - 1
                    fk = (b*ou_probs[side] - (1-ou_probs[side])) / b if b > 0 else 0
                    stake = min(bal * fk * 0.25, bal * 0.03)
                    if stake >= 10:
                        total = m.fthg + m.ftag
                        if side == "over":
                            won = total > line
                        else:
                            won = total < line
                        profit = stake*(ou_odds-1) if won else -stake
                        bal += profit
                        stats["ou"]["bets"] += 1
                        stats["ou"]["staked"] += stake
                        if won:
                            stats["ou"]["wins"] += 1
                            stats["ou"]["returned"] += stake * ou_odds

        # === 亚盘 ===
        ah_line = m.ahh if m.ahh else 0
        if ah_line != 0 and abs(ah_line) <= 2.5:
            ah_prob_home = matrix_to_ah(mat, ah_line)
            ah_odds = 1.90
            ah_market = 0.526

            ah_value = ah_prob_home - ah_market
            if ah_value > value_thresh:
                b = ah_odds - 1
                fk = (b*ah_prob_home - (1-ah_prob_home)) / b if b > 0 else 0
                stake = min(bal * fk * 0.25, bal * 0.03)
                if stake >= 10:
                    actual_diff = m.fthg - m.ftag
                    if ah_line < 0:
                        won = actual_diff > abs(ah_line)
                    else:
                        won = actual_diff > -ah_line
                    profit = stake*(ah_odds-1) if won else -stake
                    bal += profit
                    stats["ah"]["bets"] += 1
                    stats["ah"]["staked"] += stake
                    if won:
                        stats["ah"]["wins"] += 1
                        stats["ah"]["returned"] += stake * ah_odds

        # === 串关结算 ===
        # 用当前比赛结果检查之前的待处理腿
        current_id = f"{m.home_team}_vs_{m.away_team}"
        for leg in parlay_legs_pending:
            if leg["match"] == current_id and leg["won"] is None:
                sel = leg["selection"]
                if sel == "home": leg["won"] = (m.ftr == "H")
                elif sel == "draw": leg["won"] = (m.ftr == "D")
                elif sel == "away": leg["won"] = (m.ftr == "A")

        # 尝试配对已结算的腿
        settled = [l for l in parlay_legs_pending if l["won"] is not None]
        if len(settled) >= 2:
            sorted_legs = sorted(settled, key=lambda x: x["value"], reverse=True)
            pair = []
            seen = set()
            for leg in sorted_legs:
                if leg["match"] not in seen:
                    pair.append(leg)
                    seen.add(leg["match"])
                if len(pair) == 2: break

            if len(pair) == 2:
                c_odds = pair[0]["odds"] * pair[1]["odds"]
                c_prob = pair[0]["prob"] * pair[1]["prob"] * 0.9
                c_value = c_prob - (1.0/c_odds)
                if c_value > 0.02:
                    b = c_odds - 1
                    fk = (b*c_prob - (1-c_prob)) / b if b > 0 else 0
                    stake = min(bal * fk * 0.15, bal * 0.005)
                    if stake >= 5:
                        parlay_won = all(l["won"] for l in pair)
                        profit = stake*(c_odds-1) if parlay_won else -stake
                        bal += profit
                        stats["parlay"]["bets"] += 1
                        stats["parlay"]["staked"] += stake
                        if parlay_won:
                            stats["parlay"]["wins"] += 1
                            stats["parlay"]["returned"] += stake * c_odds
                        for l in pair:
                            if l in parlay_legs_pending:
                                parlay_legs_pending.remove(l)

        # 清理已结算的旧腿
        parlay_legs_pending = [l for l in parlay_legs_pending if l["won"] is None][-30:]

        if bal > peak: peak = bal
        dd = (peak - bal) / peak if peak > 0 else 0
        if dd > mdd: mdd = dd

        for mod in models: mod.update(m)

    return {
        "season": season,
        "league": league,
        "matches": len(matches),
        "bankroll_start": bankroll,
        "bankroll_end": round(bal, 2),
        "profit": round(bal - bankroll, 2),
        "mdd": round(mdd, 4),
        "stats": stats,
    }


# ================================================================
# 报告生成
# ================================================================
def format_strategy(name, s):
    if s["bets"] == 0:
        return f"  {name:8s}  无投注"
    wr = s["wins"]/s["bets"]
    roi = (s["returned"]-s["staked"])/s["staked"] if s["staked"]>0 else 0
    return f"  {name:8s}  投注={s['bets']:4d}  胜={s['wins']:4d}  胜率={wr:.1%}  ROI={roi:+.1%}  投入={s['staked']:.0f}  回报={s['returned']:.0f}"


def run_10_season_backtest(leagues, seasons, bankroll=10000.0):
    sep = "="*70
    print(f"\n{'#'*70}")
    print(f"  GTO v5.11 — 10赛季逐赛季回测")
    print(f"  联赛: {', '.join(leagues)}")
    print(f"  赛季: {', '.join(seasons)}")
    print(f"  每赛季初始资金: {bankroll:.0f}")
    print(f"{'#'*70}")

    all_results = []
    season_summaries = []

    for season in seasons:
        print(f"\n{'#'*70}")
        print(f"  赛季: {season}")
        print(f"{'#'*70}")

        season_total = {"1x2":{"bets":0,"wins":0,"staked":0,"returned":0},"ou":{"bets":0,"wins":0,"staked":0,"returned":0},"ah":{"bets":0,"wins":0,"staked":0,"returned":0},"parlay":{"bets":0,"wins":0,"staked":0,"returned":0}}
        season_profit = 0
        season_matches = 0

        for league in leagues:
            result = run_single_season(league, season, warmup_matches=30, bankroll=bankroll)
            if not result:
                continue

            all_results.append(result)
            ln = {"premier_league":"英超","la_liga":"西甲","bundesliga":"德甲","serie_a":"意甲","ligue_1":"法甲"}.get(league,league)

            print(f"\n{sep}")
            print(f"  {ln} — {season}")
            print(f"  总比赛: {result['matches']} | 最终资金: {result['bankroll_end']:.0f} | 利润: {result['profit']:+.0f} | 最大回撤: {result['mdd']:.1%}")
            print(f"{sep}")

            for strat in ["1x2","ah","ou","parlay"]:
                s = result["stats"][strat]
                sn = {"1x2":"胜平负","ah":"亚盘","ou":"大小球","parlay":"串关"}[strat]
                print(format_strategy(sn, s))
                for k in ["bets","wins","staked","returned"]:
                    season_total[strat][k] += s[k]

            season_profit += result["profit"]
            season_matches += result["matches"]

        # 赛季汇总
        total_bets = sum(s["bets"] for s in season_total.values())
        total_staked = sum(s["staked"] for s in season_total.values())
        total_returned = sum(s["returned"] for s in season_total.values())
        total_wins = sum(s["wins"] for s in season_total.values())
        overall_roi = (total_returned - total_staked) / total_staked if total_staked > 0 else 0

        print(f"\n{'#'*70}")
        print(f"  {season} 全联汇总")
        print(f"{'#'*70}")
        for strat in ["1x2","ah","ou","parlay"]:
            s = season_total[strat]
            sn = {"1x2":"胜平负","ah":"亚盘","ou":"大小球","parlay":"串关"}[strat]
            print(format_strategy(sn, s))
        print(f"  {'总计':8s}  投注={total_bets:4d}  胜={total_wins:4d}  胜率={total_wins/total_bets:.1%}  ROI={overall_roi:+.1%}" if total_bets > 0 else "")

        season_summaries.append({
            "season": season,
            "matches": season_matches,
            "total_bets": total_bets,
            "total_wins": total_wins,
            "total_staked": round(total_staked, 2),
            "total_returned": round(total_returned, 2),
            "profit": round(total_returned - total_staked, 2),
            "roi": round(overall_roi, 4),
            "strategies": {k: dict(v) for k, v in season_total.items()},
        })

    # ================================================================
    # 10赛季总复盘
    # ================================================================
    print(f"\n\n{'#'*70}")
    print(f"  10赛季总复盘")
    print(f"{'#'*70}")

    # 按赛季汇总
    print(f"\n  ── 逐赛季 ROI ──")
    print(f"  {'赛季':12s}  {'投注':>6s}  {'胜率':>6s}  {'ROI':>8s}  {'利润':>10s}")
    print(f"  {'-'*50}")
    for ss in season_summaries:
        wr = ss["total_wins"]/ss["total_bets"] if ss["total_bets"]>0 else 0
        print(f"  {ss['season']:12s}  {ss['total_bets']:6d}  {wr:6.1%}  {ss['roi']:+8.1%}  {ss['profit']:+10.0f}")

    # 按策略汇总
    print(f"\n  ── 按策略汇总 ──")
    grand = {"1x2":{"bets":0,"wins":0,"staked":0,"returned":0},"ou":{"bets":0,"wins":0,"staked":0,"returned":0},"ah":{"bets":0,"wins":0,"staked":0,"returned":0},"parlay":{"bets":0,"wins":0,"staked":0,"returned":0}}
    for ss in season_summaries:
        for strat in grand:
            for k in grand[strat]:
                grand[strat][k] += ss["strategies"].get(strat,{}).get(k,0)

    print(f"  {'策略':8s}  {'投注':>6s}  {'胜':>6s}  {'胜率':>6s}  {'ROI':>8s}  {'投入':>10s}  {'回报':>10s}")
    print(f"  {'-'*60}")
    for strat in ["1x2","ah","ou","parlay"]:
        s = grand[strat]
        sn = {"1x2":"胜平负","ah":"亚盘","ou":"大小球","parlay":"串关"}[strat]
        if s["bets"] > 0:
            wr = s["wins"]/s["bets"]
            roi = (s["returned"]-s["staked"])/s["staked"] if s["staked"]>0 else 0
            print(f"  {sn:8s}  {s['bets']:6d}  {s['wins']:6d}  {wr:6.1%}  {roi:+8.1%}  {s['staked']:10.0f}  {s['returned']:10.0f}")

    g_bets = sum(s["bets"] for s in grand.values())
    g_staked = sum(s["staked"] for s in grand.values())
    g_returned = sum(s["returned"] for s in grand.values())
    g_wins = sum(s["wins"] for s in grand.values())
    g_roi = (g_returned-g_staked)/g_staked if g_staked>0 else 0
    print(f"  {'-'*60}")
    print(f"  {'总计':8s}  {g_bets:6d}  {g_wins:6d}  {g_wins/g_bets:6.1%}  {g_roi:+8.1%}  {g_staked:10.0f}  {g_returned:10.0f}")

    # 1X2 详细分析
    print(f"\n  ── 1X2 投注方向分析 ──")
    total_sel = {"home":0,"draw":0,"away":0,"home_w":0,"draw_w":0,"away_w":0}
    for r in all_results:
        for k in total_sel:
            total_sel[k] += r["stats"]["1x2"]["selections"].get(k,0)

    for d in ["home","draw","away"]:
        dn = {"home":"主胜","draw":"平局","away":"客胜"}[d]
        bets = total_sel[d]
        wins = total_sel[d+"_w"]
        if bets > 0:
            print(f"  {dn}: {bets} 注, 胜 {wins}, 胜率 {wins/bets:.1%}")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'backtest_10_seasons.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"seasons": season_summaries, "grand_total": {k:dict(v) for k,v in grand.items()}, "per_match": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n  详细报告已保存: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--leagues",nargs="+",default=["premier_league","la_liga","bundesliga","serie_a","ligue_1"])
    p.add_argument("--seasons",nargs="+",default=["2014-15","2015-16","2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"])
    p.add_argument("--bankroll",type=float,default=10000.0)
    a = p.parse_args()
    run_10_season_backtest(a.leagues, a.seasons, a.bankroll)
