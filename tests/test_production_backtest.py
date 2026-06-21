"""
GTO v6.0 — 生产引擎回测框架

使用与生产环境相同的 58 因子引擎进行回测。
与 test_full_real_odds_backtest.py 的区别:
- 旧框架: EloDCModel + PoissonModel + FormModel + MarketModel (简化模型)
- 新框架: BacktestAdapter → GameFlowPipeline (生产引擎)

用法:
    python3 test_production_backtest.py
    python3 test_production_backtest.py --use-simplified  # 使用简化模型对比
"""

import sys
import os
import csv
import math
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.backtest_adapter import BacktestAdapter
from src.i18n.cn_names import get_cn_name, get_league_cn, get_strategy_cn, get_direction_cn


# ═══════════════════════════════════════════════════════════════
# 数据模型 (复用 test_full_real_odds_backtest.py)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RealOddsMatch:
    """真实赔率比赛数据"""
    date: str
    league: str
    home_team: str
    away_team: str
    fthg: int
    ftag: int
    ftr: str
    hthg: int = 0
    htag: int = 0
    htr: str = ""
    hs: int = 0
    as_: int = 0
    hst: int = 0
    ast: int = 0
    hc: int = 0
    ac: int = 0
    hy: int = 0
    ay: int = 0
    b365h: float = 0.0
    b365d: float = 0.0
    b365a: float = 0.0
    bwh: float = 0.0
    bwd: float = 0.0
    bwa: float = 0.0
    iwh: float = 0.0
    iwd: float = 0.0
    iwa: float = 0.0
    psh: float = 0.0
    psd: float = 0.0
    psa: float = 0.0
    whh: float = 0.0
    whd: float = 0.0
    wha: float = 0.0
    vch: float = 0.0
    vcd: float = 0.0
    vca: float = 0.0
    maxh: float = 0.0
    maxd: float = 0.0
    maxa: float = 0.0
    avgh: float = 0.0
    avgd: float = 0.0
    avga: float = 0.0
    b365ch: float = 0.0
    b365cd: float = 0.0
    b365ca: float = 0.0
    psch: float = 0.0
    pscd: float = 0.0
    psca: float = 0.0
    maxch: float = 0.0
    maxcd: float = 0.0
    maxca: float = 0.0
    avgch: float = 0.0
    avgcd: float = 0.0
    avgca: float = 0.0
    b365_over25: float = 0.0
    b365_under25: float = 0.0
    p_over25: float = 0.0
    p_under25: float = 0.0
    max_over25: float = 0.0
    max_under25: float = 0.0
    avg_over25: float = 0.0
    avg_under25: float = 0.0
    ahh: float = 0.0
    b365ahh: float = 0.0
    b365aha: float = 0.0
    pahh: float = 0.0
    paha: float = 0.0
    maxahh: float = 0.0
    maxaha: float = 0.0
    avgahh: float = 0.0
    avgaha: float = 0.0


@dataclass
class BetRecord:
    """投注记录"""
    match_date: str
    league: str
    home_team: str
    away_team: str
    strategy: str
    direction: str
    odds: float
    stake: float
    won: bool
    profit: float
    model_prob: float
    market_prob: float
    value: float
    agreement: int
    total_models: int
    prediction_source: str = ""  # "production" / "simplified"


def load_matches_from_csv(csv_path: str, league: str) -> List[RealOddsMatch]:
    """从CSV加载真实赔率数据"""
    matches = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                home_team = row.get('HomeTeam', '').strip()
                away_team = row.get('AwayTeam', '').strip()
                if not home_team or not away_team:
                    continue
                fthg = int(row.get('FTHG', 0) or 0)
                ftag = int(row.get('FTAG', 0) or 0)
                ftr = row.get('FTR', '').strip()
                if not ftr or ftr not in ('H', 'D', 'A'):
                    continue

                def safe_float(key, default=0.0):
                    val = row.get(key, '').strip()
                    if not val or val == '':
                        return default
                    try:
                        return float(val)
                    except:
                        return default

                def safe_int(key, default=0):
                    val = row.get(key, '').strip()
                    if not val or val == '':
                        return default
                    try:
                        return int(float(val))
                    except:
                        return default

                match = RealOddsMatch(
                    date=row.get('Date', ''),
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                    fthg=fthg,
                    ftag=ftag,
                    ftr=ftr,
                    hthg=safe_int('HTHG'),
                    htag=safe_int('HTAG'),
                    htr=row.get('HTR', ''),
                    hs=safe_int('HS'),
                    as_=safe_int('AS'),
                    hst=safe_int('HST'),
                    ast=safe_int('AST'),
                    hc=safe_int('HC'),
                    ac=safe_int('AC'),
                    hy=safe_int('HY'),
                    ay=safe_int('AY'),
                    b365h=safe_float('B365H'),
                    b365d=safe_float('B365D'),
                    b365a=safe_float('B365A'),
                    bwh=safe_float('BWH'),
                    bwd=safe_float('BWD'),
                    bwa=safe_float('BWA'),
                    iwh=safe_float('IWH'),
                    iwd=safe_float('IWD'),
                    iwa=safe_float('IWA'),
                    psh=safe_float('PSH'),
                    psd=safe_float('PSD'),
                    psa=safe_float('PSA'),
                    whh=safe_float('WHH'),
                    whd=safe_float('WHD'),
                    wha=safe_float('WHA'),
                    vch=safe_float('VCH'),
                    vcd=safe_float('VCD'),
                    vca=safe_float('VCA'),
                    maxh=safe_float('MaxH'),
                    maxd=safe_float('MaxD'),
                    maxa=safe_float('MaxA'),
                    avgh=safe_float('AvgH'),
                    avgd=safe_float('AvgD'),
                    avga=safe_float('AvgA'),
                    b365ch=safe_float('B365CH'),
                    b365cd=safe_float('B365CD'),
                    b365ca=safe_float('B365CA'),
                    psch=safe_float('PSCH'),
                    pscd=safe_float('PSCD'),
                    psca=safe_float('PSCA'),
                    maxch=safe_float('MaxCH'),
                    maxcd=safe_float('MaxCD'),
                    maxca=safe_float('MaxCA'),
                    avgch=safe_float('AvgCH'),
                    avgcd=safe_float('AvgCD'),
                    avgca=safe_float('AvgCA'),
                    b365_over25=safe_float('B365>2.5'),
                    b365_under25=safe_float('B365<2.5'),
                    p_over25=safe_float('P>2.5'),
                    p_under25=safe_float('P<2.5'),
                    max_over25=safe_float('Max>2.5'),
                    max_under25=safe_float('Max<2.5'),
                    avg_over25=safe_float('Avg>2.5'),
                    avg_under25=safe_float('Avg<2.5'),
                    ahh=safe_float('AHh'),
                    b365ahh=safe_float('B365AHH'),
                    b365aha=safe_float('B365AHA'),
                    pahh=safe_float('PAHH'),
                    paha=safe_float('PAHA'),
                    maxahh=safe_float('MaxAHH'),
                    maxaha=safe_float('MaxAHA'),
                    avgahh=safe_float('AvgAHH'),
                    avgaha=safe_float('AvgAHA'),
                )
                matches.append(match)
            except:
                continue
    return matches


def load_all_matches(leagues: List[str], seasons: List[str]) -> Dict[str, List[RealOddsMatch]]:
    """加载所有比赛"""
    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'historical_odds')
    all_matches = {}
    for league in leagues:
        matches = []
        for season in seasons:
            for pattern in [f"{league}_{season}.csv", f"{league}_{season.replace('-', '-')}.csv"]:
                csv_path = os.path.join(csv_dir, pattern)
                if os.path.exists(csv_path):
                    loaded = load_matches_from_csv(csv_path, league)
                    matches.extend(loaded)
                    print(f"  ✓ {league} {season}: {len(loaded)} 场")
                    break
        matches.sort(key=lambda m: m.date)
        all_matches[league] = matches
    return all_matches


def get_real_odds_1x2(m: RealOddsMatch, direction: str) -> float:
    """获取1X2真实赔率"""
    if direction == "home":
        for o in [m.psh, m.b365h, m.avgh, m.maxh]:
            if o > 1:
                return o
    elif direction == "draw":
        for o in [m.psd, m.b365d, m.avgd, m.maxd]:
            if o > 1:
                return o
    elif direction == "away":
        for o in [m.psa, m.b365a, m.avga, m.maxa]:
            if o > 1:
                return o
    return 0.0


# ═══════════════════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════════════════

def run_production_backtest(
    leagues: List[str],
    seasons: List[str],
    bankroll: float = 10000.0,
    consensus: int = 3,
    value_thresh: float = 0.05,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.03,
    use_production: bool = True,
):
    """生产引擎回测"""

    sep = '#' * 70
    engine_type = "生产引擎(58因子)" if use_production else "简化模型(Elo+DC)"
    print(f"\n{sep}")
    print(f"  GTO v6.0 回测 — {engine_type}")
    print(f"  共识={consensus}/4 | 阈值={value_thresh:.0%}")
    print(sep)

    all_matches = load_all_matches(leagues, seasons)
    total = sum(len(m) for m in all_matches.values())
    print(f"  共 {total} 场\n")

    all_bets = []
    all_results = {}

    for league in leagues:
        matches = all_matches.get(league, [])
        if not matches:
            continue

        ln = get_league_cn(league)
              

        # 创建适配器
        adapter = BacktestAdapter(league_id=league, use_production=use_production)

        # 预热
        for m in matches[:50]:
            adapter.update(m)

        bets = []
        bal = bankroll
        peak = bankroll
        mdd = 0

        print(f"  {ln}: {len(matches)}场")

        for i, m in enumerate(matches[50:], 50):
            # 获取预测
            pred = adapter.predict(m)

            if pred["home_prob"] <= 0:
                adapter.update(m)
                continue

            # 市场赔率
            odds_home = get_real_odds_1x2(m, "home")
            odds_draw = get_real_odds_1x2(m, "draw")
            odds_away = get_real_odds_1x2(m, "away")

            if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
                adapter.update(m)
                continue

            # 市场隐含概率
            mg = 1.0/odds_home + 1.0/odds_draw + 1.0/odds_away
            market_home = (1.0/odds_home) / mg
            market_draw = (1.0/odds_draw) / mg
            market_away = (1.0/odds_away) / mg

            # 计算价值
            probs = {
                "home": pred["home_prob"],
                "draw": pred["draw_prob"],
                "away": pred["away_prob"],
            }
            markets = {
                "home": market_home,
                "draw": market_draw,
                "away": market_away,
            }
            odds_map = {
                "home": odds_home,
                "draw": odds_draw,
                "away": odds_away,
            }

            # 找最佳方向
            best_dir = None
            best_value = 0
            for direction in ["home", "draw", "away"]:
                value = probs[direction] - markets[direction]
                if value > best_value:
                    best_value = value
                    best_dir = direction

            if best_dir is None or best_value < value_thresh:
                adapter.update(m)
                continue

            # 下注
            odds = odds_map[best_dir]
            cp = probs[best_dir]
            ip = markets[best_dir]

            b = odds - 1
            fk = (b * cp - (1 - cp)) / b if b > 0 else 0
            stake = min(bal * fk * kelly_fraction, bal * max_stake_pct)
            if stake < 10:
                adapter.update(m)
                continue

            rmap = {"H": "home", "D": "draw", "A": "away"}
            won = best_dir == rmap.get(m.ftr, "")
            profit = stake * (odds - 1) if won else -stake
            bal += profit

            bets.append({
                "won": won, "profit": profit, "stake": stake,
                "dir": best_dir, "value": best_value,
                "source": pred.get("source", "unknown"),
            })

            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak if peak > 0 else 0
            if dd > mdd:
                mdd = dd

            adapter.update(m)

        # 输出结果
        if bets:
            ws = sum(1 for b in bets if b["won"])
            ss = sum(b["stake"] for b in bets)
            rs = sum(b["stake"] * (b["profit"]/b["stake"] + 1) for b in bets if b["won"])
            roi = (rs - ss) / ss if ss > 0 else 0
            print(f"  投注={len(bets)} 胜={ws} 胜率={ws/len(bets):.1%} ROI={roi:+.1%} 回撤={mdd:.1%}")

            all_results[league] = {"bets": len(bets), "wins": ws, "staked": ss, "returned": rs, "mdd": mdd}
        else:
            print(f"  无投注")
            all_results[league] = {"bets": 0, "wins": 0, "staked": 0, "returned": 0, "mdd": 0}

        all_bets.extend(bets)

    # 全局汇总
    if all_bets:
        ts = sum(b["stake"] for b in all_bets)
        tr = sum(b["stake"] * (b["profit"]/b["stake"] + 1) for b in all_bets if b["won"])
        tw = sum(1 for b in all_bets if b["won"])
        roi = (tr - ts) / ts if ts > 0 else 0
        print(f"\n{sep}")
        print(f"  全联汇总: 投注={len(all_bets)} 胜={tw} 胜率={tw/len(all_bets):.1%} ROI={roi:+.1%}")
        for lg, r in all_results.items():
            ln = get_league_cn(league)
            roi_l = (r["returned"] - r["staked"]) / r["staked"] if r["staked"] > 0 else 0
            if r["bets"] > 0:
                print(f"  {ln:8s}  投注={r['bets']:3d}  胜率={r['wins']/r['bets']:.1%}  ROI={roi_l:+.1%}")

    return all_bets, all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GTO v6.0 生产引擎回测")
    p.add_argument("--leagues", nargs="+", default=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"])
    p.add_argument("--seasons", nargs="+", default=["2021-22", "2022-23", "2023-24"])
    p.add_argument("--consensus", type=int, default=3)
    p.add_argument("--value", type=float, default=0.05)
    p.add_argument("--use-simplified", action="store_true", help="使用简化模型对比")
    a = p.parse_args()

    run_production_backtest(
        leagues=a.leagues,
        seasons=a.seasons,
        consensus=a.consensus,
        value_thresh=a.value,
        use_production=not a.use_simplified,
    )
