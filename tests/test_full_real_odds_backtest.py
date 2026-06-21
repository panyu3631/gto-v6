"""
GTO v5.11 — 全策略真实赔率回测框架

要求:
1. 所有回测数据采用真实赔率，不得使用模拟赔率
2. 赛前真实赔率 (B365/Pinnacle/Max/Avg)
3. 支持 1X2 / 大小球 / 亚盘 三种策略
4. 多模型集成 (Elo+DC+Poisson+Form+Market)
5. 联赛特化参数

数据源: src/data/historical_odds/*.csv
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

# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class RealOddsMatch:
    """真实赔率比赛数据"""
    date: str
    league: str
    home_team: str
    away_team: str
    fthg: int  # 全场主队进球
    ftag: int  # 全场客队进球
    ftr: str   # H/D/A
    hthg: int = 0  # 半场主队进球
    htag: int = 0  # 半场客队进球
    htr: str = ""  # H/D/A
    
    # 比赛统计
    hs: int = 0    # 主队射门
    as_: int = 0   # 客队射门
    hst: int = 0   # 主队射正
    ast: int = 0   # 客队射正
    hc: int = 0    # 主队角球
    ac: int = 0    # 客队角球
    hy: int = 0    # 主队黄牌
    ay: int = 0    # 客队黄牌
    
    # 1X2 开盘赔率 (赛前)
    b365h: float = 0.0  # Bet365 主胜
    b365d: float = 0.0  # Bet365 平局
    b365a: float = 0.0  # Bet365 客胜
    bwh: float = 0.0    # Betway 主胜
    bwd: float = 0.0    # Betway 平局
    bwa: float = 0.0    # Betway 客胜
    iwh: float = 0.0    # Interwetten 主胜
    iwd: float = 0.0    # Interwetten 平局
    iwa: float = 0.0    # Interwetten 客胜
    psh: float = 0.0    # Pinnacle 主胜
    psd: float = 0.0    # Pinnacle 平局
    psa: float = 0.0    # Pinnacle 客胜
    whh: float = 0.0    # William Hill 主胜
    whd: float = 0.0    # William Hill 平局
    wha: float = 0.0    # William Hill 客胜
    vch: float = 0.0    # VC Bet 主胜
    vcd: float = 0.0    # VC Bet 平局
    vca: float = 0.0    # VC Bet 客胜
    maxh: float = 0.0   # 最高主胜
    maxd: float = 0.0   # 最高平局
    maxa: float = 0.0   # 最高客胜
    avgh: float = 0.0   # 平均主胜
    avgd: float = 0.0   # 平均平局
    avga: float = 0.0   # 平均客胜
    
    # 1X2 收盘赔率
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
    
    # 大小球 赔率 (开盘)
    b365_over25: float = 0.0
    b365_under25: float = 0.0
    p_over25: float = 0.0
    p_under25: float = 0.0
    max_over25: float = 0.0
    max_under25: float = 0.0
    avg_over25: float = 0.0
    avg_under25: float = 0.0
    
    # 大小球 赔率 (收盘)
    b365c_over25: float = 0.0
    b365c_under25: float = 0.0
    pc_over25: float = 0.0
    pc_under25: float = 0.0
    maxc_over25: float = 0.0
    maxc_under25: float = 0.0
    avgc_over25: float = 0.0
    avgc_under25: float = 0.0
    
    # 亚盘 (开盘)
    ahh: float = 0.0       # 亚盘让球数
    b365ahh: float = 0.0   # Bet365 主队赔率
    b365aha: float = 0.0   # Bet365 客队赔率
    pahh: float = 0.0      # Pinnacle 主队赔率
    paha: float = 0.0      # Pinnacle 客队赔率
    maxahh: float = 0.0    # 最高主队赔率
    maxaha: float = 0.0    # 最高客队赔率
    avgahh: float = 0.0    # 平均主队赔率
    avgaha: float = 0.0    # 平均客队赔率
    
    # 亚盘 (收盘)
    ahch: float = 0.0
    b365cahh: float = 0.0
    b365caha: float = 0.0
    pcahh: float = 0.0
    pcaha: float = 0.0
    maxcahh: float = 0.0
    maxcaha: float = 0.0
    avgcahh: float = 0.0
    avgcaha: float = 0.0


def load_matches_from_csv(csv_path: str, league: str) -> List[RealOddsMatch]:
    """从CSV加载真实赔率数据"""
    matches = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 基础信息
                home_team = row.get('HomeTeam', '').strip()
                away_team = row.get('AwayTeam', '').strip()
                if not home_team or not away_team:
                    continue
                
                fthg = int(row.get('FTHG', 0) or 0)
                ftag = int(row.get('FTAG', 0) or 0)
                ftr = row.get('FTR', '').strip()
                
                if not ftr or ftr not in ('H', 'D', 'A'):
                    continue
                
                # 辅助函数: 安全读取浮点数
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
                    
                    # 1X2 开盘赔率
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
                    
                    # 1X2 收盘赔率
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
                    
                    # 大小球 开盘
                    b365_over25=safe_float('B365>2.5'),
                    b365_under25=safe_float('B365<2.5'),
                    p_over25=safe_float('P>2.5'),
                    p_under25=safe_float('P<2.5'),
                    max_over25=safe_float('Max>2.5'),
                    max_under25=safe_float('Max<2.5'),
                    avg_over25=safe_float('Avg>2.5'),
                    avg_under25=safe_float('Avg<2.5'),
                    
                    # 大小球 收盘
                    b365c_over25=safe_float('B365C>2.5'),
                    b365c_under25=safe_float('B365C<2.5'),
                    pc_over25=safe_float('PC>2.5'),
                    pc_under25=safe_float('PC<2.5'),
                    maxc_over25=safe_float('MaxC>2.5'),
                    maxc_under25=safe_float('MaxC<2.5'),
                    avgc_over25=safe_float('AvgC>2.5'),
                    avgc_under25=safe_float('AvgC<2.5'),
                    
                    # 亚盘 开盘
                    ahh=safe_float('AHh'),
                    b365ahh=safe_float('B365AHH'),
                    b365aha=safe_float('B365AHA'),
                    pahh=safe_float('PAHH'),
                    paha=safe_float('PAHA'),
                    maxahh=safe_float('MaxAHH'),
                    maxaha=safe_float('MaxAHA'),
                    avgahh=safe_float('AvgAHH'),
                    avgaha=safe_float('AvgAHA'),
                    
                    # 亚盘 收盘
                    ahch=safe_float('AHCh'),
                    b365cahh=safe_float('B365CAHH'),
                    b365caha=safe_float('B365CAHA'),
                    pcahh=safe_float('PCAHH'),
                    pcaha=safe_float('PCAHA'),
                    maxcahh=safe_float('MaxCAHH'),
                    maxcaha=safe_float('MaxCAHA'),
                    avgcahh=safe_float('AvgCAHH'),
                    avgcaha=safe_float('AvgCAHA'),
                )
                matches.append(match)
            except Exception as e:
                continue
    
    return matches


def load_all_matches(leagues: List[str], seasons: List[str]) -> Dict[str, List[RealOddsMatch]]:
    """加载所有联赛和赛季的比赛数据"""
    csv_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'historical_odds')
    all_matches = {}
    
    for league in leagues:
        matches = []
        for season in seasons:
            # 尝试多种文件名格式
            patterns = [
                f"{league}_{season}.csv",
                f"{league}_{season.replace('-', '-')}.csv",
            ]
            for pattern in patterns:
                csv_path = os.path.join(csv_dir, pattern)
                if os.path.exists(csv_path):
                    loaded = load_matches_from_csv(csv_path, league)
                    matches.extend(loaded)
                    print(f"  ✓ {league} {season}: {len(loaded)} 场")
                    break
        
        # 按日期排序
        matches.sort(key=lambda m: m.date)
        all_matches[league] = matches
    
    return all_matches


# ═══════════════════════════════════════════════════════════════
# 预测模型
# ═══════════════════════════════════════════════════════════════

class EloDCModel:
    """Elo + Dixon-Coles 混合模型"""
    
    def __init__(self, league: str):
        self.league = league
        self.elo: Dict[str, float] = defaultdict(lambda: 1500.0)
        self.k = 20
        self.home_adv = 65
        
        # 联赛特化 ρ
        self.rho = {
            "premier_league": -0.08,
            "la_liga": -0.10,
            "bundesliga": -0.06,
            "serie_a": -0.13,
            "ligue_1": -0.12
        }.get(league, -0.10)
        
        self.avg_goals = {
            "premier_league": 2.65,
            "la_liga": 2.55,
            "bundesliga": 2.85,
            "serie_a": 2.50,
            "ligue_1": 2.60
        }.get(league, 2.65)
    
    def predict(self, m: RealOddsMatch) -> Optional[Dict[str, float]]:
        eh = self.elo[m.home_team]
        ea = self.elo[m.away_team]
        base = self.avg_goals / 2.0
        ef = (eh - ea) / 400.0
        lh = base * (1.0 + ef * 0.5 + self.home_adv * 0.3 / 400.0)
        la = base * (1.0 - ef * 0.5)
        s = self.avg_goals * 0.93 / max(lh + la, 0.1)
        lh, la = max(0.3, min(4.0, lh * s)), max(0.3, min(4.0, la * s))
        
        mat = {}
        for h in range(9):
            for a in range(9):
                p = poisson_pmf(h, lh) * poisson_pmf(a, la) * dc_tau(h, a, lh, la, self.rho)
                mat[(h, a)] = p
        t = sum(mat.values())
        if t > 0:
            for k in mat:
                mat[k] /= t
        
        return {
            "home": sum(v for (h, a), v in mat.items() if h > a),
            "draw": sum(v for (h, a), v in mat.items() if h == a),
            "away": sum(v for (h, a), v in mat.items() if h < a),
            "score_matrix": mat,
            "home_lambda": lh,
            "away_lambda": la,
        }
    
    def update(self, m: RealOddsMatch):
        eh = self.elo[m.home_team]
        ea = self.elo[m.away_team]
        expected = 1.0 / (1.0 + 10 ** (-(eh + self.home_adv - ea) / 400.0))
        actual = 1.0 if m.ftr == 'H' else (0.5 if m.ftr == 'D' else 0.0)
        gd = abs(m.fthg - m.ftag)
        margin = 1.0 + min(gd, 3) * 0.33
        delta = self.k * margin * (actual - expected)
        self.elo[m.home_team] += delta
        self.elo[m.away_team] -= delta


class PoissonModel:
    """基于近期表现的泊松模型"""
    
    def __init__(self, league: str):
        self.league = league
        self.avg_goals = {
            "premier_league": 2.65,
            "la_liga": 2.55,
            "bundesliga": 2.85,
            "serie_a": 2.50,
            "ligue_1": 2.60
        }.get(league, 2.65)
        self.ts: Dict[str, Dict] = defaultdict(lambda: {"hgf": [], "hga": [], "agf": [], "aga": []})
    
    def predict(self, m: RealOddsMatch) -> Optional[Dict[str, float]]:
        hs = self.ts[m.home_team]
        as_ = self.ts[m.away_team]
        if len(hs["hgf"]) < 8 or len(as_["agf"]) < 8:
            return None
        
        lg = self.avg_goals / 2.0
        lh = (sum(hs["hgf"][-10:]) / len(hs["hgf"][-10:])) * (sum(as_["aga"][-10:]) / len(as_["aga"][-10:])) / max(lg, 0.5)
        la = (sum(as_["agf"][-10:]) / len(as_["agf"][-10:])) * (sum(hs["hga"][-10:]) / len(hs["hga"][-10:])) / max(lg, 0.5)
        lh, la = max(0.3, min(4.0, lh)), max(0.3, min(4.0, la))
        
        ph = pd = pa = 0.0
        for h in range(9):
            for a in range(9):
                p = poisson_pmf(h, lh) * poisson_pmf(a, la)
                if h > a:
                    ph += p
                elif h == a:
                    pd += p
                else:
                    pa += p
        t = ph + pd + pa
        
        if t > 0:
            return {"home": ph / t, "draw": pd / t, "away": pa / t, "home_lambda": lh, "away_lambda": la}
        return None
    
    def update(self, m: RealOddsMatch):
        hs = self.ts[m.home_team]
        as_ = self.ts[m.away_team]
        hs["hgf"].append(m.fthg)
        hs["hga"].append(m.ftag)
        as_["agf"].append(m.ftag)
        as_["aga"].append(m.fthg)
        for d in [hs, as_]:
            for k in d:
                d[k] = d[k][-30:]


class FormModel:
    """近期状态模型"""
    
    def __init__(self):
        self.ts: Dict[str, Dict] = defaultdict(lambda: {"r": [], "gf": [], "ga": [], "hr": [], "ar": []})
    
    def predict(self, m: RealOddsMatch) -> Optional[Dict[str, float]]:
        hs = self.ts[m.home_team]
        as_ = self.ts[m.away_team]
        if len(hs["r"]) < 10 or len(as_["r"]) < 10:
            return None
        
        hf = sum(hs["hr"][-5:]) / max(len(hs["hr"][-5:]), 1) if hs["hr"] else sum(hs["r"][-5:]) / 5
        af = sum(as_["ar"][-5:]) / max(len(as_["ar"][-5:]), 1) if as_["ar"] else sum(as_["r"][-5:]) / 5
        hg = sum(hs["gf"][-5:]) / 5
        ag = sum(as_["gf"][-5:]) / 5
        
        score_h = hf * 0.4 + sum(hs["r"][-5:]) / 5 * 0.3 + (hg - sum(hs["ga"][-5:]) / 5) * 0.3 + 0.3
        score_a = af * 0.4 + sum(as_["r"][-5:]) / 5 * 0.3 + (ag - sum(as_["ga"][-5:]) / 5) * 0.3
        
        d = score_h - score_a
        ph = 1.0 / (1.0 + math.exp(-d * 1.5))
        pa = 1.0 / (1.0 + math.exp(d * 1.5))
        pd = max(0.1, 1.0 - ph - pa)
        t = ph + pd + pa
        
        return {"home": ph / t, "draw": pd / t, "away": pa / t}
    
    def update(self, m: RealOddsMatch):
        for team, gf, ga, ih in [(m.home_team, m.fthg, m.ftag, True), (m.away_team, m.ftag, m.fthg, False)]:
            s = self.ts[team]
            r = 3 if gf > ga else (1 if gf == ga else 0)
            s["r"].append(r)
            s["r"] = s["r"][-30:]
            s["gf"].append(gf)
            s["gf"] = s["gf"][-30:]
            s["ga"].append(ga)
            s["ga"] = s["ga"][-30:]
            if ih:
                s["hr"].append(r)
                s["hr"] = s["hr"][-20:]
            else:
                s["ar"].append(r)
                s["ar"] = s["ar"][-20:]


class MarketModel:
    """市场赔率模型 (真实赔率)"""
    
    def predict(self, m: RealOddsMatch) -> Optional[Dict[str, float]]:
        # 优先使用 Pinnacle (最接近真实概率)
        odds = self._get_best_odds(m)
        if not odds:
            return None
        
        h, d, a = odds
        if h <= 1 or d <= 1 or a <= 1:
            return None
        
        mg = 1.0 / h + 1.0 / d + 1.0 / a
        return {
            "home": (1.0 / h) / mg,
            "draw": (1.0 / d) / mg,
            "away": (1.0 / a) / mg,
        }
    
    def _get_best_odds(self, m: RealOddsMatch) -> Optional[Tuple[float, float, float]]:
        """获取最佳赔率 (优先Pinnacle, 然后B365, 最后平均)"""
        # Pinnacle
        if m.psh > 1 and m.psd > 1 and m.psa > 1:
            return (m.psh, m.psd, m.psa)
        # Bet365
        if m.b365h > 1 and m.b365d > 1 and m.b365a > 1:
            return (m.b365h, m.b365d, m.b365a)
        # 平均赔率
        if m.avgh > 1 and m.avgd > 1 and m.avga > 1:
            return (m.avgh, m.avgd, m.avga)
        # 最高赔率
        if m.maxh > 1 and m.maxd > 1 and m.maxa > 1:
            return (m.maxh, m.maxd, m.maxa)
        return None
    
    def update(self, m: RealOddsMatch):
        pass


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dc_tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    if x >= 2 or y >= 2:
        return 1.0
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lh * la * rho)
    elif x == 0 and y == 1:
        return max(0.0, 1.0 + lh * rho)
    elif x == 1 and y == 0:
        return max(0.0, 1.0 + la * rho)
    elif x == 1 and y == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def get_real_odds_1x2(m: RealOddsMatch, direction: str) -> float:
    """获取1X2真实赔率"""
    # 优先 Pinnacle > B365 > 平均 > 最高
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


def get_real_odds_overunder(m: RealOddsMatch, direction: str) -> Tuple[float, float]:
    """获取大小球真实赔率 (odds, line)"""
    # 默认 2.5 线
    if direction == "over":
        for o in [m.p_over25, m.b365_over25, m.avg_over25, m.max_over25]:
            if o > 1:
                return o, 2.5
    elif direction == "under":
        for o in [m.p_under25, m.b365_under25, m.avg_under25, m.max_under25]:
            if o > 1:
                return o, 2.5
    return 0.0, 2.5


def get_real_odds_asian(m: RealOddsMatch, direction: str) -> Tuple[float, float]:
    """获取亚盘真实赔率 (odds, handicap)"""
    line = m.ahh if m.ahh != 0 else m.ahch
    if direction == "home":
        for o in [m.pahh, m.b365ahh, m.avgahh, m.maxahh]:
            if o > 1:
                return o, line
    elif direction == "away":
        for o in [m.paha, m.b365aha, m.avgaha, m.maxaha]:
            if o > 1:
                return o, line
    return 0.0, line


def calculate_over_prob(avg_goals: float, line: float = 2.5) -> float:
    """计算大小球 over 概率"""
    prob = 0.0
    for k in range(int(line) + 1, 20):
        prob += poisson_pmf(k, avg_goals)
    return prob


# ═══════════════════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════════════════

@dataclass
class BetRecord:
    """投注记录"""
    match_date: str
    league: str
    home_team: str
    away_team: str
    strategy: str  # "1x2" / "over_under" / "asian_handicap"
    direction: str  # "home" / "draw" / "away" / "over" / "under"
    odds: float
    stake: float
    won: bool
    profit: float
    model_prob: float
    market_prob: float
    value: float
    agreement: int  # 共识模型数
    total_models: int
    handicap_line: float = 0.0


def run_full_backtest(
    leagues: List[str],
    seasons: List[str],
    bankroll: float = 10000.0,
    consensus: int = 3,
    value_thresh_1x2: float = 0.05,
    value_thresh_ou: float = 0.05,
    value_thresh_ah: float = 0.05,
    enable_1x2: bool = True,
    enable_ou: bool = True,
    enable_ah: bool = True,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.03,
):
    """全策略真实赔率回测"""
    
    sep = '#' * 70
    print(f"\n{sep}")
    print(f"  GTO v5.11 全策略真实赔率回测")
    print(f"  共识={consensus}/4 | 1X2阈值={value_thresh_1x2:.0%} | OU阈值={value_thresh_ou:.0%} | AH阈值={value_thresh_ah:.0%}")
    print(f"  Kelly={kelly_fraction:.0%} | 最大仓位={max_stake_pct:.0%}")
    print(sep)
    
    all_matches = load_all_matches(leagues, seasons)
    total = sum(len(m) for m in all_matches.values())
    print(f"  共 {total} 场\n")
    
    all_bets: List[BetRecord] = []
    all_results = {}
    
    for league in leagues:
        matches = all_matches.get(league, [])
        if not matches:
            continue
        
        ln = {
            "premier_league": "英超",
            "la_liga": "西甲",
            "bundesliga": "德甲",
            "serie_a": "意甲",
            "ligue_1": "法甲"
        }.get(league, league)
        
        # 初始化模型
        m1 = EloDCModel(league)
        m2 = PoissonModel(league)
        m3 = MarketModel()
        m4 = FormModel()
        models = [m1, m2, m3, m4]
        
        # 预热
        for m in matches[:50]:
            for mod in models:
                mod.update(m)
        
        bets = []
        bal = bankroll
        peak = bankroll
        mdd = 0
        
        print(f"  {ln}: {len(matches)}场")
        
        for i, m in enumerate(matches[50:], 50):
            # 收集各模型预测
            preds = {}
            for mod in models:
                p = mod.predict(m)
                if p:
                    preds[id(mod)] = p
            
            if len(preds) < 2:
                for mod in models:
                    mod.update(m)
                continue
            
            # ── 1X2 策略 ──
            if enable_1x2:
                bet = evaluate_1x2(m, preds, consensus, value_thresh_1x2, bal, kelly_fraction, max_stake_pct)
                if bet:
                    bets.append(bet)
                    bal += bet.profit
                    if bal > peak:
                        peak = bal
                    dd = (peak - bal) / peak if peak > 0 else 0
                    if dd > mdd:
                        mdd = dd
            
            # ── 大小球策略 ──
            if enable_ou:
                bet = evaluate_overunder(m, preds, consensus, value_thresh_ou, bal, kelly_fraction, max_stake_pct)
                if bet:
                    bets.append(bet)
                    bal += bet.profit
                    if bal > peak:
                        peak = bal
                    dd = (peak - bal) / peak if peak > 0 else 0
                    if dd > mdd:
                        mdd = dd
            
            # ── 亚盘策略 ──
            if enable_ah:
                bet = evaluate_asian(m, preds, consensus, value_thresh_ah, bal, kelly_fraction, max_stake_pct)
                if bet:
                    bets.append(bet)
                    bal += bet.profit
                    if bal > peak:
                        peak = bal
                    dd = (peak - bal) / peak if peak > 0 else 0
                    if dd > mdd:
                        mdd = dd
            
            # 更新模型
            for mod in models:
                mod.update(m)
        
        # 输出联赛结果
        if bets:
            print_league_results(ln, bets, mdd)
            all_results[league] = {
                "bets": len(bets),
                "wins": sum(1 for b in bets if b.won),
                "staked": sum(b.stake for b in bets),
                "returned": sum(b.stake + b.profit for b in bets if b.won),
                "mdd": mdd,
            }
        else:
            print(f"  {ln}: 无投注")
            all_results[league] = {"bets": 0, "wins": 0, "staked": 0, "returned": 0, "mdd": 0}
        
        all_bets.extend(bets)
    
    # 全局汇总
    print_global_summary(all_bets, all_results)
    
    return all_bets, all_results


def evaluate_1x2(
    m: RealOddsMatch,
    preds: Dict,
    consensus: int,
    value_thresh: float,
    bal: float,
    kelly_fraction: float,
    max_stake_pct: float,
) -> Optional[BetRecord]:
    """评估1X2策略"""
    dirs = ["home", "draw", "away"]
    votes = {d: 0 for d in dirs}
    probs = {d: [] for d in dirs}
    
    for pred in preds.values():
        best = max(dirs, key=lambda d: pred.get(d, 0))
        votes[best] += 1
        for d in dirs:
            probs[d].append(pred.get(d, 0))
    
    best_dir = max(dirs, key=lambda d: votes[d])
    if votes[best_dir] < consensus:
        return None
    
    cp = sum(probs[best_dir]) / len(probs[best_dir])
    odds = get_real_odds_1x2(m, best_dir)
    if odds <= 1:
        return None
    
    # 市场隐含概率
    mg = 1.0 / m.b365h + 1.0 / m.b365d + 1.0 / m.b365a if m.b365h > 1 else 0
    ip = (1.0 / odds) / mg if mg > 0 else 0
    
    value = cp - ip
    if value < value_thresh:
        return None
    
    # Kelly 公式
    b = odds - 1
    fk = (b * cp - (1 - cp)) / b if b > 0 else 0
    stake = min(bal * fk * kelly_fraction, bal * max_stake_pct)
    if stake < 10:
        return None
    
    rmap = {"H": "home", "D": "draw", "A": "away"}
    won = best_dir == rmap.get(m.ftr, "")
    profit = stake * (odds - 1) if won else -stake
    
    return BetRecord(
        match_date=m.date,
        league=m.league,
        home_team=m.home_team,
        away_team=m.away_team,
        strategy="1x2",
        direction=best_dir,
        odds=odds,
        stake=stake,
        won=won,
        profit=profit,
        model_prob=cp,
        market_prob=ip,
        value=value,
        agreement=votes[best_dir],
        total_models=len(preds),
    )


def evaluate_overunder(
    m: RealOddsMatch,
    preds: Dict,
    consensus: int,
    value_thresh: float,
    bal: float,
    kelly_fraction: float,
    max_stake_pct: float,
) -> Optional[BetRecord]:
    """评估大小球策略"""
    # 计算 over/under 概率
    over_probs = []
    under_probs = []
    
    for pred in preds.values():
        avg_goals = pred.get("home_lambda", 0) + pred.get("away_lambda", 0)
        if avg_goals <= 0:
            avg_goals = 2.65
        over_p = calculate_over_prob(avg_goals, 2.5)
        over_probs.append(over_p)
        under_probs.append(1.0 - over_p)
    
    avg_over = sum(over_probs) / len(over_probs)
    avg_under = sum(under_probs) / len(under_probs)
    
    # 投票
    over_votes = sum(1 for p in over_probs if p > 0.5)
    under_votes = len(over_probs) - over_votes
    
    if over_votes >= consensus:
        direction = "over"
        cp = avg_over
        votes = over_votes
    elif under_votes >= consensus:
        direction = "under"
        cp = avg_under
        votes = under_votes
    else:
        return None
    
    odds, line = get_real_odds_overunder(m, direction)
    if odds <= 1:
        return None
    
    # 市场隐含概率
    if m.b365_over25 > 1 and m.b365_under25 > 1:
        mg = 1.0 / m.b365_over25 + 1.0 / m.b365_under25
        ip = (1.0 / odds) / mg
    else:
        ip = 0.5
    
    value = cp - ip
    if value < value_thresh:
        return None
    
    b = odds - 1
    fk = (b * cp - (1 - cp)) / b if b > 0 else 0
    stake = min(bal * fk * kelly_fraction, bal * max_stake_pct)
    if stake < 10:
        return None
    
    actual_total = m.fthg + m.ftag
    if direction == "over":
        won = actual_total > 2.5
    else:
        won = actual_total < 2.5
    
    profit = stake * (odds - 1) if won else -stake
    
    return BetRecord(
        match_date=m.date,
        league=m.league,
        home_team=m.home_team,
        away_team=m.away_team,
        strategy="over_under",
        direction=direction,
        odds=odds,
        stake=stake,
        won=won,
        profit=profit,
        model_prob=cp,
        market_prob=ip,
        value=value,
        agreement=votes,
        total_models=len(preds),
        handicap_line=line,
    )


def evaluate_asian(
    m: RealOddsMatch,
    preds: Dict,
    consensus: int,
    value_thresh: float,
    bal: float,
    kelly_fraction: float,
    max_stake_pct: float,
) -> Optional[BetRecord]:
    """评估亚盘策略"""
    if m.ahh == 0 and m.ahch == 0:
        return None
    
    line = m.ahh if m.ahh != 0 else m.ahch
    
    # 计算主队覆盖概率
    home_cover_probs = []
    away_cover_probs = []
    
    for pred in preds.values():
        score_matrix = pred.get("score_matrix", {})
        if not score_matrix:
            continue
        
        home_cover = 0.0
        away_cover = 0.0
        for (h, a), p in score_matrix.items():
            diff = h - a
            if diff > -line:  # 主队赢盘
                home_cover += p
            elif diff < -line:  # 客队赢盘
                away_cover += p
            else:  # 走水
                home_cover += p * 0.5
                away_cover += p * 0.5
        
        home_cover_probs.append(home_cover)
        away_cover_probs.append(away_cover)
    
    if not home_cover_probs:
        return None
    
    avg_home = sum(home_cover_probs) / len(home_cover_probs)
    avg_away = sum(away_cover_probs) / len(away_cover_probs)
    
    home_votes = sum(1 for p in home_cover_probs if p > 0.5)
    away_votes = len(home_cover_probs) - home_votes
    
    if home_votes >= consensus:
        direction = "home"
        cp = avg_home
        votes = home_votes
    elif away_votes >= consensus:
        direction = "away"
        cp = avg_away
        votes = away_votes
    else:
        return None
    
    odds, _ = get_real_odds_asian(m, direction)
    if odds <= 1:
        return None
    
    # 市场隐含概率
    if m.b365ahh > 1 and m.b365aha > 1:
        mg = 1.0 / m.b365ahh + 1.0 / m.b365aha
        ip = (1.0 / odds) / mg
    else:
        ip = 0.5
    
    value = cp - ip
    if value < value_thresh:
        return None
    
    b = odds - 1
    fk = (b * cp - (1 - cp)) / b if b > 0 else 0
    stake = min(bal * fk * kelly_fraction, bal * max_stake_pct)
    if stake < 10:
        return None
    
    # 亚盘结算
    actual_diff = m.fthg - m.ftag
    if direction == "home":
        won = actual_diff > -line
    else:
        won = actual_diff < -line
    
    profit = stake * (odds - 1) if won else -stake
    
    return BetRecord(
        match_date=m.date,
        league=m.league,
        home_team=m.home_team,
        away_team=m.away_team,
        strategy="asian_handicap",
        direction=direction,
        odds=odds,
        stake=stake,
        won=won,
        profit=profit,
        model_prob=cp,
        market_prob=ip,
        value=value,
        agreement=votes,
        total_models=len(preds),
        handicap_line=line,
    )


def print_league_results(ln: str, bets: List[BetRecord], mdd: float):
    """输出联赛回测结果"""
    ws = sum(1 for b in bets if b.won)
    ss = sum(b.stake for b in bets)
    rs = sum(b.stake + b.profit for b in bets if b.won)
    roi = (rs - ss) / ss if ss > 0 else 0
    
    print(f"\n  {ln}: 投注={len(bets)} 胜={ws} 胜率={ws / len(bets):.1%} ROI={roi:+.1%} 回撤={mdd:.1%}")
    
    # 按策略分
    by_strategy = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0, "returned": 0})
    for b in bets:
        s = by_strategy[b.strategy]
        s["bets"] += 1
        if b.won:
            s["wins"] += 1
        s["staked"] += b.stake
        s["returned"] += b.stake + b.profit if b.won else 0
    
    for strategy, s in by_strategy.items():
        roi_s = (s["returned"] - s["staked"]) / s["staked"] if s["staked"] > 0 else 0
        name = {"1x2": "1X2", "over_under": "大小球", "asian_handicap": "亚盘"}.get(strategy, strategy)
        print(f"    {name:8s}  投注={s['bets']:3d}  胜率={s['wins'] / s['bets']:.1%}  ROI={roi_s:+.1%}")
    
    # 按共识级别
    by_agreement = defaultdict(lambda: {"bets": 0, "wins": 0})
    for b in bets:
        a = by_agreement[b.agreement]
        a["bets"] += 1
        if b.won:
            a["wins"] += 1
    
    for ag in sorted(by_agreement):
        d = by_agreement[ag]
        if d["bets"] > 0:
            print(f"    {ag}票共识: {d['bets']}注 胜率={d['wins'] / d['bets']:.1%}")


def print_global_summary(all_bets: List[BetRecord], all_results: Dict):
    """输出全局汇总"""
    sep = '#' * 70
    
    if not all_bets:
        print(f"\n{sep}\n  无投注记录\n{sep}")
        return
    
    total_staked = sum(b.stake for b in all_bets)
    total_returned = sum(b.stake + b.profit for b in all_bets if b.won)
    total_wins = sum(1 for b in all_bets if b.won)
    roi = (total_returned - total_staked) / total_staked if total_staked > 0 else 0
    
    print(f"\n{sep}")
    print(f"  全联汇总: 投注={len(all_bets)} 胜={total_wins} 胜率={total_wins / len(all_bets):.1%} ROI={roi:+.1%}")
    
    # 按联赛
    for league, r in all_results.items():
        ln = {
            "premier_league": "英超",
            "la_liga": "西甲",
            "bundesliga": "德甲",
            "serie_a": "意甲",
            "ligue_1": "法甲"
        }.get(league, league)
        roi_l = (r["returned"] - r["staked"]) / r["staked"] if r["staked"] > 0 else 0
        if r["bets"] > 0:
            print(f"  {ln:8s}  投注={r['bets']:3d}  胜率={r['wins'] / r['bets']:.1%}  ROI={roi_l:+.1%}")
        else:
            print(f"  {ln:8s}  无投注")
    
    # 按策略
    print(f"\n  按策略:")
    by_strategy = defaultdict(lambda: {"bets": 0, "wins": 0, "staked": 0, "returned": 0})
    for b in all_bets:
        s = by_strategy[b.strategy]
        s["bets"] += 1
        if b.won:
            s["wins"] += 1
        s["staked"] += b.stake
        s["returned"] += b.stake + b.profit if b.won else 0
    
    for strategy, s in by_strategy.items():
        roi_s = (s["returned"] - s["staked"]) / s["staked"] if s["staked"] > 0 else 0
        name = {"1x2": "1X2", "over_under": "大小球", "asian_handicap": "亚盘"}.get(strategy, strategy)
        print(f"  {name:8s}  投注={s['bets']:4d}  胜率={s['wins'] / s['bets']:.1%}  ROI={roi_s:+.1%}")


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GTO v5.11 全策略真实赔率回测")
    p.add_argument("--leagues", nargs="+", default=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"])
    p.add_argument("--seasons", nargs="+", default=["2021-22", "2022-23", "2023-24"])
    p.add_argument("--bankroll", type=float, default=10000.0)
    p.add_argument("--consensus", type=int, default=3, help="最少共识模型数")
    p.add_argument("--value-1x2", type=float, default=0.05, help="1X2价值阈值")
    p.add_argument("--value-ou", type=float, default=0.05, help="大小球价值阈值")
    p.add_argument("--value-ah", type=float, default=0.05, help="亚盘价值阈值")
    p.add_argument("--no-1x2", action="store_true", help="禁用1X2策略")
    p.add_argument("--no-ou", action="store_true", help="禁用大小球策略")
    p.add_argument("--no-ah", action="store_true", help="禁用亚盘策略")
    p.add_argument("--kelly", type=float, default=0.25, help="Kelly系数")
    p.add_argument("--max-stake", type=float, default=0.03, help="最大仓位比例")
    
    a = p.parse_args()
    
    run_full_backtest(
        leagues=a.leagues,
        seasons=a.seasons,
        bankroll=a.bankroll,
        consensus=a.consensus,
        value_thresh_1x2=a.value_1x2,
        value_thresh_ou=a.value_ou,
        value_thresh_ah=a.value_ah,
        enable_1x2=not a.no_1x2,
        enable_ou=not a.no_ou,
        enable_ah=not a.no_ah,
        kelly_fraction=a.kelly,
        max_stake_pct=a.max_stake,
    )
