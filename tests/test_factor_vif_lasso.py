"""
GTO-GameFlow v5.9.2 — 因子 VIF 分析 + LASSO 降维
- 生成大样本合成数据集
- 计算 VIF（方差膨胀因子）检测多重共线性
- LASSO 回归 + 交叉验证识别有价值的因子
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np
from typing import Dict, List, Tuple

from src.factors.registry import FACTOR_REGISTRY, get_active_factors
from src.factors.compute import FactorComputationEngine
from src.config.settings import GlobalConfig
version = GlobalConfig.version

def generate_synthetic_sample(
    n_samples: int = 5000,
    league_id: str = "premier_league",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """生成合成样本用于因子分析"""
    rng = random.Random(seed)
    np.random.seed(seed)

    engine = FactorComputationEngine(league_id)
    active = get_active_factors(league_id)
    exclude = {'F14', 'F30', 'F27'}
    factor_names = sorted([f for f in active if f not in exclude])

    n_factors = len(factor_names)
    X = np.zeros((n_samples, n_factors))
    y = np.zeros(n_samples, dtype=int)

    factor_name_to_idx = {f: i for i, f in enumerate(factor_names)}

    for i in range(n_samples):
        try:
            elo_diff = rng.gauss(0, 150)
            xi_rating = rng.gauss(0, 1.0)
            recent = [rng.choice([3.0, 3.0, 1.0, 0.0, 0.0]) for _ in range(5)]
            h2h = [rng.choice([1.0, 3.0, 0.0, 3.0, 1.0]) for _ in range(5)]
            matches_7d = rng.randint(0, 3)
            rank_diff = int(rng.gauss(0, 10))
            goal_diff = rng.gauss(0, 12)
            xg_diff = rng.gauss(0, 0.8)

            # 市场概率
            o_h = rng.uniform(1.5, 3.5)
            o_d = rng.uniform(2.5, 4.5)
            o_a = rng.uniform(2.5, 5.0)
            imp = 1/o_h + 1/o_d + 1/o_a
            mkt = {"home": (1/o_h)/imp, "draw": (1/o_d)/imp, "away": (1/o_a)/imp}

            o_h2 = o_h + rng.uniform(-0.2, 0.2)
            o_d2 = o_d + rng.uniform(-0.2, 0.2)
            o_a2 = o_a + rng.uniform(-0.2, 0.2)
            imp2 = 1/o_h2 + 1/o_d2 + 1/o_a2
            opening = {"home": (1/o_h2)/imp2, "draw": (1/o_d2)/imp2, "away": (1/o_a2)/imp2}

            deltas = engine.compute_all(
                elo_diff=elo_diff,
                xi_rating=xi_rating,
                recent_results=recent,
                h2h_results=h2h,
                matches_7d=matches_7d,
                rank_diff=rank_diff,
                goal_diff=goal_diff,
                xg_diff=xg_diff,
                market_probs=mkt,
                opening_probs=opening,
                weather=rng.uniform(-0.5, 0.5),
                ref_yellow_rate=rng.uniform(2.5, 5.5),
                coach_change_effect=rng.uniform(-0.3, 0.3),
                fatigue_penalty=rng.uniform(-0.2, 0.2),
                rotation_risk=rng.uniform(-0.15, 0.15),
                derby_boost=rng.uniform(0, 0.15),
                style_matchup_score=rng.uniform(0.3, 0.7),
                streak_momentum=rng.uniform(-0.3, 0.3),
                player_form=rng.uniform(5.0, 8.0),
                market_sentiment=rng.uniform(-0.2, 0.2),
                odds_std=rng.uniform(0.01, 0.15),
                nlp_sentiment=rng.uniform(-0.2, 0.2),
                time_decay_factor=rng.uniform(0.5, 1.0),
                league_strength_bias=rng.uniform(0.8, 1.2),
                poisson_correction=0.0,
                handicap_depth=rng.uniform(-0.5, 0.5),
                totals_trend=rng.uniform(-0.5, 0.5),
                value_signal=rng.uniform(-0.1, 0.1),
                contrarian_signal=rng.uniform(-0.2, 0.2),
                market_efficiency=rng.uniform(0.0, 0.1),
                motivation_boost=rng.uniform(-50, 50),
                financial_gap_effect=rng.uniform(-50, 50),
                winter_break_effect=rng.uniform(-0.2, 0.2),
                christmas_fatigue=rng.uniform(-0.3, 0.3),
                complacency_effect=rng.uniform(-0.1, 0.1),
                streak_momentum_league=rng.uniform(-0.3, 0.3),
                position_advantage=rng.uniform(-10, 10),
                promoted_team_delta=rng.uniform(-0.3, 0.3),
                schedule_advantage=rng.uniform(-0.3, 0.3),
                derby_intensity=rng.uniform(0, 0.15),
            )

            # 填入 X 矩阵
            for fid, d in deltas.items():
                if fid in factor_name_to_idx:
                    X[i, factor_name_to_idx[fid]] = d.get("home", 0.0)

            # 生成真实结果
            p_h = 1.0 / (1.0 + 10 ** (-(elo_diff + 65) / 400.0))
            p_h = np.clip(p_h + np.sum(X[i, :]) * 0.01, 0.1, 0.9)
            p_d = 0.25
            p_a = 1.0 - p_h - p_d
            r = rng.random()
            if r < p_h:
                y[i] = 0
            elif r < p_h + p_d:
                y[i] = 1
            else:
                y[i] = 2
        except Exception:
            X[i, :] = 0.0

    valid = np.any(X != 0, axis=1)
    return X[valid], y[valid], factor_names

def compute_vif(X: np.ndarray, factor_names: List[str]) -> Dict[str, float]:
    """计算 VIF"""
    n, p = X.shape
    vif = {}
    X1 = np.column_stack([np.ones(n), X])
    for i in range(p):
        y_i = X[:, i]
        Xo = np.delete(X1, i+1, axis=1)
        try:
            beta = np.linalg.lstsq(Xo, y_i, rcond=None)[0]
            yp = Xo @ beta
            ssr = np.sum((y_i - yp)**2)
            sst = np.sum((y_i - y_i.mean())**2)
            r2 = max(0, min(1 - ssr/sst if sst > 0 else 0, 0.999))
            vif[factor_names[i]] = 1.0/(1.0 - r2)
        except:
            vif[factor_names[i]] = float('inf')
    return vif

def lasso_analysis(X, y, factor_names, alpha=0.01):
    """LASSO 特征选择"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(
        penalty='l1', solver='saga', C=1.0/alpha,
        multi_class='multinomial', max_iter=5000, random_state=42, tol=1e-4,
    )
    model.fit(Xs, y)
    coef = model.coef_

    selected = []
    coefs = {}
    for i, fid in enumerate(factor_names):
        coefs[fid] = {"home": float(coef[0,i]), "draw": float(coef[1,i]), "away": float(coef[2,i])}
        if np.any(np.abs(coef[:, i]) > 1e-6):
            selected.append(fid)

    cv = cross_val_score(model, Xs, y, cv=5)
    return {
        "coefficients": coefs,
        "selected_factors": selected,
        "cv_mean": float(np.mean(cv)),
        "cv_std": float(np.std(cv)),
        "alpha": alpha,
    }

if __name__ == "__main__":
    print("=" * 78)
    print(f"  GTO-GameFlow {version} 因子 VIF 分析 + LASSO 降维")
    print("=" * 78)

    print("\n[1/3] 生成合成样本 (5000 场)...")
    X, y, factor_names = generate_synthetic_sample(5000, "premier_league", 42)
    print(f"  有效样本: {X.shape[0]}, 因子数: {X.shape[1]}")
    print(f"  赛果分布: 主胜={np.sum(y==0)}, 平={np.sum(y==1)}, 客胜={np.sum(y==2)}")

    print("\n[2/3] VIF 分析...")
    vif = compute_vif(X, factor_names)
    high = [(f,v) for f,v in vif.items() if v > 10]
    med = [(f,v) for f,v in vif.items() if 5 < v <= 10]
    low = [(f,v) for f,v in vif.items() if v <= 5]
    high.sort(key=lambda x: -x[1])
    med.sort(key=lambda x: -x[1])

    print(f"\n  VIF > 10 (严重共线性): {len(high)} 个")
    for f, v in high:
        fd = FACTOR_REGISTRY.get(f)
        name = fd.name_cn if fd else f
        print(f"    {f} {name:<20} VIF={v:.1f}")

    print(f"\n  5 < VIF <= 10 (中等): {len(med)} 个")
    for f, v in med:
        fd = FACTOR_REGISTRY.get(f)
        name = fd.name_cn if fd else f
        print(f"    {f} {name:<20} VIF={v:.1f}")

    print(f"\n  VIF <= 5 (低共线性): {len(low)} 个")

    print("\n[3/3] LASSO 回归...")
    # 尝试多个 alpha
    best = None
    for a in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        r = lasso_analysis(X, y, factor_names, a)
        n = len(r["selected_factors"])
        print(f"  alpha={a:.3f}: 选中 {n} 因子, CV准确率={r['cv_mean']:.3f}±{r['cv_std']:.3f}")
        if 10 <= n <= 25:
            best = r
            break
        if best is None or n > len(best["selected_factors"]):
            best = r

    if best is None:
        best = r

    print(f"\n  === 最终结果 (alpha={best['alpha']}) ===")
    print(f"  选中因子数: {len(best['selected_factors'])}")
    print(f"  CV 准确率: {best['cv_mean']:.3f} ± {best['cv_std']:.3f}")

    # 按重要性排序
    imp = []
    for fid in factor_names:
        if fid in best["coefficients"]:
            c = best["coefficients"][fid]
            m = max(abs(c["home"]), abs(c["draw"]), abs(c["away"]))
            imp.append((fid, m, c, fid in best["selected_factors"]))
    imp.sort(key=lambda x: -x[1])

    print(f"\n  {'#':>3} {'选中':>3} {'因子':<6} {'名称':<20} {'|coef|':>8} {'VIF':>7}")
    print(f"  {'-'*50}")
    for rank, (fid, importance, c, sel) in enumerate(imp[:25], 1):
        fd = FACTOR_REGISTRY.get(fid)
        name = fd.name_cn if fd else fid
        v = vif.get(fid, 0)
        mark = "✓" if sel else "✗"
        print(f"  {rank:>3} {mark:>3} {fid:<6} {name:<20} {importance:>8.4f} {v:>7.1f}")

    # 建议
    keep = []
    prune = []
    for fid in factor_names:
        v = vif.get(fid, 0)
        sel = fid in best["selected_factors"]
        imp_val = max(abs(best["coefficients"].get(fid, {}).get("home", 0)),
                      abs(best["coefficients"].get(fid, {}).get("draw", 0)),
                      abs(best["coefficients"].get(fid, {}).get("away", 0))) if fid in best["coefficients"] else 0
        if sel and v <= 10:
            keep.append(fid)
        elif v > 10 and not sel:
            prune.append(fid)
        elif not sel:
            prune.append(fid)

    print(f"\n  === 建议 ===")
    print(f"  保留: {len(keep)} 个因子 ({', '.join(keep)})")
    print(f"  精简: {len(prune)} 个因子 ({', '.join(prune)})")
    print(f"\n  原始 41 因子 → 分析 {len(factor_names)} 个 → 建议保留 {len(keep)} 个")
    print(f"  VIF>10 严重共线: {len(high)} 个")
    print(f"  LASSO 选中: {len(best['selected_factors'])} 个")
    print(f"\n{'='*78}")
    print(f"  v{version} 因子分析完成")
    print(f"{'='*78}")