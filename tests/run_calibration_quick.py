"""
GTO-GameFlow v5.3b 快速校准 (全联赛, 少量参数)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.calibration.cross_season_calibrator import (
    CrossSeasonCalibrator, CalibrationParams,
)

LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

param_grid = [
    # 基准线
    {"base_weight_mult": 1.0, "enhanced_weight_mult": 1.0, "league_weight_mult": 1.0,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    # 保守
    {"base_weight_mult": 0.8, "enhanced_weight_mult": 0.8, "league_weight_mult": 0.8,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    # 激进
    {"base_weight_mult": 1.2, "enhanced_weight_mult": 1.2, "league_weight_mult": 1.2,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    # 低阈值
    {"base_weight_mult": 1.0, "enhanced_weight_mult": 1.0, "league_weight_mult": 1.0,
     "value_threshold": 0.02, "confidence_threshold": 0.55, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    {"base_weight_mult": 1.0, "enhanced_weight_mult": 1.0, "league_weight_mult": 1.0,
     "value_threshold": 0.03, "confidence_threshold": 0.50, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    # 低 Kelly
    {"base_weight_mult": 1.0, "enhanced_weight_mult": 1.0, "league_weight_mult": 1.0,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.20,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    # 高收缩
    {"base_weight_mult": 1.0, "enhanced_weight_mult": 1.0, "league_weight_mult": 1.0,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
    # 组合优化
    {"base_weight_mult": 0.8, "enhanced_weight_mult": 0.8, "league_weight_mult": 0.8,
     "value_threshold": 0.02, "confidence_threshold": 0.55, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.50, "shrinkage_alpha_low": 0.10},
    {"base_weight_mult": 0.8, "enhanced_weight_mult": 0.8, "league_weight_mult": 0.8,
     "value_threshold": 0.03, "confidence_threshold": 0.55, "kelly_fraction": 0.20,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
]

if __name__ == "__main__":
    print("=" * 70)
    print("v5.3b 跨赛季校准: 全联赛")
    print(f"参数组: {len(param_grid)}")
    print("=" * 70)

    all_results = {}

    for league_id in LEAGUES:
        calibrator = CrossSeasonCalibrator(league_id, seed=42)
        result = calibrator.calibrate(param_grid)
        all_results[league_id] = result

    # 汇总
    print("\n\n" + "=" * 70)
    print("校准结果汇总")
    print("=" * 70)
    print(f"  {'联赛':<20} {'权重':>5} {'阈值':>6} {'训练S':>7} {'验证S':>7} "
          f"{'验证ROI':>8} {'投注':>5} {'评分':>6}")
    print(f"  {'─'*70}")

    for lid, r in all_results.items():
        print(f"  {lid:<20} {r.params.base_weight_mult:>4.1f}x "
              f"v={r.params.value_threshold:.2f} "
              f"{r.train_sharpe:>+6.2f} {r.val_sharpe:>+6.2f} "
              f"{r.val_roi:>+7.1%} {r.val_bets:>5} {r.score:>5.3f}")

    # 保存校准权重
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'reports',
                              'calibrated_weights_v53b.json')
    os.makedirs(os.path.dirname(calib_path), exist_ok=True)

    calib_data = {}
    for lid, r in all_results.items():
        calib_data[lid] = {
            "base_weight_mult": r.params.base_weight_mult,
            "enhanced_weight_mult": r.params.enhanced_weight_mult,
            "league_weight_mult": r.params.league_weight_mult,
            "value_threshold": r.params.value_threshold,
            "confidence_threshold": r.params.confidence_threshold,
            "kelly_fraction": r.params.kelly_fraction,
            "shrinkage_alpha_high": r.params.shrinkage_alpha_high,
            "shrinkage_alpha_low": r.params.shrinkage_alpha_low,
            "train_sharpe": round(r.train_sharpe, 4),
            "val_sharpe": round(r.val_sharpe, 4),
            "val_roi": round(r.val_roi, 4),
            "val_bets": r.val_bets,
            "score": round(r.score, 4),
        }

    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    print(f"\n  校准权重已保存: {calib_path}")