"""
意甲细粒度校准: 在最优区域附近搜索 + 客场价值折扣调优
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.calibration.cross_season_calibrator import (
    CrossSeasonCalibrator, CalibrationParams, CalibrationResult,
    compute_calibration_score,
)

LEAGUE = "serie_a"

# 细粒度网格: 围绕最佳区域 (0.6x, v=0.03, conf=0.60, α=0.60/0.15)
param_grid = [
    # 基准: 原校准结果
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},

    # 权重微调: 0.5-0.8
    {"base_weight_mult": 0.5, "enhanced_weight_mult": 0.5, "league_weight_mult": 0.5,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
    {"base_weight_mult": 0.7, "enhanced_weight_mult": 0.7, "league_weight_mult": 0.7,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
    {"base_weight_mult": 0.8, "enhanced_weight_mult": 0.8, "league_weight_mult": 0.8,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},

    # 提高价值阈值 (减少中差距投注)
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.04, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.05, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},

    # 提高置信度阈值 (过滤低质量信号)
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.65, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.70, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},

    # 更保守的 Kelly
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.15,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.20,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},

    # 更强的先验收缩 (减少客场偏差)
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.70, "shrinkage_alpha_low": 0.20},
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.75, "shrinkage_alpha_low": 0.25},

    # 组合: 高阈值 + 高收缩
    {"base_weight_mult": 0.6, "enhanced_weight_mult": 0.6, "league_weight_mult": 0.6,
     "value_threshold": 0.04, "confidence_threshold": 0.65, "kelly_fraction": 0.20,
     "shrinkage_alpha_high": 0.70, "shrinkage_alpha_low": 0.20},

    # 组合: 低权重 + 高阈值
    {"base_weight_mult": 0.5, "enhanced_weight_mult": 0.5, "league_weight_mult": 0.5,
     "value_threshold": 0.04, "confidence_threshold": 0.65, "kelly_fraction": 0.25,
     "shrinkage_alpha_high": 0.60, "shrinkage_alpha_low": 0.15},

    # 组合: 中权重 + 中阈值 + 低 Kelly
    {"base_weight_mult": 0.7, "enhanced_weight_mult": 0.7, "league_weight_mult": 0.7,
     "value_threshold": 0.03, "confidence_threshold": 0.60, "kelly_fraction": 0.20,
     "shrinkage_alpha_high": 0.70, "shrinkage_alpha_low": 0.20},
]

if __name__ == "__main__":
    print("=" * 70)
    print(f"v5.3b 意甲细粒度校准: {LEAGUE}")
    print(f"参数组: {len(param_grid)}")
    print("=" * 70)

    calibrator = CrossSeasonCalibrator(LEAGUE, seed=42)
    result = calibrator.calibrate(param_grid)

    # 保存
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'reports',
                              'calibrated_weights_v53b.json')
    calib_data = {}
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib_data = json.load(f)

    calib_data[LEAGUE] = {
        "base_weight_mult": result.params.base_weight_mult,
        "enhanced_weight_mult": result.params.enhanced_weight_mult,
        "league_weight_mult": result.params.league_weight_mult,
        "value_threshold": result.params.value_threshold,
        "confidence_threshold": result.params.confidence_threshold,
        "kelly_fraction": result.params.kelly_fraction,
        "shrinkage_alpha_high": result.params.shrinkage_alpha_high,
        "shrinkage_alpha_low": result.params.shrinkage_alpha_low,
        "train_sharpe": round(result.train_sharpe, 4),
        "val_sharpe": round(result.val_sharpe, 4),
        "val_roi": round(result.val_roi, 4),
        "val_bets": result.val_bets,
        "score": round(result.score, 4),
    }

    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    print(f"\n  意甲最优参数已保存到 {calib_path}")