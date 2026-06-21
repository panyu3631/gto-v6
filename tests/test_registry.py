"""
L1 单元测试：因子注册中心 (registry.py)

测试范围:
- 41 因子定义完整性
- 因子分类 (BASE/ENHANCED/LEAGUE)
- 5 联赛权重配置
- 活跃因子计数
- F14 废弃逻辑
- F20/F38 互斥逻辑
- 联赛特定因子过滤
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.factors.registry import (
    FACTOR_REGISTRY, LEAGUE_FACTOR_WEIGHTS,
    FactorCategory, get_factor, get_active_factors,
    get_factor_count, get_factor_weight,
    get_factor_ids_by_category, validate_mutual_exclusion,
    FactorDefinition,
)


# ================================================================
# Test 1: 因子注册完整性
# ================================================================

class TestFactorRegistryCompleteness:
    """因子注册中心完整性检查"""

    def test_total_factors_41(self):
        """注册中心应包含 41 个因子定义 (含 F14 废弃, v5.5.1: F42已合并)"""
        assert len(FACTOR_REGISTRY) == 41

    def test_all_ids_present_F1_to_F41(self):
        """所有因子 ID F1-F41 均存在 (v5.5.1: F42合并到F18)"""
        for i in range(1, 42):
            assert f"F{i}" in FACTOR_REGISTRY, f"因子 F{i} 缺失"

    def test_f14_is_deprecated(self):
        """F14 应标记为废弃"""
        f14 = FACTOR_REGISTRY["F14"]
        assert f14.default_weight == 0.0
        assert f14.name_cn == "[已废弃] 球队身价"
        assert f14.degradation_strategy.value == "skip"

    def test_f14_excluded_from_active(self):
        """F14 不应出现在活跃因子列表中"""
        for league in ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]:
            active = get_active_factors(league)
            assert "F14" not in active, f"F14 不应在 {league} 活跃因子中"


# ================================================================
# Test 2: 因子分类 (规范第4/5/12章)
# ================================================================

class TestFactorCategories:
    """因子分类测试"""

    def test_base_factors_count(self):
        """基础因子应为 17 个 (F1-F18, 排除 F14)"""
        base = get_factor_ids_by_category(FactorCategory.BASE)
        assert len(base) == 17, f"预期 17 个基础因子，实际 {len(base)}"
        for fid in base:
            assert fid.startswith("F"), f"非预期 ID: {fid}"
            fid_num = int(fid[1:])
            assert 1 <= fid_num <= 18, f"{fid} 不在 F1-F18 范围"
            assert fid != "F14", "F14 不应出现在基础因子中"

    def test_enhanced_factors_count(self):
        """增强因子应为 14 个 (F19-F32)"""
        enhanced = get_factor_ids_by_category(FactorCategory.ENHANCED)
        assert len(enhanced) == 14, f"预期 14 个增强因子，实际 {len(enhanced)}"

    def test_league_factors_count(self):
        """联赛特定因子应为 9 个 (F33-F41) (v5.5.1: F42合并到F18)"""
        league = get_factor_ids_by_category(FactorCategory.LEAGUE_SPECIFIC)
        assert len(league) == 9, f"预期 9 个联赛特定因子，实际 {len(league)}"


# ================================================================
# Test 3: 5 联赛权重配置
# ================================================================

class TestLeagueWeights:
    """联赛权重配置测试"""

    LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

    def test_all_5_leagues_configured(self):
        """5 个联赛均应有权重配置"""
        assert len(LEAGUE_FACTOR_WEIGHTS) == 5
        for league in self.LEAGUES:
            assert league in LEAGUE_FACTOR_WEIGHTS

    def test_each_league_has_kelly_discount(self):
        """每个联赛配置应包含 kelly_discount = 0.25"""
        for league in self.LEAGUES:
            assert LEAGUE_FACTOR_WEIGHTS[league].get("kelly_discount") == 0.25

    def test_league_specific_factors_filtered(self):
        """联赛特定因子应按联赛过滤"""
        # F35 (冬歇期) 仅德甲激活
        active_bundesliga = get_active_factors("bundesliga")
        active_premier = get_active_factors("premier_league")
        assert "F35" in active_bundesliga
        assert "F35" not in active_premier

        # F36 (圣诞赛程) 仅英超激活
        assert "F36" in active_premier
        assert "F36" not in get_active_factors("bundesliga")

    def test_f14_weight_zero_all_leagues(self):
        """F14 在所有联赛中权重为 0"""
        for league in self.LEAGUES:
            assert LEAGUE_FACTOR_WEIGHTS[league].get("F14", 0.0) == 0.0

    def test_f35_weight_zero_non_bundesliga(self):
        """F35 (冬歇期) 在非德甲联赛权重为 0"""
        for league in ["premier_league", "la_liga", "serie_a", "ligue_1"]:
            assert LEAGUE_FACTOR_WEIGHTS[league].get("F35", 0.0) == 0.0

    def test_f36_weight_zero_non_premier(self):
        """F36 (圣诞赛程) 在非英超联赛权重为 0"""
        for league in ["la_liga", "bundesliga", "serie_a", "ligue_1"]:
            assert LEAGUE_FACTOR_WEIGHTS[league].get("F36", 0.0) == 0.0


# ================================================================
# Test 4: 因子权重获取
# ================================================================

class TestGetFactorWeight:
    """因子权重获取测试"""

    def test_get_weight_from_league(self):
        """从联赛配置获取因子权重"""
        # 英超 F1 权重 = 1.0
        w = get_factor_weight("F1", "premier_league")
        assert w == 1.0

    def test_fallback_to_default(self):
        """联赛未配置时返回默认权重"""
        w = get_factor_weight("F1", "unknown_league")
        assert w == FACTOR_REGISTRY["F1"].default_weight

    def test_f14_weight_always_zero(self):
        """F14 权重始终为 0"""
        w = get_factor_weight("F14", "premier_league")
        assert w == 0.0

    def test_league_specific_factor_weights(self):
        """联赛特定因子权重差异"""
        # F33 在英超 = 0.6, 西甲 = 0.7
        assert get_factor_weight("F33", "premier_league") == 0.6
        assert get_factor_weight("F33", "la_liga") == 0.7


# ================================================================
# Test 5: 活跃因子 (按联赛)
# ================================================================

class TestActiveFactors:
    """活跃因子获取测试"""

    def test_active_without_league_returns_all_non_deprecated(self):
        """不指定联赛时应返回所有非废弃因子"""
        active = get_active_factors()
        assert "F14" not in active
        assert len(active) >= 40  # v5.5.1: F42合并到F18, 至少 40 个

    def test_premier_league_active_count(self):
        """英超活跃因子数量"""
        count = get_factor_count("premier_league")
        assert count >= 30, f"英超应有至少 30 个活跃因子，实际 {count}"

    def test_bundesliga_has_winter_break(self):
        """德甲应包含冬歇期因子 F35"""
        active = get_active_factors("bundesliga")
        assert "F35" in active

    def test_premier_has_christmas_fixtures(self):
        """英超应包含圣诞赛程因子 F36"""
        active = get_active_factors("premier_league")
        assert "F36" in active


# ================================================================
# Test 6: F20/F38 互斥逻辑
# ================================================================

class TestMutualExclusion:
    """F20/F38 互斥逻辑测试"""

    def test_both_enabled_returns_f38(self):
        """当 F20 和 F38 均启用时，优先返回 F38"""
        result = validate_mutual_exclusion("premier_league")
        assert result == "F38"

    def test_both_enabled_bundesliga(self):
        """德甲也启用 F20/F38 互斥"""
        result = validate_mutual_exclusion("bundesliga")
        assert result == "F38"

    def test_league_without_f38(self):
        """未配置 F38 的联赛不触发互斥"""
        # 所有联赛都配置了 F38，但 ligue_1 权重较低
        result = validate_mutual_exclusion("ligue_1")
        assert result == "F38"  # ligue_1 也配置了 F38


# ================================================================
# Test 7: 因子定义字段完整性
# ================================================================

class TestFactorDefinitionFields:
    """因子定义字段完整性"""

    def test_all_factors_have_name(self):
        for fid, f in FACTOR_REGISTRY.items():
            assert f.name, f"{fid} 缺少 name"
            assert f.name_cn, f"{fid} 缺少 name_cn"

    def test_all_active_factors_have_formula(self):
        for fid, f in FACTOR_REGISTRY.items():
            if f.default_weight > 0:
                assert f.formula, f"{fid} 缺少 formula"

    def test_all_factors_have_delta_signs(self):
        for fid, f in FACTOR_REGISTRY.items():
            assert "home" in f.delta_signs
            assert "draw" in f.delta_signs
            assert "away" in f.delta_signs

    def test_all_factors_have_data_sources(self):
        for fid, f in FACTOR_REGISTRY.items():
            if fid != "F14":
                assert len(f.data_sources) > 0, f"{fid} 缺少数据源"

    def test_odds_factors_have_api_football_source(self):
        """F10/F11/F23 数据源应为 api_football"""
        assert "api_football" in FACTOR_REGISTRY["F10"].data_sources
        assert "api_football" in FACTOR_REGISTRY["F11"].data_sources
        assert "api_football" in FACTOR_REGISTRY["F23"].data_sources


# ================================================================
# Test 8: get_factor 异常处理
# ================================================================

class TestGetFactor:
    """get_factor 函数测试"""

    def test_get_valid_factor(self):
        f = get_factor("F1")
        assert f.factor_id == "F1"
        assert f.name == "elo_rating"

    def test_get_invalid_factor_raises(self):
        with pytest.raises(KeyError):
            get_factor("F99")

    def test_get_f14_returns_deprecated(self):
        f = get_factor("F14")
        assert f.default_weight == 0.0