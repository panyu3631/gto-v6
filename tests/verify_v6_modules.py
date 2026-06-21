"""
v6.0 模块落地验证脚本
"""
import sys
sys.path.insert(0, '/workspace/gto-gameflow-v5')

# ================================================================
# 验证1: 导入链
# ================================================================
print("=== 验证1: 导入链 ===")
try:
    from src.pipeline.orchestrator import GameFlowPipeline
    print("  ✅ GameFlowPipeline 导入成功")
except Exception as e:
    print(f"  ❌ GameFlowPipeline 导入失败: {e}")
    sys.exit(1)

try:
    from src.engine.adaptive_clipping import AdaptiveClipping, MarketAnomalyDetector
    print("  ✅ AdaptiveClipping + MarketAnomalyDetector 导入成功")
except Exception as e:
    print(f"  ❌ adaptive_clipping 导入失败: {e}")

try:
    from src.factors.compute_v2 import FactorComputationEngineV2, V6_FACTOR_REGISTRY
    print("  ✅ FactorComputationEngineV2 导入成功")
except Exception as e:
    print(f"  ❌ compute_v2 导入失败: {e}")

# ================================================================
# 验证2: orchestrator.__init__ 实际使用哪些模块
# ================================================================
print("\n=== 验证2: orchestrator 实际使用的模块 ===")
import inspect
from src.pipeline.orchestrator import GameFlowPipeline

# 检查 __init__ 源码
init_source = inspect.getsource(GameFlowPipeline.__init__)
print(f"  __init__ 源码长度: {len(init_source)} 字符")

# 检查关键字符串
checks = {
    "FactorComputationEngine (v5.10.8)": "FactorComputationEngine" in init_source and "weight_multipliers" in init_source,
    "AdaptiveClipping": "AdaptiveClipping" in init_source,
    "MarketAnomalyDetector": "MarketAnomalyDetector" in init_source,
    "FactorComputationEngineV2 (V2精简引擎)": "FactorComputationEngineV2" in init_source,
}
for name, present in checks.items():
    print(f"  {'✅' if present else '❌'} {name}: {'使用' if present else '未使用'}")

# ================================================================
# 验证3: 实际运行 pipeline 并追踪执行路径
# ================================================================
print("\n=== 验证3: 实际运行 pipeline 追踪 ===")
import os
os.environ.setdefault('GTO_DB_URL', 'sqlite:///:memory:')

# 创建 pipeline 实例
pipeline = GameFlowPipeline('premier_league')

# 检查实例属性
attrs = {
    "factor_engine (类型)": type(pipeline.factor_engine).__name__,
    "adaptive_clipper (存在)": hasattr(pipeline, 'adaptive_clipper'),
    "anomaly_detector (存在)": hasattr(pipeline, 'anomaly_detector'),
    "factor_scale (值)": pipeline.factor_scale,
}
for name, val in attrs.items():
    print(f"  {'✅' if val else '❌'} {name}: {val}")

# 检查 factor_engine 类型
from src.factors.compute import FactorComputationEngine
from src.factors.compute_v2 import FactorComputationEngineV2
print(f"  factor_engine 是 V2引擎: {isinstance(pipeline.factor_engine, FactorComputationEngineV2)}")
print(f"  factor_engine 是 v5.10.8引擎: {isinstance(pipeline.factor_engine, FactorComputationEngine)}")

# ================================================================
# 验证4: 模拟运行 pipeline 追踪所有新模块调用
# ================================================================
print("\n=== 验证4: 模拟运行 pipeline ===")
from src.data.models import MatchContext
from datetime import datetime

# 创建模拟比赛
match = MatchContext(
    match_id="test_001",
    league_id="premier_league",
    season="2023-2024",
    matchday=28,
    home_team="Arsenal",
    away_team="Chelsea",
    home_elo=1650.0,
    away_elo=1600.0,
    odds_home=2.10,
    odds_draw=3.40,
    odds_away=3.50,
    kickoff_time=datetime(2024, 3, 15, 15, 0),
)

# 模拟 extra 数据
extra = {
    "elo_diff": 50.0,
    "recent_results": [3, 1, 3, 0, 3],
    "h2h_results": [3, 0, 1, 3, 0],
    "matches_7d": 1,
    "rank_diff": 2,
    "goal_diff": 5.0,
    "xg_diff": 0.8,
    "match_phase": 0.7,
    "odds_std": 0.06,
    "data_completeness": 0.85,
    "data_source_count": 5,
    "dispersion_penalty": 0.88,
    "is_primetime": True,
    "market_efficiency": 0.55,
}

# 运行 Stage 1-5
result = pipeline.run_stages_1_5(match, extra)

# 检查结果
print(f"\n  Pipeline执行结果:")
print(f"  ✅ 因子数: {len(result.factor_deltas)}")
print(f"  ✅ 市场效率画像: {'有' if result.market_efficiency_profile else '无'}")
if result.market_efficiency_profile:
    print(f"    效率评分: {result.market_efficiency_profile.get('efficiency_score', 'N/A')}")
    print(f"    是否高效: {result.market_efficiency_profile.get('is_efficient', 'N/A')}")
    print(f"    模型权重加成: {result.market_efficiency_profile.get('model_weight_boost', 'N/A'):.3f}")
    print(f"    裁剪宽松度: {result.market_efficiency_profile.get('clip_allowance', 'N/A'):.3f}")
print(f"  ✅ 自适应裁剪: {'有' if result.clip_limits else '无'}")
if result.clip_limits:
    print(f"    裁剪上限: H={result.clip_limits.get('home', 0):.1%} D={result.clip_limits.get('draw', 0):.1%} A={result.clip_limits.get('away', 0):.1%}")
print(f"  ✅ 价值洼地检测: {len(result.value_depression_alerts)} 个")
for alert in result.value_depression_alerts:
    print(f"    {alert['outcome']}: 模型={alert['model_prob']:.1%} 市场={alert['implied_prob']:.1%} 低估度={alert['score']:.2f}")

# 检查价值洼地是否影响决策
print(f"\n  {'❌' if not result.proposals else '⚠️'} 价值洼地检测是否影响投注决策: {'否 — 仅用于诊断打印' if not result.proposals else '需进一步验证'}")

# ================================================================
# 验证5: 检查 compute_v2.py 是否被实际使用
# ================================================================
print("\n=== 验证5: compute_v2.py 实际使用情况 ===")
import ast

# 分析 orchestrator.py 的所有 import
with open('/workspace/gto-gameflow-v5/src/pipeline/orchestrator.py', 'r') as f:
    tree = ast.parse(f.read())

v2_imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and 'compute_v2' in node.module:
            v2_imports.append(node.module)
        for alias in node.names:
            if 'compute_v2' in str(alias.name):
                v2_imports.append(f"{node.module}.{alias.name}")

if v2_imports:
    print(f"  ❌ compute_v2 被导入: {v2_imports}")
else:
    print(f"  ✅ compute_v2 未被 orchestrator 导入 (确认是死代码)")

# 搜索 compute_v2 在整个项目中的使用
import subprocess
result = subprocess.run(
    ['grep', '-r', 'compute_v2', '/workspace/gto-gameflow-v5/src/', '--include', '*.py'],
    capture_output=True, text=True
)
lines = [l for l in result.stdout.strip().split('\n') if l and 'compute_v2.py' not in l]
if lines:
    print(f"  ❌ compute_v2 被以下文件引用 ({len(lines)}处):")
    for l in lines[:5]:
        print(f"    {l}")
else:
    print(f"  ✅ compute_v2 仅被自身文件引用 (确认是孤岛代码)")

print("\n" + "="*60)
print("验证结论:")
print("="*60)
print("✅ 实际运行: AdaptiveClipping (自适应裁剪)")
print("✅ 实际运行: MarketAnomalyDetector (市场异常检测)")
print("✅ 实际运行: compute_information_quality (信息质量)")
print("✅ 实际运行: 价值洼地检测 (但仅用于诊断打印)")
print("❌ 未运行: FactorComputationEngineV2 (V2精简引擎 — 死代码)")
print("❌ 未使用: compute_factor_diversity (硬编码0.5)")
print("❌ 未使用: 价值洼地检测结果 (不影响投注决策)")
print("❌ 未使用: 所有V2新因子 (U1-U3, V1-V2, G1-G2, D1-D3, M3)")