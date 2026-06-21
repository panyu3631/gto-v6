# GTO-GameFlow v5.10.8 — 赛前数据采集与因子计算架构

> **最后更新**: 2026-06-18  
> **版本**: v5.10.8

---

## 一、赛前数据采集模块总览

GTO-GameFlow 的数据采集分为**三大数据源**，每个模块独立运作，最终汇聚到 Walk-Forward 回测引擎。

```
┌─────────────────────────────────────────────────────────────────┐
│                    数据源架构 (Data Source Architecture)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ 1. CSV 历史数据   │  │ 2. 外部实时API   │  │ 3. 外部数据抓取│ │
│  │  (回测主力)       │  │  (生产环境)       │  │  (增强数据)    │ │
│  ├──────────────────┤  ├──────────────────┤  ├───────────────┤ │
│  │ OddsProvider     │  │ FootballDataClient│  │ XGFetcher     │ │
│  │ EnhancedData-    │  │ ApiFootballClient │  │ (Understat)   │ │
│  │   Provider       │  │                   │  │               │ │
│  │ MatchStats-      │  │ DailyDataPipeline │  │ ExternalData- │ │
│  │   Enricher       │  │ (生产定时任务)     │  │   Collector   │ │
│  │                  │  │                   │  │ (FBref)       │ │
│  └────────┬─────────┘  └────────┬──────────┘  └───────┬───────┘ │
│           │                     │                      │         │
│           └─────────────────────┼──────────────────────┘         │
│                                 ▼                                │
│                    ┌──────────────────────┐                      │
│                    │ Walk-Forward 回测引擎 │                      │
│                    │ (test_phase6_walk_   │                      │
│                    │  forward.py)         │                      │
│                    └──────────┬───────────┘                      │
│                               ▼                                   │
│                    ┌──────────────────────┐                      │
│                    │ GameFlowPipeline     │                      │
│                    │ (orchestrator.py)    │                      │
│                    └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、三大数据源详解

### 2.1 CSV 历史数据 — 回测主力数据源 (v5.10.8 已全面激活)

**数据目录**: `src/data/historical_odds/`  
**覆盖范围**: 五大联赛，2014-15 至 2023-24 赛季，共 55 个 CSV 文件  
**每场数据列数**: 106 列

#### 已读取的 CSV 列 (按模块)

| 模块 | 读取的列 | 用途 |
|------|---------|------|
| **OddsProvider** | PSH/PSD/PSA, AvgH/AvgD/AvgA, BbAHh, Avg>2.5 等 | 赔率数据 |
| **EnhancedDataProvider** | Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, Referee, AHh, B365>2.5, B365<2.5, B365H/D/A, BWH/D/A, IWH/D/A, PSH/D/A, WHH/D/A, VCH/D/A | 积分表 + 18个因子 |
| **MatchStatsEnricher** (v5.10.8 NEW) | 上述全部 + HTHG, HTAG, HTR, B365CH/CD/CA, B365C>2.5, B365C<2.5, B365CAHH, B365CAHA | 14个高维衍生特征 |

#### 数据流路径

```
CSV (106列)
  │
  ├── OddsProvider._parse_csv_row()
  │     └── MatchOddsBundle {odds_home/draw/away, asian_odds, totals_odds}
  │
  ├── EnhancedDataProvider._load_all_data()
  │     ├── 积分表构建 (_build_league_tables)
  │     ├── 24个因子计算 (get_enhanced_data)
  │     └── 动态 data_completeness
  │
  └── MatchStatsEnricher._load_all_data() [v5.10.8 NEW]
        ├── 球队滚动统计 (射门/犯规/角球/黄红牌/半场)
        ├── 裁判统计 (主胜率/黄牌率)
        ├── 赔率漂移统计 (开盘→收盘)
        └── 14个衍生特征 (get_enriched_stats)
```

---

### 2.2 外部实时 API — 生产环境数据源

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| **FootballDataClient** | `src/data/api_client.py` | 生产环境 | football-data.org API v4，获取积分表/赛程/赛果/交锋 |
| **ApiFootballClient** | `src/data/api_client.py` | 生产环境 | API-Football v3，获取赔率/实时比分/球队统计 |
| **DataSourceManager** | `src/data/api_client.py` | 生产环境 | 多源切换和降级管理 |
| **DailyDataPipeline** | `src/data/pipeline.py` | 生产环境 | 每日定时任务，拉取→分析→结算→保存 |

**重要说明**: 这些 API 客户端仅在 `DailyDataPipeline`（生产环境）中使用，**Walk-Forward 回测不使用**。回测直接从 CSV 文件读取历史数据，确保：
- 无未来信息泄露
- 可复现
- 不依赖外部 API 可用性

---

### 2.3 外部数据抓取 — 增强数据源

| 模块 | 文件 | 数据源 | 状态 | 说明 |
|------|------|--------|------|------|
| **XGFetcher** | `src/data/xg_fetcher.py` | Understat.com | ⚠️ 数据格式变更 | 2026年 Understat 改为动态加载，HTTP 解析失效 |
| **ExternalDataCollector** | `src/data/external_data_collector.py` | FBref.com | ⚠️ 被墙 (403) | 需要代理或浏览器才能访问 |
| **RealOddsEnhancer** | `src/data/real_orthogonal_loader.py` | CSV + 合成 | 部分可用 | 赔率部分真实，14个字段仍用随机数 |

**当前替代方案**: 由于外部数据源暂时不可用，v5.10.8 已将所有可用的 CSV 数据（106列）全面激活，通过 `MatchStatsEnricher` 提取了 14 个此前未使用的高维衍生特征。

---

## 三、因子体系 (v5.10.8)

### 3.1 因子总览

| 章节 | 因子范围 | 数量 | 状态 |
|------|---------|------|------|
| 第4章 基础因子 | F1-F18 (F14废弃) | 17 | 全部激活 |
| 第5章 增强因子 | F19-F32 | 14 | 全部激活 |
| 第12章 联赛特定 | F33-F41 | 9 | 全部激活 |
| **第13章 比赛统计衍生** (NEW) | **F42-F55** | **14** | **v5.10.8 新增** |

**总计: 54 个活跃因子** (53 + 1个废弃F14)

### 3.2 v5.10.8 新增因子 (F42-F55)

| 因子ID | 名称 | 数据来源 | 计算逻辑 | 权重 |
|--------|------|---------|---------|------|
| F42 | 半场动量 | CSV HTHG/HTAG/HTR | 半场领先→全场胜率 + 半场落后逆转率 | 0.35 |
| F43 | 射门效率差 | CSV HST/HS | 射正率×转化率复合差值 | 0.40 |
| F44 | 控场优势 | CSV HS/AS/HC/AC/HF/AF | 射门比+角球比+犯规比倒数加权 | 0.35 |
| F45 | 纪律指数 | CSV HY/AY/HR/AR/HF/AF | 黄牌/犯规比+红牌风险 | 0.25 |
| F46 | 赔率漂移信号 | CSV B365CH/B365H | 开盘→收盘赔率变动方向 | 0.45 |
| F47 | 市场分歧 | CSV 6家博彩商 | 赔率标准差，分歧大→平局概率高 | 0.30 |
| F48 | 裁判主场偏置 | CSV Referee | 裁判历史主胜率 vs 联赛平均 | 0.30 |
| F49 | 逆转韧性 | CSV HTHG/HTAG/FTR | 半场落后→全场不输转化率 | 0.25 |
| F50 | 连胜动量(增强) | CSV FTR | 连续胜/负场次累计 | 0.40 |
| F51 | 进球波动率 | CSV FTHG/FTAG | 近期进球数标准差 | 0.20 |
| F52 | 角球优势 | CSV HC/AC | 角球比差值 | 0.20 |
| F53 | 射正率差 | CSV HST/HS | 射正/射门比差值 | 0.30 |
| F54 | 亚盘赔率漂移 | CSV B365CAHH/B365AHH | 亚盘开盘→收盘水位变动 | 0.35 |
| F55 | 大小球赔率漂移 | CSV B365C>2.5/B365>2.5 | 大小球开盘→收盘水位变动 | 0.25 |

### 3.3 数据完整度 (data_completeness)

`data_completeness` 动态计算非零因子数 / 可用因子总数：

- v5.10.7: 24/33 ≈ 72.7%
- **v5.10.8: 38/47 ≈ 80.9%** (+14个新特征，全部有真实数据)

---

## 四、Walk-Forward 回测数据流

### 4.1 完整数据流

```
test_phase6_walk_forward.py (主入口)
  │
  ├── [初始化] OddsProvider (赔率)
  ├── [初始化] EloColdStart → EloTracker (实力评分)
  ├── [初始化] EnhancedDataProvider (积分表+24因子)
  ├── [初始化] MatchStatsEnricher (14个衍生特征) [v5.10.8 NEW]
  │
  └── [每窗口] run_unified_window()
        │
        ├── [每场比赛]
        │     ├── OddsProvider.get_match_odds() → MatchOddsBundle
        │     ├── EloTracker.get_elos() → home_elo, away_elo
        │     ├── TeamStatsTracker → recent_form, goal_diff
        │     ├── 构建 MatchContext
        │     ├── 构建 extra_data (基础字段)
        │     ├── EnhancedDataProvider.get_enhanced_data(stats_enricher=...)
        │     │     ├── 24个因子 (F5-F41)
        │     │     ├── MatchStatsEnricher.get_enriched_stats()
        │     │     │     └── 14个衍生特征 (F42-F55)
        │     │     └── 动态 data_completeness
        │     │
        │     └── GameFlowPipeline.run_full(match_ctx, extra_data)
        │           ├── FactorComputationEngine.compute_all()
        │           │     └── 54个因子的 delta 计算
        │           ├── UnifiedBayesianShrinkage
        │           │     └── 信号分解 + 贝叶斯收缩
        │           ├── MarketRealismIntegrator
        │           │     └── 市场真实化
        │           └── UnifiedDecisionGate + UnifiedBankrollManager
        │                 └── 统一决策 + 资金管理
        │
        └── [窗口结束] LASSO 因子选择
              └── RollingLassoSelector.select()
```

### 4.2 严格避免未来信息泄露

所有模块均遵循 **"仅使用 match_date 之前的数据"** 原则：

- **EnhancedDataProvider**: `_get_table_before()` 仅累计 `date < match_date` 的比赛
- **MatchStatsEnricher**: `_get_team_stats_before()` 截断到 `date < match_date`
- **EloTracker**: 按比赛日顺序更新，仅回看历史
- **TeamStatsTracker**: 按比赛日顺序累积，仅回看历史

---

## 五、赛前数据覆盖度

### 5.1 已激活的赛前数据

| 数据类别 | 来源 | 覆盖因子 | 方法 |
|---------|------|---------|------|
| 球队实力 | CSV + Elo | F1, F3 | Elo 评分追踪 |
| 近期战绩 | CSV | F4, F20, F38 | 近5场结果 |
| 历史交锋 | CSV | F5 | 跨赛季 H2H |
| 赛程密度 | CSV | F6, F41 | 过去7天比赛数 |
| 联赛排名 | CSV | F7, F39 | 积分表排名差 |
| 进球数据 | CSV | F8, F27, F29, F51 | 进球差/泊松修正/大小球趋势/波动率 |
| 射门数据 | CSV | F9, F42, F43, F53 | xG代理/射门效率/射正率 |
| 赔率数据 | CSV | F10, F11, F23, F30, F31, F32, F46, F47 | 市场概率/离散度/价值信号/漂移/分歧 |
| 欧战疲劳 | CSV (日期推算) | F16 | 周中比赛检测 |
| 德比战 | 静态映射 | F18 | 德比配对表 |
| 风格匹配 | CSV | F19, F44, F45, F52 | 射门/犯规/角球/纪律模式 |
| 联赛特定 | CSV + 静态 | F33-F37, F40 | 保级/争冠/冬歇期/圣诞/升班马 |
| 裁判数据 | CSV | F13, F48 | 裁判黄牌率/主场偏置 |
| 半场数据 | CSV | 通过 F42, F49 | 半场动量/逆转韧性 |
| 收盘赔率 | CSV | F46, F54, F55 | 赔率漂移/亚盘漂移/大小球漂移 |

### 5.2 仍需外部数据源的因子

| 因子 | 名称 | 所需数据 | 当前状态 |
|------|------|---------|---------|
| F2 | 核心伤停 | xi_rating | 使用默认值 6.0 |
| F12 | 天气 | weather | 使用默认值 0.0 |
| F15 | 教练更替 | coach_change | 使用默认值 0.0 |
| F17 | 轮换预测 | rotation_risk | 使用默认值 0.0 |
| F21 | 核心球员状态 | player_form | 使用默认值 6.5 |
| F22 | 市场情绪 | market_sentiment | 使用默认值 0.0 |
| F24 | 新闻 NLP | nlp_sentiment | 使用默认值 0.0 |
| F34 | 财力差距 | financial_gap | 使用默认值 0.0 |

---

## 六、文件结构

```
src/data/
  ├── historical_odds/          # 55个CSV文件 (106列/场)
  │     ├── premier_league_*.csv
  │     ├── la_liga_*.csv
  │     ├── bundesliga_*.csv
  │     ├── serie_a_*.csv
  │     └── ligue_1_*.csv
  │
  ├── odds_provider.py          # 赔率数据提供器 (CSV → MatchOddsBundle)
  ├── enhanced_data_provider.py # 增强数据提供器 (CSV → 24个因子)
  ├── match_stats_enricher.py   # [NEW v5.10.8] 比赛统计增强器 (CSV → 14个衍生特征)
  │
  ├── api_client.py             # 实时API客户端 (FootballData + API-Football)
  ├── pipeline.py               # 生产环境每日管道 (DailyDataPipeline)
  │
  ├── xg_fetcher.py             # Understat xG 获取器 (数据格式已变更)
  ├── external_data_collector.py # FBref xG 采集器 (被墙)
  ├── real_orthogonal_loader.py # 真实数据 + 合成数据 (部分可用)
  ├── orthogonal_sources.py     # ⚠️ 已废弃 — 合成随机数据 (仅用于单元测试)
  │
  └── loader.py                 # 数据加载器 (辅助工具)

src/factors/
  ├── registry.py               # 因子注册中心 (54个因子定义)
  └── compute.py                # 因子计算引擎 (54个因子计算逻辑)

src/pipeline/
  └── orchestrator.py           # GameFlowPipeline (数据 → 因子 → 概率 → 决策)

tests/
  └── test_phase6_walk_forward.py  # Walk-Forward 回测主入口
```

---

## 七、使用方式

### 运行 Walk-Forward 回测

```bash
cd /workspace/gto-gameflow-v5
python tests/test_phase6_walk_forward.py
```

### 单独测试 MatchStatsEnricher

```python
from src.data.match_stats_enricher import MatchStatsEnricher
from datetime import datetime

enricher = MatchStatsEnricher(
    csv_dir='src/data/historical_odds',
    leagues=['premier_league'],
    seasons=['2022-23', '2023-24'],
)

stats = enricher.get_enriched_stats(
    'premier_league', '2023-24',
    'Arsenal', 'Tottenham',
    datetime(2023, 9, 24),
)
# 返回 14 个衍生特征值
```

### 单独测试 EnhancedDataProvider

```python
from src.data.enhanced_data_provider import EnhancedDataProvider

provider = EnhancedDataProvider(
    csv_dir='src/data/historical_odds',
    leagues=['premier_league'],
    seasons=['2022-23', '2023-24'],
)

extra = provider.get_enhanced_data(
    'premier_league', '2023-24',
    'Arsenal', 'Tottenham',
    datetime(2023, 9, 24),
    stats_enricher=enricher,  # v5.10.8: 可选
)
# 返回 39 个键 (含 data_completeness)
```