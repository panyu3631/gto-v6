# GTO v6.0 项目进度文档

> 最后更新: 2026-06-21 11:18 UTC
> 本文档记录完整项目进度，可直接用于恢复工作。

---

## 一、项目概述

**目标**: 足球博彩预测模型，五大联赛，1X2/大小球/亚盘/串关
**技术栈**: Python，生产引擎58因子系统
**代码路径**: `/home/work/.openclaw/workspace/gto/足球大模型/gto-gameflow-v5/`

---

## 二、已完成工作

### Phase 0: 框架加固 ✅

| 任务 | 文件 | 说明 |
|------|------|------|
| FactorInputs数据类 | `src/factors/factor_inputs.py` | 58个参数打包，避免签名错位 |
| 因子Elo分类 | `src/factors/registry.py` | 新增EloCategory枚举，统一管理 |
| F10/F11/F14禁用 | `src/factors/compute.py` | 跳过计算，节省资源 |
| BacktestAdapter | `src/pipeline/backtest_adapter.py` | 让回测使用生产引擎 |
| 生产引擎回测 | `tests/test_production_backtest.py` | 支持生产/简化两种模式 |

### Phase 1: 新增模块 ✅

| 模块 | 文件 | 功能 |
|------|------|------|
| 大小球动态线路 | `src/strategies/dynamic_totals.py` | 5条线路edge比较，自动选最佳 |
| 串关筛选器 | `src/strategies/parlay_filter.py` | 独立阈值，2/3串1，双选腿 |
| 亚盘模块 | `src/strategies/asian_handicap_module.py` | quarter-line，走水处理 |

### Phase 2: 联赛参数优化 ✅

| 联赛 | 参数 | 旧值 | 新值 | 真实值 |
|------|------|------|------|--------|
| 英超 | avg_goals | 2.85 | 2.98 | 2.98 |
| 英超 | draw_rate | 0.24 | 0.225 | 0.225 |
| 德甲 | avg_goals | 3.05 | 3.17 | 3.17 |
| 德甲 | draw_rate | 0.22 | 0.249 | 0.249 |
| 法甲 | avg_goals | 2.70 | 2.78 | 2.78 |
| 法甲 | draw_rate | 0.30 | 0.258 | 0.258 |

### Phase 0-4: 回测框架重构 ✅

- BacktestAdapter 让回测使用生产引擎
- 支持生产/简化两种模式对比

### P0-P3: 因子数据激活 ✅

| 优先级 | 内容 | 状态 |
|--------|------|------|
| P0 | CSV比赛统计提取 (F42-F58) | ✅ |
| P0 | CSV赔率数据提取 (F46/F47/F54/F55) | ✅ |
| P0 | 积分榜计算 (F7/F33/F37/F39) | ✅ |
| P1 | 进球分布修正 (F27/F29/F26) | ✅ |
| P2 | xG数据 (F9) | ✅ 网页抓取 |
| P2 | 教练更替 (F15) | ✅ 网页抓取 |
| P2 | 升班马 (F40) | ✅ 网页抓取 |
| P3 | 天气 (F12) | ✅ Open-Meteo API |

---

## 三、当前回测结果

### 生产引擎 (58因子)

| 联赛 | 投注 | 胜率 | ROI |
|------|------|------|-----|
| 英超 | 438 | 50.9% | **+3.2%** |
| 西甲 | 116 | 50.9% | **+11.3%** |
| 德甲 | 168 | 44.0% | -7.3% |
| 意甲 | 125 | 43.2% | -12.1% |
| 法甲 | 5 | 40.0% | -24.3% |
| **总计** | **852** | **48.4%** | **+1.1%** |

### 因子激活率

| 联赛 | 非零因子 | 激活率 |
|------|----------|--------|
| 英超 | 26-29 | 49-55% |
| 西甲 | 28-32 | 53-60% |
| 德甲 | 26-30 | 49-57% |
| 意甲 | 26-30 | 49-57% |
| 法甲 | 26-31 | 49-58% |

---

## 四、抓取的数据文件

| 文件 | 内容 | 路径 |
|------|------|------|
| 积分榜 | 5个联赛2023-24赛季 | `data/scraped/standings_*.json` |
| 球队名映射 | 抓取名↔CSV名 | `data/scraped/team_name_mapping.json` |
| xG数据 | 英超20支球队 | `data/scraped/xg_data.json` |
| 教练更替 | 英超3支球队 | `data/scraped/manager_changes.json` |
| 升班马 | 5个联赛 | `data/scraped/promoted_teams.json` |

---

## 五、仍为零的因子

### 实战可激活（回测中无法获取）

| 因子 | 说明 | 实战数据来源 |
|------|------|-------------|
| F2 | 伤病 | 赛前爬取伤病新闻 |
| F21 | 球员状态 | 赛前爬取球员评分 |
| F34 | 财力差距 | Transfermarkt球队身价 |
| F36 | 圣诞赛程 | 赛程表 |
| F16 | 欧战影响 | 欧战赛程 |
| F17 | 轮换预测 | 赛程+伤病 |
| F18 | 德比战 | 历史对阵 |
| F30/F31/F32 | 赔率价值信号 | 模型内部实时计算 |

### 需要NLP处理

| 因子 | 说明 |
|------|------|
| F22 | 市场情绪 (新闻NLP) |
| F24 | 新闻NLP情感 |

---

## 六、待执行工作

### 高优先级

1. **大小球策略回测** — 用动态线路模块回测大小球
2. **串关策略回测** — 用串关筛选器回测
3. **亚盘策略回测** — 用亚盘模块回测
4. **德甲/意甲/法甲优化** — 当前亏损，需联赛特化调参

### 已完成（NGINX架构借鉴）

5. **YAML配置文件系统** — `config/model_config.yaml` + `src/config/config_loader.py`
6. **因子接口标准化** — `src/factors/base_factor.py`
7. **数据源抽象接口** — `src/data/base_source.py`
8. **策略插件化** — `src/strategies/base_strategy.py`
9. **健康检查监控** — `src/monitoring/health_monitor.py`
10. **A/B测试框架** — `src/testing/ab_test.py`
11. **事件驱动实时系统** — `src/realtime/event_engine.py`
12. **热更新模块** — `src/realtime/hot_updater.py`
13. **世界杯模块** — `src/worldcup/worldcup_module.py`

### 中优先级

5. **串关轮次制回测** — 按日期分组，同轮次生成串关
6. **F27/F42/F45/F48/F49修复** — 这些CSV因子仍为零，需检查
7. **更多xG数据** — 其他4个联赛的xG数据

### 低优先级

8. **新闻NLP集成** — F22/F24
9. **实时数据管道** — 生产部署时的实时数据采集
10. **因子有效性监控** — 因子激活率监控和报警

---

## 七、关键文件清单

### 核心代码
```
src/factors/factor_inputs.py      — 因子输入数据类
src/factors/registry.py           — 因子注册中心 (含Elo分类)
src/factors/compute.py            — 因子计算引擎
src/pipeline/backtest_adapter.py  — 回测适配器 (核心)
src/pipeline/orchestrator.py      — 生产引擎编排器
src/config/league_params.py       — 联赛参数配置
```

### 新增模块
```
src/strategies/dynamic_totals.py      — 大小球动态线路
src/strategies/parlay_filter.py       — 串关筛选器
src/strategies/asian_handicap_module.py — 亚盘模块
src/data/external_fetcher.py          — 外部数据获取器
src/data/web_scraper.py               — 网页抓取器
```

### 回测框架
```
tests/test_production_backtest.py     — 生产引擎回测
tests/test_full_real_odds_backtest.py — 真实赔率回测 (简化模型)
tests/test_ensemble_backtest.py       — 集成回测
```

### 数据文件
```
data/scraped/standings_*.json         — 积分榜
data/scraped/team_name_mapping.json   — 球队名映射
data/scraped/xg_data.json             — xG数据
data/scraped/manager_changes.json     — 教练更替
data/scraped/promoted_teams.json      — 升班马
```

---

## 八、执行规则 (AGENTS.md)

1. 读完即改 — 读取后必须产出实际变更
2. 单次完成 — 一个文件一次改完
3. 禁止空转 — 连续3个无产出动作必须停下
4. 执行前检查 — 没有变更就不要执行
5. 代码修改流程 — 读取→写入→验证→完成
6. 禁止多余操作 — 编辑后不验证、编辑前不读取
7. 禁止反复询问 — 用户说执行就执行
8. 禁止重复行为 — 无效动作只允许一次
9. 更新迭代流程 — 先文字头脑风暴→确认→执行
10. 禁止多余验证 — 不运行grep/sed确认结果

---

## 十、世界杯模块

### 功能
- 小组赛预测 — 32支球队，8组
- 淘汰赛预测 — 16强→8强→4强→决赛
- 特殊因子 — 国家队经验、大赛基因、赛程密度
- 历史数据 — 世界杯历史战绩

### 使用方式
```python
from src.worldcup.worldcup_module import get_world_cup_module
wc = get_world_cup_module()

# 预测比赛
pred = wc.predict_match("Argentina", "France", stage="final")

# 预测小组出线
group = ["Argentina", "France", "Mexico", "Australia"]
results = wc.predict_group(group)
```

### 预测示例
| 比赛 | 阶段 | 主胜 | 平局 | 客胜 |
|------|------|------|------|------|
| Argentina vs France | 决赛 | 42.3% | 16.2% | 41.5% |

---

## 十一、恢复指令

如果需要从当前进度继续：

1. 读取本文档了解进度
2. 检查 `src/pipeline/backtest_adapter.py` 确认代码完整
3. 检查 `data/scraped/` 确认数据文件存在
4. 运行 `python3 tests/test_production_backtest.py --consensus 4` 验证回测
5. 根据"待执行工作"继续下一步
