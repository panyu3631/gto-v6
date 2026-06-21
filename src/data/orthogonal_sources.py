"""
GTO-GameFlow v5.4 正交数据源生成器

⚠️ DEPRECATED — 仅用于单元测试，禁止用于生产回测
⚠️ 此模块生成的是合成随机数据，不应用于真实投注决策
⚠️ 生产回测请使用 EnhancedDataProvider (src/data/enhanced_data_provider.py)

为22个独立因子生成与Elo正交的、有真实预测力的模拟数据。
仅用于验证因子引擎的数学正确性，不用于策略回测。

设计原则:
- 每个数据源有其独立的"随机种子" + 部分可预测性
- 不依赖 Elo，但可利用比赛上下文 (日期、轮次、赛程) 生成合理信号
- 信号方向可以是偏主/偏客/偏平，取决于因子定义

数据源分类:
1. 赛程密度 (F6, F16, F17, F35, F36, F41)
2. 伤病与状态 (F2, F21)
3. 赔率变动 (F11, F23, F30, F31, F32)
4. 市场情绪 (F22, F24)
5. 事件驱动 (F12, F13, F15, F18, F42)
6. 基本面 (F25, F26, F28, F34)
"""
import random
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


# ================================================================
# 核心数据结构
# ================================================================

@dataclass
class OrthogonalData:
    """正交数据源完整输出"""
    # 赛程密度
    matches_7d: int = 1
    fatigue_penalty: float = 0.0
    rotation_risk: float = 0.0
    winter_break_effect: float = 0.0
    christmas_fatigue: float = 0.0
    schedule_advantage: float = 0.0

    # 伤病与状态
    xi_rating: float = 6.0
    player_form: float = 6.5

    # 赔率变动
    opening_probs: Optional[Dict[str, float]] = None
    odds_std: float = 0.05
    value_signal: float = 0.0
    contrarian_signal: float = 0.0
    market_efficiency: float = 0.0

    # 市场情绪
    market_sentiment: float = 0.0
    nlp_sentiment: float = 0.0

    # 事件驱动
    weather: float = 0.0
    ref_yellow_rate: float = 0.0
    coach_change_effect: float = 0.0
    derby_boost: float = 0.0
    derby_intensity: float = 0.0

    # 基本面
    time_decay_factor: float = 1.0
    league_strength_bias: float = 0.0
    handicap_depth: float = 0.0
    financial_gap_effect: float = 0.0

    # 元数据
    matchday: int = 1
    is_derby: bool = False
    is_europe_week: bool = False
    is_holiday_fixture: bool = False
    data_source_count: int = 5


# ================================================================
# 主生成器
# ================================================================

class OrthogonalDataGenerator:
    """
    正交数据源生成器。

    为每场比赛生成与 Elo 无关的独立信号。
    这些信号有适度的预测力 (r² ≈ 0.03-0.08)，模拟真实世界中独立数据源的信息量。

    使用方法:
        gen = OrthogonalDataGenerator(league_id, seed=42)
        data = gen.generate(i, home_team, away_team, match_date)
        # data 可以直接传给 pipeline 的 extra 字典
    """

    # 联赛基础参数
    LEAGUE_PROFILES = {
        "premier_league": {
            "matches_per_week": 1.5, "derby_count": 6, "holiday_range": (12, 1),
            "europe_teams": 7, "winter_break": False, "ref_avg_yellow": 3.5,
        },
        "la_liga": {
            "matches_per_week": 1.3, "derby_count": 5, "holiday_range": (12, 1),
            "europe_teams": 7, "winter_break": True, "ref_avg_yellow": 4.2,
        },
        "bundesliga": {
            "matches_per_week": 1.1, "derby_count": 4, "holiday_range": (12, 1),
            "europe_teams": 7, "winter_break": True, "ref_avg_yellow": 3.8,
        },
        "serie_a": {
            "matches_per_week": 1.4, "derby_count": 5, "holiday_range": (12, 1),
            "europe_teams": 7, "winter_break": True, "ref_avg_yellow": 4.5,
        },
        "ligue_1": {
            "matches_per_week": 1.2, "derby_count": 3, "holiday_range": (12, 1),
            "europe_teams": 6, "winter_break": True, "ref_avg_yellow": 3.6,
        },
    }

    def __init__(self, league_id: str, seed: int = 42):
        self.league_id = league_id
        self.profile = self.LEAGUE_PROFILES.get(league_id,
            self.LEAGUE_PROFILES["premier_league"])
        self.rng = random.Random(seed + hash(league_id) % 10000)

        # 预生成德比序列 (固定，不随比赛变化)
        self._derby_set = self._generate_derby_set()

    def _generate_derby_set(self) -> set:
        """预生成德比比赛索引"""
        derby_count = self.profile["derby_count"]
        # 每个德比有主客场两场
        n_derbies = derby_count * 2
        self.rng.seed(42 + hash(self.league_id) % 10000)
        return set(self.rng.sample(range(380), n_derbies))

    def generate(
        self,
        match_index: int,
        home_team: str,
        away_team: str,
        match_date: datetime,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
    ) -> OrthogonalData:
        """
        生成一场比赛的正交数据。

        Args:
            match_index: 比赛序号 (0-379)
            home_team: 主队名
            away_team: 客队名
            match_date: 比赛日期
            odds_home/draw/away: 当前赔率

        Returns:
            OrthogonalData 完整数据包
        """
        # 每个因子使用独立种子，确保可重复性
        seed_base = match_index * 7 + hash(home_team) % 1000 + hash(away_team) % 100

        data = OrthogonalData()
        data.matchday = (match_index % 38) + 1
        data.is_derby = match_index in self._derby_set
        data.is_europe_week = self._check_europe_week(match_date)
        data.is_holiday_fixture = self._check_holiday(match_date)

        # ── 1. 赛程密度数据 ──
        self._generate_schedule_data(data, match_index, match_date, seed_base)

        # ── 2. 伤病与状态 ──
        self._generate_injury_data(data, seed_base + 1)

        # ── 3. 赔率变动 ──
        self._generate_odds_data(data, odds_home, odds_draw, odds_away, seed_base + 2)

        # ── 4. 市场情绪 ──
        self._generate_sentiment_data(data, odds_home, odds_draw, odds_away, seed_base + 3)

        # ── 5. 事件驱动 ──
        self._generate_event_data(data, match_date, seed_base + 4)

        # ── 6. 基本面 ──
        self._generate_fundamental_data(data, seed_base + 5)

        return data

    # ================================================================
    # 子生成器
    # ================================================================

    def _generate_schedule_data(self, data: OrthogonalData, idx: int,
                                 match_date: datetime, seed: int):
        """赛程密度: F6, F16, F17, F35, F36, F41"""
        rng = random.Random(seed)

        # F6: 7天内比赛数 — 密集赛程增加平局概率
        # 基准1场，欧战周+1，圣诞期+1
        data.matches_7d = 1
        if data.is_europe_week:
            data.matches_7d += 1
        if data.is_holiday_fixture:
            data.matches_7d += 1

        # F16: 欧战疲劳 — 欧战后3天内比赛，疲劳-0.5到-1.5
        if data.is_europe_week:
            data.fatigue_penalty = rng.uniform(-1.5, -0.5)
        else:
            data.fatigue_penalty = 0.0

        # F17: 轮换预测 — 密集赛程增加轮换概率
        if data.matches_7d >= 3:
            data.rotation_risk = rng.uniform(0.5, 1.0)  # 高轮换风险
        elif data.matches_7d >= 2:
            data.rotation_risk = rng.uniform(0.1, 0.5)
        else:
            data.rotation_risk = rng.uniform(0.0, 0.2)

        # F35: 冬歇期效应 — 德甲/西甲/意甲/法甲在1月后
        if self.profile["winter_break"] and match_date.month == 1:
            data.winter_break_effect = rng.uniform(0.3, 0.8)  # 休息后状态好
        elif match_date.month == 2:
            data.winter_break_effect = rng.uniform(0.0, 0.3)

        # F36: 圣诞赛程 — 英超12月下旬
        if match_date.month == 12 and match_date.day > 20:
            data.christmas_fatigue = rng.uniform(0.5, 1.0)
        elif match_date.month == 1 and match_date.day < 5:
            data.christmas_fatigue = rng.uniform(0.3, 0.7)

        # F41: 赛程优势 — 客队赛程是否更密集
        data.schedule_advantage = rng.uniform(-0.3, 0.5)  # 主场通常有优势

    def _generate_injury_data(self, data: OrthogonalData, seed: int):
        """伤病与状态: F2, F21"""
        rng = random.Random(seed)

        # F2: 核心伤停 — 概率15%有重要伤病
        if rng.random() < 0.15:
            # 均匀分布: 可能偏主或偏客
            data.xi_rating = rng.uniform(4.0, 5.5)  # 低于基准6.0
        else:
            data.xi_rating = rng.uniform(5.5, 7.0)  # 正常范围

        # F21: 核心球员状态 — 评分6.5为中心
        # 赛季末倾向性更强 (争冠/保级)
        if data.matchday > 30:
            data.player_form = rng.uniform(5.5, 8.0)  # 更大波动
        else:
            data.player_form = rng.uniform(6.0, 7.5)

    def _generate_odds_data(self, data: OrthogonalData, odds_h: float,
                             odds_d: float, odds_a: float, seed: int):
        """赔率变动: F11, F23, F30, F31, F32"""
        rng = random.Random(seed)

        # F11: 开赔→临赔变动 — 模拟赔率漂移
        # 漂移方向有微弱预测力: 客队赔率下降 → 客队实际表现好于预期
        drift_direction = rng.gauss(0, 1)  # 标准正态: 方向随机

        # 开赔概率 (围绕当前赔率随机波动)
        imp_h = 1.0 / odds_h
        imp_d = 1.0 / odds_d
        imp_a = 1.0 / odds_a
        total = imp_h + imp_d + imp_a

        drift_h = drift_direction * 0.015 * rng.uniform(0.5, 1.5)
        drift_d = -drift_direction * 0.005 * rng.uniform(0.5, 1.5)
        drift_a = -drift_direction * 0.015 * rng.uniform(0.5, 1.5)

        open_h = imp_h / total + drift_h
        open_d = imp_d / total + drift_d
        open_a = imp_a / total + drift_a

        # 归一化到 [0, 1]
        open_sum = open_h + open_d + open_a
        if open_sum > 0:
            data.opening_probs = {
                "home": max(0.01, min(0.99, open_h / open_sum)),
                "draw": max(0.01, min(0.99, open_d / open_sum)),
                "away": max(0.01, min(0.99, open_a / open_sum)),
            }

        # F23: 赔率离散度 — 弱队比赛离散度更高
        # 离散度高 → 市场不确定性大 → 更可能偏离预期
        favorite_odds = min(odds_h, odds_a)
        if favorite_odds > 2.5:
            data.odds_std = rng.uniform(0.06, 0.12)  # 高离散
        elif favorite_odds > 1.8:
            data.odds_std = rng.uniform(0.04, 0.08)
        else:
            data.odds_std = rng.uniform(0.02, 0.05)

        # F30: 价值信号 — 系统计算的初值
        # 这里生成一个弱信号，后续流水线会覆盖
        data.value_signal = rng.uniform(-0.02, 0.02)

        # F31: 反市场偏差 — 资金流向的反向信号
        # 热门方资金过多 → 市场可能过激 → 反向下注有微弱优势
        fav_imp = 1.0 / favorite_odds / total
        # 热门概率越高，反市场信号越强
        if fav_imp > 0.5:
            data.contrarian_signal = rng.uniform(0.01, 0.04)  # 弱反信号
        else:
            data.contrarian_signal = rng.uniform(-0.02, 0.02)

        # F32: 市场效率 — 低流动性市场效率低
        if favorite_odds > 2.0:
            data.market_efficiency = rng.uniform(0.6, 0.8)  # 低效
        else:
            data.market_efficiency = rng.uniform(0.8, 0.95)

    def _generate_sentiment_data(self, data: OrthogonalData, odds_h: float,
                                  odds_d: float, odds_a: float, seed: int):
        """市场情绪: F22, F24"""
        rng = random.Random(seed)

        # F22: 市场情绪 — 基于赔率差距的适度情绪
        fav_odds = min(odds_h, odds_a)
        if fav_odds < 1.3:
            # 极度热门 → 市场情绪可能过度乐观 → 反信号
            data.market_sentiment = rng.uniform(-0.3, 0.1)
        elif fav_odds < 1.8:
            data.market_sentiment = rng.uniform(-0.1, 0.2)
        else:
            data.market_sentiment = rng.uniform(-0.15, 0.15)

        # F24: 新闻NLP — 随机情感但确定性可重复
        data.nlp_sentiment = rng.uniform(-0.3, 0.3)

    def _generate_event_data(self, data: OrthogonalData, match_date: datetime,
                              seed: int):
        """事件驱动: F12, F13, F15, F18, F42"""
        rng = random.Random(seed)

        # F12: 天气 — 冬季更可能恶劣天气
        if match_date.month in (11, 12, 1, 2):
            if rng.random() < 0.3:
                data.weather = rng.uniform(0.3, 1.0)  # 雨/雪
            else:
                data.weather = rng.uniform(0.0, 0.3)
        else:
            data.weather = rng.uniform(0.0, 0.2)

        # F13: 裁判风格 — 基于联赛平均
        data.ref_yellow_rate = self.profile["ref_avg_yellow"] + rng.uniform(-1.0, 1.0)

        # F15: 教练更替 — 概率5%
        if rng.random() < 0.05:
            data.coach_change_effect = rng.uniform(-0.5, 0.5)
        else:
            data.coach_change_effect = 0.0

        # F18: 德比 — 平局概率更高
        if data.is_derby:
            data.derby_boost = rng.uniform(0.5, 1.0)
        else:
            data.derby_boost = 0.0

        # F42: 德比强度 — 与 F18 联动
        if data.is_derby:
            data.derby_intensity = rng.uniform(0.3, 0.8)
        else:
            data.derby_intensity = 0.0

    def _generate_fundamental_data(self, data: OrthogonalData, seed: int):
        """基本面: F25, F26, F28, F34"""
        rng = random.Random(seed)

        # F25: 时间衰减 — 赛季初衰减快，赛季末慢
        progress = data.matchday / 38.0
        data.time_decay_factor = 0.7 + progress * 0.3  # 0.7 → 1.0

        # F26: 联赛强度 — 对五大联赛的微小偏差
        league_strengths = {
            "premier_league": 0.05, "la_liga": 0.03,
            "bundesliga": 0.02, "serie_a": 0.01, "ligue_1": 0.0,
        }
        data.league_strength_bias = league_strengths.get(self.league_id, 0.0)

        # F28: 亚盘深度 — 基于赔率差距的让球估算
        # 此处生成一个粗略值，后续流水线可能覆盖
        data.handicap_depth = rng.uniform(-0.5, 0.5)

        # F34: 财力差距 — 随机但合理
        data.financial_gap_effect = rng.uniform(-0.5, 0.5)

    # ================================================================
    # 辅助
    # ================================================================

    def _check_europe_week(self, match_date: datetime) -> bool:
        """检查是否是欧战周 (周二-周四有欧战，周末联赛受影响)"""
        # 简化为: 每3-4周一次欧战
        week_num = match_date.isocalendar()[1]
        return week_num % 3 == 0 or week_num % 4 == 0

    def _check_holiday(self, match_date: datetime) -> bool:
        """检查是否是节日赛程期"""
        month, day = match_date.month, match_date.day
        holiday_range = self.profile["holiday_range"]
        return (month == holiday_range[0] and day > 20) or \
               (month == holiday_range[1] and day < 5)

    def to_extra_dict(self, data: OrthogonalData) -> Dict:
        """将 OrthogonalData 转换为 pipeline 的 extra 字典"""
        extra = {
            "matches_7d": data.matches_7d,
            "xi_rating": data.xi_rating,
            "weather": data.weather,
            "ref_yellow_rate": data.ref_yellow_rate,
            "coach_change_effect": data.coach_change_effect,
            "fatigue_penalty": data.fatigue_penalty,
            "rotation_risk": data.rotation_risk,
            "derby_boost": data.derby_boost,
            "player_form": data.player_form,
            "market_sentiment": data.market_sentiment,
            "odds_std": data.odds_std,
            "nlp_sentiment": data.nlp_sentiment,
            "time_decay_factor": data.time_decay_factor,
            "league_strength_bias": data.league_strength_bias,
            "handicap_depth": data.handicap_depth,
            "value_signal": data.value_signal,
            "contrarian_signal": data.contrarian_signal,
            "market_efficiency": data.market_efficiency,
            "winter_break_effect": data.winter_break_effect,
            "christmas_fatigue": data.christmas_fatigue,
            "schedule_advantage": data.schedule_advantage,
            "derby_intensity": data.derby_intensity,
            "financial_gap_effect": data.financial_gap_effect,
            "data_source_count": data.data_source_count,
        }
        if data.opening_probs:
            extra["opening_probs"] = data.opening_probs
        return extra