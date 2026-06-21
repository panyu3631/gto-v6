"""
GTO-GameFlow v5.10.9 — NLP新闻情绪因子

建筑说明书:
- 为每个赛前事件提取情绪分数 (正面/负面/中性)
- 作为独立因子注入到因子引擎
- 默认使用空实现 (不产生信号), 待接入真实NLP模型后启用

集成点:
1. orchestrator.py Stage 1: 调用 news_sentiment.compute() 获取情绪分数
2. 情绪分数作为额外因子注入 factor_deltas["NLP_SENT"]
3. 需要的数据源: 赛前新闻文本、伤病报告、转会消息、教练发言

未来的NLP模型选项:
- 微调 BERT/RoBERTa 体育新闻分类模型
- 使用 GPT-4/Claude API 提取结构化情绪 (需要API密钥)
- 使用预训练的情感分析模型 (VADER, TextBlob 作为baseline)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class NewsSentimentSignal:
    """单个新闻事件的情绪信号"""
    source: str           # 新闻来源
    headline: str         # 标题
    sentiment_score: float  # -1.0 (极负面) 到 +1.0 (极正面)
    relevance: float      # 0.0 (无关) 到 1.0 (高度相关)
    team_affected: str    # 影响的球队
    category: str         # 类别: injury, transfer, coach, morale, other


@dataclass
class MatchSentimentResult:
    """一场比赛的NLP情绪综合结果"""
    home_sentiment: float = 0.0   # 主队情绪 (-1.0 到 +1.0)
    away_sentiment: float = 0.0   # 客队情绪
    home_confidence: float = 0.0  # 主队情绪置信度 (0=无数据)
    away_confidence: float = 0.0  # 客队情绪置信度
    signals: List[NewsSentimentSignal] = field(default_factory=list)
    total_articles: int = 0


class NewsSentimentProvider:
    """
    NLP新闻情绪提供器。

    默认实现: 空情绪 (不产生信号)
    真实实现: 接入NLP模型后替换此类
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "default"):
        self.api_key = api_key
        self.model = model
        self._enabled = api_key is not None

    def is_enabled(self) -> bool:
        return self._enabled

    def compute_match_sentiment(
        self,
        league_id: str,
        home_team: str,
        away_team: str,
        match_date,
        existing_extra: Optional[Dict] = None,
    ) -> MatchSentimentResult:
        """
        计算一场比赛的NLP情绪分数。

        这是集成点: 在此方法中调用NLP模型获取情绪数据。

        Args:
            league_id: 联赛ID
            home_team: 主队名
            away_team: 客队名
            match_date: 比赛日期
            existing_extra: 现有的extra数据

        Returns:
            MatchSentimentResult: 情绪综合结果
        """
        if not self._enabled:
            return MatchSentimentResult()

        # TODO: 接入真实NLP模型
        # 1. 获取赛前新闻 (通过 NewsAPI / RSS / Web Scraping)
        # 2. 对每条新闻运行情绪分析
        # 3. 按球队聚合情绪分数
        # 4. 返回 MatchSentimentResult

        return MatchSentimentResult()


def compute_sentiment_factor(
    sentiment_result: MatchSentimentResult,
    elo_diff: float,
) -> Optional[Dict[str, float]]:
    """
    将情绪结果转换为因子delta。

    逻辑:
    - 主队正面情绪 > 客队 → 主胜信号 + (home_sentiment - away_sentiment) * 0.2
    - 客队正面情绪 > 主队 → 客胜信号 + (away_sentiment - home_sentiment) * 0.2
    - 平局: 情绪差异小 → 轻微平局信号

    强度系数 0.2 确保情绪因子不会主导其他因子。

    Returns:
        {"home": +0.04, "draw": 0.0, "away": -0.04} 或 None (无信号)
    """
    if not sentiment_result.signals:
        return None

    home_s = sentiment_result.home_sentiment
    away_s = sentiment_result.away_sentiment
    home_conf = sentiment_result.home_confidence
    away_conf = sentiment_result.away_confidence

    if home_conf < 0.3 and away_conf < 0.3:
        return None  # 置信度不足

    sentiment_diff = (home_s * home_conf - away_s * away_conf) * 0.2

    if abs(sentiment_diff) < 0.01:
        return None

    result = {"home": 0.0, "draw": 0.0, "away": 0.0}
    if sentiment_diff > 0:
        result["home"] = sentiment_diff
        result["away"] = -sentiment_diff * 0.5
    else:
        result["away"] = -sentiment_diff
        result["home"] = sentiment_diff * 0.5

    return result