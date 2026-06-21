"""
GTO v6.0 — 实时赔率搜索模块

通过网页抓取获取实时赔率，无需API密钥。

使用方式:
    searcher = OddsSearcher()
    odds = searcher.search_odds("Manchester City", "Arsenal")
"""

from __future__ import annotations
import re
import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OddsResult:
    """赔率结果"""
    home_team: str
    away_team: str
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    odds_over_25: float = 0.0
    odds_under_25: float = 0.0
    source: str = ""
    confidence: float = 0.0


class OddsSearcher:
    """实时赔率搜索器"""
    
    def __init__(self):
        self._cache: Dict[str, OddsResult] = {}
    
    def search_odds(self, home_team: str, away_team: str) -> Optional[OddsResult]:
        """
        搜索比赛赔率。
        
        参数:
            home_team: 主队名
            away_team: 客队名
        
        返回:
            OddsResult 或 None
        """
        cache_key = f"{home_team}_vs_{away_team}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 尝试从多个源获取赔率
        odds = None
        
        # 源1: 从赔率比较网站抓取
        if not odds:
            odds = self._scrape_from_odds_comparison(home_team, away_team)
        
        # 源2: 从体育新闻网站抓取
        if not odds:
            odds = self._scrape_from_sports_news(home_team, away_team)
        
        # 源3: 使用默认赔率（基于Elo评分估算）
        if not odds:
            odds = self._estimate_odds_from_elo(home_team, away_team)
        
        if odds:
            self._cache[cache_key] = odds
        
        return odds
    
    def _scrape_from_odds_comparison(self, home_team: str, away_team: str) -> Optional[OddsResult]:
        """从赔率比较网站抓取"""
        try:
            # 构建搜索URL
            query = f"{home_team} vs {away_team} odds"
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
            
            # 解析赔率
            odds = self._parse_odds_from_html(html, home_team, away_team)
            
            return odds
        
        except Exception as e:
            logger.debug(f"赔率抓取失败: {e}")
            return None
    
    def _scrape_from_sports_news(self, home_team: str, away_team: str) -> Optional[OddsResult]:
        """从体育新闻网站抓取"""
        # 简化实现
        return None
    
    def _estimate_odds_from_elo(self, home_team: str, away_team: str) -> Optional[OddsResult]:
        """基于Elo评分估算赔率"""
        # 使用默认Elo评分
        default_elo = 1500.0
        
        # 简化的Elo到概率转换
        elo_diff = 0  # 默认无差异
        
        # 计算期望胜率
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        expected_draw = 0.25  # 默认平局率
        expected_away = 1.0 - expected_home - expected_draw
        
        # 转换为赔率（含边际）
        margin = 1.05  # 5%边际
        
        if expected_home > 0:
            odds_home = margin / expected_home
        else:
            odds_home = 10.0
        
        if expected_draw > 0:
            odds_draw = margin / expected_draw
        else:
            odds_draw = 10.0
        
        if expected_away > 0:
            odds_away = margin / expected_away
        else:
            odds_away = 10.0
        
        return OddsResult(
            home_team=home_team,
            away_team=away_team,
            odds_home=round(odds_home, 2),
            odds_draw=round(odds_draw, 2),
            odds_away=round(odds_away, 2),
            source="elo_estimate",
            confidence=0.3,
        )
    
    def _parse_odds_from_html(self, html: str, home_team: str, away_team: str) -> Optional[OddsResult]:
        """从HTML解析赔率"""
        odds = OddsResult(home_team=home_team, away_team=away_team)
        
        # 尝试多种赔率格式
        patterns = [
            # 格式1: 主胜 1.85 平局 3.50 客胜 4.20
            r'(?:主胜|主队|1)\s*[:：]?\s*(\d+\.?\d*)\s*(?:平局|平|X)\s*[:：]?\s*(\d+\.?\d*)\s*(?:客胜|客队|2)\s*[:：]?\s*(\d+\.?\d*)',
            # 格式2: 1.85/3.50/4.20
            r'(\d+\.?\d*)\s*[/／]\s*(\d+\.?\d*)\s*[/／]\s*(\d+\.?\d*)',
            # 格式3: 1.85 3.50 4.20
            r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                o1 = float(match.group(1))
                o2 = float(match.group(2))
                o3 = float(match.group(3))
                
                # 验证赔率合理性
                if self._validate_odds(o1, o2, o3):
                    # 通常最小的是主胜赔率
                    if o1 <= o2 and o1 <= o3:
                        odds.odds_home = o1
                        odds.odds_draw = o2
                        odds.odds_away = o3
                    elif o3 <= o1 and o3 <= o2:
                        odds.odds_home = o3
                        odds.odds_draw = o1
                        odds.odds_away = o2
                    else:
                        odds.odds_home = o1
                        odds.odds_draw = o2
                        odds.odds_away = o3
                    
                    odds.source = "web_scrape"
                    odds.confidence = self._calculate_confidence(odds)
                    return odds
        
        return None
    
    def _validate_odds(self, o1: float, o2: float, o3: float) -> bool:
        """验证赔率是否合理"""
        # 赔率应该大于1
        if o1 <= 1 or o2 <= 1 or o3 <= 1:
            return False
        
        # 赔率应该在合理范围内
        if o1 > 50 or o2 > 50 or o3 > 50:
            return False
        
        # 计算边际
        margin = 1.0/o1 + 1.0/o2 + 1.0/o3
        
        # 边际应该在合理范围内
        if margin < 0.8 or margin > 1.3:
            return False
        
        return True
    
    def _calculate_confidence(self, odds: OddsResult) -> float:
        """计算赔率置信度"""
        if odds.odds_home <= 1 or odds.odds_draw <= 1 or odds.odds_away <= 1:
            return 0.0
        
        # 检查赔率是否合理
        margin = 1.0/odds.odds_home + 1.0/odds.odds_draw + 1.0/odds.odds_away
        
        if margin < 0.9 or margin > 1.2:
            return 0.3  # 边际异常
        
        return 0.6  # 默认置信度
    
    def search_batch(self, matches: List[Dict]) -> Dict[str, OddsResult]:
        """
        批量搜索赔率。
        
        参数:
            matches: 比赛列表 [{"home": str, "away": str}]
        
        返回:
            {match_id: OddsResult}
        """
        results = {}
        
        for match in matches:
            home = match.get("home", "")
            away = match.get("away", "")
            match_id = match.get("id", f"{home}_vs_{away}")
            
            odds = self.search_odds(home, away)
            if odds:
                results[match_id] = odds
        
        return results


# 便捷函数
def search_match_odds(home_team: str, away_team: str) -> Optional[OddsResult]:
    """搜索比赛赔率"""
    searcher = OddsSearcher()
    return searcher.search_odds(home_team, away_team)
