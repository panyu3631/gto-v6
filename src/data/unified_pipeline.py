"""
GTO v6.0 — 统一数据管道

整合多个数据源，提供统一的数据获取接口。

数据源优先级:
1. 缓存数据 (1小时内有效)
2. API数据 (需要API密钥)
3. 网页抓取 (公开数据)
4. 历史数据 (CSV文件)

使用方式:
    pipeline = UnifiedDataPipeline(league_id="premier_league")
    matches = pipeline.get_upcoming_matches()
    odds = pipeline.get_match_odds(match_id)
"""

from __future__ import annotations
import json
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'realtime')
HISTORICAL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'scraped')
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class MatchData:
    """比赛数据"""
    match_id: str
    league_id: str
    home_team: str
    away_team: str
    kickoff_time: str
    status: str = "upcoming"
    
    # 1X2赔率
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    
    # 大小球赔率
    odds_over_25: float = 0.0
    odds_under_25: float = 0.0
    
    # 亚盘
    handicap_line: float = 0.0
    odds_home_ah: float = 0.0
    odds_away_ah: float = 0.0
    
    # 球队数据
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_rank: int = 10
    away_rank: int = 10
    
    # 元数据
    source: str = ""
    fetched_at: str = ""


class UnifiedDataPipeline:
    """统一数据管道"""
    
    def __init__(self, league_id: str):
        self.league_id = league_id
        self._cache: Dict[str, Any] = {}
        self._cache_dir = os.path.join(DATA_DIR, league_id)
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # 加载历史数据
        self._standings = self._load_standings()
        self._elo_ratings = self._load_elo_ratings()
    
    def get_upcoming_matches(self, days: int = 7) -> List[MatchData]:
        """获取未来N天的比赛"""
        # 尝试从缓存获取
        cached = self._load_cache("upcoming_matches")
        if cached:
            return [MatchData(**m) for m in cached]
        
        # 尝试从ESPN获取
        matches = self._fetch_from_espn()
        
        # 如果ESPN没有数据，使用模拟数据（用于测试）
        if not matches:
            matches = self._get_simulated_matches()
        
        # 添加球队数据
        for match in matches:
            self._enrich_match_data(match)
        
        # 缓存结果
        if matches:
            self._save_cache("upcoming_matches", [self._match_to_dict(m) for m in matches])
        
        return matches
    
    def get_match_odds(self, match_id: str) -> Optional[Dict]:
        """获取比赛赔率"""
        # 尝试从缓存获取
        cached = self._load_cache(f"odds_{match_id}")
        if cached:
            return cached
        
        # 尝试从Odds API获取
        odds = self._fetch_odds(match_id)
        if odds:
            self._save_cache(f"odds_{match_id}", odds)
        
        return odds
    
    def get_team_stats(self, team: str) -> Optional[Dict]:
        """获取球队统计"""
        # 从历史数据获取
        return self._standings.get(team, None)
    
    def _fetch_from_espn(self) -> List[MatchData]:
        """从ESPN获取赛程"""
        import urllib.request
        
        league_ids = {
            "premier_league": "eng.1",
            "la_liga": "esp.1",
            "bundesliga": "ger.1",
            "serie_a": "ita.1",
            "ligue_1": "fra.1",
        }
        
        espn_id = league_ids.get(self.league_id, "")
        if not espn_id:
            return []
        
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/scoreboard"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            matches = []
            for event in data.get("events", []):
                competition = event.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                
                if len(competitors) < 2:
                    continue
                
                home = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away = next((c for c in competitors if c.get("homeAway") == "away"), {})
                
                match = MatchData(
                    match_id=event.get("id", ""),
                    league_id=self.league_id,
                    home_team=home.get("team", {}).get("shortDisplayName", ""),
                    away_team=away.get("team", {}).get("shortDisplayName", ""),
                    kickoff_time=event.get("date", ""),
                    status="upcoming",
                    source="espn",
                    fetched_at=datetime.now().isoformat(),
                )
                matches.append(match)
            
            logger.info(f"从ESPN获取到 {len(matches)} 场比赛")
            return matches
        
        except Exception as e:
            logger.warning(f"ESPN获取失败: {e}")
            return []
    
    def _fetch_odds(self, match_id: str) -> Optional[Dict]:
        """获取赔率数据"""
        # 从Odds API获取（需要API密钥）
        from src.data.api_config import get_api_key, is_provider_enabled
        
        if is_provider_enabled("odds_api"):
            api_key = get_api_key("odds_api")
            if api_key:
                return self._fetch_from_odds_api(match_id, api_key)
        
        return None
    
    def _fetch_from_odds_api(self, match_id: str, api_key: str) -> Optional[Dict]:
        """从The Odds API获取赔率"""
        import urllib.request
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={api_key}&regions=uk&markets=h2h"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            # 查找匹配的比赛
            for game in data:
                if game.get("id") == match_id:
                    bookmakers = game.get("bookmakers", [])
                    if bookmakers:
                        # 使用第一个博彩公司的赔率
                        markets = bookmakers[0].get("markets", [])
                        if markets:
                            outcomes = markets[0].get("outcomes", [])
                            odds = {}
                            for outcome in outcomes:
                                name = outcome.get("name", "")
                                price = outcome.get("price", 0)
                                if "home" in name.lower() or name == game.get("home_team"):
                                    odds["home"] = price
                                elif "away" in name.lower() or name == game.get("away_team"):
                                    odds["away"] = price
                                elif "draw" in name.lower():
                                    odds["draw"] = price
                            return odds
            
            return None
        
        except Exception as e:
            logger.warning(f"Odds API获取失败: {e}")
            return None
    
    def _get_simulated_matches(self) -> List[MatchData]:
        """获取模拟比赛数据（用于测试）"""
        # 使用历史数据创建模拟比赛
        if not self._standings:
            return []
        
        teams = list(self._standings.keys())
        if len(teams) < 2:
            return []
        
        # 创建一些模拟比赛
        matches = []
        for i in range(0, min(10, len(teams)), 2):
            if i + 1 < len(teams):
                home = teams[i]
                away = teams[i + 1]
                
                match = MatchData(
                    match_id=f"sim_{self.league_id}_{i}",
                    league_id=self.league_id,
                    home_team=home,
                    away_team=away,
                    kickoff_time=(datetime.now() + timedelta(days=i//2)).isoformat(),
                    status="upcoming",
                    source="simulated",
                    fetched_at=datetime.now().isoformat(),
                )
                matches.append(match)
        
        return matches
    
    def _enrich_match_data(self, match: MatchData):
        """添加球队数据"""
        # 添加Elo评分
        match.home_elo = self._elo_ratings.get(match.home_team, 1500.0)
        match.away_elo = self._elo_ratings.get(match.away_team, 1500.0)
        
        # 添加排名
        home_stats = self._standings.get(match.home_team, {})
        away_stats = self._standings.get(match.away_team, {})
        match.home_rank = home_stats.get("position", 10)
        match.away_rank = away_stats.get("position", 10)
    
    def _load_standings(self) -> Dict:
        """加载积分榜数据"""
        path = os.path.join(HISTORICAL_DIR, f"standings_{self.league_id}_2023-24.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_elo_ratings(self) -> Dict[str, float]:
        """加载Elo评分"""
        # 从积分榜计算Elo评分
        elo = {}
        for team, stats in self._standings.items():
            points = stats.get("points", 0)
            # 简化计算: 基于积分估算Elo
            elo[team] = 1500 + (points - 50) * 2
        return elo
    
    def _load_cache(self, key: str) -> Optional[Any]:
        """加载缓存"""
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            # 检查缓存是否过期（1小时）
            mtime = os.path.getmtime(cache_path)
            if time.time() - mtime < 3600:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        return None
    
    def _save_cache(self, key: str, data: Any):
        """保存缓存"""
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        with open(cache_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _match_to_dict(self, match: MatchData) -> Dict:
        """转换为字典"""
        return {
            "match_id": match.match_id,
            "league_id": match.league_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "kickoff_time": match.kickoff_time,
            "status": match.status,
            "odds_home": match.odds_home,
            "odds_draw": match.odds_draw,
            "odds_away": match.odds_away,
            "odds_over_25": match.odds_over_25,
            "odds_under_25": match.odds_under_25,
            "handicap_line": match.handicap_line,
            "odds_home_ah": match.odds_home_ah,
            "odds_away_ah": match.odds_away_ah,
            "home_elo": match.home_elo,
            "away_elo": match.away_elo,
            "home_rank": match.home_rank,
            "away_rank": match.away_rank,
            "source": match.source,
            "fetched_at": match.fetched_at,
        }


def get_unified_pipeline(league_id: str = "premier_league") -> UnifiedDataPipeline:
    """获取统一数据管道"""
    return UnifiedDataPipeline(league_id)
