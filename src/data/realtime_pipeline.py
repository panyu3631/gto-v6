"""
GTO v6.0 — 实时数据管道

数据源:
1. 赛程 — 从公开网站抓取
2. 赔率 — 从赔率比较网站抓取
3. 球队数据 — 从统计网站抓取

使用方式:
    pipeline = DataPipeline(league_id="premier_league")
    matches = pipeline.get_upcoming_matches()
    odds = pipeline.get_odds(match_id)
"""

from __future__ import annotations
import json
import os
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'realtime')
os.makedirs(DATA_DIR, exist_ok=True)

# 联赛ID映射
LEAGUE_IDS = {
    "premier_league": {"espn": "eng.1", "football_data": "PL", "name": "英超"},
    "la_liga": {"espn": "esp.1", "football_data": "PD", "name": "西甲"},
    "bundesliga": {"espn": "ger.1", "football_data": "BL1", "name": "德甲"},
    "serie_a": {"espn": "ita.1", "football_data": "SA", "name": "意甲"},
    "ligue_1": {"espn": "fra.1", "football_data": "FL1", "name": "法甲"},
}


@dataclass
class LiveMatch:
    """实时比赛数据"""
    match_id: str
    league_id: str
    home_team: str
    away_team: str
    kickoff_time: str
    status: str = "upcoming"  # upcoming, live, finished
    
    # 赔率
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
    
    # 元数据
    source: str = ""
    fetched_at: str = ""


class DataPipeline:
    """实时数据管道"""
    
    def __init__(self, league_id: str):
        self.league_id = league_id
        self.league_info = LEAGUE_IDS.get(league_id, {})
        self._cache: Dict[str, Any] = {}
        self._cache_dir = os.path.join(DATA_DIR, league_id)
        os.makedirs(self._cache_dir, exist_ok=True)
    
    def get_upcoming_matches(self, days: int = 7) -> List[LiveMatch]:
        """获取未来N天的比赛"""
        matches = []
        
        # 尝试从缓存获取
        cached = self._load_cache("upcoming_matches")
        if cached:
            return [LiveMatch(**m) for m in cached]
        
        # 从多个源尝试获取
        matches = self._fetch_from_espn()
        if not matches:
            matches = self._fetch_from_football_data()
        if not matches:
            matches = self._fetch_from_sofascore()
        
        # 缓存结果
        if matches:
            self._save_cache("upcoming_matches", [self._match_to_dict(m) for m in matches])
        
        return matches
    
    def get_odds(self, match_id: str) -> Optional[Dict]:
        """获取比赛赔率"""
        # 尝试从缓存获取
        cached = self._load_cache(f"odds_{match_id}")
        if cached:
            return cached
        
        # 从赔率网站获取
        odds = self._fetch_odds_from_oddsportal(match_id)
        if odds:
            self._save_cache(f"odds_{match_id}", odds)
        
        return odds
    
    def get_team_stats(self, team: str) -> Optional[Dict]:
        """获取球队统计"""
        cached = self._load_cache(f"team_{team}")
        if cached:
            return cached
        
        stats = self._fetch_team_stats(team)
        if stats:
            self._save_cache(f"team_{team}", stats)
        
        return stats
    
    def _fetch_from_espn(self) -> List[LiveMatch]:
        """从ESPN获取赛程"""
        import urllib.request
        
        espn_id = self.league_info.get("espn", "")
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
                
                # 获取赔率
                odds_data = competition.get("odds", [{}])
                odds = odds_data[0] if odds_data else {}
                
                match = LiveMatch(
                    match_id=event.get("id", ""),
                    league_id=self.league_id,
                    home_team=home.get("team", {}).get("shortDisplayName", ""),
                    away_team=away.get("team", {}).get("shortDisplayName", ""),
                    kickoff_time=event.get("date", ""),
                    status="upcoming",
                    odds_home=float(odds.get("homeTeamOdds", {}).get("moneyLine", 0) or 0),
                    odds_draw=float(odds.get("drawOdds", {}).get("moneyLine", 0) or 0),
                    odds_away=float(odds.get("awayTeamOdds", {}).get("moneyLine", 0) or 0),
                    source="espn",
                    fetched_at=datetime.now().isoformat(),
                )
                matches.append(match)
            
            logger.info(f"从ESPN获取到 {len(matches)} 场比赛")
            return matches
        
        except Exception as e:
            logger.warning(f"ESPN获取失败: {e}")
            return []
    
    def _fetch_from_football_data(self) -> List[LiveMatch]:
        """从football-data.org获取赛程"""
        import urllib.request
        
        fd_id = self.league_info.get("football_data", "")
        if not fd_id:
            return []
        
        try:
            url = f"https://api.football-data.org/v4/competitions/{fd_id}/matches?status=SCHEDULED"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            matches = []
            for match_data in data.get("matches", []):
                home_team = match_data.get("homeTeam", {}).get("name", "")
                away_team = match_data.get("awayTeam", {}).get("name", "")
                
                match = LiveMatch(
                    match_id=str(match_data.get("id", "")),
                    league_id=self.league_id,
                    home_team=home_team,
                    away_team=away_team,
                    kickoff_time=match_data.get("utcDate", ""),
                    status="upcoming",
                    source="football_data",
                    fetched_at=datetime.now().isoformat(),
                )
                matches.append(match)
            
            logger.info(f"从football-data.org获取到 {len(matches)} 场比赛")
            return matches
        
        except Exception as e:
            logger.warning(f"football-data.org获取失败: {e}")
            return []
    
    def _fetch_from_sofascore(self) -> List[LiveMatch]:
        """从Sofascore获取赛程"""
        import urllib.request
        
        try:
            # Sofascore API
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{today}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            matches = []
            league_name = self.league_info.get("name", "")
            
            for event in data.get("events", []):
                tournament = event.get("tournament", {})
                if league_name not in tournament.get("name", ""):
                    continue
                
                home_team = event.get("homeTeam", {}).get("name", "")
                away_team = event.get("awayTeam", {}).get("name", "")
                
                match = LiveMatch(
                    match_id=str(event.get("id", "")),
                    league_id=self.league_id,
                    home_team=home_team,
                    away_team=away_team,
                    kickoff_time=datetime.fromtimestamp(event.get("startTimestamp", 0)).isoformat(),
                    status="upcoming",
                    source="sofascore",
                    fetched_at=datetime.now().isoformat(),
                )
                matches.append(match)
            
            logger.info(f"从Sofascore获取到 {len(matches)} 场比赛")
            return matches
        
        except Exception as e:
            logger.warning(f"Sofascore获取失败: {e}")
            return []
    
    def _fetch_odds_from_oddsportal(self, match_id: str) -> Optional[Dict]:
        """从OddsPortal获取赔率"""
        # OddsPortal需要浏览器渲染，简化版返回None
        return None
    
    def _fetch_team_stats(self, team: str) -> Optional[Dict]:
        """获取球队统计"""
        # 从已抓取的数据中获取
        standings_path = os.path.join(DATA_DIR, '..', 'scraped', f'standings_{self.league_id}_2023-24.json')
        if os.path.exists(standings_path):
            with open(standings_path, 'r') as f:
                standings = json.load(f)
            return standings.get(team, None)
        return None
    
    def _load_cache(self, key: str) -> Optional[Any]:
        """加载缓存"""
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            # 检查缓存是否过期（1小时）
            mtime = os.path.getmtime(cache_path)
            if datetime.now().timestamp() - mtime < 3600:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        return None
    
    def _save_cache(self, key: str, data: Any):
        """保存缓存"""
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        with open(cache_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _match_to_dict(self, match: LiveMatch) -> Dict:
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
            "source": match.source,
            "fetched_at": match.fetched_at,
        }


class MultiLeaguePipeline:
    """多联赛数据管道"""
    
    def __init__(self, leagues: List[str] = None):
        self.leagues = leagues or list(LEAGUE_IDS.keys())
        self.pipelines: Dict[str, DataPipeline] = {}
        for league in self.leagues:
            self.pipelines[league] = DataPipeline(league)
    
    def get_all_upcoming_matches(self, days: int = 7) -> Dict[str, List[LiveMatch]]:
        """获取所有联赛的未来比赛"""
        all_matches = {}
        for league, pipeline in self.pipelines.items():
            matches = pipeline.get_upcoming_matches(days)
            if matches:
                all_matches[league] = matches
        return all_matches
    
    def get_all_odds(self) -> Dict[str, Dict]:
        """获取所有比赛的赔率"""
        all_odds = {}
        for league, pipeline in self.pipelines.items():
            matches = pipeline.get_upcoming_matches()
            for match in matches:
                odds = pipeline.get_odds(match.match_id)
                if odds:
                    all_odds[match.match_id] = odds
        return all_odds


# 全局数据管道
_pipeline = None


def get_data_pipeline(league_id: str = None) -> DataPipeline:
    """获取数据管道"""
    if league_id:
        return DataPipeline(league_id)
    return DataPipeline("premier_league")


def get_multi_league_pipeline() -> MultiLeaguePipeline:
    """获取多联赛数据管道"""
    return MultiLeaguePipeline()
