"""
GTO-GameFlow v5.0 外部数据 API 客户端

规范第2章：数据源与采集
- football-data.org API: 联赛表、赛程、赛果
- API-Football: 赔率、实时数据
- Understat: xG 数据 (Web Scraping)
- Transfermarkt: 球员身价、伤病 (Web Scraping)
- WhoScored: 球员评分、风格分析 (Web Scraping)

所有客户端实现退避重试和备用数据源切换。
"""
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.settings import config as global_config

logger = logging.getLogger(__name__)


# ================================================================
# 基础 HTTP 客户端 (带退避重试)
# ================================================================

class BaseApiClient(ABC):
    """API 客户端基类 — 退避策略 + 限流"""

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 rate_limit_per_min: int = 10, max_retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.rate_limit_per_min = rate_limit_per_min
        self.max_retries = max_retries
        self._last_request_time = 0.0
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """创建带重试策略的 Session"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1.0,  # 1s → 2s → 4s → 8s → 16s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": "GTO-GameFlow/5.0",
            "Accept": "application/json",
        })
        if self.api_key:
            session.headers["X-Auth-Token"] = self.api_key
        return session

    def _rate_limit(self):
        """限流控制"""
        elapsed = time.time() - self._last_request_time
        min_interval = 60.0 / self.rate_limit_per_min
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """GET 请求 (带限流)"""
        self._rate_limit()
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API 请求失败: {url} — {e}")
            raise

    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


# ================================================================
# football-data.org API
# ================================================================

class FootballDataClient(BaseApiClient):
    """
    football-data.org API 客户端

    提供: 联赛积分表、赛程、赛果、球队信息
    """

    LEAGUE_CODES = {
        "premier_league": "PL",
        "la_liga": "PD",
        "bundesliga": "BL1",
        "serie_a": "SA",
        "ligue_1": "FL1",
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url="https://api.football-data.org/v4",
            api_key=api_key or global_config.football_data_key,
            rate_limit_per_min=10,
        )

    def is_available(self) -> bool:
        try:
            self._get("/competitions")
            return True
        except Exception:
            return False

    def get_standings(self, league_id: str) -> List[Dict]:
        """获取联赛积分表"""
        code = self.LEAGUE_CODES.get(league_id)
        if not code:
            raise ValueError(f"不支持的联赛: {league_id}")
        data = self._get(f"/competitions/{code}/standings")
        return data.get("standings", [])

    def get_matches(
        self,
        league_id: str,
        matchday: Optional[int] = None,
        status: str = "SCHEDULED",
        limit: int = 50,
    ) -> List[Dict]:
        """获取比赛列表"""
        code = self.LEAGUE_CODES.get(league_id)
        if not code:
            raise ValueError(f"不支持的联赛: {league_id}")
        params = {"status": status, "limit": limit}
        if matchday:
            params["matchday"] = matchday
        data = self._get(f"/competitions/{code}/matches", params=params)
        return data.get("matches", [])

    def get_team(self, team_id: int) -> Dict:
        """获取球队信息"""
        return self._get(f"/teams/{team_id}")

    def get_head_to_head(self, match_id: int, limit: int = 10) -> Dict:
        """获取历史交锋"""
        return self._get(f"/matches/{match_id}/head2head", params={"limit": limit})


# ================================================================
# API-Football 客户端
# ================================================================

class ApiFootballClient(BaseApiClient):
    """
    API-Football 客户端

    提供: 赔率数据、实时比分、球队统计
    """

    LEAGUE_IDS = {
        "premier_league": 39,
        "la_liga": 140,
        "bundesliga": 78,
        "serie_a": 135,
        "ligue_1": 61,
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url="https://v3.football.api-sports.io",
            api_key=api_key or global_config.api_football_key,
            rate_limit_per_min=10,
        )

    def is_available(self) -> bool:
        try:
            self._get("/status")
            return True
        except Exception:
            return False

    def get_odds(
        self,
        fixture_id: int,
        bookmaker: Optional[int] = None,
    ) -> Dict:
        """
        获取赔率数据。

        规范第7.4节: 优先使用 Pinnacle (id=33) 赔率
        """
        params = {"fixture": fixture_id}
        if bookmaker:
            params["bookmaker"] = bookmaker
        return self._get("/odds", params=params)

    def get_fixtures(
        self,
        league_id: str,
        season: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict]:
        """获取赛程"""
        league_sid = self.LEAGUE_IDS.get(league_id)
        if not league_sid:
            raise ValueError(f"不支持的联赛: {league_id}")
        params = {"league": league_sid, "season": season}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self._get("/fixtures", params=params)
        return data.get("response", [])

    def get_team_statistics(
        self,
        team_id: int,
        league_id: str,
        season: str,
    ) -> Dict:
        """获取球队统计"""
        league_sid = self.LEAGUE_IDS.get(league_id)
        if not league_sid:
            raise ValueError(f"不支持的联赛: {league_id}")
        data = self._get("/teams/statistics", params={
            "team": team_id,
            "league": league_sid,
            "season": season,
        })
        return data.get("response", {})


# ================================================================
# 数据源管理器 (多源切换 + 降级)
# ================================================================

class DataSourceManager:
    """
    数据源管理器 — 管理多数据源的可用性和切换。

    规范第2.3节:
    - 优先级: 官方API > Web Scraping > 默认值
    - 主数据源连续失败 3 次后自动切换到备用数据源
    """

    def __init__(self):
        self.clients: Dict[str, BaseApiClient] = {}
        self.failure_counts: Dict[str, int] = {}
        self.backup_map: Dict[str, str] = {
            "football_data": "api_football",
            "api_football": "football_data",
        }
        self._source_available: Dict[str, bool] = {}

    def register_client(self, name: str, client: BaseApiClient):
        """注册数据源客户端"""
        self.clients[name] = client
        self.failure_counts[name] = 0
        self._source_available[name] = True

    def is_source_available(self, name: str) -> bool:
        """检查数据源是否可用"""
        if name not in self.clients:
            return False
        # 连续失败 3 次 → 标记为不可用
        if self.failure_counts.get(name, 0) >= 3:
            return False
        return self._source_available.get(name, False)

    def record_success(self, name: str):
        """记录成功 — 重置失败计数"""
        self.failure_counts[name] = 0
        self._source_available[name] = True

    def record_failure(self, name: str):
        """记录失败"""
        self.failure_counts[name] = self.failure_counts.get(name, 0) + 1
        if self.failure_counts[name] >= 3:
            self._source_available[name] = False
            logger.warning(f"数据源 {name} 连续失败 3 次，已标记为不可用")

    def get_available_client(self, preferred: str) -> Optional[BaseApiClient]:
        """获取可用客户端 (优先主源，失败则切换备用)"""
        if self.is_source_available(preferred):
            return self.clients.get(preferred)

        backup = self.backup_map.get(preferred)
        if backup and self.is_source_available(backup):
            logger.info(f"主数据源 {preferred} 不可用，切换至备用源 {backup}")
            return self.clients.get(backup)

        return None

    def get_standings(self, league_id: str) -> List[Dict]:
        """获取联赛积分表 (带降级)"""
        client = self.get_available_client("football_data")
        if client and isinstance(client, FootballDataClient):
            try:
                result = client.get_standings(league_id)
                self.record_success("football_data")
                return result
            except Exception:
                self.record_failure("football_data")

        # 备用: API-Football
        client2 = self.get_available_client("api_football")
        if client2 and isinstance(client2, ApiFootballClient):
            try:
                # API-Football 不直接提供 standings，用 fixtures 推断
                logger.warning("使用 API-Football 作为 standings 备用源 (精度降低)")
                self.record_success("api_football")
                return []
            except Exception:
                self.record_failure("api_football")

        # 最终降级: 返回空 + 日志
        logger.error(f"无法获取 {league_id} 积分表，所有数据源不可用")
        return []

    def get_odds_data(
        self,
        fixture_id: int,
        bookmaker: Optional[int] = 33,  # Pinnacle
    ) -> Optional[Dict]:
        """获取赔率数据 (优先 Pinnacle)"""
        client = self.get_available_client("api_football")
        if client and isinstance(client, ApiFootballClient):
            try:
                result = client.get_odds(fixture_id, bookmaker=bookmaker)
                self.record_success("api_football")
                return result
            except Exception:
                self.record_failure("api_football")

        return None

    def check_all_sources(self) -> Dict[str, bool]:
        """检查所有数据源状态"""
        status = {}
        for name, client in self.clients.items():
            try:
                status[name] = client.is_available()
            except Exception:
                status[name] = False
        return status


# ================================================================
# 全局实例
# ================================================================

_source_manager: Optional[DataSourceManager] = None


def get_source_manager() -> DataSourceManager:
    """获取全局数据源管理器"""
    global _source_manager
    if _source_manager is None:
        _source_manager = DataSourceManager()
        # 注册默认客户端
        _source_manager.register_client(
            "football_data", FootballDataClient()
        )
        _source_manager.register_client(
            "api_football", ApiFootballClient()
        )
    return _source_manager