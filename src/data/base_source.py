"""
GTO v6.0 — 数据源抽象接口

所有数据源必须实现此接口，确保：
1. 统一的数据获取方式
2. 支持多种数据来源（CSV/API/爬虫）
3. 数据缓存和降级策略
4. 数据质量检查

使用方式:
    class MyDataSource(BaseDataSource):
        def fetch(self, key: str) -> Optional[Dict]:
            return {"data": ...}
"""

from __future__ import annotations
import json
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """数据源基类 — 所有数据源必须继承此类"""
    
    source_id: str = ""
    source_name: str = ""
    data_type: str = ""  # standings, xg, injuries, odds, etc.
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._cache: Dict[str, Any] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    @abstractmethod
    def fetch(self, key: str, **kwargs) -> Optional[Any]:
        """
        获取数据。
        
        参数:
            key: 数据键（如联赛ID、球队名）
            **kwargs: 额外参数
        
        返回:
            数据字典或 None（如果获取失败）
        """
        pass
    
    def fetch_cached(self, key: str, **kwargs) -> Optional[Any]:
        """获取数据（优先使用缓存）"""
        cache_key = f"{self.source_id}_{key}"
        
        # 检查内存缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 检查文件缓存
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._cache[cache_key] = data
                    return data
                except Exception as e:
                    logger.warning(f"读取缓存失败: {e}")
        
        # 从源获取
        data = self.fetch(key, **kwargs)
        if data:
            self._cache[cache_key] = data
            self._save_to_cache(cache_key, data)
        
        return data
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """保存到文件缓存"""
        if not self.cache_dir:
            return
        
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        if self.cache_dir:
            for f in os.listdir(self.cache_dir):
                if f.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, f))
    
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        return True


class DataSourceManager:
    """数据源管理器 — 管理所有数据源"""
    
    def __init__(self):
        self._sources: Dict[str, BaseDataSource] = {}
    
    def register(self, source: BaseDataSource):
        """注册数据源"""
        self._sources[source.source_id] = source
        logger.debug(f"数据源已注册: {source.source_id}")
    
    def get(self, source_id: str) -> Optional[BaseDataSource]:
        """获取数据源"""
        return self._sources.get(source_id)
    
    def fetch(self, source_id: str, key: str, **kwargs) -> Optional[Any]:
        """从指定数据源获取数据"""
        source = self._sources.get(source_id)
        if source:
            return source.fetch_cached(key, **kwargs)
        return None
    
    def list_sources(self) -> List[str]:
        """列出所有数据源"""
        return list(self._sources.keys())
    
    def list_available(self) -> List[str]:
        """列出可用数据源"""
        return [sid for sid, s in self._sources.items() if s.is_available()]


# 具体数据源实现

class CSVStandingsSource(BaseDataSource):
    """CSV积分榜数据源"""
    
    source_id = "csv_standings"
    source_name = "CSV积分榜"
    data_type = "standings"
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.data_dir = data_dir
    
    def fetch(self, key: str, **kwargs) -> Optional[Dict]:
        """获取积分榜数据"""
        season = kwargs.get('season', '2023-24')
        path = os.path.join(self.data_dir, f"standings_{key}_{season}.json")
        
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


class ScrapedXGSource(BaseDataSource):
    """网页抓取xG数据源"""
    
    source_id = "scraped_xg"
    source_name = "网页抓取xG"
    data_type = "xg"
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.data_dir = data_dir
    
    def fetch(self, key: str, **kwargs) -> Optional[Dict]:
        """获取xG数据"""
        path = os.path.join(self.data_dir, "xg_data.json")
        
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(key, None)
        return None


class WeatherAPISource(BaseDataSource):
    """天气API数据源"""
    
    source_id = "weather_api"
    source_name = "Open-Meteo天气"
    data_type = "weather"
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.cities = {
            "premier_league": {"lat": 51.5, "lon": -0.12},
            "la_liga": {"lat": 40.4, "lon": -3.7},
            "bundesliga": {"lat": 52.52, "lon": 13.4},
            "serie_a": {"lat": 41.9, "lon": 12.5},
            "ligue_1": {"lat": 48.85, "lon": 2.35},
        }
    
    def fetch(self, key: str, **kwargs) -> Optional[Dict]:
        """获取天气数据"""
        import urllib.request
        
        city = self.cities.get(key)
        if not city:
            return None
        
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={city['lat']}&longitude={city['lon']}&current_weather=true"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            current = data.get("current_weather", {})
            temp = current.get("temperature", 15)
            wind = current.get("windspeed", 0)
            
            # 计算影响因子
            temp_score = max(0, min(1, 1 - abs(temp - 18) / 20))
            wind_score = max(0, 1 - wind / 30)
            impact = (temp_score * 0.5 + wind_score * 0.5) * 2 - 1
            
            return {"temp": temp, "wind": wind, "impact": impact}
        except Exception as e:
            logger.warning(f"获取天气失败: {e}")
            return None


# 全局数据源管理器
_manager = None


def get_data_source_manager() -> DataSourceManager:
    """获取全局数据源管理器"""
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager
