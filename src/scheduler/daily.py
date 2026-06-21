"""
GTO v6.0 — 每日自动更新调度器

自动执行:
1. 每日采集赛程和赔率
2. 赛后更新比赛结果和统计
3. 更新积分榜
4. 更新天气数据
5. 累积历史数据

使用方式:
    scheduler = DailyScheduler()
    scheduler.run_daily()
"""

from __future__ import annotations
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'collected')
os.makedirs(DATA_DIR, exist_ok=True)


class DailyScheduler:
    """每日自动更新调度器"""
    
    def __init__(self):
        self.leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1", "worldcup"]
        self.last_run = self._load_last_run()
    
    def run_daily(self):
        """执行每日更新"""
        logger.info("开始每日更新...")
        
        for league in self.leagues:
            try:
                self._update_league(league)
            except Exception as e:
                logger.error(f"更新 {league} 失败: {e}")
        
        self._save_last_run()
        logger.info("每日更新完成")
    
    def _update_league(self, league_id: str):
        """更新单个联赛"""
        from src.data.collector import get_collector
        from src.data.factor_adapter import get_factor_adapter
        
        collector = get_collector(league_id)
        adapter = get_factor_adapter(league_id)
        
        # 1. 采集赛程
        matches = collector.collect_upcoming_matches()
        logger.info(f"  {league_id}: 获取 {len(matches)} 场比赛")
        
        # 2. 采集积分榜
        standings = collector.collect_standings()
        if standings:
            logger.info(f"  {league_id}: 获取 {len(standings)} 支球队积分榜")
        
        # 3. 采集天气（未来比赛）
        for match in matches[:5]:  # 只获取前5场
            if match.city:
                weather = collector.collect_weather(match.city)
                if weather:
                    logger.debug(f"  {league_id}: 获取 {match.city} 天气")
        
        # 4. 更新历史数据
        # 从已结束比赛累积
        for match in matches:
            if match.status == "finished" and match.home_score is not None:
                adapter.update_history(
                    match.home_team,
                    match.home_score,
                    match.away_score,
                    match.shots_home,
                    match.sot_home,
                    match.corners_home,
                    match.yellows_home,
                    match.ht_home,
                )
                adapter.update_history(
                    match.away_team,
                    match.away_score,
                    match.home_score,
                    match.shots_away,
                    match.sot_away,
                    match.corners_away,
                    match.yellows_away,
                    match.ht_away,
                )
                adapter.update_elo(
                    match.home_team,
                    match.away_team,
                    match.home_score,
                    match.away_score,
                )
    
    def _load_last_run(self) -> str:
        """加载上次运行时间"""
        path = os.path.join(DATA_DIR, "last_run.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get("last_run", "")
        return ""
    
    def _save_last_run(self):
        """保存运行时间"""
        path = os.path.join(DATA_DIR, "last_run.json")
        with open(path, 'w') as f:
            json.dump({"last_run": datetime.now().isoformat()}, f)
    
    def get_status(self) -> Dict:
        """获取调度器状态"""
        return {
            "last_run": self.last_run,
            "leagues": self.leagues,
            "data_dir": DATA_DIR,
        }


# 全局实例
_scheduler = None


def get_scheduler() -> DailyScheduler:
    """获取调度器"""
    global _scheduler
    if _scheduler is None:
        _scheduler = DailyScheduler()
    return _scheduler


def run_daily_update():
    """运行每日更新"""
    scheduler = get_scheduler()
    scheduler.run_daily()
