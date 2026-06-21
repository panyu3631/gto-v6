"""
GTO v6.0 — 热更新模块

运行时更新因子权重和配置，无需重启。

使用方式:
    updater = HotUpdater()
    updater.update_factor_weight("F1", 1.2)
    updater.update_league_param("premier_league", "elo_k", 25)
    updater.reload_config()
"""

from __future__ import annotations
import json
import os
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HotUpdater:
    """热更新管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._watchers: List[Callable] = []
        self._update_history: List[Dict] = []
        self._lock = threading.Lock()
    
    def update_factor_weight(self, factor_id: str, weight: float):
        """更新因子权重"""
        with self._lock:
            from src.config.config_loader import get_config
            config = get_config()
            config.update_factor_weight(factor_id, weight)
            
            self._record_update("factor_weight", {
                "factor_id": factor_id,
                "weight": weight,
            })
            
            self._notify_watchers({
                "type": "factor_weight",
                "factor_id": factor_id,
                "weight": weight,
            })
            
            logger.info(f"因子权重已更新: {factor_id} = {weight}")
    
    def enable_factor(self, factor_id: str):
        """启用因子"""
        with self._lock:
            from src.config.config_loader import get_config
            config = get_config()
            config.enable_factor(factor_id)
            
            self._record_update("factor_enable", {"factor_id": factor_id})
            logger.info(f"因子已启用: {factor_id}")
    
    def disable_factor(self, factor_id: str, reason: str = ""):
        """禁用因子"""
        with self._lock:
            from src.config.config_loader import get_config
            config = get_config()
            config.disable_factor(factor_id, reason)
            
            self._record_update("factor_disable", {
                "factor_id": factor_id,
                "reason": reason,
            })
            logger.info(f"因子已禁用: {factor_id}")
    
    def update_league_param(self, league_id: str, param: str, value: Any):
        """更新联赛参数"""
        with self._lock:
            from src.config.config_loader import get_config
            config = get_config()
            config.update_league_param(league_id, param, value)
            
            self._record_update("league_param", {
                "league_id": league_id,
                "param": param,
                "value": value,
            })
            
            self._notify_watchers({
                "type": "league_param",
                "league_id": league_id,
                "param": param,
                "value": value,
            })
            
            logger.info(f"联赛参数已更新: {league_id}.{param} = {value}")
    
    def update_strategy_param(self, strategy: str, param: str, value: Any):
        """更新策略参数"""
        with self._lock:
            from src.config.config_loader import get_config
            config = get_config()
            
            if 'strategies' not in config._config:
                config._config['strategies'] = {}
            if strategy not in config._config['strategies']:
                config._config['strategies'][strategy] = {}
            config._config['strategies'][strategy][param] = value
            
            self._record_update("strategy_param", {
                "strategy": strategy,
                "param": param,
                "value": value,
            })
            
            logger.info(f"策略参数已更新: {strategy}.{param} = {value}")
    
    def reload_config(self):
        """重新加载配置"""
        with self._lock:
            from src.config.config_loader import reload_config
            reload_config()
            
            self._record_update("config_reload", {})
            logger.info("配置已重新加载")
    
    def save_config(self, path: Optional[str] = None):
        """保存当前配置"""
        with self._lock:
            from src.config.config_loader import get_config
            config = get_config()
            config.save(path)
            
            self._record_update("config_save", {"path": path})
            logger.info(f"配置已保存: {path or 'default'}")
    
    def register_watcher(self, callback: Callable):
        """注册更新监听器"""
        self._watchers.append(callback)
    
    def _notify_watchers(self, update: Dict):
        """通知监听器"""
        for watcher in self._watchers:
            try:
                watcher(update)
            except Exception as e:
                logger.warning(f"监听器通知失败: {e}")
    
    def _record_update(self, update_type: str, details: Dict):
        """记录更新历史"""
        self._update_history.append({
            "type": update_type,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })
        # 保留最近100条
        if len(self._update_history) > 100:
            self._update_history = self._update_history[-100:]
    
    def get_update_history(self) -> List[Dict]:
        """获取更新历史"""
        return list(self._update_history)
    
    def batch_update(self, updates: List[Dict]):
        """批量更新"""
        for update in updates:
            update_type = update.get("type")
            
            if update_type == "factor_weight":
                self.update_factor_weight(update["factor_id"], update["weight"])
            elif update_type == "factor_enable":
                self.enable_factor(update["factor_id"])
            elif update_type == "factor_disable":
                self.disable_factor(update["factor_id"], update.get("reason", ""))
            elif update_type == "league_param":
                self.update_league_param(update["league_id"], update["param"], update["value"])
            elif update_type == "strategy_param":
                self.update_strategy_param(update["strategy"], update["param"], update["value"])
            elif update_type == "config_reload":
                self.reload_config()
            else:
                logger.warning(f"未知更新类型: {update_type}")


class ConfigWatcher:
    """配置文件监听器"""
    
    def __init__(self, config_path: str, callback: Callable):
        self.config_path = config_path
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_modified = 0.0
    
    def start(self):
        """启动监听"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"配置监听已启动: {self.config_path}")
    
    def stop(self):
        """停止监听"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _watch_loop(self):
        """监听循环"""
        while self._running:
            try:
                if os.path.exists(self.config_path):
                    mtime = os.path.getmtime(self.config_path)
                    if mtime > self._last_modified:
                        self._last_modified = mtime
                        self.callback()
                        logger.info("配置文件已变更，触发重新加载")
                
                time.sleep(5)
            except Exception as e:
                logger.warning(f"配置监听错误: {e}")
                time.sleep(10)


# 全局热更新器
_updater = None


def get_hot_updater() -> HotUpdater:
    """获取全局热更新器"""
    global _updater
    if _updater is None:
        _updater = HotUpdater()
    return _updater
