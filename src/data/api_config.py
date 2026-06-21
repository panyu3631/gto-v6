"""
GTO v6.0 — 数据源配置

配置各种数据源的API密钥和端点。
填入API密钥后即可使用对应数据源。
"""

from __future__ import annotations
import os
import json
from typing import Dict, Optional

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.json')


# 默认配置
DEFAULT_CONFIG = {
    # football-data.org (免费注册获取)
    "football_data": {
        "enabled": False,
        "api_key": "",
        "base_url": "https://api.football-data.org/v4",
        "rate_limit": 10,  # 每分钟请求数
    },
    
    # API-Football (付费)
    "api_football": {
        "enabled": False,
        "api_key": "",
        "base_url": "https://v3.football.api-sports.io",
        "rate_limit": 100,
    },
    
    # Odds API (免费层可用)
    "odds_api": {
        "enabled": False,
        "api_key": "",
        "base_url": "https://api.the-odds-api.com/v4",
        "rate_limit": 500,  # 每月请求数
    },
    
    # ESPN (公开API，无需密钥)
    "espn": {
        "enabled": True,
        "api_key": "",
        "base_url": "https://site.api.espn.com/apis/site/v2",
        "rate_limit": 100,
    },
    
    # Sofascore (公开API)
    "sofascore": {
        "enabled": True,
        "api_key": "",
        "base_url": "https://api.sofascore.com/api/v1",
        "rate_limit": 100,
    },
    
    # Open-Meteo (免费天气API)
    "open_meteo": {
        "enabled": True,
        "api_key": "",
        "base_url": "https://api.open-meteo.com/v1",
        "rate_limit": 10000,
    },
}


def load_config() -> Dict:
    """加载API配置"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(config: Dict):
    """保存API配置"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def get_api_key(provider: str) -> Optional[str]:
    """获取API密钥"""
    config = load_config()
    provider_config = config.get(provider, {})
    if provider_config.get("enabled", False):
        return provider_config.get("api_key", "")
    return None


def is_provider_enabled(provider: str) -> bool:
    """检查数据源是否启用"""
    config = load_config()
    return config.get(provider, {}).get("enabled", False)


def enable_provider(provider: str, api_key: str = ""):
    """启用数据源"""
    config = load_config()
    if provider not in config:
        config[provider] = DEFAULT_CONFIG.get(provider, {})
    config[provider]["enabled"] = True
    if api_key:
        config[provider]["api_key"] = api_key
    save_config(config)


def disable_provider(provider: str):
    """禁用数据源"""
    config = load_config()
    if provider in config:
        config[provider]["enabled"] = False
    save_config(config)


# 数据源能力说明
PROVIDER_CAPABILITIES = {
    "football_data": {
        "name": "Football-Data.org",
        "data": ["赛程", "积分榜", "比赛结果"],
        "odds": False,
        "free": True,
        "registration": "需要免费注册获取API Key",
        "url": "https://www.football-data.org/client/register",
    },
    "api_football": {
        "name": "API-Football",
        "data": ["赛程", "积分榜", "比赛结果", "伤病", "阵容", "xG"],
        "odds": True,
        "free": False,
        "registration": "需要付费订阅",
        "url": "https://www.api-football.com/",
    },
    "odds_api": {
        "name": "The Odds API",
        "data": ["赛程"],
        "odds": True,
        "free": True,
        "registration": "免费层每月500次请求",
        "url": "https://the-odds-api.com/",
    },
    "espn": {
        "name": "ESPN API",
        "data": ["赛程", "积分榜", "比赛结果"],
        "odds": False,
        "free": True,
        "registration": "无需注册",
    },
    "sofascore": {
        "name": "Sofascore API",
        "data": ["赛程", "积分榜", "比赛结果", "xG", "球员数据"],
        "odds": False,
        "free": True,
        "registration": "无需注册",
    },
    "open_meteo": {
        "name": "Open-Meteo",
        "data": ["天气"],
        "odds": False,
        "free": True,
        "registration": "无需注册",
    },
}


def print_provider_status():
    """打印数据源状态"""
    config = load_config()
    
    print("\n数据源状态:")
    print("-" * 60)
    
    for provider, capabilities in PROVIDER_CAPABILITIES.items():
        provider_config = config.get(provider, {})
        enabled = provider_config.get("enabled", False)
        has_key = bool(provider_config.get("api_key", ""))
        
        status = "✅ 启用" if enabled else "❌ 禁用"
        key_status = "有密钥" if has_key else "无密钥"
        
        print(f"\n  {capabilities['name']}:")
        print(f"    状态: {status}")
        print(f"    密钥: {key_status}")
        print(f"    数据: {', '.join(capabilities['data'])}")
        print(f"    赔率: {'是' if capabilities['odds'] else '否'}")
        print(f"    免费: {'是' if capabilities['free'] else '否'}")


if __name__ == "__main__":
    print_provider_status()
