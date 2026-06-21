"""
GTO v6.0 — 配置加载器

从 YAML 配置文件加载所有模型参数。
修改配置文件即可调整模型行为，无需改代码。
"""

from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, 'model_config.yaml')


def _load_yaml_simple(path: str) -> Dict:
    """简单的YAML解析（不依赖PyYAML）"""
    result = {}
    current_section = None
    current_subsection = None
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 计算缩进级别
            indent = len(line) - len(line.lstrip())
            content = line.strip()
            
            # 顶级section
            if indent == 0 and content.endswith(':'):
                current_section = content[:-1]
                result[current_section] = {}
                current_subsection = None
                continue
            
            # 二级section
            if indent == 2 and content.endswith(':'):
                current_subsection = content[:-1]
                if current_section:
                    result[current_section][current_subsection] = {}
                continue
            
            # 键值对
            if ':' in content:
                key, _, value = content.partition(':')
                key = key.strip()
                value = value.strip()
                
                # 解析值
                parsed_value = _parse_value(value)
                
                if current_subsection and current_section:
                    result[current_section][current_subsection][key] = parsed_value
                elif current_section:
                    result[current_section][key] = parsed_value
    
    return result


def _parse_value(value: str) -> Any:
    """解析YAML值"""
    if not value:
        return None
    
    # 移除引号
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    # 布尔值
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    
    # None
    if value.lower() in ('null', 'none', '~'):
        return None
    
    # 数字
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # 列表 [a, b, c]
    if value.startswith('[') and value.endswith(']'):
        items = value[1:-1].split(',')
        return [_parse_value(item.strip()) for item in items]
    
    # 字典 {a: b, c: d}
    if value.startswith('{') and value.endswith('}'):
        result = {}
        items = value[1:-1].split(',')
        for item in items:
            if ':' in item:
                k, _, v = item.partition(':')
                result[k.strip()] = _parse_value(v.strip())
        return result
    
    return value


class ModelConfig:
    """模型配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG
        self._config = {}
        self.load()
    
    def load(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            self._config = _load_yaml_simple(self.config_path)
            logger.info(f"配置已加载: {self.config_path}")
        else:
            logger.warning(f"配置文件不存在: {self.config_path}")
    
    def get_league_config(self, league_id: str) -> Dict:
        """获取联赛配置"""
        return self._config.get('leagues', {}).get(league_id, {})
    
    def get_factor_config(self, factor_id: str) -> Dict:
        """获取因子配置"""
        return self._config.get('factors', {}).get(factor_id, {})
    
    def get_strategy_config(self, strategy: str) -> Dict:
        """获取策略配置"""
        return self._config.get('strategies', {}).get(strategy, {})
    
    def get_ensemble_config(self) -> Dict:
        """获取集成配置"""
        return self._config.get('ensemble', {})
    
    def get_backtest_config(self) -> Dict:
        """获取回测配置"""
        return self._config.get('backtest', {})
    
    def is_factor_enabled(self, factor_id: str) -> bool:
        """检查因子是否启用"""
        config = self.get_factor_config(factor_id)
        return config.get('enabled', True)
    
    def get_factor_weight(self, factor_id: str) -> float:
        """获取因子权重"""
        config = self.get_factor_config(factor_id)
        return config.get('weight', 0.0)
    
    def get_enabled_factors(self) -> Dict[str, Dict]:
        """获取所有启用的因子"""
        factors = self._config.get('factors', {})
        return {k: v for k, v in factors.items() if v.get('enabled', True)}
    
    def get_disabled_factors(self) -> Dict[str, Dict]:
        """获取所有禁用的因子"""
        factors = self._config.get('factors', {})
        return {k: v for k, v in factors.items() if not v.get('enabled', True)}
    
    def update_league_param(self, league_id: str, param: str, value: Any):
        """更新联赛参数"""
        if 'leagues' not in self._config:
            self._config['leagues'] = {}
        if league_id not in self._config['leagues']:
            self._config['leagues'][league_id] = {}
        self._config['leagues'][league_id][param] = value
    
    def update_factor_weight(self, factor_id: str, weight: float):
        """更新因子权重"""
        if 'factors' not in self._config:
            self._config['factors'] = {}
        if factor_id not in self._config['factors']:
            self._config['factors'][factor_id] = {}
        self._config['factors'][factor_id]['weight'] = weight
    
    def enable_factor(self, factor_id: str):
        """启用因子"""
        if 'factors' not in self._config:
            self._config['factors'] = {}
        if factor_id not in self._config['factors']:
            self._config['factors'][factor_id] = {}
        self._config['factors'][factor_id]['enabled'] = True
    
    def disable_factor(self, factor_id: str, reason: str = ""):
        """禁用因子"""
        if 'factors' not in self._config:
            self._config['factors'] = {}
        if factor_id not in self._config['factors']:
            self._config['factors'][factor_id] = {}
        self._config['factors'][factor_id]['enabled'] = False
        if reason:
            self._config['factors'][factor_id]['reason'] = reason
    
    def save(self, path: Optional[str] = None):
        """保存配置到文件"""
        save_path = path or self.config_path
        # 简单的YAML输出
        with open(save_path, 'w', encoding='utf-8') as f:
            _write_yaml(f, self._config, indent=0)
        logger.info(f"配置已保存: {save_path}")
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return dict(self._config)


def _write_yaml(f, data, indent=0):
    """写入YAML格式"""
    prefix = '  ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                f.write(f"{prefix}{key}:\n")
                _write_yaml(f, value, indent + 1)
            elif isinstance(value, list):
                f.write(f"{prefix}{key}: {value}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")
    elif isinstance(data, list):
        for item in data:
            f.write(f"{prefix}- {item}\n")


# 全局配置实例
_config_instance = None


def get_config(config_path: Optional[str] = None) -> ModelConfig:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ModelConfig(config_path)
    return _config_instance


def reload_config():
    """重新加载配置"""
    global _config_instance
    if _config_instance:
        _config_instance.load()
