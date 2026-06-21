"""
GTO v6.0 — 健康检查监控

监控内容:
1. 因子激活率 — 每个因子是否正常工作
2. 数据质量 — 数据是否完整、是否有异常值
3. 模型性能 — 预测准确率、ROI
4. 系统状态 — 内存、CPU、运行时间

使用方式:
    monitor = HealthMonitor()
    monitor.check_factors(factor_deltas)
    monitor.check_data(data)
    monitor.get_report()
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """健康状态"""
    component: str
    status: str  # "healthy" / "warning" / "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthMonitor:
    """健康检查监控器"""
    
    def __init__(self):
        self._statuses: List[HealthStatus] = []
        self._factor_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "non_zero": 0, "errors": 0})
        self._prediction_stats = {"total": 0, "correct": 0, "roi_sum": 0.0}
        self._start_time = time.time()
    
    def check_factors(self, factor_deltas: Dict[str, Dict[str, float]]) -> List[HealthStatus]:
        """检查因子激活状态"""
        statuses = []
        total = len(factor_deltas)
        non_zero = 0
        zero_factors = []
        
        for fid, deltas in factor_deltas.items():
            self._factor_stats[fid]["total"] += 1
            
            is_non_zero = any(abs(v) > 0.001 for v in deltas.values())
            if is_non_zero:
                non_zero += 1
                self._factor_stats[fid]["non_zero"] += 1
            else:
                zero_factors.append(fid)
        
        activation_rate = non_zero / total if total > 0 else 0
        
        if activation_rate < 0.3:
            status = "critical"
            message = f"因子激活率过低: {activation_rate:.1%}"
        elif activation_rate < 0.5:
            status = "warning"
            message = f"因子激活率偏低: {activation_rate:.1%}"
        else:
            status = "healthy"
            message = f"因子激活率正常: {activation_rate:.1%}"
        
        statuses.append(HealthStatus(
            component="factors",
            status=status,
            message=message,
            details={
                "total": total,
                "non_zero": non_zero,
                "activation_rate": activation_rate,
                "zero_factors": zero_factors,
            }
        ))
        
        return statuses
    
    def check_data(self, data: Dict[str, Any]) -> List[HealthStatus]:
        """检查数据质量"""
        statuses = []
        
        # 检查赔率
        odds_home = data.get("odds_home", 0)
        odds_draw = data.get("odds_draw", 0)
        odds_away = data.get("odds_away", 0)
        
        if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
            statuses.append(HealthStatus(
                component="odds",
                status="critical",
                message="赔率数据无效",
                details={"odds_home": odds_home, "odds_draw": odds_draw, "odds_away": odds_away}
            ))
        else:
            # 检查赔率是否合理
            margin = 1.0/odds_home + 1.0/odds_draw + 1.0/odds_away
            if margin > 1.15:
                statuses.append(HealthStatus(
                    component="odds",
                    status="warning",
                    message=f"赔率边际过高: {margin:.2%}",
                    details={"margin": margin}
                ))
            else:
                statuses.append(HealthStatus(
                    component="odds",
                    status="healthy",
                    message="赔率数据正常",
                    details={"margin": margin}
                ))
        
        # 检查概率分布
        home_prob = data.get("home_prob", 0)
        draw_prob = data.get("draw_prob", 0)
        away_prob = data.get("away_prob", 0)
        total_prob = home_prob + draw_prob + away_prob
        
        if abs(total_prob - 1.0) > 0.05:
            statuses.append(HealthStatus(
                component="probabilities",
                status="warning",
                message=f"概率分布异常: 总和={total_prob:.3f}",
                details={"home": home_prob, "draw": draw_prob, "away": away_prob, "total": total_prob}
            ))
        else:
            statuses.append(HealthStatus(
                component="probabilities",
                status="healthy",
                message="概率分布正常",
                details={"total": total_prob}
            ))
        
        return statuses
    
    def record_prediction(self, predicted: str, actual: str, profit: float):
        """记录预测结果"""
        self._prediction_stats["total"] += 1
        if predicted == actual:
            self._prediction_stats["correct"] += 1
        self._prediction_stats["roi_sum"] += profit
    
    def get_prediction_stats(self) -> Dict:
        """获取预测统计"""
        total = self._prediction_stats["total"]
        correct = self._prediction_stats["correct"]
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "total_roi": self._prediction_stats["roi_sum"],
        }
    
    def get_factor_report(self) -> Dict[str, Dict]:
        """获取因子报告"""
        report = {}
        for fid, stats in self._factor_stats.items():
            total = stats["total"]
            non_zero = stats["non_zero"]
            report[fid] = {
                "total": total,
                "non_zero": non_zero,
                "activation_rate": non_zero / total if total > 0 else 0,
                "error_rate": stats["errors"] / total if total > 0 else 0,
            }
        return report
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": uptime,
            "uptime_human": _format_duration(uptime),
            "prediction_stats": self.get_prediction_stats(),
        }
    
    def get_full_report(self) -> Dict:
        """获取完整报告"""
        return {
            "system": self.get_system_status(),
            "factors": self.get_factor_report(),
            "recent_statuses": [
                {"component": s.component, "status": s.status, "message": s.message}
                for s in self._statuses[-10:]
            ]
        }
    
    def add_status(self, status: HealthStatus):
        """添加状态"""
        self._statuses.append(status)
        # 保留最近100条
        if len(self._statuses) > 100:
            self._statuses = self._statuses[-100:]


def _format_duration(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.0f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


# 全局监控器
_monitor = None


def get_health_monitor() -> HealthMonitor:
    """获取全局健康监控器"""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor
