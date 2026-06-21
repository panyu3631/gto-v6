"""
GTO-GameFlow v5.5 — PostgreSQL 连接池管理器

功能:
- 连接池配置优化 (size, overflow, timeout, recycle)
- 健康检查与自动重连
- 读写分离支持 (预留)
- 慢查询日志 (通过 SQLAlchemy 事件)
- 连接指标统计

使用方式:
    from src.data.pg_pool import PGPoolManager

    pool = PGPoolManager(db_url="postgresql://gto:pass@localhost:5432/gto_gameflow")
    pool.health_check()  # 启动时检查
    with pool.get_session() as session:
        ...

    # 或替换现有 DatabaseManager:
    db = DatabaseManager(pool_manager=pool)
"""

import logging
import time
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 慢查询日志 (SQLAlchemy 事件)
# ═══════════════════════════════════════════════════════════════

SLOW_QUERY_THRESHOLD_MS = 200  # 超过 200ms 记录警告


@event.listens_for(Engine, "before_cursor_execute")
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """记录查询开始时间"""
    conn.info.setdefault("query_start_time", []).append(time.perf_counter())


@event.listens_for(Engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """检测慢查询"""
    start_times = conn.info.get("query_start_time", [])
    if start_times:
        elapsed = (time.perf_counter() - start_times.pop()) * 1000
        if elapsed > SLOW_QUERY_THRESHOLD_MS:
            # 截断长语句
            short_stmt = statement[:200] + "..." if len(statement) > 200 else statement
            logger.warning(
                "slow_query",
                extra={
                    "duration_ms": round(elapsed, 2),
                    "statement": short_stmt,
                    "parameters": str(parameters)[:100] if parameters else None,
                },
            )


# ═══════════════════════════════════════════════════════════════
# 连接池管理器
# ═══════════════════════════════════════════════════════════════

class PGPoolManager:
    """
    PostgreSQL 连接池管理器。

    参数:
        db_url: PostgreSQL 连接字符串
        pool_size: 连接池大小 (默认 5)
        max_overflow: 最大溢出连接数 (默认 10)
        pool_timeout: 获取连接超时秒数 (默认 30)
        pool_recycle: 连接回收秒数 (默认 3600，即 1 小时)
        echo: 是否打印 SQL 语句 (默认 False)
        health_check_interval: 健康检查间隔秒数 (默认 60)
    """

    def __init__(
        self,
        db_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
        health_check_interval: int = 60,
    ):
        self.db_url = db_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._echo = echo
        self._health_check_interval = health_check_interval

        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._healthy: bool = False
        self._last_health_check: float = 0.0
        self._stats: Dict[str, Any] = {
            "connections_created": 0,
            "slow_queries": 0,
            "health_checks": 0,
            "health_failures": 0,
            "last_error": None,
        }
        self._lock = threading.Lock()

    @property
    def engine(self) -> Engine:
        """延迟创建引擎"""
        if self._engine is None:
            connect_args = {
                "application_name": "gto-gameflow-v5.5",
                "options": "-c statement_timeout=30000",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
            # SQLite 不支持 PostgreSQL 特有参数
            if "sqlite" in self.db_url:
                connect_args = {"check_same_thread": False}

            self._engine = create_engine(
                self.db_url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_recycle=self._pool_recycle,
                pool_pre_ping=True,  # 连接前检查有效性
                echo=self._echo,
                connect_args=connect_args,
            )
            self._session_factory = sessionmaker(bind=self._engine)
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        if self._session_factory is None:
            _ = self.engine  # 触发创建
        return self._session_factory

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        上下文管理器 — 自动 commit/rollback。

        使用方式:
            with pool.session_scope() as session:
                session.add(obj)
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """
        数据库健康检查。

        返回:
            True 表示连接正常，False 表示连接失败。
        """
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return self._healthy

        with self._lock:
            self._stats["health_checks"] += 1
            self._last_health_check = now

            try:
                with self.session_scope() as session:
                    result = session.execute(text("SELECT 1 AS alive")).scalar()
                    if result == 1:
                        # 检查连接池状态
                        pool = self.engine.pool
                        self._stats["pool_size"] = pool.size()
                        self._stats["pool_checkedin"] = pool.checkedin()
                        self._stats["pool_overflow"] = pool.overflow()
                        self._healthy = True
                        logger.debug(
                            "pg_health_check_passed",
                            extra={
                                "pool_size": pool.size(),
                                "checkedin": pool.checkedin(),
                                "overflow": pool.overflow(),
                            },
                        )
                        return True
            except Exception as e:
                self._stats["health_failures"] += 1
                self._stats["last_error"] = str(e)
                self._healthy = False
                logger.error(
                    "pg_health_check_failed",
                    extra={"error": str(e)},
                )
                return False

        return self._healthy

    def wait_for_db(self, max_retries: int = 30, retry_interval: float = 2.0) -> bool:
        """
        等待数据库就绪 (启动时使用)。

        参数:
            max_retries: 最大重试次数
            retry_interval: 重试间隔秒数

        返回:
            True 表示数据库已就绪，False 表示超时。
        """
        logger.info("pg_waiting_for_db", extra={"max_retries": max_retries})

        for attempt in range(1, max_retries + 1):
            try:
                with self.session_scope() as session:
                    result = session.execute(text("SELECT 1")).scalar()
                    if result == 1:
                        self._healthy = True
                        logger.info(
                            "pg_db_ready",
                            extra={"attempt": attempt, "elapsed_s": attempt * retry_interval},
                        )
                        return True
            except Exception as e:
                logger.debug(
                    "pg_retry",
                    extra={"attempt": attempt, "error": str(e)},
                )
                if attempt < max_retries:
                    time.sleep(retry_interval)

        logger.error("pg_db_timeout", extra={"max_retries": max_retries})
        return False

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        stats = dict(self._stats)
        if self._engine:
            pool = self._engine.pool
            stats.update({
                "pool_size": pool.size(),
                "pool_checkedin": pool.checkedin(),
                "pool_overflow": pool.overflow(),
                "pool_total": pool.size() + pool.overflow(),
            })
        return stats

    def reset_pool(self):
        """重置连接池 (处理连接断开后恢复)"""
        if self._engine:
            self._engine.dispose()
            self._session_factory = None
            self._healthy = False
            logger.info("pg_pool_reset")

    def close(self):
        """关闭连接池"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._healthy = False
            logger.info("pg_pool_closed")


# ═══════════════════════════════════════════════════════════════
# 数据库迁移管理 (Alembic 兼容)
# ═══════════════════════════════════════════════════════════════

def get_current_revision(engine: Engine) -> Optional[str]:
    """
    获取当前数据库迁移版本。

    如果 alembic_version 表不存在，返回 None。
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            ).scalar()
            return result
    except Exception:
        return None


def stamp_revision(engine: Engine, revision: str = "head"):
    """
    标记当前数据库版本 (用于首次迁移时跳过历史)。
    """
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("migrations/alembic.ini")
    alembic_cfg.attributes["connection"] = engine
    command.stamp(alembic_cfg, revision)
    logger.info("pg_stamped_revision", extra={"revision": revision})