"""
GTO-GameFlow v5.5 — Alembic 环境配置

用于自动生成和运行数据库迁移。

使用方式:
    # 生成迁移 (自动检测模型变更)
    alembic -c migrations/alembic.ini revision --autogenerate -m "描述"

    # 升级到最新版本
    alembic -c migrations/alembic.ini upgrade head

    # 回滚一个版本
    alembic -c migrations/alembic.ini downgrade -1

    # 查看迁移历史
    alembic -c migrations/alembic.ini history
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 读取 Alembic 配置
config = context.config

# 设置日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 从环境变量覆盖数据库 URL
db_url = os.getenv("GTO_DB_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# 导入所有模型元数据
from src.data.database import Base
from src.data.database import (  # noqa: F401 — 确保模型被导入
    TeamElo,
    BetHistory,
    BankrollLog,
    FactorResult,
    MatchResult,
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    离线模式 — 生成 SQL 脚本而非直接执行。
    使用: alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # PostgreSQL 特定
        compare_type=True,       # 检测列类型变更
        compare_server_default=True,  # 检测默认值变更
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    在线模式 — 直接连接数据库执行迁移。
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()