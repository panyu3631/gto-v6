"""
GTO-GameFlow v5.5 — 数据库迁移 CLI

提供命令行接口用于数据库迁移管理。

使用方式:
    # 初始化数据库 (创建表 + 标记迁移版本)
    python -m src.data.migrate init

    # 生成自动迁移 (检测模型变更)
    python -m src.data.migrate generate -m "添加新字段"

    # 升级到最新版本
    python -m src.data.migrate upgrade

    # 回滚一个版本
    python -m src.data.migrate downgrade

    # 查看当前版本
    python -m src.data.migrate current

    # 查看迁移历史
    python -m src.data.migrate history

    # 生成 SQL 脚本 (不需数据库连接)
    python -m src.data.migrate sql

    # 健康检查
    python -m src.data.migrate health
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("migrate")


def cmd_init(args):
    """初始化数据库 — 创建表并标记迁移版本"""
    from src.data.database import init_database
    from src.data.pg_pool import get_current_revision, stamp_revision

    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if not db_url:
        print("错误: 请设置 GTO_DB_URL 环境变量或使用 --db-url 参数")
        sys.exit(1)

    logger.info(f"连接数据库: {db_url[:50]}...")

    try:
        db, repo = init_database(db_url)
        logger.info("数据库表创建成功")

        # 检查是否已有迁移版本
        rev = get_current_revision(db.engine)
        if rev:
            logger.info(f"当前迁移版本: {rev}")
        else:
            # 标记为最新版本
            stamp_revision(db.engine, "head")
            logger.info("已标记迁移版本为 head")

        logger.info("数据库初始化完成 ✓")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        sys.exit(1)


def cmd_upgrade(args):
    """升级到最新版本"""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.error("请安装 alembic: pip install alembic")
        sys.exit(1)

    alembic_cfg = Config("migrations/alembic.ini")
    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    if args.revision:
        command.upgrade(alembic_cfg, args.revision)
    else:
        command.upgrade(alembic_cfg, "head")
    logger.info("迁移升级完成 ✓")


def cmd_downgrade(args):
    """回滚迁移"""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.error("请安装 alembic: pip install alembic")
        sys.exit(1)

    alembic_cfg = Config("migrations/alembic.ini")
    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.downgrade(alembic_cfg, args.revision or "-1")
    logger.info("迁移回滚完成 ✓")


def cmd_generate(args):
    """生成自动迁移"""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.error("请安装 alembic: pip install alembic")
        sys.exit(1)

    alembic_cfg = Config("migrations/alembic.ini")
    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.revision(alembic_cfg, autogenerate=True, message=args.message)
    logger.info("迁移文件已生成 ✓")


def cmd_current(args):
    """查看当前版本"""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.error("请安装 alembic: pip install alembic")
        sys.exit(1)

    alembic_cfg = Config("migrations/alembic.ini")
    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.current(alembic_cfg)


def cmd_history(args):
    """查看迁移历史"""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.error("请安装 alembic: pip install alembic")
        sys.exit(1)

    alembic_cfg = Config("migrations/alembic.ini")
    command.history(alembic_cfg)


def cmd_sql(args):
    """生成 SQL 迁移脚本"""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        logger.error("请安装 alembic: pip install alembic")
        sys.exit(1)

    alembic_cfg = Config("migrations/alembic.ini")
    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_cfg, "head", sql=True)


def cmd_health(args):
    """数据库健康检查"""
    from src.data.database import DatabaseManager
    from src.data.pg_pool import PGPoolManager

    db_url = args.db_url or os.getenv("GTO_DB_URL")
    if not db_url:
        print("错误: 请设置 GTO_DB_URL 环境变量或使用 --db-url 参数")
        sys.exit(1)

    if "postgresql" in db_url:
        pool = PGPoolManager(db_url)
        db = DatabaseManager(db_url, pool_manager=pool)
    else:
        db = DatabaseManager(db_url)

    # 连接测试
    ok = db.health_check()
    if ok:
        print("✓ 数据库连接正常")
        if hasattr(db, '_pool_manager') and db._pool_manager:
            stats = db._pool_manager.get_pool_stats()
            print(f"  连接池大小: {stats.get('pool_size', 'N/A')}")
            print(f"  空闲连接: {stats.get('pool_checkedin', 'N/A')}")
            print(f"  溢出连接: {stats.get('pool_overflow', 'N/A')}")
    else:
        print("✗ 数据库连接失败")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="GTO-GameFlow v5.5 数据库迁移管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.data.migrate init                    # 初始化数据库
  python -m src.data.migrate generate -m "添加字段"   # 生成迁移
  python -m src.data.migrate upgrade                  # 升级
  python -m src.data.migrate health                   # 健康检查
        """,
    )
    parser.add_argument("--db-url", help="数据库连接 URL (覆盖环境变量 GTO_DB_URL)")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # init
    p_init = subparsers.add_parser("init", help="初始化数据库 (创建表 + 标记迁移版本)")
    p_init.set_defaults(func=cmd_init)

    # upgrade
    p_up = subparsers.add_parser("upgrade", help="升级到最新版本")
    p_up.add_argument("revision", nargs="?", help="目标版本 (默认 head)")
    p_up.set_defaults(func=cmd_upgrade)

    # downgrade
    p_down = subparsers.add_parser("downgrade", help="回滚迁移")
    p_down.add_argument("revision", nargs="?", help="目标版本 (默认 -1)")
    p_down.set_defaults(func=cmd_downgrade)

    # generate
    p_gen = subparsers.add_parser("generate", help="生成自动迁移")
    p_gen.add_argument("-m", "--message", required=True, help="迁移描述")
    p_gen.set_defaults(func=cmd_generate)

    # current
    p_cur = subparsers.add_parser("current", help="查看当前版本")
    p_cur.set_defaults(func=cmd_current)

    # history
    p_hist = subparsers.add_parser("history", help="查看迁移历史")
    p_hist.set_defaults(func=cmd_history)

    # sql
    p_sql = subparsers.add_parser("sql", help="生成 SQL 脚本")
    p_sql.set_defaults(func=cmd_sql)

    # health
    p_health = subparsers.add_parser("health", help="数据库健康检查")
    p_health.set_defaults(func=cmd_health)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()