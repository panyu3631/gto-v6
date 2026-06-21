"""
GTO-GameFlow v5.5 — PostgreSQL 迁移测试

测试内容:
1. PGPoolManager 创建与连接
2. DatabaseManager 集成 PGPoolManager
3. 健康检查
4. 迁移脚本 (init/upgrade/downgrade)
5. SQLite 回退兼容
"""

import sys
import os
import time
import tempfile
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARN)


def test_pg_pool_manager_creation():
    """测试 PGPoolManager 创建"""
    from src.data.pg_pool import PGPoolManager

    pool = PGPoolManager(
        db_url="postgresql://gto:test@localhost:5432/gto_gameflow",
        pool_size=3,
        max_overflow=5,
        pool_timeout=10,
        pool_recycle=1800,
    )
    assert pool._pool_size == 3
    assert pool._max_overflow == 5
    assert pool._pool_timeout == 10
    assert pool._pool_recycle == 1800
    assert pool._healthy is False
    assert pool._engine is None
    print("PASS: test_pg_pool_manager_creation")


def test_pg_pool_manager_stats():
    """测试连接池统计"""
    from src.data.pg_pool import PGPoolManager

    pool = PGPoolManager("postgresql://gto:test@localhost:5432/gto_gameflow")
    stats = pool.get_pool_stats()
    assert "connections_created" in stats
    assert "health_checks" in stats
    assert "health_failures" in stats
    assert stats["health_checks"] == 0
    print("PASS: test_pg_pool_manager_stats")


def test_pg_pool_manager_reset():
    """测试连接池重置 (无实际连接)"""
    from src.data.pg_pool import PGPoolManager

    pool = PGPoolManager("postgresql://gto:test@localhost:5432/gto_gameflow")
    pool.reset_pool()
    assert pool._healthy is False
    assert pool._engine is None
    print("PASS: test_pg_pool_manager_reset")


def test_database_manager_sqlite():
    """测试 DatabaseManager 与 SQLite 集成"""
    from src.data.database import DatabaseManager, Base

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(f"sqlite:///{db_path}")
        db.init_db()

        # 验证表已创建
        with db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            tables = [r[0] for r in result]
            assert "team_elo" in tables
            assert "bet_history" in tables
            assert "bankroll_log" in tables
            assert "factor_results" in tables
            assert "match_results" in tables

        # 健康检查
        assert db.health_check() is True

        db.close()
        print("PASS: test_database_manager_sqlite")
    finally:
        os.unlink(db_path)


def test_database_manager_health_check():
    """测试健康检查"""
    from src.data.database import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(f"sqlite:///{db_path}")
        db.init_db()
        assert db.health_check() is True
        db.close()

        # 模拟关闭后健康检查失败
        db2 = DatabaseManager(f"sqlite:///{db_path}")
        assert db2.health_check() is True  # SQLite 自动创建连接
        db2.close()
        print("PASS: test_database_manager_health_check")
    finally:
        os.unlink(db_path)


def test_database_manager_with_pool_manager():
    """测试 DatabaseManager 与 PGPoolManager 集成"""
    from src.data.database import DatabaseManager
    from src.data.pg_pool import PGPoolManager

    pool = PGPoolManager("postgresql://gto:test@localhost:5432/gto_gameflow")
    db = DatabaseManager(pool_manager=pool)

    # 未连接时不应创建引擎
    assert db._engine is None
    assert pool._engine is None

    # 健康检查无连接时应返回 False
    ok = db.health_check()
    assert ok is False  # 无实际连接，应该失败

    # 关闭应安全
    db.close()
    print("PASS: test_database_manager_with_pool_manager")


def test_repository_sqlite():
    """测试 Repository 与 SQLite 完整 CRUD"""
    from src.data.database import DatabaseManager, Repository

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(f"sqlite:///{db_path}")
        db.init_db()
        repo = Repository(db)

        # Elo 操作
        assert repo.get_elo("TestTeam", "test_league") == 1500.0
        repo.upsert_elo("TestTeam", "test_league", 1600.0)
        assert repo.get_elo("TestTeam", "test_league") == 1600.0

        # 投注操作
        repo.save_bet({
            "bet_id": "B001",
            "match_id": "M001",
            "league_id": "test_league",
            "selection": "home_win",
            "odds": 2.10,
            "stake": 100.0,
        })
        repo.settle_bet("B001", "win", 110.0)

        # 资金快照
        repo.save_bankroll_snapshot({
            "balance": 10100.0,
            "total_staked": 100.0,
            "total_returned": 210.0,
            "total_bets": 1,
            "total_wins": 1,
        })

        # 因子结果
        repo.save_factor_results("M001", "test_league", {
            "E01": {"home": 0.05, "draw": -0.02, "away": -0.03},
            "E02": {"home": 0.10, "draw": 0.00, "away": -0.10},
        })

        # 比赛结果
        from datetime import datetime
        repo.save_match_result({
            "match_id": "M001",
            "league_id": "test_league",
            "season": "2024/25",
            "matchday": 1,
            "kickoff_time": datetime.now(),
            "home_team": "TeamA",
            "away_team": "TeamB",
            "home_goals": 2,
            "away_goals": 1,
            "is_complete": True,
        })

        # 验证查询
        bets = repo.get_bet_history(limit=10)
        assert len(bets) == 1
        assert bets[0].result == "win"

        bankroll = repo.get_latest_bankroll()
        assert bankroll is not None
        assert bankroll.balance == 10100.0

        factors = repo.get_factor_results("M001")
        assert len(factors) == 2

        pending = repo.get_pending_matches()
        assert len(pending) == 0

        db.close()
        print("PASS: test_repository_sqlite")
    finally:
        os.unlink(db_path)


def test_migration_script_structure():
    """测试迁移脚本结构"""
    # 验证迁移文件存在
    migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
    assert os.path.isdir(migrations_dir), f"migrations 目录不存在: {migrations_dir}"
    assert os.path.isfile(os.path.join(migrations_dir, "alembic.ini")), "alembic.ini 不存在"
    assert os.path.isfile(os.path.join(migrations_dir, "env.py")), "env.py 不存在"
    assert os.path.isfile(os.path.join(migrations_dir, "script.py.mako")), "script.py.mako 不存在"

    versions_dir = os.path.join(migrations_dir, "versions")
    assert os.path.isdir(versions_dir), f"versions 目录不存在: {versions_dir}"

    init_schema = os.path.join(versions_dir, "0001_initial_schema.py")
    assert os.path.isfile(init_schema), "初始迁移脚本不存在"

    # 验证迁移脚本包含 upgrade/downgrade
    with open(init_schema) as f:
        content = f.read()
    assert "def upgrade()" in content
    assert "def downgrade()" in content
    assert 'revision: str = "0001_initial"' in content

    print("PASS: test_migration_script_structure")


def test_dotenv_template():
    """测试 .env 模板"""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
    assert os.path.isfile(env_path), ".env.example 不存在"

    with open(env_path) as f:
        content = f.read()

    assert "GTO_DB_URL" in content
    assert "GTO_ALERT_EMAIL_SENDER" in content
    assert "GTO_LOG_LEVEL" in content
    assert "GTO_VERSION" in content
    assert "739252249@qq.com" in content
    print("PASS: test_dotenv_template")


def test_docker_compose():
    """测试 Docker Compose 文件"""
    compose_path = os.path.join(
        os.path.dirname(__file__), "..", "deploy", "docker", "docker-compose.yml"
    )
    assert os.path.isfile(compose_path), "docker-compose.yml 不存在"

    with open(compose_path) as f:
        content = f.read()

    assert "postgres:16" in content
    assert "gto_gameflow" in content
    assert "healthcheck" in content
    assert "adminer" in content
    print("PASS: test_docker_compose")


def test_init_sql():
    """测试初始化 SQL 脚本"""
    init_sql_path = os.path.join(
        os.path.dirname(__file__), "..", "deploy", "docker", "init-scripts", "01-init.sql"
    )
    assert os.path.isfile(init_sql_path), "01-init.sql 不存在"

    with open(init_sql_path) as f:
        content = f.read()

    assert "gto_app" in content
    assert "gto_v5" in content
    assert "uuid-ossp" in content
    print("PASS: test_init_sql")


def test_migrate_cli_import():
    """测试迁移 CLI 可导入"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.data.migrate import main as _migrate_main
    from src.data.migrate import cmd_init, cmd_health
    assert callable(cmd_init)
    assert callable(cmd_health)
    print("PASS: test_migrate_cli_import")


def test_pg_pool_session_scope():
    """测试 PGPoolManager session_scope 上下文管理器"""
    from src.data.pg_pool import PGPoolManager
    from src.data.database import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # 使用 SQLite URL 测试 session_scope
        pool = PGPoolManager(f"sqlite:///{db_path}")
        # 直接使用 pool 创建表
        from src.data.database import Base
        Base.metadata.create_all(pool.engine)

        with pool.session_scope() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1

        pool.close()
        print("PASS: test_pg_pool_session_scope")
    finally:
        os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GTO-GameFlow v5.5 — PostgreSQL 迁移测试")
    print("=" * 60)
    print()

    tests = [
        ("PGPoolManager 创建", test_pg_pool_manager_creation),
        ("PGPoolManager 统计", test_pg_pool_manager_stats),
        ("PGPoolManager 重置", test_pg_pool_manager_reset),
        ("DatabaseManager SQLite", test_database_manager_sqlite),
        ("DatabaseManager 健康检查", test_database_manager_health_check),
        ("DatabaseManager + PGPoolManager", test_database_manager_with_pool_manager),
        ("Repository CRUD", test_repository_sqlite),
        ("PGPoolManager session_scope", test_pg_pool_session_scope),
        ("迁移脚本结构", test_migration_script_structure),
        (".env 模板", test_dotenv_template),
        ("Docker Compose", test_docker_compose),
        ("初始化 SQL", test_init_sql),
        ("迁移 CLI", test_migrate_cli_import),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"结果: {passed} 通过, {failed} 失败, {len(tests)} 总计")
    print()

    if failed == 0:
        print("提示: 启动 PostgreSQL 容器:")
        print("  docker compose -f deploy/docker/docker-compose.yml up -d")
        print()
        print("初始化数据库:")
        print("  python -m src.data.migrate init")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())