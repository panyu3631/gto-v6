-- GTO-GameFlow v5.5 — PostgreSQL 初始化脚本
-- 在容器首次启动时自动执行 (docker-entrypoint-initdb.d)
-- 
-- 生产环境建议:
--   1. 使用 Alembic 管理迁移 (见 /migrations/)
--   2. 此文件仅用于开发环境快速启动

-- 创建系统用户 (用于应用连接)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'gto_app') THEN
        CREATE ROLE gto_app WITH LOGIN PASSWORD 'gto_app_2026';
    END IF;
END
$$;

-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE gto_gameflow TO gto_app;
GRANT ALL PRIVILEGES ON DATABASE gto_gameflow TO gto;

-- 创建 schema (可选，用于多租户分离)
CREATE SCHEMA IF NOT EXISTS gto_v5;
ALTER SCHEMA gto_v5 OWNER TO gto;

-- 基础扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 设置默认搜索路径
ALTER DATABASE gto_gameflow SET search_path TO gto_v5, public;

-- 创建索引优化建议 (将在应用启动时由 SQLAlchemy 自动创建表)
-- 以下是手动优化建议，供参考:

-- 1. 为 bet_history 表添加部分索引 (仅 pending 状态)
-- CREATE INDEX idx_bet_pending ON bet_history (placed_at) WHERE result = 'pending';

-- 2. 为 match_results 添加 BRIN 索引 (时间序列)
-- CREATE INDEX idx_match_kickoff_brin ON match_results USING BRIN (kickoff_time);

-- 3. 为 factor_results 添加覆盖索引
-- CREATE INDEX idx_factor_lookup ON factor_results (match_id, factor_id) INCLUDE (delta_home, delta_draw, delta_away);

-- 4. 物化视图 — 每日投注统计 (可选)
-- CREATE MATERIALIZED VIEW daily_bet_stats AS
-- SELECT
--     date_trunc('day', placed_at) AS bet_date,
--     league_id,
--     COUNT(*) AS bet_count,
--     SUM(stake) AS total_staked,
--     SUM(CASE WHEN result = 'win' THEN profit_loss ELSE 0 END) AS total_profit,
--     COUNT(CASE WHEN result = 'win' THEN 1 END) * 100.0 / NULLIF(COUNT(CASE WHEN result IN ('win', 'loss') THEN 1 END), 0) AS win_rate
-- FROM bet_history
-- GROUP BY 1, 2;