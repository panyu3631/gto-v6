"""GTO-GameFlow v5.5 — Alembic 迁移脚本模板

每个迁移文件通过 `alembic revision --autogenerate -m "描述"` 自动生成。
此文件作为模板参考，展示了必须的结构。
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# 迁移标识符
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升级到当前版本 — 创建所有表"""
    # === team_elo ===
    op.create_table(
        "team_elo",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_name", sa.String(128), nullable=False),
        sa.Column("league_id", sa.String(64), nullable=False),
        sa.Column("elo_rating", sa.Float(), nullable=False, server_default="1500.0"),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_team_league", "team_elo", ["team_name", "league_id"], unique=True)
    op.create_index("ix_team_elo_team_name", "team_elo", ["team_name"])
    op.create_index("ix_team_elo_league_id", "team_elo", ["league_id"])

    # === bet_history ===
    op.create_table(
        "bet_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("bet_id", sa.String(128), nullable=False),
        sa.Column("match_id", sa.String(128), nullable=False),
        sa.Column("league_id", sa.String(64), nullable=False),
        sa.Column("selection", sa.String(16), nullable=False),
        sa.Column("odds", sa.Float(), nullable=False),
        sa.Column("stake", sa.Float(), nullable=False),
        sa.Column("placed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("result", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("profit_loss", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("settled_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bet_history_bet_id", "bet_history", ["bet_id"], unique=True)
    op.create_index("idx_bet_match", "bet_history", ["match_id"])
    op.create_index("idx_bet_placed", "bet_history", ["placed_at"])
    op.create_index("ix_bet_history_league_id", "bet_history", ["league_id"])

    # === bankroll_log ===
    op.create_table(
        "bankroll_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("balance", sa.Float(), nullable=False),
        sa.Column("total_staked", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_returned", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_bets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_wins", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("consecutive_losses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_drawdown", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("snapshot_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_bankroll_snapshot", "bankroll_log", ["snapshot_at"])

    # === factor_results ===
    op.create_table(
        "factor_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.String(128), nullable=False),
        sa.Column("league_id", sa.String(64), nullable=False),
        sa.Column("factor_id", sa.String(8), nullable=False),
        sa.Column("delta_home", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("delta_draw", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("delta_away", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("weight", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("computed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_factor_match", "factor_results", ["match_id", "factor_id"])
    op.create_index("ix_factor_results_match_id", "factor_results", ["match_id"])
    op.create_index("ix_factor_results_league_id", "factor_results", ["league_id"])
    op.create_index("ix_factor_results_factor_id", "factor_results", ["factor_id"])

    # === match_results ===
    op.create_table(
        "match_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.String(128), nullable=False),
        sa.Column("league_id", sa.String(64), nullable=False),
        sa.Column("season", sa.String(16), nullable=False),
        sa.Column("matchday", sa.Integer(), nullable=False),
        sa.Column("kickoff_time", sa.DateTime(), nullable=False),
        sa.Column("home_team", sa.String(128), nullable=False),
        sa.Column("away_team", sa.String(128), nullable=False),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column("home_elo_before", sa.Float(), nullable=True),
        sa.Column("away_elo_before", sa.Float(), nullable=True),
        sa.Column("odds_home", sa.Float(), nullable=True),
        sa.Column("odds_draw", sa.Float(), nullable=True),
        sa.Column("odds_away", sa.Float(), nullable=True),
        sa.Column("model_prob_home", sa.Float(), nullable=True),
        sa.Column("model_prob_draw", sa.Float(), nullable=True),
        sa.Column("model_prob_away", sa.Float(), nullable=True),
        sa.Column("is_complete", sa.Boolean(), nullable=False, server_default="false"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_match_results_match_id", "match_results", ["match_id"], unique=True)
    op.create_index("idx_match_league_season", "match_results", ["league_id", "season"])
    op.create_index("ix_match_results_league_id", "match_results", ["league_id"])


def downgrade() -> None:
    """回滚当前版本 — 删除所有表"""
    op.drop_table("match_results")
    op.drop_table("factor_results")
    op.drop_table("bankroll_log")
    op.drop_table("bet_history")
    op.drop_table("team_elo")