"""${revision_id} — ${message}

Revision ID: ${revision_id}
Revises: ${down_revision | n, none}
Create Date: ${create_date}

自动生成迁移模板。执行 `alembic revision --autogenerate -m "描述"` 来生成实际迁移文件。
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# 迁移标识符 (由 alembic 自动填充)
revision: str = "${revision_id}"
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """升级迁移"""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """回滚迁移"""
    ${downgrades if downgrades else "pass"}