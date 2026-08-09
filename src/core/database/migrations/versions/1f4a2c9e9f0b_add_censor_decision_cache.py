"""add censor decision cache

Revision ID: 1f4a2c9e9f0b
Revises: eb70c684445e
Create Date: 2026-08-09 10:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1f4a2c9e9f0b"
down_revision: Union[str, None] = "eb70c684445e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "censor_decision_cache",
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("config_version_hash", sa.String(), nullable=False),
        sa.Column("model_version_hash", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("risk_tier", sa.String(), nullable=False),
        sa.Column("reason_codes_json", sa.Text(), nullable=False),
        sa.Column("confidence_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("content_hash", "config_version_hash"),
    )
    op.create_index(
        "ix_censor_decision_cache_hash_version",
        "censor_decision_cache",
        ["content_hash", "config_version_hash"],
        unique=False,
    )
    op.create_index(
        "ix_censor_decision_cache_last_accessed_at",
        "censor_decision_cache",
        ["last_accessed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_censor_decision_cache_last_accessed_at", table_name="censor_decision_cache")
    op.drop_index("ix_censor_decision_cache_hash_version", table_name="censor_decision_cache")
    op.drop_table("censor_decision_cache")

