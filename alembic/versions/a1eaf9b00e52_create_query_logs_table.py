"""create_query_logs_table

Revision ID: a1eaf9b00e52
Revises: 19ac790eabbb
Create Date: 2026-07-29 16:32:14.635014

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1eaf9b00e52'
down_revision: Union[str, Sequence[str], None] = '19ac790eabbb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    import pgvector.sqlalchemy
    from sqlalchemy.dialects import postgresql

    op.create_table(
        'query_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('resume_id', sa.Integer(), nullable=True),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('query_embedding', pgvector.sqlalchemy.Vector(dim=3072), nullable=False),
        sa.Column('generated_response', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('meta_info', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('query_logs')
