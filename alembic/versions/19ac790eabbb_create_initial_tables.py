"""create_initial_tables

Revision ID: 19ac790eabbb
Revises: 
Create Date: 2026-07-29 15:58:28.535006

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '19ac790eabbb'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    import pgvector.sqlalchemy
    from sqlalchemy.dialects import postgresql

    # Execute extension creation as a fallback
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        'resumes',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('meta_info', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_hash')
    )

    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('resume_id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=3072), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('start_index', sa.Integer(), nullable=True),
        sa.Column('meta_info', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('document_chunks')
    op.drop_table('resumes')
