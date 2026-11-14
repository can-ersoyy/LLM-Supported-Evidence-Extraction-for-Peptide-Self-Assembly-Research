"""add_run_identifier_to_paper_metadata_and_paper_content

Revision ID: 888e2404c1f3
Revises: efcf95150de9
Create Date: 2026-11-11 22:21:26.982027

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '888e2404c1f3'
down_revision: Union[str, Sequence[str], None] = 'efcf95150de9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('paper_metadata', schema=None) as batch_op:
        batch_op.add_column(sa.Column('run_identifier', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_paper_metadata_run_identifier'), ['run_identifier'], unique=False)

    with op.batch_alter_table('paper_content', schema=None) as batch_op:
        batch_op.add_column(sa.Column('run_identifier', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_paper_content_run_identifier'), ['run_identifier'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('paper_content', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_paper_content_run_identifier'))
        batch_op.drop_column('run_identifier')

    with op.batch_alter_table('paper_metadata', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_paper_metadata_run_identifier'))
        batch_op.drop_column('run_identifier')

