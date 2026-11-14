"""add chirality to peptide_entry

Revision ID: 197b89114257
Revises: efcf95150de9
Create Date: 2026-11-11 22:16:50.631728

"""
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '197b89114257'
down_revision: str | Sequence[str] | None = 'efcf95150de9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('peptide_entry', schema=None) as batch_op:
        batch_op.add_column(sa.Column('chirality', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('peptide_entry', schema=None) as batch_op:
        batch_op.drop_column('chirality')
