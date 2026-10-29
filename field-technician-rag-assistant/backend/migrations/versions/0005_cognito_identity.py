"""Make app_user a projection of a Cognito user rather than a credential store.

Three changes, each forced by moving identity out of the application:

* `cognito_sub` -- the pool's stable identifier for the user. Nullable because
  the rows that already exist predate it and are matched by email until their
  owner next signs in, at which point the sub is recorded.
* `password_hash` nullable -- there is no password here any more. The column is
  kept rather than dropped so that an existing row is not destroyed by a
  migration, and so the change is reversible.
* the role check constraint gains `customer` -- the pool has had an equally
  named group since it was created, and a token carrying it would otherwise fail
  the insert with a constraint violation rather than a useful error.

WHY EVERY STEP CHECKS FIRST

The early revisions build their tables with `Base.metadata.create_all`, which
reads the *live* domain models rather than a snapshot of them. So adding
`cognito_sub` to the model changed what revision 0003 produces: on a database
created from scratch today, `app_user` already has the column, the constraint
already names three roles, and `password_hash` is already nullable by the time
this revision runs. On the database that has been running since September none
of that is true.

Both must work. A blind `add_column` succeeds in production and fails on every
fresh database with `DuplicateColumn` -- which is a migration that passes review,
passes deployment, and then breaks the first time anyone restores a backup or
stands up a second environment. Hence the inspection.
"""
from alembic import op
import sqlalchemy as sa

revision = "0005_cognito_identity"
down_revision = "0004_observability"
branch_labels = None
depends_on = None

_ROLES_NEW = "role IN ('admin','technician','customer')"
_ROLES_OLD = "role IN ('admin','technician')"


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _columns() -> dict[str, dict]:
    return {c["name"]: c for c in _inspector().get_columns("app_user")}


def _has_constraint(name: str) -> bool:
    insp = _inspector()
    checks = {c["name"] for c in insp.get_check_constraints("app_user")}
    uniques = {c["name"] for c in insp.get_unique_constraints("app_user")}
    return name in checks or name in uniques


def upgrade() -> None:
    cols = _columns()

    if "cognito_sub" not in cols:
        op.add_column("app_user", sa.Column("cognito_sub", sa.Text(), nullable=True))
    if not _has_constraint("uq_app_user_cognito_sub"):
        op.create_unique_constraint("uq_app_user_cognito_sub", "app_user", ["cognito_sub"])

    if not cols.get("password_hash", {}).get("nullable", True):
        op.alter_column("app_user", "password_hash", existing_type=sa.Text(), nullable=True)

    # Recreated rather than altered: Postgres has no ALTER CONSTRAINT for a
    # check. Dropping before adding is safe inside the migration's transaction --
    # if the new one fails to apply, neither is left behind.
    if _has_constraint("ck_app_user_role"):
        op.drop_constraint("ck_app_user_role", "app_user", type_="check")
    op.create_check_constraint("ck_app_user_role", "app_user", _ROLES_NEW)


def downgrade() -> None:
    # Any customer row would violate the narrower constraint, so move them to the
    # nearest role that exists in the old world rather than failing the migration.
    op.execute("UPDATE app_user SET role = 'technician' WHERE role = 'customer'")
    if _has_constraint("ck_app_user_role"):
        op.drop_constraint("ck_app_user_role", "app_user", type_="check")
    op.create_check_constraint("ck_app_user_role", "app_user", _ROLES_OLD)

    # Rows created under Cognito have no password. Going back means they cannot
    # sign in either way, so give them an unusable value rather than a NULL that
    # the NOT NULL below would reject.
    op.execute("UPDATE app_user SET password_hash = '!' WHERE password_hash IS NULL")
    op.alter_column("app_user", "password_hash", existing_type=sa.Text(), nullable=False)

    if _has_constraint("uq_app_user_cognito_sub"):
        op.drop_constraint("uq_app_user_cognito_sub", "app_user", type_="unique")
    if "cognito_sub" in _columns():
        op.drop_column("app_user", "cognito_sub")
