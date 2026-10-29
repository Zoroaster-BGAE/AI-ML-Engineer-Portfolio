"""User administration, applied to the Cognito pool rather than a local table.

The endpoints keep the shapes the admin console already calls, so the change is
invisible to it, but every write now lands in Cognito. That is the point: if
creating a user inserted a row here, the account would not exist anywhere you
could sign in from, and disabling one would leave a working login behind.

`access_expires` is the exception and stays local. It is a subscription end date,
which Cognito has no concept of -- it can disable an account but not schedule it.
So the listing is Cognito's users, joined to whatever local row exists for the
fields only this application knows about.
"""
from datetime import datetime

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.cognito import Admin
from app.auth.dependencies import require_admin
from app.db import get_session
from app.domain.support import AppUser

router = APIRouter(prefix="/api/admin/users", tags=["admin"], dependencies=[Depends(require_admin)])

_ROLE = Field(pattern="^(admin|technician|customer)$")


class UserCreate(BaseModel):
    email: str
    # Cognito enforces the pool's own password policy on top of this, and returns
    # InvalidPasswordException if it is not met -- surfaced as a 400 below rather
    # than duplicating the policy here, where the two could disagree.
    password: str = Field(min_length=6)
    role: str = _ROLE
    access_expires: datetime | None = None


class UserPatch(BaseModel):
    is_active: bool | None = None
    role: str | None = Field(default=None, pattern="^(admin|technician|customer)$")
    access_expires: datetime | None = None
    clear_expiry: bool = False
    # Local only: golden-example reviewer (the lead technician). Not a Cognito group, see AppUser.is_reviewer.
    is_reviewer: bool | None = None


class PasswordReset(BaseModel):
    password: str = Field(min_length=6)


class AdminUserOut(BaseModel):
    """A Cognito user, plus the local fields that only exist here."""

    username: str
    email: str
    role: str
    is_active: bool
    status: str
    access_expires: datetime | None = None
    has_local_row: bool
    is_reviewer: bool = False


def _admin() -> Admin:
    try:
        return Admin()
    except Exception as exc:
        raise HTTPException(503, f"Cognito is not available: {exc}") from exc


def _local_by_email(db: Session) -> dict[str, AppUser]:
    return {u.email: u for u in db.scalars(select(AppUser))}


def _as_out(c: dict, local: AppUser | None) -> AdminUserOut:
    return AdminUserOut(
        username=c["username"],
        email=c["email"],
        role=(c["groups"] or ["technician"])[0],
        is_active=c["enabled"],
        status=c["status"],
        access_expires=local.access_expires if local else None,
        has_local_row=local is not None,
        is_reviewer=bool(local and local.is_reviewer),
    )


def _cognito_error(exc: ClientError) -> HTTPException:
    """Turn a boto exception into something the console can display.

    Without this the console shows a botocore repr, and the two cases an
    administrator actually hits -- a duplicate email and a password the pool
    rejects -- are indistinguishable from a bug.
    """
    code = exc.response.get("Error", {}).get("Code", "")
    msg = exc.response.get("Error", {}).get("Message", str(exc))
    if code == "UsernameExistsException":
        return HTTPException(409, "A user with this email already exists")
    if code in ("InvalidPasswordException", "InvalidParameterException"):
        return HTTPException(400, msg)
    if code == "UserNotFoundException":
        return HTTPException(404, "User not found")
    return HTTPException(502, f"Cognito rejected the request: {msg}")


def _resolve(a: Admin, ident: str) -> dict:
    """Find a user by email or by Cognito username, and return their pool record.

    Both are accepted because they are different strings and only one is usable.
    This pool signs in by email, so Cognito assigns an opaque uuid as `Username`
    and every administrative API takes that uuid -- while an administrator, a URL
    written by hand, and the console's own listing all think in email addresses.
    Accepting only the uuid produced a 404 for every sensible request.
    """
    ident = ident.strip()
    users = a.list_users()
    for u in users:
        if u["username"] == ident:
            return u
    lowered = ident.lower()
    for u in users:
        if u["email"].lower() == lowered:
            return u
    raise HTTPException(404, "User not found")


@router.get("", response_model=list[AdminUserOut])
def list_users(db: Session = Depends(get_session)) -> list[AdminUserOut]:
    local = _local_by_email(db)
    try:
        users = _admin().list_users()
    except ClientError as exc:
        raise _cognito_error(exc) from exc
    return [_as_out(c, local.get(c["email"])) for c in sorted(users, key=lambda u: u["email"])]


@router.post("", response_model=AdminUserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_session)) -> AdminUserOut:
    """Create the account in Cognito. No local row is written.

    The row appears by itself when they first sign in -- see
    app/auth/dependencies.py. Writing one here would be the start of two
    directories to keep in step, which is exactly what this design removes.
    """
    email = body.email.strip().lower()
    a = _admin()
    try:
        username = a.create_user(email, body.password, body.role)
    except ClientError as exc:
        raise _cognito_error(exc) from exc

    # Only a subscription end date needs a row before first login, because it
    # has to be enforceable from the moment the account exists.
    if body.access_expires is not None:
        db.add(AppUser(email=email, role=body.role, access_expires=body.access_expires))
        db.commit()

    return AdminUserOut(
        username=username, email=email, role=body.role, is_active=True,
        status="CONFIRMED", access_expires=body.access_expires,
        has_local_row=body.access_expires is not None,
    )


@router.patch("/{ident}", response_model=AdminUserOut)
def patch_user(ident: str, body: UserPatch, admin: AppUser = Depends(require_admin),
               db: Session = Depends(get_session)) -> AdminUserOut:
    a = _admin()
    try:
        target = _resolve(a, ident)
    except ClientError as exc:
        raise _cognito_error(exc) from exc
    username = target["username"]
    groups = target["groups"]

    # Locking yourself out is a support call, and the check has to happen here
    # because Cognito will happily let an admin disable their own account.
    if target["email"] == admin.email and (body.is_active is False or (body.role and body.role != "admin")):
        raise HTTPException(400, "Admins cannot disable or demote their own account")

    try:
        if body.role:
            a.set_role(username, body.role)
            groups = [body.role]
        if body.is_active is not None:
            a.set_enabled(username, body.is_active)
    except ClientError as exc:
        raise _cognito_error(exc) from exc

    local = db.scalar(select(AppUser).where(AppUser.email == target["email"]))
    if body.clear_expiry or body.access_expires is not None or body.is_reviewer is not None:
        if local is None:
            local = AppUser(email=target["email"], role=body.role or (groups or ["technician"])[0])
            db.add(local)
        if body.clear_expiry or body.access_expires is not None:
            local.access_expires = None if body.clear_expiry else body.access_expires
        if body.is_reviewer is not None:
            local.is_reviewer = body.is_reviewer
        db.commit()

    return AdminUserOut(
        username=username, email=target["email"], role=(groups or ["technician"])[0],
        is_active=body.is_active if body.is_active is not None else target["enabled"],
        status=target["status"],
        access_expires=local.access_expires if local else None,
        has_local_row=local is not None,
        is_reviewer=bool(local and local.is_reviewer),
    )


@router.post("/{ident}/reset-password", status_code=204)
def reset_password(ident: str, body: PasswordReset) -> None:
    """Set a new password and sign the user out of every device.

    The sign-out is the part that matters: without it the old session keeps
    working on its refresh token for 30 days, which is the opposite of what
    resetting a password is for.
    """
    a = _admin()
    try:
        a.set_password(_resolve(a, ident)["username"], body.password)
    except ClientError as exc:
        raise _cognito_error(exc) from exc


@router.delete("/{ident}", status_code=204)
def delete_user(ident: str, admin: AppUser = Depends(require_admin),
                db: Session = Depends(get_session)) -> None:
    """Remove the account from Cognito, and the local row with it.

    The row is deleted only when nothing references it. Chat sessions and agent
    runs carry a foreign key to it, and those are the record of what the
    assistant was asked and answered -- worth more than the tidiness of removing
    a row whose Cognito account is gone.
    """
    a = _admin()
    target = _resolve(a, ident)
    if target["email"] == admin.email:
        raise HTTPException(400, "Admins cannot delete their own account")

    try:
        a.delete_user(target["username"])
    except ClientError as exc:
        raise _cognito_error(exc) from exc

    local = db.scalar(select(AppUser).where(AppUser.email == target["email"]))
    if local is not None:
        local.is_active = False
        db.commit()
