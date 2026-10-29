"""Caller identity from a Cognito id token, and the role guards built on it.

The `app_user` row is a local projection of a Cognito user, not a second user
directory. It exists because the rest of the schema needs something to point at:
`chat_session`, `agent_run` and `retrieval_hit` all carry a foreign key to it.

That is why rows are created on demand here rather than by an administrator.
Adding someone in Cognito is the whole of adding a user -- their row appears the
first time they call the API. Without that, every account would have to be
created in two places and the two would drift, which is the failure mode that
makes "we use Cognito" quietly untrue.

Role and email are refreshed from the token on every request, so moving someone
between groups in Cognito takes effect on their next call rather than whenever
somebody remembers to update the database.
"""
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.cognito import TokenError, role_from_claims, verify_id_token
from app.db import get_session
from app.domain.support import AppUser

_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
                     db: Session = Depends(get_session)) -> AppUser:
    if credentials is None:
        raise _unauthorized()

    try:
        claims = verify_id_token(credentials.credentials)
    except TokenError as exc:
        raise _unauthorized(str(exc)) from exc

    sub = claims["sub"]
    email = claims["email"].strip().lower()
    role = role_from_claims(claims)

    # By sub first: it is the pool's stable identifier and survives an email
    # change. By email second, which is how the six accounts that predate this
    # code are matched -- they have rows with no sub recorded yet.
    user = db.scalar(select(AppUser).where(AppUser.cognito_sub == sub))
    if user is None:
        user = db.scalar(select(AppUser).where(AppUser.email == email))

    if user is None:
        user = AppUser(cognito_sub=sub, email=email, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    # Cognito is authoritative for all three, so reconcile rather than trust the
    # row. Written only on an actual change, to keep a read-only request from
    # producing a write on every call.
    changed = False
    if user.cognito_sub != sub:
        user.cognito_sub = sub
        changed = True
    if user.email != email:
        user.email = email
        changed = True
    if user.role != role:
        user.role = role
        changed = True
    if changed:
        db.commit()

    # Kept as a local override even though Cognito's own disable is the normal
    # path. It is the one way to cut someone off faster than their token expires,
    # since a signature already issued cannot be withdrawn.
    if not user.is_active:
        raise _unauthorized("Account is disabled")

    if user.access_expires is not None and datetime.now(UTC) > user.access_expires:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access expired")

    return user


def require_admin(user: AppUser = Depends(get_current_user)) -> AppUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only")
    return user


def require_reviewer(user: AppUser = Depends(get_current_user)) -> AppUser:
    """Admins, and users an admin flagged as golden-example reviewers (the lead technician)."""
    if not user.can_review:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Reviewers only")
    return user
