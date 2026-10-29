"""Token verification against the Cognito user pool, and user administration in it.

Cognito is the identity provider. This application issues no tokens and stores no
passwords; it verifies what Cognito signed and reads the caller's role out of the
`cognito:groups` claim.

Two halves live here:

* `verify_id_token` -- offline verification. Cognito signs with RS256, so the
  public half of the signing key is enough and it is published at a well-known
  URL. Nothing here calls AWS on the request path.
* `Admin` -- the thin wrapper the admin console goes through, so that creating or
  disabling a user changes Cognito rather than a local table. It uses the task
  role's IAM permissions, so there is no credential to hold.

WHY THE ID TOKEN AND NOT THE ACCESS TOKEN

Both are signed by the same key and both carry `cognito:groups`. Only the id
token carries `email`, and email is how this application and the people
administering it identify a user. The access token's `username` is the pool's
opaque uuid.

WHAT REVOCATION MEANS HERE

Verification is offline, so a token stays valid until it expires no matter what
happens to the account behind it -- disabling a user cannot reach a signature
already made. The bound is therefore the token's lifetime, which is configured
to five minutes. Refreshing is what enforces the account state: redeeming a
refresh token is a call to Cognito, and Cognito refuses it for a disabled user.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import jwt
from jwt import PyJWKClient

from app.config import settings
from app.obs.logging import get_logger

log = get_logger()

# Group name -> the role stored on app_user. Order matters: a user in several
# groups gets the most capable one, which is the only sane reading of "this
# person is both an admin and a technician".
_ROLE_BY_PRECEDENCE = ("admin", "technician", "customer")
_DEFAULT_ROLE = "technician"


class TokenError(Exception):
    """The token is missing, malformed, expired, or not ours."""


@lru_cache(maxsize=1)
def _jwk_client() -> PyJWKClient:
    """One client for the process, caching the pool's signing keys.

    PyJWKClient fetches the key set on first use and keeps it, refetching only
    when it sees a `kid` it does not know -- which is what happens when Cognito
    rotates a signing key. That is the behaviour wanted: no fetch per request,
    and no manual cache invalidation to get wrong.
    """
    if not settings.cognito_configured:
        raise TokenError("Cognito is not configured on this deployment")
    return PyJWKClient(settings.cognito_jwks_url, cache_keys=True)


def verify_id_token(token: str) -> dict[str, Any]:
    """Return the claims of a valid Cognito id token, or raise TokenError.

    Every check here matters, and the two easy ones to leave out are the ones
    that make verification meaningless:

    * `audience` -- without it, an id token minted for *any* app client in this
      pool is accepted, including one created later for something unrelated.
    * `token_use` -- PyJWT has no concept of it, so it must be checked by hand.
      An access token from this pool is signed by the same key and passes every
      standard check; accepting one would let a token issued for calling
      Cognito's own API act as a login here.
    """
    try:
        key = _jwk_client().get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=settings.cognito_client_id,
            issuer=settings.cognito_issuer,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except TokenError:
        raise
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    except Exception as exc:  # network failure reaching the key set, most likely
        log.warning("jwks_unavailable", error=str(exc)[:200])
        raise TokenError("could not verify the token signature") from exc

    if claims.get("token_use") != "id":
        raise TokenError(f"expected an id token, got token_use={claims.get('token_use')!r}")
    if not claims.get("email"):
        raise TokenError("token carries no email claim")

    return claims


def role_from_claims(claims: dict[str, Any]) -> str:
    """The app role for a verified token, from its Cognito group membership.

    A user in no group gets `technician`: the least privileged role that can
    still use the product. Falling back to `admin` would turn a forgotten
    group assignment into a privilege grant, and falling back to nothing would
    make the failure look like a broken login rather than missing setup.
    """
    groups = claims.get("cognito:groups") or []
    for role in _ROLE_BY_PRECEDENCE:
        if role in groups:
            return role
    return _DEFAULT_ROLE


class Admin:
    """User administration against the pool, for the admin console.

    Deliberately not a general Cognito client: it exposes the six operations the
    console needs and nothing else, so the IAM policy backing it can stay the
    same shape.
    """

    def __init__(self) -> None:
        import boto3  # imported here so the module stays importable without AWS

        self._c = boto3.client("cognito-idp", region_name=settings.cognito_region,
                               endpoint_url=settings.cognito_endpoint or None)
        self._pool = settings.cognito_user_pool_id

    # -- reading ----------------------------------------------------------

    def list_users(self) -> list[dict[str, Any]]:
        """Every user in the pool, with their groups resolved.

        One call per user for the groups, which is fine at this scale and would
        not be at thousands. If that day comes, list each group's members
        instead -- three calls total regardless of user count.
        """
        users: list[dict[str, Any]] = []
        for page in self._c.get_paginator("list_users").paginate(UserPoolId=self._pool):
            for u in page["Users"]:
                attrs = {a["Name"]: a["Value"] for a in u.get("Attributes", [])}
                users.append({
                    "username": u["Username"],
                    "email": attrs.get("email", ""),
                    "enabled": u.get("Enabled", False),
                    "status": u.get("UserStatus", ""),
                    "groups": self.groups_of(u["Username"]),
                })
        return users

    def groups_of(self, username: str) -> list[str]:
        r = self._c.admin_list_groups_for_user(UserPoolId=self._pool, Username=username)
        return [g["GroupName"] for g in r.get("Groups", [])]

    # -- writing ----------------------------------------------------------

    def create_user(self, email: str, password: str, role: str) -> str:
        """Create a confirmed user with a usable password, and return the username.

        `SUPPRESS` because this pool has no verified sending address configured,
        so asking Cognito to email an invitation fails the whole call. The
        password is therefore set here and communicated by whoever is
        administering the account.

        Setting it `--permanent` skips FORCE_CHANGE_PASSWORD. The alternative is
        a first-login password change flow, which this frontend does not have;
        a temporary password would lock every new user out at the login screen
        with a message the UI cannot act on.
        """
        r = self._c.admin_create_user(
            UserPoolId=self._pool,
            Username=email,
            UserAttributes=[
                {"Name": "email", "Value": email},
                {"Name": "email_verified", "Value": "true"},
            ],
            MessageAction="SUPPRESS",
        )
        username = r["User"]["Username"]
        self._c.admin_set_user_password(
            UserPoolId=self._pool, Username=username, Password=password, Permanent=True,
        )
        self.set_role(username, role)
        return username

    def set_role(self, username: str, role: str) -> None:
        """Make `role` the user's only group, so the mapping stays single-valued."""
        for g in self.groups_of(username):
            if g != role:
                self._c.admin_remove_user_from_group(UserPoolId=self._pool, Username=username, GroupName=g)
        if role not in self.groups_of(username):
            self._c.admin_add_user_to_group(UserPoolId=self._pool, Username=username, GroupName=role)

    def set_enabled(self, username: str, enabled: bool) -> None:
        """Enable or disable, signing the user out of everything on the way down.

        The global sign-out is what makes disabling take effect within the token
        lifetime rather than at the end of the 30-day refresh window: it
        invalidates the refresh tokens, so thefive-minute access token cannot be
        renewed once it expires.
        """
        if enabled:
            self._c.admin_enable_user(UserPoolId=self._pool, Username=username)
        else:
            self._c.admin_disable_user(UserPoolId=self._pool, Username=username)
            self._c.admin_user_global_sign_out(UserPoolId=self._pool, Username=username)

    def set_password(self, username: str, password: str) -> None:
        self._c.admin_set_user_password(
            UserPoolId=self._pool, Username=username, Password=password, Permanent=True,
        )
        self._c.admin_user_global_sign_out(UserPoolId=self._pool, Username=username)

    def delete_user(self, username: str) -> None:
        self._c.admin_delete_user(UserPoolId=self._pool, Username=username)
