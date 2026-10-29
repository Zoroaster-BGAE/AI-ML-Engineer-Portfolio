"""Who am I, and what the browser needs to know to sign in.

There is no login endpoint. The frontend authenticates against Cognito directly
and arrives here holding an id token, so no password ever reaches this service
and there is nothing here to attack with one. Changing a password is Cognito's
forgot-password flow, or an administrator through /api/admin/users.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.schemas import UserOut
from app.auth.dependencies import get_current_user
from app.config import settings
from app.domain.support import AppUser

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AuthConfig(BaseModel):
    """What the browser needs to talk to Cognito. All of it is public."""

    user_pool_id: str
    client_id: str
    region: str
    configured: bool
    # Set only in local development, where sign-in goes to a Cognito emulator instead of AWS.
    endpoint: str | None = None


@router.get("/config", response_model=AuthConfig)
def auth_config() -> AuthConfig:
    """Unauthenticated by necessity -- it is read *before* anyone can log in.

    Served at runtime rather than compiled into the bundle because Vite inlines
    environment variables at build time: baking these in would mean a rebuild
    and a redeploy to point the app at a different pool, and would put the values
    in the image where they are harder to see than in the task definition.

    Nothing secret is exposed. The client id travels in every login request the
    browser makes, and the pool id is in the issuer of every token it receives.
    The app client has no secret at all, which is what makes a public SPA client
    safe in the first place.
    """
    return AuthConfig(
        user_pool_id=settings.cognito_user_pool_id,
        client_id=settings.cognito_client_id,
        region=settings.cognito_region,
        configured=settings.cognito_configured,
        endpoint=settings.cognito_public_endpoint or settings.cognito_endpoint or None,
    )


@router.get("/me", response_model=UserOut)
def me(user: AppUser = Depends(get_current_user)) -> AppUser:
    return user
