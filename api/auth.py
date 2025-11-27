from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from brain_service.core.dependencies import get_auth_service, get_settings, get_user_service, get_xp_service
from brain_service.core.settings import Settings
from brain_service.models.db import User
from brain_service.services.auth_service import AuthService
from brain_service.services.user_service import UserAlreadyExistsError, UserService
from brain_service.services.xp_service import XpService, XpProfile

router = APIRouter()


def _get_client_ip(request: Request) -> Optional[str]:
    client = request.client
    return client.host if client else None


def _build_user_payload(user: User, xp_profile: XpProfile | None = None) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "phone_number": user.phone_number,
        "role": user.role,
        "is_active": user.is_active,
        **(xp_profile.to_payload() if xp_profile else {}),
    }


def _set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    max_age = settings.JWT_REFRESH_TOKEN_EXPIRES_DAYS * 24 * 60 * 60
    response.set_cookie(
        settings.REFRESH_TOKEN_COOKIE_NAME,
        token,
        httponly=settings.REFRESH_TOKEN_COOKIE_HTTPONLY,
        secure=settings.REFRESH_TOKEN_COOKIE_SECURE,
        samesite=settings.REFRESH_TOKEN_COOKIE_SAMESITE,
        max_age=max_age,
        path=settings.REFRESH_TOKEN_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        settings.REFRESH_TOKEN_COOKIE_NAME,
        path=settings.REFRESH_TOKEN_COOKIE_PATH,
    )


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    phone_number: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    response: Response,
    fastapi_request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
    xp_service: XpService = Depends(get_xp_service),
):
    user = await auth_service.authenticate_user(
        request.username,
        request.password,
        ip_address=_get_client_ip(fastapi_request),
        user_agent=fastapi_request.headers.get("user-agent"),
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token, refresh_token = await auth_service.start_session(
        user,
        ip_address=_get_client_ip(fastapi_request),
        user_agent=fastapi_request.headers.get("user-agent"),
    )
    _set_refresh_cookie(response, refresh_token, settings)
    xp_profile = await xp_service.get_profile(str(user.id))
    return LoginResponse(access_token=access_token, user=_build_user_payload(user, xp_profile))


@router.post("/auth/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    response: Response,
    fastapi_request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings),
    xp_service: XpService = Depends(get_xp_service),
):
    try:
        user = await user_service.create_user(
            request.username,
            request.password,
            phone_number=request.phone_number,
            created_manually=False,
        )
    except UserAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or phone already taken",
        )

    access_token, refresh_token = await auth_service.start_session(
        user,
        ip_address=_get_client_ip(fastapi_request),
        user_agent=fastapi_request.headers.get("user-agent"),
    )
    _set_refresh_cookie(response, refresh_token, settings)
    xp_profile = await xp_service.get_profile(str(user.id))
    return LoginResponse(access_token=access_token, user=_build_user_payload(user, xp_profile))


@router.post("/auth/refresh", response_model=LoginResponse)
async def refresh(
    response: Response,
    fastapi_request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
    xp_service: XpService = Depends(get_xp_service),
):
    refresh_token = fastapi_request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing")

    try:
        access_token, new_refresh_token, user = await auth_service.refresh_session(
            refresh_token,
            ip_address=_get_client_ip(fastapi_request),
            user_agent=fastapi_request.headers.get("user-agent"),
        )
    except ValueError:
        _clear_refresh_cookie(response, settings)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    _set_refresh_cookie(response, new_refresh_token, settings)
    xp_profile = await xp_service.get_profile(str(user.id))
    return LoginResponse(access_token=access_token, user=_build_user_payload(user, xp_profile))


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    fastapi_request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
):
    refresh_token = fastapi_request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if refresh_token:
        try:
            await auth_service.revoke_refresh_token(refresh_token)
        except ValueError:
            pass
    _clear_refresh_cookie(response, settings)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
