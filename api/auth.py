from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from brain_service.core.dependencies import get_auth_service
from brain_service.services.auth_service import AuthService
from brain_service.models.db import User

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _serialize_user(user: User) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "role": user.role,
        "is_active": user.is_active,
    }


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    user = await auth_service.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    token = auth_service.issue_token(user)
    return LoginResponse(
        access_token=token,
        user=_serialize_user(user),
    )
