from typing import List, Literal, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from brain_service.core.dependencies import get_user_service, require_admin_user
from brain_service.services.user_service import (
    UserAlreadyExistsError,
    UserNotFoundError,
    UserService,
)
from brain_service.models.db import User, UserApiKey

router = APIRouter()


class UserApiKeyResponse(BaseModel):
    id: str
    provider: str
    last_four: str
    daily_limit: Optional[int]
    usage_today: int
    last_reset_at: Optional[str]
    is_active: bool
    created_at: str

    @classmethod
    def from_model(cls, key: UserApiKey) -> "UserApiKeyResponse":
        return cls(
            id=str(key.id),
            provider=key.provider,
            last_four=key.last_four,
            daily_limit=key.daily_limit,
            usage_today=key.usage_today,
            last_reset_at=key.last_reset_at.isoformat() if key.last_reset_at else None,
            is_active=key.is_active,
            created_at=key.created_at.isoformat(),
        )


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool
    created_manually: bool
    created_at: str
    updated_at: str
    last_login_at: Optional[str]
    api_keys: List[UserApiKeyResponse]

    @classmethod
    def from_model(
        cls,
        user: User,
        api_keys: Optional[List[UserApiKey]] = None,
    ) -> "UserResponse":
        key_models = api_keys if api_keys is not None else getattr(user, "api_keys", [])
        key_responses = [UserApiKeyResponse.from_model(key) for key in key_models]
        return cls(
            id=str(user.id),
            username=user.username,
            role=user.role,
            is_active=user.is_active,
            created_manually=user.created_manually,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
            api_keys=key_responses,
        )


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: Literal["admin", "member"] = "member"
    is_active: bool = True
    created_manually: bool = True


class UpdateUserRequest(BaseModel):
    password: Optional[str] = None
    role: Optional[Literal["admin", "member"]] = None
    is_active: Optional[bool] = None


class CreateApiKeyRequest(BaseModel):
    provider: Literal["openrouter", "openai"] = "openrouter"
    api_key: str
    daily_limit: Optional[int] = None


class UpdateApiKeyRequest(BaseModel):
    daily_limit: Optional[int] = None
    is_active: Optional[bool] = None


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    users = await user_service.list_users()
    responses: List[UserResponse] = []
    for user in users:
        keys = await user_service.list_api_keys(user.id)
        responses.append(UserResponse.from_model(user, keys))
    return responses


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    try:
        user = await user_service.create_user(
            username=request.username,
            password=request.password,
            role=request.role,
            is_active=request.is_active,
            created_manually=request.created_manually,
        )
    except UserAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists",
        ) from None

    return UserResponse.from_model(user, api_keys=[])


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id")

    # For password changes we use username to keep compatibility
    user = await user_service.get_user_by_id(user_uuid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    updated = await user_service.update_user(
        user.username,
        password=request.password,
        role=request.role,
        is_active=request.is_active,
    )
    keys = await user_service.list_api_keys(updated.id)
    return UserResponse.from_model(updated, keys)


@router.get("/users/{user_id}/api-keys", response_model=List[UserApiKeyResponse])
async def list_user_api_keys(
    user_id: str,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id")

    keys = await user_service.list_api_keys(user_uuid)
    return [UserApiKeyResponse.from_model(key) for key in keys]


@router.post(
    "/users/{user_id}/api-keys",
    response_model=UserApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_api_key(
    user_id: str,
    request: CreateApiKeyRequest,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id")

    try:
        key = await user_service.create_api_key(
            user_uuid,
            provider=request.provider,
            api_key=request.api_key,
            daily_limit=request.daily_limit,
        )
    except UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return UserApiKeyResponse.from_model(key)


@router.patch(
    "/users/{user_id}/api-keys/{key_id}",
    response_model=UserApiKeyResponse,
)
async def update_user_api_key(
    user_id: str,
    key_id: str,
    request: UpdateApiKeyRequest,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    try:
        user_uuid = uuid.UUID(user_id)
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid identifier")

    # Ensure key belongs to user
    keys = await user_service.list_api_keys(user_uuid)
    target = next((k for k in keys if k.id == key_uuid), None)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    try:
        updated = await user_service.update_api_key(
            key_uuid,
            daily_limit=request.daily_limit,
            is_active=request.is_active,
        )
    except UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found") from None

    return UserApiKeyResponse.from_model(updated)


@router.delete(
    "/users/{user_id}/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_user_api_key(
    user_id: str,
    key_id: str,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    try:
        user_uuid = uuid.UUID(user_id)
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid identifier")

    keys = await user_service.list_api_keys(user_uuid)
    target = next((k for k in keys if k.id == key_uuid), None)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    await user_service.delete_api_key(key_uuid)
    return None
