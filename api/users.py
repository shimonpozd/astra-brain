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
from brain_service.models.db import User, UserApiKey, UserSession, UserLoginEvent

router = APIRouter()


def _parse_uuid(value: str, name: str = "identifier") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name}",
        )


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
    phone_number: Optional[str]
    api_keys: List[UserApiKeyResponse]
    active_session_count: int
    total_session_count: int

    @classmethod
    def from_model(
        cls,
        user: User,
        api_keys: Optional[List[UserApiKey]] = None,
        *,
        active_session_count: int = 0,
        total_session_count: int = 0,
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
            phone_number=user.phone_number,
            api_keys=key_responses,
            active_session_count=active_session_count,
            total_session_count=total_session_count,
        )


class UserSessionResponse(BaseModel):
    id: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str
    last_used_at: Optional[str]
    expires_at: str

    @classmethod
    def from_model(cls, session: UserSession) -> "UserSessionResponse":
        return cls(
            id=str(session.id),
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            is_active=session.is_active,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            last_used_at=session.last_used_at.isoformat() if session.last_used_at else None,
            expires_at=session.expires_at.isoformat(),
        )


class UserLoginEventResponse(BaseModel):
    id: str
    username: Optional[str]
    success: bool
    reason: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: str

    @classmethod
    def from_model(cls, event: UserLoginEvent) -> "UserLoginEventResponse":
        return cls(
            id=str(event.id),
            username=event.username,
            success=event.success,
            reason=event.reason,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            created_at=event.created_at.isoformat(),
        )


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: Literal["admin", "member"] = "member"
    is_active: bool = True
    created_manually: bool = True
    phone_number: Optional[str] = None


class UpdateUserRequest(BaseModel):
    password: Optional[str] = None
    role: Optional[Literal["admin", "member"]] = None
    is_active: Optional[bool] = None
    phone_number: Optional[str] = None


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
        sessions = await user_service.list_sessions_for_user(user.id)
        active_sessions = sum(1 for session in sessions if session.is_active)
        responses.append(
            UserResponse.from_model(
                user,
                keys,
                active_session_count=active_sessions,
                total_session_count=len(sessions),
            )
        )
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
            phone_number=request.phone_number,
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
        phone_number=request.phone_number,
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


@router.get("/users/{user_id}/sessions", response_model=List[UserSessionResponse])
async def list_user_sessions(
    user_id: str,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    user_uuid = _parse_uuid(user_id, "user id")
    user = await user_service.get_user_by_id(user_uuid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    sessions = await user_service.list_sessions_for_user(user_uuid)
    return [UserSessionResponse.from_model(session) for session in sessions]


@router.delete(
    "/users/{user_id}/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_user_session(
    user_id: str,
    session_id: str,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    user_uuid = _parse_uuid(user_id, "user id")
    session_uuid = _parse_uuid(session_id, "session id")
    user = await user_service.get_user_by_id(user_uuid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    session = await user_service.get_session_by_id(session_uuid)
    if not session or session.user_id != user_uuid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    await user_service.revoke_session(session_uuid)
    return None


@router.get("/users/{user_id}/login-events", response_model=List[UserLoginEventResponse])
async def list_user_login_events(
    user_id: str,
    limit: int = 20,
    _: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    user_uuid = _parse_uuid(user_id, "user id")
    user = await user_service.get_user_by_id(user_uuid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    events = await user_service.list_login_events(user_uuid, limit=limit)
    return [UserLoginEventResponse.from_model(event) for event in events]
