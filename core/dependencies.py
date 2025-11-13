from fastapi import Depends, Header, HTTPException, Request, status
import redis.asyncio as redis

from .settings import Settings
from brain_service.models.db import User
from brain_service.services.auth_service import AuthService
from brain_service.services.user_service import UserService

def get_redis_client(request: Request) -> redis.Redis:
    """Dependency to get the Redis client from the application state."""
    if not hasattr(request.app.state, 'redis_client') or request.app.state.redis_client is None:
        raise HTTPException(status_code=503, detail="Redis client is not available.")
    return request.app.state.redis_client

def get_http_client(request: Request):
    """Dependency to get the HTTPX client from the application state."""
    return request.app.state.http_client

def get_sefaria_service(request: Request):
    """Dependency to get the SefariaService instance."""
    return request.app.state.sefaria_service

def get_sefaria_index_service(request: Request):
    """Dependency to get the SefariaIndexService instance."""
    return request.app.state.sefaria_index_service

def get_tool_registry(request: Request):
    """Dependency to get the ToolRegistry instance."""
    return request.app.state.tool_registry

def get_memory_service(request: Request):
    """Dependency to get the MemoryService instance."""
    return request.app.state.memory_service

def get_chat_service(request: Request):
    """Dependency to get the ChatService instance."""
    return request.app.state.chat_service

def get_study_service(request: Request):
    """Dependency to get the StudyService instance."""
    return request.app.state.study_service

def get_config_service(request: Request):
    """Dependency to get the ConfigService instance."""
    return request.app.state.config_service

def get_lexicon_service(request: Request):
    """Dependency to get the LexiconService instance."""
    return request.app.state.lexicon_service

def get_session_service(request: Request):
    """Dependency to get the SessionService instance."""
    return request.app.state.session_service

def get_translation_service(request: Request):
    """Dependency to get the TranslationService instance."""
    return request.app.state.translation_service

def get_navigation_service(request: Request):
    """Dependency to get the NavigationService instance."""
    return request.app.state.navigation_service


def get_user_service(request: Request) -> UserService:
    service = getattr(request.app.state, "user_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="User service unavailable")
    return service


def get_auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Auth service unavailable")
    return service


async def get_current_user(
    request: Request,
    authorization: str = Header(None, alias="Authorization"),
) -> User:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header")

    auth_service = get_auth_service(request)
    try:
        user = await auth_service.resolve_user(token)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive or not found")

    return user


def require_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_admin_token(request: Request, x_admin_token: str = Header(None)):
    """Dependency to require admin token for protected endpoints."""
    expected = getattr(request.app.state.settings, "ADMIN_TOKEN", None)
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
    return x_admin_token
