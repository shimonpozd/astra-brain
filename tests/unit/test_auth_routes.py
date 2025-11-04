import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain_service.api import auth as auth_router
from brain_service.api import users as users_router
from brain_service.core import dependencies
from brain_service.services.user_service import UserAlreadyExistsError


class DummyUser:
    def __init__(self, user_id: uuid.UUID, username: str, role: str, is_active: bool = True):
        self.id = user_id
        self.username = username
        self.role = role
        self.is_active = is_active


class DummyAuthService:
    def __init__(self):
        self._users = {
            "alice": ("wonderland", DummyUser(uuid.uuid4(), "alice", "member")),
        }

    async def authenticate_user(self, username: str, password: str):
        record = self._users.get(username)
        if not record:
            return None
        stored_password, user = record
        if password != stored_password:
            return None
        return user

    def issue_token(self, user: DummyUser) -> str:
        return f"token-for-{user.username}"


class DummyUserService:
    def __init__(self):
        self._users = {
            "admin": DummyUser(uuid.uuid4(), "admin", "admin"),
        }

    async def list_users(self):
        return list(self._users.values())

    async def create_user(self, username: str, password: str, role: str = "member", is_active: bool = True):
        key = username.lower()
        if key in self._users:
            raise UserAlreadyExistsError(username)
        user = DummyUser(uuid.uuid4(), key, role, is_active)
        self._users[key] = user
        return user


@pytest.fixture
def test_client():
    app = FastAPI()
    app.include_router(auth_router.router, prefix="/api")
    app.include_router(users_router.router, prefix="/api")

    auth_service = DummyAuthService()
    user_service = DummyUserService()
    admin_user = DummyUser(uuid.uuid4(), "admin", "admin")

    app.dependency_overrides[dependencies.get_auth_service] = lambda: auth_service
    app.dependency_overrides[dependencies.get_user_service] = lambda: user_service
    app.dependency_overrides[dependencies.require_admin_user] = lambda: admin_user

    client = TestClient(app)
    return client, auth_service, user_service


def test_login_success(test_client):
    client, _, _ = test_client
    response = client.post("/api/auth/login", json={"username": "alice", "password": "wonderland"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"] == "token-for-alice"
    assert payload["token_type"] == "bearer"
    assert payload["user"]["username"] == "alice"


def test_login_failure(test_client):
    client, _, _ = test_client
    response = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
    assert response.status_code == 401


def test_list_users_requires_admin(test_client):
    client, _, user_service = test_client
    response = client.get("/api/users")
    assert response.status_code == 200
    payload = response.json()
    returned_usernames = {user["username"] for user in payload}
    assert set(returned_usernames) == {user.username for user in user_service._users.values()}


def test_create_user_and_conflict(test_client):
    client, _, _ = test_client

    create_resp = client.post("/api/users", json={"username": "bob", "password": "secret", "role": "member"})
    assert create_resp.status_code == 201
    assert create_resp.json()["username"] == "bob"

    conflict_resp = client.post("/api/users", json={"username": "bob", "password": "secret", "role": "member"})
    assert conflict_resp.status_code == 409
