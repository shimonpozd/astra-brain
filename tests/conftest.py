"""
Pytest configuration and shared fixtures for brain_service tests.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import redis.asyncio as redis

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock(spec=redis.Redis)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.scan_iter = AsyncMock(return_value=iter([]))
    redis_mock.pipeline.return_value = AsyncMock()
    return redis_mock


@pytest.fixture
def mock_http_client():
    """Mock HTTPX client for testing."""
    http_mock = AsyncMock()
    http_mock.get = AsyncMock()
    http_mock.post = AsyncMock()
    http_mock.aclose = AsyncMock()
    return http_mock


@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI Request object for testing."""
    request_mock = Mock()
    request_mock.client.host = "127.0.0.1"
    request_mock.headers = {}
    request_mock.query_params = {}
    request_mock.app.state = Mock()
    return request_mock


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "What can you help me with?"},
        {"role": "assistant", "content": "I can help you with various tasks and answer questions."}
    ]


@pytest.fixture
def sample_stm_data():
    """Sample STM data for testing."""
    return {
        "summary_v1": "User asked about capabilities and got helpful responses.",
        "salient_facts": [
            "Assistant is helpful and responsive",
            "User is interested in learning about capabilities"
        ],
        "open_loops": [
            "User might want to know more about specific capabilities"
        ],
        "ts_updated": 1234567890.0
    }


@pytest.fixture
def sample_tool_schema():
    """Sample tool schema for testing."""
    return {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool for unit testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input parameter"
                    }
                },
                "required": ["input"]
            }
        }
    }



























