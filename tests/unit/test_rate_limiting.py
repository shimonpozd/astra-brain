import pytest
import time
from unittest.mock import AsyncMock, Mock
from brain_service.core.rate_limiting import RateLimiter, setup_rate_limiter, get_rate_limiter


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.pipeline.return_value = AsyncMock()
        return redis_mock
    
    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create RateLimiter instance with mocked Redis."""
        return RateLimiter(redis_client=mock_redis, default_limit=5, window_seconds=60)
    
    @pytest.mark.asyncio
    async def test_is_allowed_within_limit(self, rate_limiter, mock_redis):
        """Test that requests within limit are allowed."""
        # Mock pipeline execution to return count < limit
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [None, 3, None, None]  # count = 3, limit = 5
        mock_redis.pipeline.return_value = mock_pipeline
        
        result = await rate_limiter.is_allowed("test_key")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_allowed_exceeds_limit(self, rate_limiter, mock_redis):
        """Test that requests exceeding limit are not allowed."""
        # Mock pipeline execution to return count >= limit
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [None, 6, None, None]  # count = 6, limit = 5
        mock_redis.pipeline.return_value = mock_pipeline
        
        result = await rate_limiter.is_allowed("test_key")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_allowed_custom_limit(self, rate_limiter, mock_redis):
        """Test rate limiting with custom limit."""
        # Mock pipeline execution to return count < custom limit
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [None, 2, None, None]  # count = 2, custom limit = 3
        mock_redis.pipeline.return_value = mock_pipeline
        
        result = await rate_limiter.is_allowed("test_key", limit=3)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_allowed_redis_error(self, rate_limiter, mock_redis):
        """Test behavior when Redis raises an error."""
        # Mock Redis to raise an exception
        mock_redis.pipeline.side_effect = Exception("Redis error")
        
        # Should allow request on error
        result = await rate_limiter.is_allowed("test_key")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_no_redis_client(self):
        """Test behavior when Redis client is None."""
        rate_limiter = RateLimiter(redis_client=None, default_limit=5, window_seconds=60)
        
        # Should allow all requests when Redis is not available
        result = await rate_limiter.is_allowed("test_key")
        assert result is True


class TestRateLimitingSetup:
    """Test cases for rate limiting setup functions."""
    
    def test_setup_rate_limiter(self, mock_redis):
        """Test setting up the global rate limiter."""
        setup_rate_limiter(mock_redis, default_limit=10, window_seconds=120)
        
        rate_limiter = get_rate_limiter()
        assert rate_limiter is not None
        assert rate_limiter.default_limit == 10
        assert rate_limiter.window_seconds == 120
    
    def test_get_rate_limiter_before_setup(self):
        """Test getting rate limiter before setup."""
        # Reset global state
        import brain_service.core.rate_limiting as rl_module
        rl_module._rate_limiter = None
        
        rate_limiter = get_rate_limiter()
        assert rate_limiter is None


class TestRateLimitingIntegration:
    """Integration tests for rate limiting with FastAPI."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_dependency_allowed(self):
        """Test rate limit dependency when request is allowed."""
        from fastapi import Request
        from brain_service.core.rate_limiting import rate_limit_dependency
        
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.client.host = "192.168.1.1"
        
        # Setup rate limiter that allows requests
        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [None, 1, None, None]  # count = 1, limit = 5
        mock_redis.pipeline.return_value = mock_pipeline
        
        setup_rate_limiter(mock_redis, default_limit=5, window_seconds=60)
        
        # Create dependency
        dependency = rate_limit_dependency()
        
        # Should not raise exception
        result = await dependency(mock_request)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_dependency_exceeded(self):
        """Test rate limit dependency when limit is exceeded."""
        from fastapi import Request, HTTPException
        from brain_service.core.rate_limiting import rate_limit_dependency
        
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.client.host = "192.168.1.1"
        
        # Setup rate limiter that blocks requests
        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [None, 10, None, None]  # count = 10, limit = 5
        mock_redis.pipeline.return_value = mock_pipeline
        
        setup_rate_limiter(mock_redis, default_limit=5, window_seconds=60)
        
        # Create dependency
        dependency = rate_limit_dependency()
        
        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await dependency(mock_request)
        
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)


































