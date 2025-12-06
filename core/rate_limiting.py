import time
import logging
from typing import Dict, Optional
from fastapi import Request, HTTPException
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Simple rate limiter using Redis for distributed rate limiting.
    """
    
    def __init__(self, redis_client: redis.Redis, default_limit: int = 10, window_seconds: int = 60):
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.window_seconds = window_seconds
    
    async def is_allowed(self, key: str, limit: Optional[int] = None) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            key: Unique identifier for rate limiting (e.g., user_id, ip_address)
            limit: Custom limit for this check (uses default if None)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        if not self.redis_client:
            # If Redis is not available, allow all requests
            logger.warning("Redis not available, skipping rate limiting")
            return True
        
        limit = limit or self.default_limit
        current_time = int(time.time())
        window_start = current_time - self.window_seconds
        
        try:
            # Use Redis sorted set for sliding window rate limiting
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration for the key
            pipe.expire(key, self.window_seconds)
            
            results = await pipe.execute()
            current_count = results[1]  # Count after cleanup
            
            is_allowed = current_count < limit
            
            if not is_allowed:
                logger.warning(f"Rate limit exceeded for key {key}: {current_count}/{limit}")
            
            return is_allowed
            
        except Exception as e:
            logger.error(f"Rate limiting error for key {key}: {e}")
            # On error, allow the request
            return True

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> Optional[RateLimiter]:
    """Get the global rate limiter instance."""
    return _rate_limiter

def setup_rate_limiter(redis_client: redis.Redis, default_limit: int = 10, window_seconds: int = 60):
    """Setup the global rate limiter."""
    global _rate_limiter
    _rate_limiter = RateLimiter(redis_client, default_limit, window_seconds)
    logger.info(f"Rate limiter configured: {default_limit} requests per {window_seconds} seconds")

async def check_rate_limit(request: Request, limit: Optional[int] = None) -> bool:
    """
    Check rate limit for a request.
    
    Args:
        request: FastAPI request object
        limit: Custom limit for this check
        
    Returns:
        True if request is allowed
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    rate_limiter = get_rate_limiter()
    if not rate_limiter:
        return True
    
    # Use client IP as rate limiting key
    client_ip = request.client.host if request.client else "unknown"
    
    is_allowed = await rate_limiter.is_allowed(f"rate_limit:{client_ip}", limit)
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    return True

def rate_limit_dependency(limit: Optional[int] = None):
    """
    Create a dependency for rate limiting.
    
    Args:
        limit: Custom rate limit for this endpoint
        
    Returns:
        FastAPI dependency function
    """
    async def _rate_limit_check(request: Request):
        await check_rate_limit(request, limit)
        return True
    
    return _rate_limit_check



































