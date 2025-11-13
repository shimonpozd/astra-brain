from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

from .config_loader import ensure_config_root

ensure_config_root()
from config import get_config

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load settings from TOML config
        self._load_from_toml()
    
    def _load_from_toml(self):
        """Load settings from TOML configuration."""
        try:
            config = get_config()
            
            # Load STM settings
            if 'memory' in config and 'stm' in config['memory']:
                stm_config = config['memory']['stm']
                self.STM_TTL_SEC = stm_config.get('ttl_seconds', 86400)
                self.STM_TRIGGER_MSGS = stm_config.get('trigger_messages', 8)
                self.STM_TRIGGER_TOKENS = stm_config.get('trigger_tokens', 2000)
            
            # Load brain service settings
            if 'services' in config and 'brain' in config['services']:
                brain_config = config['services']['brain']
                self.BRAIN_PORT = brain_config.get('port', 7030)
                self.ADMIN_TOKEN = brain_config.get('admin_token', 'super-secret-token')
                self.CORS_ORIGINS = brain_config.get('cors_origins', 'http://localhost:5173')
                
                # Load rate limiting settings
                if 'rate_limiting' in brain_config:
                    rate_config = brain_config['rate_limiting']
                    self.RATE_LIMIT_ENABLED = rate_config.get('enabled', True)
                    self.RATE_LIMIT_DEFAULT = rate_config.get('default_limit', 10)
                    self.RATE_LIMIT_WINDOW = rate_config.get('window_seconds', 60)
                    self.RATE_LIMIT_LLM = rate_config.get('llm_limit', 5)
                
                # Load Sefaria settings
                if 'sefaria' in brain_config:
                    sefaria_config = brain_config['sefaria']
                    self.SEFARIA_API_URL = sefaria_config.get('api_url', 'http://localhost:8000/api/')
                    self.SEFARIA_API_KEY = sefaria_config.get('api_key', None)
                    self.SEFARIA_CACHE_TTL = sefaria_config.get('cache_ttl_seconds', 60)
                self.DATABASE_URL = brain_config.get('database_url', self.DATABASE_URL)
                jwt_cfg = brain_config.get('jwt', {})
                self.JWT_SECRET = jwt_cfg.get('secret', self.JWT_SECRET)
                self.JWT_ALGORITHM = jwt_cfg.get('algorithm', self.JWT_ALGORITHM)
                self.JWT_ACCESS_TOKEN_EXPIRES_MINUTES = jwt_cfg.get(
                    'access_token_expires_minutes',
                    self.JWT_ACCESS_TOKEN_EXPIRES_MINUTES,
                )
                self.JWT_REFRESH_TOKEN_EXPIRES_DAYS = jwt_cfg.get(
                    'refresh_token_expires_days',
                    self.JWT_REFRESH_TOKEN_EXPIRES_DAYS,
                )
                refresh_cfg = jwt_cfg.get('refresh_cookie', {})
                self.REFRESH_TOKEN_COOKIE_NAME = refresh_cfg.get(
                    'name',
                    self.REFRESH_TOKEN_COOKIE_NAME,
                )
                self.REFRESH_TOKEN_COOKIE_PATH = refresh_cfg.get(
                    'path',
                    self.REFRESH_TOKEN_COOKIE_PATH,
                )
                self.REFRESH_TOKEN_COOKIE_SAMESITE = refresh_cfg.get(
                    'samesite',
                    self.REFRESH_TOKEN_COOKIE_SAMESITE,
                )
                self.REFRESH_TOKEN_COOKIE_SECURE = refresh_cfg.get(
                    'secure',
                    self.REFRESH_TOKEN_COOKIE_SECURE,
                )
                self.REFRESH_TOKEN_COOKIE_HTTPONLY = refresh_cfg.get(
                    'httponly',
                    self.REFRESH_TOKEN_COOKIE_HTTPONLY,
                )

            # Load Redis URL from brain_service.services
            if 'services' in config:
                services_config = config['services']
                self.REDIS_URL = services_config.get('redis_url', 'redis://localhost:6379/0')
                self.DATABASE_URL = services_config.get('database_url', self.DATABASE_URL)
                self.SEFARIA_MCP_URL = services_config.get('sefaria_mcp_url', self.SEFARIA_MCP_URL)
                self.SEFARIA_MCP_TIMEOUT_SEC = services_config.get(
                    'sefaria_mcp_timeout_sec',
                    self.SEFARIA_MCP_TIMEOUT_SEC,
                )
                self.API_KEY_SECRET = services_config.get(
                    'api_key_secret',
                    self.API_KEY_SECRET,
                )

        except Exception as e:
            # Fallback to defaults if TOML loading fails
            print(f"Warning: Could not load TOML config: {e}")
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default values if TOML loading fails."""
        self.BRAIN_PORT = 7030
        self.REDIS_URL = "redis://localhost:6379/0"
        self.DATABASE_URL = "postgresql+asyncpg://astra:astra@localhost:5432/astra_brain"
        self.ADMIN_TOKEN = "super-secret-token"
        self.STM_TTL_SEC = 86400
        self.STM_TRIGGER_MSGS = 8
        self.STM_TRIGGER_TOKENS = 2000
        self.SEFARIA_API_URL = "http://localhost:8000/api/"
        self.SEFARIA_API_KEY = None
        self.SEFARIA_CACHE_TTL = 60
        self.CORS_ORIGINS = "http://localhost:5173"
        self.RATE_LIMIT_ENABLED = True
        self.RATE_LIMIT_DEFAULT = 10
        self.RATE_LIMIT_WINDOW = 60
        self.RATE_LIMIT_LLM = 5
        self.LOG_LEVEL = "INFO"
        self.LOG_JSON = False
        self.SEFARIA_MCP_URL = "http://sefaria.org:8088/sse"
        self.SEFARIA_MCP_TIMEOUT_SEC = 30
        self.JWT_SECRET = "change-me"
        self.JWT_ALGORITHM = "HS256"
        self.JWT_ACCESS_TOKEN_EXPIRES_MINUTES = 60
        self.JWT_REFRESH_TOKEN_EXPIRES_DAYS = 30
        self.API_KEY_SECRET = "change-me-api-key-secret"
        self.REFRESH_TOKEN_COOKIE_NAME = "astra_refresh_token"
        self.REFRESH_TOKEN_COOKIE_PATH = "/api"
        self.REFRESH_TOKEN_COOKIE_SAMESITE = "lax"
        self.REFRESH_TOKEN_COOKIE_SECURE = True
        self.REFRESH_TOKEN_COOKIE_HTTPONLY = True

    # Default values (will be overridden by TOML config)
    BRAIN_PORT: int = 7030
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql+asyncpg://astra:astra@localhost:5432/astra_brain"
    ADMIN_TOKEN: str = "super-secret-token"
    
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    
    STREAM_FORMAT: str = "ndjson"
    MAX_TOOL_STEPS: int = 3

    STM_TTL_SEC: int = 86400
    STM_TRIGGER_MSGS: int = 8
    STM_TRIGGER_TOKENS: int = 2000

    SEFARIA_API_URL: str = "http://localhost:8000/api/"
    SEFARIA_API_KEY: Optional[str] = None
    SEFARIA_CACHE_TTL: int = 60
    
    CORS_ORIGINS: str = "http://localhost:5173"
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: int = 10  # requests per window
    RATE_LIMIT_WINDOW: int = 60   # seconds
    RATE_LIMIT_LLM: int = 5       # stricter limit for LLM endpoints
    
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    SEFARIA_MCP_URL: str = "http://sefaria.org:8088/sse"
    SEFARIA_MCP_TIMEOUT_SEC: int = 30
    JWT_SECRET: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRES_MINUTES: int = 60
    API_KEY_SECRET: str = "change-me-api-key-secret"
    JWT_REFRESH_TOKEN_EXPIRES_DAYS: int = 30
    REFRESH_TOKEN_COOKIE_NAME: str = "astra_refresh_token"
    REFRESH_TOKEN_COOKIE_PATH: str = "/api"
    REFRESH_TOKEN_COOKIE_SAMESITE: str = "lax"
    REFRESH_TOKEN_COOKIE_SECURE: bool = True
    REFRESH_TOKEN_COOKIE_HTTPONLY: bool = True
