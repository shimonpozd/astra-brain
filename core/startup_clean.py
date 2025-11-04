from contextlib import asynccontextmanager
import httpx
import redis.asyncio as redis
from fastapi import FastAPI
from sqlalchemy import text

from .settings import Settings
from .logging_config import setup_logging
from ..services.sefaria_service import SefariaService
from ..services.sefaria_index_service import SefariaIndexService
from ..services.memory_service import MemoryService
from ..services.summary_service import SummaryService
from ..services.llm_service import LLMService
from ..services.chat_service import ChatService
from ..services.study_service import StudyService
from ..services.config_service import ConfigService
from ..services.lexicon_service import LexiconService
from ..services.session_service import SessionService
from ..services.translation_service import TranslationService
from ..domain.chat.tools import ToolRegistry
from .rate_limiting import setup_rate_limiter
from .database import create_engine, create_session_factory, shutdown_engine
from ..models.db import Base
from ..services.user_service import UserService
from ..services.auth_service import AuthService
from .config_loader import ensure_config_root

ensure_config_root()
from config import get_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load legacy .toml config
    get_config(force_reload=True)

    # Load settings and configure logging
    settings = Settings()
    setup_logging(settings)
    app.state.settings = settings

    app.state.db_engine = create_engine(settings.DATABASE_URL)
    app.state.db_session_factory = create_session_factory(app.state.db_engine)
    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_manually BOOLEAN NOT NULL DEFAULT FALSE"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ NULL"
            )
        )
    app.state.user_service = UserService(
        app.state.db_session_factory, encryption_secret=settings.API_KEY_SECRET
    )
    app.state.auth_service = AuthService(
        app.state.user_service,
        jwt_secret=settings.JWT_SECRET,
        jwt_algorithm=settings.JWT_ALGORITHM,
        jwt_expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRES_MINUTES,
    )

    # Initialize and store clients
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    try:
        app.state.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await app.state.redis_client.ping()
        print("Successfully connected to Redis.")
    except Exception as e:
        print(f"Could not connect to Redis: {e}")
        app.state.redis_client = None

    # Instantiate and load index service
    app.state.sefaria_index_service = SefariaIndexService(
        http_client=app.state.http_client,
        sefaria_api_url=settings.SEFARIA_API_URL,
        sefaria_api_key=settings.SEFARIA_API_KEY
    )
    await app.state.sefaria_index_service.load()

    # Instantiate config service
    app.state.config_service = ConfigService(
        redis_client=app.state.redis_client
    )
    await app.state.config_service.start_listening()

    initial_study_config = await fetch_study_config(app.state.config_service)

    # Instantiate lexicon service
    app.state.lexicon_service = LexiconService(
        http_client=app.state.http_client,
        redis_client=app.state.redis_client,
        sefaria_api_url=settings.SEFARIA_API_URL,
        cache_ttl_sec=settings.SEFARIA_CACHE_TTL
    )

    # Instantiate session service
    app.state.session_service = SessionService(
        redis_client=app.state.redis_client,
        user_service=app.state.user_service,
    )

    # Instantiate translation service
    app.state.translation_service = TranslationService(
        sefaria_service=None,  # Will be set after sefaria service is created
        llm_service=None  # Will be set after LLM service is created
    )

    # Instantiate other services
    app.state.sefaria_service = SefariaService(
        http_client=app.state.http_client, 
        redis_client=app.state.redis_client,
        sefaria_api_url=settings.SEFARIA_API_URL,
        sefaria_api_key=settings.SEFARIA_API_KEY,
        cache_ttl_sec=settings.SEFARIA_CACHE_TTL
    )

    # Instantiate LLM service
    app.state.llm_service = LLMService(
        http_client=app.state.http_client,
        memory_service=None,  # Will be set after memory service is created
        settings=settings
    )

    # Instantiate memory service with configuration
    config = get_config()
    app.state.memory_service = MemoryService(
        redis_client=app.state.redis_client,
        ttl_sec=settings.STM_TTL_SEC,
        config=config
    )

    # Instantiate summary service
    app.state.summary_service = SummaryService(
        llm_service=app.state.llm_service,
        config=config,
        redis_client=app.state.redis_client
    )

    # Update memory service with summary service
    app.state.memory_service.summary_service = app.state.summary_service

    # Update translation service with sefaria and LLM services
    app.state.translation_service.sefaria_service = app.state.sefaria_service
    app.state.translation_service.llm_service = app.state.llm_service

    # Setup rate limiting
    if settings.RATE_LIMIT_ENABLED and app.state.redis_client:
        setup_rate_limiter(
            redis_client=app.state.redis_client,
            default_limit=settings.RATE_LIMIT_DEFAULT,
            window_seconds=settings.RATE_LIMIT_WINDOW
        )

    # Instantiate and populate the tool registry
    app.state.tool_registry = ToolRegistry()

    # Instantiate chat service
    app.state.chat_service = ChatService(
        redis_client=app.state.redis_client,
        tool_registry=app.state.tool_registry,
        memory_service=app.state.memory_service,
        user_service=app.state.user_service,
    )

    # Instantiate study service
    app.state.study_service = StudyService(
        redis_client=app.state.redis_client,
        sefaria_service=app.state.sefaria_service,
        sefaria_index_service=app.state.sefaria_index_service,
        tool_registry=app.state.tool_registry,
        memory_service=app.state.memory_service,
        study_config=initial_study_config,
    )

    async def _on_study_config_update(new_config):
        app.state.study_service.update_study_config(new_config)

    await register_study_config_listener(app.state.config_service, _on_study_config_update)

    # Register Sefaria tools
    sefaria_get_text_schema = {
        "type": "function",
        "function": {
            "name": "sefaria_get_text",
            "description": "Get a specific text segment or commentary by its reference (ref). Returns both Hebrew and English text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tref": {
                        "type": "string",
                        "description": "The text reference, e.g., 'Genesis 1:1' or 'Rashi on Genesis 1:1:1'"
                    }
                },
                "required": ["tref"]
            }
        }
    }
    app.state.tool_registry.register(
        name="sefaria_get_text", 
        handler=app.state.sefaria_service.get_text,
        schema=sefaria_get_text_schema
    )

    sefaria_get_related_links_schema = {
        "type": "function",
        "function": {
            "name": "sefaria_get_related_links",
            "description": "Get related links and commentaries for a text reference. Useful for finding connected texts, commentaries, and cross-references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "The text reference to find links for, e.g., 'Shabbat 2a:1'"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of categories to filter by (e.g., ['Commentary', 'Mishnah'])"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of links to return (default: 120)",
                        "default": 120
                    }
                },
                "required": ["ref"]
            }
        }
    }
    app.state.tool_registry.register(
        name="sefaria_get_related_links",
        handler=app.state.sefaria_service.get_related_links,
        schema=sefaria_get_related_links_schema
    )


    # Lexicon tool for word definitions
    sefaria_get_lexicon_schema = {
        "type": "function",
        "function": {
            "name": "sefaria_get_lexicon",
            "description": "Get word definition and linguistic explanation from Sefaria lexicon. Use when user asks about meaning of Hebrew/Aramaic words.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "Hebrew or Aramaic word to look up"
                    }
                },
                "required": ["word"]
            }
        }
    }
    app.state.tool_registry.register(
        name="sefaria_get_lexicon",
        handler=app.state.lexicon_service.get_word_definition_for_tool,
        schema=sefaria_get_lexicon_schema
    )

    # Wikipedia search tool
    wikipedia_search_schema = {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for biographical information, historical context, and general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
    app.state.tool_registry.register(
        name="wikipedia_search",
        handler=app.state.wiki_service.search_wikipedia,
        schema=wikipedia_search_schema
    )

    # Chabadpedia search tool
    chabadpedia_search_schema = {
        "type": "function",
        "function": {
            "name": "chabadpedia_search",
            "description": "Search Chabadpedia for Chabad-Lubavitch related information and Chassidic perspective.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in Hebrew or English"
                    }
                },
                "required": ["query"]
            }
        }
    }
    app.state.tool_registry.register(
        name="chabadpedia_search",
        handler=app.state.wiki_service.search_chabadpedia,
        schema=chabadpedia_search_schema
    )

    print("Startup complete. Yielding to application.")
    yield

    # Shutdown logic
    print("Shutdown sequence initiated.")
    
    # Stop config service listener
    if hasattr(app.state, "config_service"):
        await app.state.config_service.stop_listening()
    
    await app.state.http_client.aclose()
    if app.state.redis_client:
        await app.state.redis_client.aclose()
    await shutdown_engine(getattr(app.state, "db_engine", None))
    print("Clients closed. Shutdown complete.")




