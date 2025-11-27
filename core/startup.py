from contextlib import asynccontextmanager
import logging
import httpx
import redis.asyncio as redis
from fastapi import FastAPI
from typing import Any, Dict, List, Optional

from .settings import Settings
from .logging_config import setup_logging
from brain_service.services.sefaria_service import SefariaService
from brain_service.services.sefaria_index_service import SefariaIndexService
from brain_service.services.sefaria_mcp_service import SefariaMCPService
from brain_service.services.memory_service import MemoryService
from brain_service.services.summary_service import SummaryService
from brain_service.services.llm_service import LLMService
from brain_service.services.chat_service import ChatService
from brain_service.services.study import fetch_study_config, register_study_config_listener
from brain_service.services.study_service import StudyService
from brain_service.services.config_service import ConfigService
from brain_service.services.lexicon_service import LexiconService
from brain_service.services.session_service import SessionService
from brain_service.services.translation_service import TranslationService
from brain_service.services.wiki_service import WikiService
from brain_service.services.xp_service import XpService
from brain_service.services.achievement_service import AchievementService
from brain_service.services.navigation_service import NavigationService
from domain.chat.tools import ToolRegistry
from .rate_limiting import setup_rate_limiter
from .database import create_engine, create_session_factory, shutdown_engine
from brain_service.models.db import Base
from brain_service.services.user_service import UserService
from brain_service.services.auth_service import AuthService
from .config_loader import ensure_config_root

ensure_config_root()
from config import get_config
from sqlalchemy import text

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load legacy .toml config
    get_config(force_reload=True)

    # Load settings and configure logging
    settings = Settings()
    setup_logging(settings)
    app.state.settings = settings

    # Database
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
        await conn.execute(
            text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number VARCHAR(32)"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_phone_number ON users (phone_number) WHERE phone_number IS NOT NULL"
            )
        )
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    refresh_token_hash TEXT NOT NULL,
                    ip_address VARCHAR(64),
                    user_agent VARCHAR(512),
                    expires_at TIMESTAMPTZ NOT NULL,
                    last_used_at TIMESTAMPTZ,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_user_sessions_user_id ON user_sessions (user_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_user_sessions_expires_at ON user_sessions (expires_at)"
            )
        )
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS user_login_events (
                    id UUID PRIMARY KEY,
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    username VARCHAR(64),
                    success BOOLEAN NOT NULL DEFAULT FALSE,
                    ip_address VARCHAR(64),
                    user_agent VARCHAR(512),
                    reason VARCHAR(128),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_user_login_events_user_id ON user_login_events (user_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_user_login_events_created_at ON user_login_events (created_at)"
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
        refresh_token_expires_days=settings.JWT_REFRESH_TOKEN_EXPIRES_DAYS,
    )

    if not await app.state.user_service.has_admin():
        logger.warning(
            "No admin user detected. Use the bootstrap endpoint or CLI to create one."
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

    # Ensure XP table exists
    async with app.state.db_engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS user_xp (
                    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    xp_total BIGINT NOT NULL DEFAULT 0,
                    level INT NOT NULL DEFAULT 1,
                    xp_in_level BIGINT NOT NULL DEFAULT 0,
                    xp_to_next BIGINT NOT NULL DEFAULT 300,
                    last_level_up_at TIMESTAMPTZ NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION user_xp_set_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                  NEW.updated_at = NOW();
                  RETURN NEW;
                END;
                $$ language 'plpgsql';
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_user_xp_set_updated_at'
                  ) THEN
                    CREATE TRIGGER trg_user_xp_set_updated_at
                    BEFORE UPDATE ON user_xp
                    FOR EACH ROW EXECUTE FUNCTION user_xp_set_updated_at();
                  END IF;
                END$$;
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS user_achievement_stats (
                    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    discipline_total BIGINT NOT NULL DEFAULT 0,
                    lexicon_total BIGINT NOT NULL DEFAULT 0,
                    rambam_refs BIGINT NOT NULL DEFAULT 0,
                    daf_refs BIGINT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_user_achievements_updated_at') THEN
                    CREATE TRIGGER trg_user_achievements_updated_at
                    BEFORE UPDATE ON user_achievement_stats
                    FOR EACH ROW EXECUTE PROCEDURE user_xp_set_updated_at();
                  END IF;
                END$$;
                """
            )
        )

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

    # Instantiate wiki service
    app.state.wiki_service = WikiService(
        http_client=app.state.http_client,
        redis_client=app.state.redis_client,
        cache_ttl_sec=settings.SEFARIA_CACHE_TTL,  # Reuse same TTL
        max_chars=4000,
        top_k=3
    )

    # XP & achievements services
    app.state.achievement_service = AchievementService(app.state.db_session_factory)
    app.state.xp_service = XpService(app.state.redis_client, app.state.db_session_factory, app.state.achievement_service)

    # Instantiate other services
    app.state.sefaria_service = SefariaService(
        http_client=app.state.http_client, 
        redis_client=app.state.redis_client,
        sefaria_api_url=settings.SEFARIA_API_URL,
        sefaria_api_key=settings.SEFARIA_API_KEY,
        cache_ttl_sec=settings.SEFARIA_CACHE_TTL
    )

    app.state.sefaria_mcp_service = None
    try:
        if settings.SEFARIA_MCP_URL:
            app.state.sefaria_mcp_service = SefariaMCPService(
                endpoint=settings.SEFARIA_MCP_URL,
                timeout=float(settings.SEFARIA_MCP_TIMEOUT_SEC),
            )
            logger.info(
                "Sefaria MCP service ready",
                extra={"endpoint": settings.SEFARIA_MCP_URL},
            )
    except Exception as exc:
        logger.warning(
            "Sefaria MCP service disabled: %s",
            exc,
            exc_info=True,
        )
        app.state.sefaria_mcp_service = None

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

    # Instantiate navigation service
    app.state.navigation_service = NavigationService(
        redis_client=app.state.redis_client,
        sefaria_service=app.state.sefaria_service
    )


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

    if app.state.sefaria_mcp_service:
        sefaria_text_search_schema = {
            "type": "function",
            "function": {
                "name": "sefaria_text_search",
                "description": "Full-text search across the Sefaria library. Prefer Hebrew keywords when possible; use filters to narrow by category.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search phrase to look for within Sefaria texts (Hebrew recommended).",
                        },
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of category filter paths (e.g., ['Tanakh', 'Commentary']).",
                        },
                        "size": {
                            "type": "integer",
                            "description": "Maximum number of matches to return (default 10).",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

        async def sefaria_text_search_handler(
            query: str,
            filters: Optional[List[str]] = None,
            size: int = 10,
            session_id: str = "unknown",
        ) -> Dict[str, Any]:
            result = await app.state.sefaria_mcp_service.text_search(
                query=query,
                filters=filters,
                size=size,
            )
            hits = None
            data = result.get("data")
            if isinstance(data, dict):
                hits_field = data.get("hits") or data.get("results")
                if isinstance(hits_field, list):
                    hits = len(hits_field)
            logger.info(
                "sefaria_text_search completed",
                extra={
                    "query": query,
                    "filters": filters,
                    "size": size,
                    "hits": hits,
                    "session_id": session_id,
                },
            )
            result["session_id"] = session_id
            return result

        app.state.tool_registry.register(
            name="sefaria_text_search",
            handler=sefaria_text_search_handler,
            schema=sefaria_text_search_schema,
        )

        sefaria_semantic_search_schema = {
            "type": "function",
            "function": {
                "name": "sefaria_semantic_search",
                "description": "Semantic search over English-encoded sources to surface conceptually related passages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "English description of the idea you want to find in Sefaria.",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional metadata filters (e.g., document_categories, authors, eras, topics, places).",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

        async def sefaria_semantic_search_handler(
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            session_id: str = "unknown",
        ) -> Dict[str, Any]:
            result = await app.state.sefaria_mcp_service.english_semantic_search(
                query=query,
                filters=filters,
            )
            matches = None
            data = result.get("data")
            if isinstance(data, dict):
                matches_field = data.get("results") or data.get("hits")
                if isinstance(matches_field, list):
                    matches = len(matches_field)
            logger.info(
                "sefaria_semantic_search completed",
                extra={
                    "query": query,
                    "filters": filters,
                    "matches": matches,
                    "session_id": session_id,
                },
            )
            result["session_id"] = session_id
            return result

        app.state.tool_registry.register(
            name="sefaria_semantic_search",
            handler=sefaria_semantic_search_handler,
            schema=sefaria_semantic_search_schema,
        )

        sefaria_topic_details_schema = {
            "type": "function",
            "function": {
                "name": "sefaria_topic_details",
                "description": "Retrieve structured information about a Sefaria topic, including summaries, linked sources, and related topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_slug": {
                            "type": "string",
                            "description": "Topic slug from Sefaria (e.g., 'shabbat', 'teshuvah').",
                        },
                        "with_links": {
                            "type": "boolean",
                            "description": "Include curated topical links (default false).",
                        },
                        "with_refs": {
                            "type": "boolean",
                            "description": "Include source references attached to the topic (default false).",
                        },
                    },
                    "required": ["topic_slug"],
                },
            },
        }

        async def sefaria_topic_details_handler(
            topic_slug: str,
            with_links: Optional[bool] = None,
            with_refs: Optional[bool] = None,
            session_id: str = "unknown",
        ) -> Dict[str, Any]:
            result = await app.state.sefaria_mcp_service.get_topic_details(
                topic_slug=topic_slug,
                with_links=with_links,
                with_refs=with_refs,
            )
            data = result.get("data")
            links_count = refs_count = None
            if isinstance(data, dict):
                links = data.get("links")
                refs = data.get("refs")
                if isinstance(links, list):
                    links_count = len(links)
                if isinstance(refs, list):
                    refs_count = len(refs)
            logger.info(
                "sefaria_topic_details completed",
                extra={
                    "topic_slug": topic_slug,
                    "with_links": with_links,
                    "with_refs": with_refs,
                    "links_count": links_count,
                    "refs_count": refs_count,
                    "session_id": session_id,
                },
            )
            result["session_id"] = session_id
            return result

        app.state.tool_registry.register(
            name="sefaria_topic_details",
            handler=sefaria_topic_details_handler,
            schema=sefaria_topic_details_schema,
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

    # Navigation tools
    navigate_to_ref_schema = {
        "type": "function",
        "function": {
            "name": "navigate_to_ref",
            "description": "Navigate focus reader to a specific text reference. Use when you want to show the user a different text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tref": {
                        "type": "string",
                        "description": "Text reference to navigate to (e.g., 'Shabbat 21b:1', 'Rashi on Genesis 1:1:1')"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional explanation for why you're navigating (e.g., 'to see the original source', 'for context')"
                    }
                },
                "required": ["tref"]
            }
        }
    }
    async def navigate_to_ref_handler(session_id: str = 'unknown', **kwargs):
        return await app.state.navigation_service.navigate_to_ref(session_id, **kwargs)
    
    app.state.tool_registry.register(
        name="navigate_to_ref",
        handler=navigate_to_ref_handler,
        schema=navigate_to_ref_schema
    )

    load_commentary_schema = {
        "type": "function",
        "function": {
            "name": "load_commentary_to_workbench",
            "description": "Load a commentary into the left or right workbench panel. Use to show relevant commentaries alongside the main text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "commentary_ref": {
                        "type": "string",
                        "description": "Commentary reference (e.g., 'Rashi on Shabbat 21b:1:1', 'Tosafot on Shabbat 21b:1:2')"
                    },
                    "panel": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which workbench panel to load the commentary into"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional explanation for loading this commentary"
                    }
                },
                "required": ["commentary_ref", "panel"]
            }
        }
    }
    async def load_commentary_handler(session_id: str = 'unknown', **kwargs):
        return await app.state.navigation_service.load_commentary_to_workbench(session_id, **kwargs)
    
    app.state.tool_registry.register(
        name="load_commentary_to_workbench",
        handler=load_commentary_handler,
        schema=load_commentary_schema
    )

    clear_workbench_schema = {
        "type": "function",
        "function": {
            "name": "clear_workbench_panel",
            "description": "Clear the specified workbench panel. Use to remove clutter or make space for new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "panel": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which workbench panel to clear"
                    }
                },
                "required": ["panel"]
            }
        }
    }
    async def clear_workbench_handler(session_id: str = 'unknown', **kwargs):
        return await app.state.navigation_service.clear_workbench_panel(session_id, **kwargs)
    
    app.state.tool_registry.register(
        name="clear_workbench_panel",
        handler=clear_workbench_handler,
        schema=clear_workbench_schema
    )

    print("Startup complete. Yielding to application.")
    yield

    # Shutdown logic
    print("Shutdown sequence initiated.")
    
    # Stop config service listener
    if hasattr(app.state, 'config_service'):
        await app.state.config_service.stop_listening()
    
    if getattr(app.state, "sefaria_mcp_service", None):
        await app.state.sefaria_mcp_service.close()
    await app.state.http_client.aclose()
    if app.state.redis_client:
        await app.state.redis_client.aclose()
    await shutdown_engine(getattr(app.state, "db_engine", None))
    print("Clients closed. Shutdown complete.")
