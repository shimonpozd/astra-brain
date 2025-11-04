import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class ConfigService:
    """
    Configuration service with hot-reload functionality.
    
    Provides centralized configuration management with Redis pub/sub
    for real-time configuration updates across the application.
    """
    
    def __init__(self, redis_client: redis.Redis, config_channel: str = "astra_config_channel"):
        self.redis_client = redis_client
        self.config_channel = config_channel
        self._config_cache: Dict[str, Any] = {}
        self._listeners: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._pubsub = None
        self._listen_task = None
        
    async def get_config_section(self, section: str, default: Any = None) -> Any:
        """
        Get a configuration section.
        
        Args:
            section: Section name (e.g., "llm", "sefaria", "stm")
            default: Default value if section not found
            
        Returns:
            Configuration section data or default
        """
        if not self._config_cache:
            await self._load_config()
            
        return self._config_cache.get(section, default)
    
    async def update_config_section(self, section: str, data: Dict[str, Any]) -> bool:
        """
        Update a configuration section and broadcast changes.
        
        Args:
            section: Section name to update
            data: New configuration data
            
        Returns:
            True if update was successful
        """
        try:
            # Update local cache
            self._config_cache[section] = data
            
            # Broadcast change via Redis pub/sub
            message = {
                "type": "config_update",
                "section": section,
                "data": data
            }
            
            await self.redis_client.publish(
                self.config_channel,
                json.dumps(message, ensure_ascii=False)
            )
            
            logger.info("Configuration section updated", extra={
                "section": section,
                "keys": list(data.keys()) if isinstance(data, dict) else None
            })
            
            return True
            
        except Exception as e:
            logger.error("Failed to update configuration section", extra={
                "section": section,
                "error": str(e)
            })
            return False
    
    async def register_listener(self, section: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for configuration changes in a specific section.
        
        Args:
            section: Section name to listen for
            callback: Function to call when section is updated
        """
        self._listeners[section] = callback
        logger.debug("Configuration listener registered", extra={"section": section})
    
    async def start_listening(self) -> None:
        """
        Start listening for configuration updates via Redis pub/sub.
        """
        if self._listen_task:
            return
            
        try:
            self._pubsub = self.redis_client.pubsub()
            await self._pubsub.subscribe(self.config_channel)
            
            self._listen_task = asyncio.create_task(self._listen_for_updates())
            
            logger.info("Configuration service started listening", extra={
                "channel": self.config_channel
            })
            
        except Exception as e:
            logger.error("Failed to start configuration listener", extra={
                "error": str(e)
            })
    
    async def stop_listening(self) -> None:
        """
        Stop listening for configuration updates.
        """
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
            
        if self._pubsub:
            await self._pubsub.unsubscribe(self.config_channel)
            await self._pubsub.aclose()
            self._pubsub = None
            
        logger.info("Configuration service stopped listening")
    
    async def _load_config(self) -> None:
        """
        Load initial configuration from the central config system.
        
        This integrates with the existing config/__init__.py system.
        """
        try:
            # Import the central config system
            from config import get_config
            
            # Load the full configuration
            full_config = get_config(force_reload=True)
            
            if isinstance(full_config, dict):
                self._config_cache = full_config.copy()
                logger.info("Configuration loaded from central system", extra={
                    "sections": list(full_config.keys())
                })
            else:
                logger.warning("Central config returned non-dict, using empty config")
                self._config_cache = {}
                
        except Exception as e:
            logger.error("Failed to load central configuration", extra={
                "error": str(e)
            })
            self._config_cache = {}
    
    async def _listen_for_updates(self) -> None:
        """
        Listen for configuration updates via Redis pub/sub.
        """
        try:
            async for message in self._pubsub.listen():
                if message["type"] != "message":
                    continue
                    
                try:
                    raw_data = message["data"]
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode("utf-8", errors="replace")

                    data = json.loads(raw_data)
                    
                    if data.get("type") == "config_update":
                        section = data.get("section")
                        section_data = data.get("data")
                        
                        if section and section_data is not None:
                            # Update local cache
                            self._config_cache[section] = section_data
                            
                            # Notify listeners
                            if section in self._listeners:
                                try:
                                    await self._listeners[section](section_data)
                                except Exception as e:
                                    logger.error("Configuration listener failed", extra={
                                        "section": section,
                                        "error": str(e)
                                    })
                            
                            logger.info("Configuration updated from pub/sub", extra={
                                "section": section
                            })
                            
                except json.JSONDecodeError as e:
                    if raw_data == "config_updated":
                        logger.info("Legacy config update notification received, reloading configuration")
                        await self.reload_config()
                    else:
                        logger.warning("Invalid JSON in config update message", extra={
                            "error": str(e)
                        })
                except Exception as e:
                    logger.error("Error processing config update", extra={
                        "error": str(e)
                    })
                    
        except asyncio.CancelledError:
            logger.debug("Configuration listener cancelled")
            raise
        except Exception as e:
            logger.error("Configuration listener error", extra={
                "error": str(e)
            })
    
    async def reload_config(self) -> bool:
        """
        Force reload configuration from the central system.
        
        Returns:
            True if reload was successful
        """
        try:
            await self._load_config()
            
            # Notify all listeners about the reload
            for section, callback in self._listeners.items():
                section_data = self._config_cache.get(section)
                if section_data is not None:
                    try:
                        await callback(section_data)
                    except Exception as e:
                        logger.error("Configuration listener failed during reload", extra={
                            "section": section,
                            "error": str(e)
                        })
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to reload configuration", extra={
                "error": str(e)
            })
            return False
    
    def get_cached_config(self) -> Dict[str, Any]:
        """
        Get the current cached configuration.
        
        Returns:
            Current configuration cache
        """
        return self._config_cache.copy()



























