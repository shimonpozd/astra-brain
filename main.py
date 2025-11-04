from fastapi import FastAPI
import sys
import os

# Add the brain_service directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.startup import lifespan
from core.middleware import logging_middleware, setup_cors_middleware
from core.exceptions import setup_exception_handlers
from core.settings import Settings

from api import admin, chat, study, actions, tts, audio, auth, users

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(
    title="Brain Service v2", 
    version="24.9.0", 
    lifespan=lifespan
)

# Load settings for middleware configuration
settings = Settings()

# Setup CORS middleware (must be added before other middleware)
setup_cors_middleware(app, settings.CORS_ORIGINS)

# Add logging middleware
app.middleware("http")(logging_middleware)

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(study.router, prefix="/api/study", tags=["study"])
app.include_router(actions.router, prefix="/api/actions", tags=["actions"])
app.include_router(tts.router, prefix="/api/tts", tags=["tts"])
app.include_router(audio.router, prefix="/api", tags=["audio"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(users.router, prefix="/api", tags=["users"])

@app.get("/health")
async def health():
    """
    Health check endpoint to confirm the service is running.
    """
    return {"status": "healthy", "service": "brain_v2"}

# Application is now fully configured with:
# - CORS middleware for cross-origin requests
# - Logging middleware for request tracking
# - Exception handlers for error management
# - All API routers included
