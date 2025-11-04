import time
import uuid
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from .logging_config import request_id_var

async def logging_middleware(request: Request, call_next) -> Response:
    """
    Middleware to add a request_id to each request and log the request/response.
    """
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Set the request ID in the context variable for other parts of the app to use
    request_id_var.set(request_id)

    start_time = time.time()
    
    # Proceed with the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000  # in milliseconds

    # Add custom headers to the response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = str(process_time)
    
    return response

def setup_cors_middleware(app, cors_origins: str):
    """
    Setup CORS middleware for the FastAPI app.
    
    Args:
        app: FastAPI application instance
        cors_origins: Comma-separated list of allowed origins
    """
    origins = [origin.strip() for origin in cors_origins.split(",")]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )
