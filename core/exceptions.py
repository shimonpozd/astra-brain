from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class ServiceError(Exception):
    """Base exception for service-related errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class NotFoundError(ServiceError):
    """Raised when a resource is not found."""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)

class BadInputError(ServiceError):
    """Raised for invalid user input."""
    def __init__(self, message: str = "Invalid input"):
        super().__init__(message, status_code=400)


def setup_exception_handlers(app: FastAPI):
    """Add custom exception handlers to the FastAPI app."""

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError):
        logger.warning(f"Service error occurred: {exc.message}", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=exc.status_code,
            content={"type": "error", "data": {"message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(request: Request, exc: Exception):
        logger.error(f"An unexpected error occurred: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"type": "error", "data": {"message": "An internal server error occurred."}},
        )
    
    logger.info("Exception handlers configured.")
