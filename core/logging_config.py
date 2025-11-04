import logging
import json
import time
import contextvars

# Define context variables
request_id_var = contextvars.ContextVar("request_id", default=None)
session_id_var = contextvars.ContextVar("session_id", default=None)

class JsonContextFormatter(logging.Formatter):
    """A custom formatter to add context variables to the log record."""

    def format(self, record: logging.LogRecord) -> str:
        # Create a dictionary with standard attributes
        log_record = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Add context variables if they are set
        request_id = request_id_var.get()
        if request_id:
            log_record['request_id'] = request_id
        
        session_id = session_id_var.get()
        if session_id:
            log_record['session_id'] = session_id

        # Add any extra fields passed to the logger
        if hasattr(record, 'extra') and isinstance(record.extra, dict):
            log_record.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)

def setup_logging(settings):
    """Set up the root logger with a JSON formatter."""
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set the logging level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Create a stream handler (writes to stderr by default)
    handler = logging.StreamHandler()

    # Set the formatter based on settings
    if settings.LOG_JSON:
        formatter = JsonContextFormatter()
    else:
        # Basic formatter for non-JSON logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger.addHandler(handler)

    # Suppress verbose logs from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Logging configured successfully.")
