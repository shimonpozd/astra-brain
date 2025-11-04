import logging
from typing import Any, Dict


def _ensure_handler(logger: logging.Logger) -> None:
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(service)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class _ServiceLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: Dict[str, Any]):
        kwargs.setdefault("extra", {}).setdefault("service", self.extra.get("service", "brain"))
        return msg, kwargs


def get_logger(name: str, service: str = "brain") -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    _ensure_handler(logger)
    return _ServiceLoggerAdapter(logger, {"service": service})
