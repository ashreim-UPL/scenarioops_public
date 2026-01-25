import functools
import json
import logging
import time
import traceback
import uuid
from typing import Any, Callable, TypeVar, ParamSpec



# Type variables for decorator
P = ParamSpec("P")
R = TypeVar("R")

class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data) # type: ignore
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def configure_logging() -> None:
    """Configures the root logger for JSON output."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)

def observe(operation_id: bool = True) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for RED metrics (Rate, Error, Duration).
    
    Args:
        operation_id: If True, generates a unique ID for the operation context.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger = get_logger(func.__module__)
            op_id = str(uuid.uuid4()) if operation_id else "n/a"
            start_time = time.perf_counter()
            
            # Rate (Entry)
            logger.info(
                f"Starting {func.__name__}", 
                extra={"extra_data": {"event": "start", "operation_id": op_id}}
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                # Duration & Success
                logger.info(
                    f"Completed {func.__name__}",
                    extra={
                        "extra_data": {
                            "event": "complete", 
                            "operation_id": op_id, 
                            "duration_seconds": duration,
                            "status": "success"
                        }
                    }
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                # Error
                logger.error(
                    f"Failed {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_data": {
                            "event": "error", 
                            "operation_id": op_id, 
                            "duration_seconds": duration,
                            "status": "error",
                            "error_type": type(e).__name__
                        }
                    }
                )
                raise
        return wrapper
    return decorator
