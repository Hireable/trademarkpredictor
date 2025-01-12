import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

from src.config.settings import get_settings

class CustomFormatter(logging.Formatter):
    """
    Custom formatter adding color to log levels and timestamps.
    """
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Purple
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Add color to level name if terminal supports it
        if sys.stdout.isatty():  # Only apply colors when outputting to terminal
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

def get_log_formatter(format_type: str = "detailed") -> logging.Formatter:
    """
    Get a log formatter based on the specified format type.
    
    Args:
        format_type: Type of format ("basic", "detailed", or "json")
        
    Returns:
        Configured logging.Formatter instance
    """
    if format_type == "basic":
        return CustomFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    elif format_type == "json":
        return logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","message":"%(message)s"}'
        )
    else:  # detailed
        return CustomFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def setup_file_handler(
    log_dir: Union[str, Path],
    logger_name: str,
    max_bytes: int,
    backup_count: int
) -> RotatingFileHandler:
    """
    Set up a rotating file handler for logging.
    
    Args:
        log_dir: Directory for log files
        logger_name: Name of the logger (used for file naming)
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        
    Returns:
        Configured RotatingFileHandler
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{logger_name}.log"
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes * 1024 * 1024,  # Convert MB to bytes
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    # Use detailed formatting for file logs
    handler.setFormatter(get_log_formatter("detailed"))
    return handler

def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_type: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with both console and file output.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional override for log level
        format_type: Optional override for format type
        
    Returns:
        Configured logging.Logger instance
    """
    settings = get_settings()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Only set up handlers if they haven't been set up already
    if not logger.handlers:
        # Set log level
        log_level = level or settings.log_level
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        format_type = format_type or settings.log_format
        console_handler.setFormatter(get_log_formatter(format_type))
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = setup_file_handler(
            settings.log_dir,
            name.split('.')[-1],  # Use the last part of the logger name
            settings.max_log_size_mb,
            settings.log_backup_count
        )
        logger.addHandler(file_handler)
        
        logger.debug(f"Logger '{name}' initialized with level {log_level}")
    
    return logger

# Initialize a default logger for the utils package
logger = setup_logger(__name__)