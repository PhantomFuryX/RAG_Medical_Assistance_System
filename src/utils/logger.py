import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import datetime
from logging.config import dictConfig

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file paths
GENERAL_LOG_FILE = os.path.join(LOGS_DIR, "medical_assistant.log")
ERROR_LOG_FILE = os.path.join(LOGS_DIR, "errors.log")
API_LOG_FILE = os.path.join(LOGS_DIR, "api.log")
DB_LOG_FILE = os.path.join(LOGS_DIR, "database.log")
LLM_LOG_FILE = os.path.join(LOGS_DIR, "llm.log")
SERVER_LOG_FILE = os.path.join(LOGS_DIR, "server.log")
ACCESS_LOG_FILE = os.path.join(LOGS_DIR, "access.log")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(filename)s::%(funcName)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Configure root logger
def configure_root_logger():
    """Configure the root logger with console and file handlers"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates when reconfiguring
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(console_handler)
    
    # File handler (rotating by size)
    file_handler = RotatingFileHandler(
        GENERAL_LOG_FILE, 
        maxBytes=MAX_LOG_SIZE, 
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(file_handler)
    
    # Error file handler (only logs errors and above)
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE, 
        maxBytes=MAX_LOG_SIZE, 
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(error_handler)
    
    return root_logger

# Get logger for a specific module
def get_logger(module_name, log_file=None):
    """
    Get a logger for a specific module with optional dedicated log file
    
    Args:
        module_name: Name of the module (e.g., 'api', 'db', 'llm')
        log_file: Optional path to a dedicated log file for this module
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)
    
    # If a specific log file is provided, add a file handler for it
    if log_file and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        handler = RotatingFileHandler(
            log_file, 
            maxBytes=MAX_LOG_SIZE, 
            backupCount=3
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
    
    return logger

# Specialized loggers
def get_api_logger():
    """Get logger for API-related modules"""
    return get_logger("api", API_LOG_FILE)

def get_db_logger():
    """Get logger for database-related modules"""
    return get_logger("db", DB_LOG_FILE)

def get_llm_logger():
    """Get logger for LLM-related modules"""
    return get_logger("llm", LLM_LOG_FILE)

def get_server_logger():
    """Get logger for server-related modules"""
    return get_logger("server", SERVER_LOG_FILE)

def get_access_logger():
    """Get logger for access logs"""
    return get_logger("access", ACCESS_LOG_FILE)

# Create a log configuration dictionary for Uvicorn
def get_uvicorn_log_config():
    """
    Get a logging configuration dictionary for Uvicorn
    
    Returns:
        Dictionary with logging configuration
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": LOG_FORMAT,
                "datefmt": DATE_FORMAT
            },
            "access": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": DATE_FORMAT
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "server_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "default",
                "filename": SERVER_LOG_FILE,
                "maxBytes": MAX_LOG_SIZE,
                "backupCount": 5
            },
            "access_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "access",
                "filename": ACCESS_LOG_FILE,
                "maxBytes": MAX_LOG_SIZE,
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "default",
                "filename": ERROR_LOG_FILE,
                "maxBytes": MAX_LOG_SIZE,
                "backupCount": 5
            }
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "server_file"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "server_file", "error_file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "access_file"],
                "propagate": False
            },
            "watchfiles": {
                "level": "WARNING",  # Reduce noise from watchfiles
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "server_file", "error_file"]
        }
    }

# Save the Uvicorn log config to a file
def save_uvicorn_log_config(filename="log_config.json"):
    """Save the Uvicorn log config to a JSON file"""
    import json
    
    config = get_uvicorn_log_config()
    
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)
    
    return filename
# Daily log rotation
def setup_daily_log_rotation():
    """Set up daily log rotation for all log files"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_files = [GENERAL_LOG_FILE, ERROR_LOG_FILE, API_LOG_FILE, DB_LOG_FILE, LLM_LOG_FILE]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            # Create daily backup directory
            daily_dir = os.path.join(LOGS_DIR, "daily", today)
            os.makedirs(daily_dir, exist_ok=True)
            
            # Get base filename
            base_name = os.path.basename(log_file)
            
            # Copy to daily directory
            try:
                with open(log_file, 'r') as src, open(os.path.join(daily_dir, base_name), 'w') as dst:
                    dst.write(src.read())
            except Exception as e:
                print(f"Error rotating log {log_file}: {str(e)}")

def configure_logging():
    """Configure logging to prevent watchfiles feedback loop"""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "default",
                "filename": "logs/app.log",
                "maxBytes": 10485760,
                "backupCount": 5,
                "delay": True  # Only create the file when first log is written
            }
        },
        "loggers": {
            "watchfiles": {
                "level": "WARNING",  # Change this from INFO to WARNING
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
    
    dictConfig(log_config)
    
    
# Initialize logging
configure_root_logger()

# Example usage
if __name__ == "__main__":
    # Test the logger
    root_logger = logging.getLogger()
    root_logger.info("This is a test message from the root logger")
    root_logger.error("This is an error message from the root logger")
    
    api_logger = get_api_logger()
    api_logger.info("This is a test message from the API logger")
    
    db_logger = get_db_logger()
    db_logger.info("This is a test message from the DB logger")
    
    llm_logger = get_llm_logger()
    llm_logger.info("This is a test message from the LLM logger")