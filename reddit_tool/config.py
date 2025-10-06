"""Configuration management for reddit sentiment tool."""

import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
    
    # Also check config/.env for backward compatibility
    config_env = Path("config/.env")
    if config_env.exists():
        load_dotenv(config_env)
    
    config = {
        # Reddit OAuth
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", "").strip(),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        "funnel_host": os.getenv("FUNNEL_HOST", "").strip(),
        "redirect_uri": os.getenv("REDIRECT_URI", "").strip(),
        "user_agent": os.getenv("USER_AGENT", "reddit-sentiment-tool:v0.1 (by /u/unknown)").strip(),
        
        # App behavior
        "subreddit_default": os.getenv("SUBREDDIT_DEFAULT", "python").strip(),
        "data_dir": Path(os.getenv("DATA_DIR", "./data")),
        "report_dir": Path(os.getenv("REPORT_DIR", "./reports")),
        "use_transformer": os.getenv("USE_TRANSFORMER", "false").lower() in ("true", "1", "yes"),
        
        # Logging
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
        
        # Rate limiting
        "requests_per_minute": int(os.getenv("REQUESTS_PER_MINUTE", "60")),
    }
    
    # Validate required fields
    required_fields = ["reddit_client_id", "reddit_client_secret", "user_agent"]
    missing_fields = [field for field in required_fields if not config[field]]
    
    if missing_fields:
        raise ValueError(
            f"Missing required configuration: {missing_fields}. "
            "Please check your .env file or environment variables."
        )
    
    # Ensure directories exist
    config["data_dir"].mkdir(parents=True, exist_ok=True)
    config["report_dir"].mkdir(parents=True, exist_ok=True)
    (config["data_dir"] / "raw").mkdir(exist_ok=True)
    (config["data_dir"] / "processed").mkdir(exist_ok=True)
    
    return config


def get_secrets_dir() -> Path:
    """Get the secrets directory path."""
    secrets_dir = Path("secrets")
    secrets_dir.mkdir(exist_ok=True)
    return secrets_dir


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir
