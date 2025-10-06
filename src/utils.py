import os
from dotenv import load_dotenv


def load_settings():
    # Loads environment variables and provides typed defaults
    load_dotenv(dotenv_path=os.path.join("config", ".env"), override=False)

    cfg = {
        "client_id": os.getenv("REDDIT_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "reddit-sentiment/0.1"),
        "subreddits": os.getenv("SUBREDDITS", "python").strip(),
        "fetch_limit": int(os.getenv("FETCH_LIMIT", "200")),
        "mode": os.getenv("MODE", "both").strip().lower(),
    }

    # Basic validation
    missing = [k for k in ["client_id","client_secret","user_agent"] if not cfg[k]]
    if missing:
        raise ValueError(f"Missing required Reddit credentials in .env: {missing}")
    if cfg["mode"] not in {"submissions","comments","both"}:
        raise ValueError("MODE must be one of: submissions|comments|both")
    return cfg
