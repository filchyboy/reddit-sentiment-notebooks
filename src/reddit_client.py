import praw
from .utils import load_settings


def get_reddit():
    cfg = load_settings()
    reddit = praw.Reddit(
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
        user_agent=cfg["user_agent"],
    )
    # simple auth check (doesn't make a request)
    return reddit
