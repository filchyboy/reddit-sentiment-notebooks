from typing import List, Dict, Any
import pandas as pd
from praw.models import Submission, Comment
from .reddit_client import get_reddit
from .utils import load_settings


def fetch_submissions(subs: List[str], limit: int) -> pd.DataFrame:
    reddit = get_reddit()
    rows = []
    for name in subs:
        for s in reddit.subreddit(name).new(limit=limit):
            assert isinstance(s, Submission)
            rows.append({
                "type": "submission",
                "subreddit": s.subreddit.display_name,
                "id": s.id,
                "created_utc": s.created_utc,
                "author": str(s.author) if s.author else None,
                "title": s.title or "",
                "selftext": s.selftext or "",
                "url": s.url or "",
                "score": s.score,
                "num_comments": s.num_comments,
                "permalink": f"https://reddit.com{s.permalink}",
            })
    return pd.DataFrame(rows)


def fetch_comments(subs: List[str], limit: int) -> pd.DataFrame:
    reddit = get_reddit()
    rows: List[Dict[str, Any]] = []
    for name in subs:
        # Pull latest submissions then expand comments—keeps within API patterns
        for s in reddit.subreddit(name).new(limit=min(limit, 200)):  # cap to avoid deep crawls
            s.comments.replace_more(limit=0)
            for c in s.comments.list():
                assert isinstance(c, Comment)
                rows.append({
                    "type": "comment",
                    "subreddit": s.subreddit.display_name,
                    "submission_id": s.id,
                    "comment_id": c.id,
                    "created_utc": c.created_utc,
                    "author": str(c.author) if c.author else None,
                    "body": c.body or "",
                    "score": c.score,
                    "permalink": f"https://reddit.com{c.permalink}",
                    "submission_title": s.title or "",
                })
                if len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def fetch_all(mode: str, subreddits: str, limit: int) -> pd.DataFrame:
    subs = [s.strip() for s in subreddits.split("+") if s.strip()]
    frames = []
    if mode in {"submissions","both"}:
        frames.append(fetch_submissions(subs, limit))
    if mode in {"comments","both"}:
        frames.append(fetch_comments(subs, limit))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
