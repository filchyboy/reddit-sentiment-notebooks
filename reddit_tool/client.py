"""Reddit API client wrapper with rate limiting and error handling."""

import time
from typing import List, Dict, Any, Optional, Iterator
import logging

import praw
import pandas as pd
from praw.models import Submission, Comment
from rich.console import Console

from .auth import get_reddit_auth
from .config import load_config

console = Console()
logger = logging.getLogger(__name__)


class RedditClient:
    """Reddit API client with OAuth, rate limiting, and error handling."""
    
    def __init__(self):
        self.config = load_config()
        self.auth = get_reddit_auth()
        self._reddit = None
        self._last_request_time = 0
        self._min_request_interval = 60.0 / self.config["requests_per_minute"]
    
    def _get_reddit_instance(self) -> praw.Reddit:
        """Get authenticated Reddit instance."""
        if self._reddit is None:
            access_token = self.auth.get_valid_access_token()
            if not access_token:
                raise Exception(
                    "No valid access token available. Please run 'python -m reddit_tool auth' first."
                )
            
            self._reddit = praw.Reddit(
                client_id=self.config["reddit_client_id"],
                client_secret=self.config["reddit_client_secret"],
                user_agent=self.config["user_agent"],
                access_token=access_token
            )
        
        return self._reddit
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _retry_on_failure(self, func, max_retries: int = 3, backoff_factor: float = 2.0):
        """Retry function on failure with exponential backoff."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = backoff_factor ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def fetch_submissions(
        self, 
        subreddit: str, 
        limit: int = 100, 
        listing_type: str = "new",
        time_filter: str = "all"
    ) -> List[Dict[str, Any]]:
        """Fetch submissions from a subreddit."""
        reddit = self._get_reddit_instance()
        
        def _fetch():
            sub = reddit.subreddit(subreddit)
            
            if listing_type == "new":
                submissions = sub.new(limit=limit)
            elif listing_type == "hot":
                submissions = sub.hot(limit=limit)
            elif listing_type == "top":
                submissions = sub.top(limit=limit, time_filter=time_filter)
            elif listing_type == "rising":
                submissions = sub.rising(limit=limit)
            else:
                raise ValueError(f"Invalid listing_type: {listing_type}")
            
            results = []
            for submission in submissions:
                results.append(self._submission_to_dict(submission))
            
            return results
        
        return self._retry_on_failure(_fetch)
    
    def fetch_comments(
        self, 
        subreddit: str, 
        limit: int = 100,
        submission_limit: int = 50,
        top_level_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch comments from submissions in a subreddit."""
        reddit = self._get_reddit_instance()
        
        def _fetch():
            sub = reddit.subreddit(subreddit)
            submissions = sub.new(limit=submission_limit)
            
            results = []
            comment_count = 0
            
            for submission in submissions:
                if comment_count >= limit:
                    break
                
                # Replace "more comments" with actual comments (limited to avoid deep recursion)
                submission.comments.replace_more(limit=0)
                
                comments_to_process = (
                    submission.comments.list()[:10] if top_level_only 
                    else submission.comments.list()
                )
                
                for comment in comments_to_process:
                    if comment_count >= limit:
                        break
                    
                    if isinstance(comment, Comment):
                        results.append(self._comment_to_dict(comment, submission))
                        comment_count += 1
            
            return results
        
        return self._retry_on_failure(_fetch)
    
    def _submission_to_dict(self, submission: Submission) -> Dict[str, Any]:
        """Convert PRAW Submission to dictionary."""
        return {
            "id": submission.id,
            "created_utc": submission.created_utc,
            "title": submission.title or "",
            "selftext": submission.selftext or "",
            "author": str(submission.author) if submission.author else "[deleted]",
            "score": submission.score,
            "num_comments": submission.num_comments,
            "permalink": f"https://reddit.com{submission.permalink}",
            "url": submission.url or "",
            "subreddit": submission.subreddit.display_name,
            "upvote_ratio": getattr(submission, "upvote_ratio", None),
            "is_self": submission.is_self,
            "link_flair_text": submission.link_flair_text,
            "over_18": submission.over_18,
        }
    
    def _comment_to_dict(self, comment: Comment, submission: Submission) -> Dict[str, Any]:
        """Convert PRAW Comment to dictionary."""
        return {
            "id": comment.id,
            "link_id": submission.id,
            "parent_id": comment.parent_id,
            "created_utc": comment.created_utc,
            "body": comment.body or "",
            "author": str(comment.author) if comment.author else "[deleted]",
            "score": comment.score,
            "permalink": f"https://reddit.com{comment.permalink}",
            "subreddit": comment.subreddit.display_name,
            "is_submitter": comment.is_submitter,
            "stickied": comment.stickied,
            "submission_title": submission.title or "",
            "submission_author": str(submission.author) if submission.author else "[deleted]",
        }
    
    def test_connection(self) -> bool:
        """Test Reddit API connection."""
        try:
            reddit = self._get_reddit_instance()
            # Simple test - get user info
            user = reddit.user.me()
            console.print(f"✅ Connected to Reddit as: {user.name}")
            return True
        except Exception as e:
            console.print(f"❌ Reddit connection failed: {e}")
            return False


def create_dataframes(
    submissions: List[Dict[str, Any]], 
    comments: List[Dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create pandas DataFrames from submission and comment data."""
    
    # Create submissions DataFrame
    if submissions:
        submissions_df = pd.DataFrame(submissions)
        # Convert timestamp to datetime
        submissions_df["created_datetime"] = pd.to_datetime(
            submissions_df["created_utc"], unit="s"
        )
    else:
        submissions_df = pd.DataFrame()
    
    # Create comments DataFrame
    if comments:
        comments_df = pd.DataFrame(comments)
        # Convert timestamp to datetime
        comments_df["created_datetime"] = pd.to_datetime(
            comments_df["created_utc"], unit="s"
        )
    else:
        comments_df = pd.DataFrame()
    
    return submissions_df, comments_df
