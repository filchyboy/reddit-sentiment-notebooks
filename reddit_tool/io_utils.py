"""I/O utilities for reading and writing data files."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from rich.console import Console

from .config import load_config

console = Console()


class DataManager:
    """Manages data I/O operations for the Reddit sentiment tool."""
    
    def __init__(self):
        self.config = load_config()
        self.data_dir = self.config["data_dir"]
        self.report_dir = self.config["report_dir"]
    
    def get_raw_data_path(self, subreddit: str, data_type: str = "submissions") -> Path:
        """Get path for raw data files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{subreddit}_{data_type}_{timestamp}.jsonl"
        path = self.data_dir / "raw" / subreddit
        path.mkdir(parents=True, exist_ok=True)
        return path / filename
    
    def get_processed_data_path(self, subreddit: str, data_type: str = "posts", format: str = "parquet") -> Path:
        """Get path for processed data files."""
        path = self.data_dir / "processed"
        path.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            return path / f"{subreddit}_{data_type}.parquet"
        elif format == "csv":
            return path / f"{subreddit}_{data_type}.csv"
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_report_path(self, subreddit: str, report_type: str = "ngrams") -> Path:
        """Get path for report files."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        return self.report_dir / f"{subreddit}_{report_type}.csv"
    
    def save_jsonl(self, data: List[Dict[str, Any]], filepath: Path) -> None:
        """Save data as JSONL (JSON Lines) format."""
        console.print(f"💾 Saving {len(data)} records to {filepath}")
        
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        
        console.print(f"✅ Saved to {filepath}")
    
    def load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load data from JSONL format."""
        if not filepath.exists():
            console.print(f"⚠️  File not found: {filepath}")
            return []
        
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        console.print(f"📖 Loaded {len(data)} records from {filepath}")
        return data
    
    def save_dataframe(self, df: pd.DataFrame, filepath: Path, format: str = "parquet") -> None:
        """Save DataFrame to file."""
        if df.empty:
            console.print(f"⚠️  Empty DataFrame, skipping save to {filepath}")
            return
        
        console.print(f"💾 Saving DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) to {filepath}")
        
        if format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            df.to_csv(filepath, index=False, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        console.print(f"✅ Saved to {filepath}")
    
    def load_dataframe(self, filepath: Path, format: str = None) -> pd.DataFrame:
        """Load DataFrame from file."""
        if not filepath.exists():
            console.print(f"⚠️  File not found: {filepath}")
            return pd.DataFrame()
        
        # Auto-detect format from extension if not specified
        if format is None:
            if filepath.suffix == ".parquet":
                format = "parquet"
            elif filepath.suffix == ".csv":
                format = "csv"
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        
        if format == "parquet":
            df = pd.read_parquet(filepath)
        elif format == "csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        console.print(f"📖 Loaded DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) from {filepath}")
        return df
    
    def save_raw_data(self, subreddit: str, submissions: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> Dict[str, Path]:
        """Save raw data to JSONL files."""
        saved_files = {}
        
        if submissions:
            submissions_path = self.get_raw_data_path(subreddit, "submissions")
            self.save_jsonl(submissions, submissions_path)
            saved_files["submissions"] = submissions_path
        
        if comments:
            comments_path = self.get_raw_data_path(subreddit, "comments")
            self.save_jsonl(comments, comments_path)
            saved_files["comments"] = comments_path
        
        return saved_files
    
    def save_processed_data(self, subreddit: str, posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> Dict[str, Path]:
        """Save processed DataFrames to Parquet and CSV."""
        saved_files = {}
        
        if not posts_df.empty:
            # Save as Parquet (primary format)
            parquet_path = self.get_processed_data_path(subreddit, "posts", "parquet")
            self.save_dataframe(posts_df, parquet_path, "parquet")
            saved_files["posts_parquet"] = parquet_path
            
            # Save as CSV (for compatibility)
            csv_path = self.get_processed_data_path(subreddit, "posts", "csv")
            self.save_dataframe(posts_df, csv_path, "csv")
            saved_files["posts_csv"] = csv_path
        
        if not comments_df.empty:
            # Save as Parquet (primary format)
            parquet_path = self.get_processed_data_path(subreddit, "comments", "parquet")
            self.save_dataframe(comments_df, parquet_path, "parquet")
            saved_files["comments_parquet"] = parquet_path
            
            # Save as CSV (for compatibility)
            csv_path = self.get_processed_data_path(subreddit, "comments", "csv")
            self.save_dataframe(comments_df, csv_path, "csv")
            saved_files["comments_csv"] = csv_path
        
        return saved_files
    
    def load_processed_data(self, subreddit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data for a subreddit."""
        posts_path = self.get_processed_data_path(subreddit, "posts", "parquet")
        comments_path = self.get_processed_data_path(subreddit, "comments", "parquet")
        
        posts_df = self.load_dataframe(posts_path) if posts_path.exists() else pd.DataFrame()
        comments_df = self.load_dataframe(comments_path) if comments_path.exists() else pd.DataFrame()
        
        return posts_df, comments_df
    
    def list_available_subreddits(self) -> List[str]:
        """List subreddits with processed data."""
        processed_dir = self.data_dir / "processed"
        if not processed_dir.exists():
            return []
        
        subreddits = set()
        for file in processed_dir.glob("*_posts.parquet"):
            subreddit = file.stem.replace("_posts", "")
            subreddits.add(subreddit)
        
        return sorted(list(subreddits))
    
    def get_file_info(self, filepath: Path) -> Dict[str, Any]:
        """Get information about a file."""
        if not filepath.exists():
            return {"exists": False}
        
        stat = filepath.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime),
        }
