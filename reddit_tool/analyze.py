"""Data analysis and KPI calculation utilities."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from rich.console import Console

console = Console()


class RedditAnalyzer:
    """Analyzes Reddit data and calculates KPIs."""
    
    def __init__(self):
        pass
    
    def calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for the dataset."""
        if df.empty:
            return {}
        
        stats = {
            'total_records': len(df),
            'date_range': self._get_date_range(df),
            'unique_authors': df['author'].nunique() if 'author' in df.columns else 0,
            'subreddits': df['subreddit'].unique().tolist() if 'subreddit' in df.columns else [],
        }
        
        # Add type-specific stats
        if 'title' in df.columns:  # Submissions
            stats.update({
                'avg_score': df['score'].mean() if 'score' in df.columns else 0,
                'avg_comments': df['num_comments'].mean() if 'num_comments' in df.columns else 0,
                'self_posts_ratio': df['is_self'].mean() if 'is_self' in df.columns else 0,
            })
        elif 'body' in df.columns:  # Comments
            stats.update({
                'avg_score': df['score'].mean() if 'score' in df.columns else 0,
                'avg_comment_length': df['body'].str.len().mean() if 'body' in df.columns else 0,
            })
        
        return stats
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get date range information from DataFrame."""
        if 'created_datetime' not in df.columns:
            return {}
        
        dates = pd.to_datetime(df['created_datetime'])
        return {
            'start_date': dates.min().isoformat() if not dates.empty else None,
            'end_date': dates.max().isoformat() if not dates.empty else None,
            'days_span': (dates.max() - dates.min()).days if not dates.empty else 0,
        }
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze posting patterns over time."""
        if df.empty or 'created_datetime' not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        df['created_datetime'] = pd.to_datetime(df['created_datetime'])
        
        # Extract time components
        df['date'] = df['created_datetime'].dt.date
        df['hour'] = df['created_datetime'].dt.hour
        df['day_of_week'] = df['created_datetime'].dt.day_name()
        
        # Daily counts
        daily_stats = (
            df.groupby('date')
            .agg({
                'score': ['count', 'mean', 'sum'],
                'sentiment_compound': 'mean' if 'sentiment_compound' in df.columns else lambda x: np.nan
            })
            .round(2)
        )
        
        # Flatten column names
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        daily_stats = daily_stats.reset_index()
        
        return daily_stats
    
    def analyze_top_content(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """Analyze top content by various metrics."""
        results = {}
        
        if df.empty:
            return results
        
        # Top by score
        if 'score' in df.columns:
            top_score = df.nlargest(top_n, 'score')[
                ['score', 'author', 'created_datetime'] + 
                (['title'] if 'title' in df.columns else ['body'])
            ].copy()
            results['top_by_score'] = top_score
        
        # Top authors by post count
        if 'author' in df.columns:
            author_stats = (
                df.groupby('author')
                .agg({
                    'score': ['count', 'mean', 'sum'],
                    'sentiment_compound': 'mean' if 'sentiment_compound' in df.columns else lambda x: np.nan
                })
                .round(2)
            )
            author_stats.columns = ['_'.join(col).strip() for col in author_stats.columns]
            author_stats = author_stats.reset_index().nlargest(top_n, 'score_count')
            results['top_authors'] = author_stats
        
        # Most controversial (high engagement, mixed sentiment)
        if 'score' in df.columns and 'num_comments' in df.columns:
            df_copy = df.copy()
            df_copy['engagement'] = df_copy['score'] + df_copy['num_comments']
            controversial = df_copy.nlargest(top_n, 'engagement')[
                ['score', 'num_comments', 'engagement', 'author', 'title']
            ]
            results['most_engaging'] = controversial
        
        return results
    
    def analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        if df.empty or 'sentiment_compound' not in df.columns:
            return {}
        
        sentiment_stats = {
            'overall_sentiment': {
                'mean_compound': df['sentiment_compound'].mean(),
                'std_compound': df['sentiment_compound'].std(),
                'positive_ratio': (df['sentiment_compound'] > 0.05).mean(),
                'negative_ratio': (df['sentiment_compound'] < -0.05).mean(),
                'neutral_ratio': (df['sentiment_compound'].between(-0.05, 0.05)).mean(),
            }
        }
        
        # Sentiment by time if datetime available
        if 'created_datetime' in df.columns:
            df_copy = df.copy()
            df_copy['created_datetime'] = pd.to_datetime(df_copy['created_datetime'])
            df_copy['date'] = df_copy['created_datetime'].dt.date
            
            daily_sentiment = (
                df_copy.groupby('date')['sentiment_compound']
                .agg(['mean', 'std', 'count'])
                .reset_index()
            )
            
            sentiment_stats['daily_trends'] = daily_sentiment
        
        return sentiment_stats
    
    def create_summary_report(self, posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive summary report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'posts': {},
            'comments': {},
            'combined': {}
        }
        
        # Analyze posts
        if not posts_df.empty:
            console.print("📊 Analyzing posts...")
            report['posts'] = {
                'basic_stats': self.calculate_basic_stats(posts_df),
                'temporal_patterns': self.analyze_temporal_patterns(posts_df).to_dict('records'),
                'top_content': {k: v.to_dict('records') for k, v in self.analyze_top_content(posts_df).items()},
                'sentiment_trends': self.analyze_sentiment_trends(posts_df),
            }
        
        # Analyze comments
        if not comments_df.empty:
            console.print("📊 Analyzing comments...")
            report['comments'] = {
                'basic_stats': self.calculate_basic_stats(comments_df),
                'temporal_patterns': self.analyze_temporal_patterns(comments_df).to_dict('records'),
                'top_content': {k: v.to_dict('records') for k, v in self.analyze_top_content(comments_df).items()},
                'sentiment_trends': self.analyze_sentiment_trends(comments_df),
            }
        
        # Combined analysis
        if not posts_df.empty or not comments_df.empty:
            combined_df = pd.concat([posts_df, comments_df], ignore_index=True)
            report['combined'] = {
                'basic_stats': self.calculate_basic_stats(combined_df),
                'sentiment_trends': self.analyze_sentiment_trends(combined_df),
            }
        
        return report
    
    def calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engagement metrics."""
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Basic engagement metrics
        if 'score' in df.columns and 'num_comments' in df.columns:
            df['engagement_score'] = df['score'] + df['num_comments']
            df['comment_to_score_ratio'] = df['num_comments'] / (df['score'] + 1)  # +1 to avoid division by zero
        
        # Sentiment-based metrics
        if 'sentiment_compound' in df.columns:
            df['sentiment_category'] = pd.cut(
                df['sentiment_compound'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['negative', 'neutral', 'positive']
            )
        
        # Time-based metrics
        if 'created_datetime' in df.columns:
            df['created_datetime'] = pd.to_datetime(df['created_datetime'])
            df['hours_since_creation'] = (
                datetime.now() - df['created_datetime']
            ).dt.total_seconds() / 3600
            
            # Engagement rate (score per hour)
            if 'score' in df.columns:
                df['engagement_rate'] = df['score'] / (df['hours_since_creation'] + 1)
        
        return df


def analyze_subreddit_data(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function to analyze subreddit data."""
    analyzer = RedditAnalyzer()
    return analyzer.create_summary_report(posts_df, comments_df)
