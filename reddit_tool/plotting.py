"""Visualization utilities for Reddit sentiment analysis."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from rich.console import Console

console = Console()

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class RedditPlotter:
    """Creates visualizations for Reddit sentiment analysis."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, title: str = "Sentiment Distribution") -> Figure:
        """Plot sentiment distribution histogram."""
        if df.empty or 'sentiment_compound' not in df.columns:
            return self._create_empty_plot("No sentiment data available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of compound scores
        ax1.hist(df['sentiment_compound'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax1.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='Positive threshold')
        ax1.axvline(x=-0.05, color='orange', linestyle='--', alpha=0.7, label='Negative threshold')
        ax1.set_xlabel('Sentiment Compound Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Compound Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar chart of sentiment labels
        if 'sentiment_label' in df.columns:
            sentiment_counts = df['sentiment_label'].value_counts()
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            bar_colors = [colors.get(label, 'blue') for label in sentiment_counts.index]
            
            ax2.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, alpha=0.7)
            ax2.set_xlabel('Sentiment Label')
            ax2.set_ylabel('Count')
            ax2.set_title(f'{title} - Label Counts')
            
            # Add percentage labels on bars
            total = sentiment_counts.sum()
            for i, (label, count) in enumerate(sentiment_counts.items()):
                percentage = (count / total) * 100
                ax2.text(i, count + total * 0.01, f'{percentage:.1f}%', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_over_time(self, df: pd.DataFrame, title: str = "Sentiment Over Time") -> Figure:
        """Plot sentiment trends over time."""
        if df.empty or 'created_datetime' not in df.columns or 'sentiment_compound' not in df.columns:
            return self._create_empty_plot("No temporal sentiment data available")
        
        df = df.copy()
        df['created_datetime'] = pd.to_datetime(df['created_datetime'])
        df['date'] = df['created_datetime'].dt.date
        
        # Daily sentiment averages
        daily_sentiment = df.groupby('date')['sentiment_compound'].agg(['mean', 'std', 'count']).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Sentiment trend line
        ax1.plot(daily_sentiment['date'], daily_sentiment['mean'], marker='o', linewidth=2, markersize=4)
        ax1.fill_between(daily_sentiment['date'], 
                        daily_sentiment['mean'] - daily_sentiment['std'], 
                        daily_sentiment['mean'] + daily_sentiment['std'], 
                        alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Average Sentiment Score')
        ax1.set_title(f'{title} - Daily Average')
        ax1.grid(True, alpha=0.3)
        
        # Volume over time
        ax2.bar(daily_sentiment['date'], daily_sentiment['count'], alpha=0.7, color='lightblue')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Posts/Comments')
        ax2.set_title('Daily Volume')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_score_distribution(self, df: pd.DataFrame, title: str = "Score Distribution") -> Figure:
        """Plot score distribution."""
        if df.empty or 'score' not in df.columns:
            return self._create_empty_plot("No score data available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of scores
        scores = df['score']
        ax1.hist(scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - All Scores')
        ax1.grid(True, alpha=0.3)
        
        # Log scale for better visualization of distribution
        positive_scores = scores[scores > 0]
        if len(positive_scores) > 0:
            ax2.hist(positive_scores, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xscale('log')
            ax2.set_xlabel('Score (log scale)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{title} - Positive Scores (Log Scale)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_ngram_frequencies(self, ngram_data: Dict[int, pd.DataFrame], title: str = "N-gram Frequencies") -> Figure:
        """Plot n-gram frequency charts."""
        if not ngram_data:
            return self._create_empty_plot("No n-gram data available")
        
        n_plots = len(ngram_data)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for i, (n, df) in enumerate(ngram_data.items()):
            if df.empty:
                continue
            
            ax = axes[i]
            top_ngrams = df.head(20)  # Show top 20
            
            bars = ax.barh(range(len(top_ngrams)), top_ngrams['frequency'], alpha=0.7)
            ax.set_yticks(range(len(top_ngrams)))
            ax.set_yticklabels(top_ngrams['ngram'])
            ax.set_xlabel('Frequency')
            ax.set_title(f'{title} - Top {n}-grams')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add frequency labels on bars
            for j, (bar, freq) in enumerate(zip(bars, top_ngrams['frequency'])):
                ax.text(freq + max(top_ngrams['frequency']) * 0.01, j, str(freq), 
                       va='center', ha='left')
            
            # Invert y-axis to show highest frequency at top
            ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_author_activity(self, df: pd.DataFrame, top_n: int = 15, title: str = "Top Authors") -> Figure:
        """Plot top authors by activity."""
        if df.empty or 'author' not in df.columns:
            return self._create_empty_plot("No author data available")
        
        # Calculate author statistics
        author_stats = df.groupby('author').agg({
            'score': ['count', 'sum', 'mean'],
            'sentiment_compound': 'mean' if 'sentiment_compound' in df.columns else lambda x: 0
        }).round(2)
        
        author_stats.columns = ['_'.join(col).strip() for col in author_stats.columns]
        author_stats = author_stats.reset_index()
        top_authors = author_stats.nlargest(top_n, 'score_count')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Post count
        bars1 = ax1.barh(range(len(top_authors)), top_authors['score_count'], alpha=0.7, color='lightblue')
        ax1.set_yticks(range(len(top_authors)))
        ax1.set_yticklabels(top_authors['author'])
        ax1.set_xlabel('Number of Posts/Comments')
        ax1.set_title(f'{title} - Post Count')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # Average sentiment (if available)
        if 'sentiment_compound_mean' in top_authors.columns:
            colors = ['red' if x < -0.05 else 'green' if x > 0.05 else 'gray' 
                     for x in top_authors['sentiment_compound_mean']]
            bars2 = ax2.barh(range(len(top_authors)), top_authors['sentiment_compound_mean'], 
                           alpha=0.7, color=colors)
            ax2.set_yticks(range(len(top_authors)))
            ax2.set_yticklabels(top_authors['author'])
            ax2.set_xlabel('Average Sentiment Score')
            ax2.set_title(f'{title} - Average Sentiment')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax2.axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
            ax2.axvline(x=-0.05, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def _create_empty_plot(self, message: str) -> Figure:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16, 
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    def save_figure(self, fig: Figure, filename: str, dpi: int = 300) -> Path:
        """Save figure to file."""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        console.print(f"📊 Saved plot: {filepath}")
        return filepath
    
    def create_all_plots(self, posts_df: pd.DataFrame, comments_df: pd.DataFrame, 
                        ngram_data: Dict[int, pd.DataFrame], subreddit: str) -> List[Path]:
        """Create all standard plots for a subreddit analysis."""
        saved_plots = []
        
        # Combine data for overall plots
        combined_df = pd.concat([posts_df, comments_df], ignore_index=True) if not posts_df.empty or not comments_df.empty else pd.DataFrame()
        
        if not combined_df.empty:
            # Sentiment distribution
            fig = self.plot_sentiment_distribution(combined_df, f"r/{subreddit} Sentiment Distribution")
            saved_plots.append(self.save_figure(fig, f"{subreddit}_sentiment_distribution.png"))
            plt.close(fig)
            
            # Sentiment over time
            fig = self.plot_sentiment_over_time(combined_df, f"r/{subreddit} Sentiment Over Time")
            saved_plots.append(self.save_figure(fig, f"{subreddit}_sentiment_timeline.png"))
            plt.close(fig)
            
            # Score distribution
            fig = self.plot_score_distribution(combined_df, f"r/{subreddit} Score Distribution")
            saved_plots.append(self.save_figure(fig, f"{subreddit}_score_distribution.png"))
            plt.close(fig)
            
            # Author activity
            fig = self.plot_author_activity(combined_df, title=f"r/{subreddit} Top Authors")
            saved_plots.append(self.save_figure(fig, f"{subreddit}_top_authors.png"))
            plt.close(fig)
        
        # N-gram frequencies
        if ngram_data:
            fig = self.plot_ngram_frequencies(ngram_data, f"r/{subreddit} N-gram Frequencies")
            saved_plots.append(self.save_figure(fig, f"{subreddit}_ngram_frequencies.png"))
            plt.close(fig)
        
        return saved_plots
