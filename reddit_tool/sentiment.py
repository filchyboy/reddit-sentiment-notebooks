"""Sentiment analysis using VADER and optional transformer models."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import warnings

import pandas as pd
import numpy as np
import nltk
from rich.console import Console

from .config import load_config

console = Console()
logger = logging.getLogger(__name__)

# Download VADER lexicon if needed
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except LookupError:
    console.print("📥 Downloading VADER sentiment lexicon...")
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer

# Optional transformer imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    console.print("⚠️  Transformers not available. Only VADER sentiment will be used.")


class SentimentAnalyzer:
    """Sentiment analysis with VADER and optional transformer support."""
    
    def __init__(self, use_transformer: bool = False, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.config = load_config()
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize transformer if requested
        self.transformer_pipeline = None
        if self.use_transformer:
            self._load_transformer_model()
    
    def _load_transformer_model(self) -> None:
        """Load transformer model for sentiment analysis."""
        try:
            console.print(f"🤖 Loading transformer model: {self.model_name}")
            
            # Create cache directory
            cache_dir = Path(".cache/transformers")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model with caching
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                cache_dir=str(cache_dir),
                device=0 if torch.cuda.is_available() else -1
            )
            
            console.print("✅ Transformer model loaded successfully")
            
        except Exception as e:
            console.print(f"❌ Failed to load transformer model: {e}")
            console.print("🔄 Falling back to VADER only")
            self.use_transformer = False
            self.transformer_pipeline = None
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        if not text or pd.isna(text):
            return {
                'sentiment_compound': 0.0,
                'sentiment_pos': 0.0,
                'sentiment_neu': 1.0,
                'sentiment_neg': 0.0
            }
        
        scores = self.vader.polarity_scores(str(text))
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_pos': scores['pos'],
            'sentiment_neu': scores['neu'],
            'sentiment_neg': scores['neg']
        }
    
    def analyze_transformer(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Analyze sentiment using transformer model in batches."""
        if not self.transformer_pipeline:
            raise ValueError("Transformer model not loaded")
        
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean texts for transformer
            cleaned_batch = []
            for text in batch:
                if not text or pd.isna(text):
                    cleaned_batch.append("")
                else:
                    # Truncate very long texts
                    text_str = str(text)[:512]
                    cleaned_batch.append(text_str)
            
            try:
                # Get predictions
                batch_results = self.transformer_pipeline(cleaned_batch)
                
                # Convert to standardized format
                for result in batch_results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    # Map labels to sentiment scores
                    if 'positive' in label or label == 'label_2':
                        sentiment_score = score
                    elif 'negative' in label or label == 'label_0':
                        sentiment_score = -score
                    else:  # neutral or label_1
                        sentiment_score = 0.0
                    
                    results.append({
                        'transformer_label': result['label'],
                        'transformer_score': score,
                        'transformer_sentiment': sentiment_score
                    })
            
            except Exception as e:
                logger.warning(f"Transformer analysis failed for batch {i//batch_size + 1}: {e}")
                # Add empty results for failed batch
                for _ in batch:
                    results.append({
                        'transformer_label': 'UNKNOWN',
                        'transformer_score': 0.0,
                        'transformer_sentiment': 0.0
                    })
        
        return results
    
    def get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound score to sentiment label."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'combined_text') -> pd.DataFrame:
        """Add sentiment analysis to DataFrame."""
        if df.empty:
            return df
        
        df = df.copy()
        console.print(f"🎭 Analyzing sentiment for {len(df)} texts...")
        
        # Ensure text column exists
        if text_column not in df.columns:
            if 'title' in df.columns and 'selftext' in df.columns:
                # Combine title and selftext for submissions
                df[text_column] = (
                    df['title'].fillna('').astype(str) + ' ' + 
                    df['selftext'].fillna('').astype(str)
                ).str.strip()
            elif 'body' in df.columns:
                # Use body for comments
                df[text_column] = df['body'].fillna('').astype(str)
            else:
                raise ValueError(f"Text column '{text_column}' not found and cannot be created")
        
        # VADER sentiment analysis
        console.print("🔍 Running VADER sentiment analysis...")
        vader_results = df[text_column].apply(self.analyze_vader)
        
        # Extract VADER scores
        for key in ['sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg']:
            df[key] = vader_results.apply(lambda x: x[key])
        
        # Add sentiment labels
        df['sentiment_label'] = df['sentiment_compound'].apply(self.get_sentiment_label)
        
        # Transformer sentiment analysis (if enabled)
        if self.use_transformer:
            console.print("🤖 Running transformer sentiment analysis...")
            try:
                texts = df[text_column].fillna('').astype(str).tolist()
                transformer_results = self.analyze_transformer(texts)
                
                # Add transformer results
                transformer_df = pd.DataFrame(transformer_results)
                df = pd.concat([df.reset_index(drop=True), transformer_df], axis=1)
                
            except Exception as e:
                console.print(f"❌ Transformer analysis failed: {e}")
        
        console.print("✅ Sentiment analysis complete")
        return df
    
    def summarize_sentiment(self, df: pd.DataFrame, group_by: List[str] = None) -> pd.DataFrame:
        """Create sentiment summary statistics."""
        if df.empty or 'sentiment_label' not in df.columns:
            return pd.DataFrame()
        
        if group_by is None:
            group_by = ['subreddit'] if 'subreddit' in df.columns else []
        
        # Add data type if available
        if 'title' in df.columns:
            df = df.copy()
            df['data_type'] = 'submission'
        elif 'body' in df.columns:
            df = df.copy()
            df['data_type'] = 'comment'
        
        if 'data_type' in df.columns and 'data_type' not in group_by:
            group_by.append('data_type')
        
        # Group by specified columns and sentiment label
        group_cols = group_by + ['sentiment_label']
        
        if not group_cols:
            # No grouping, just count sentiment labels
            summary = df['sentiment_label'].value_counts().reset_index()
            summary.columns = ['sentiment_label', 'count']
        else:
            summary = (
                df.groupby(group_cols, dropna=False)
                .size()
                .reset_index(name='count')
            )
        
        # Calculate percentages
        if group_by:
            total_by_group = summary.groupby(group_by)['count'].transform('sum')
            summary['percentage'] = (summary['count'] / total_by_group * 100).round(2)
        else:
            total = summary['count'].sum()
            summary['percentage'] = (summary['count'] / total * 100).round(2)
        
        # Add average sentiment scores
        if group_by:
            avg_scores = (
                df.groupby(group_by)
                .agg({
                    'sentiment_compound': 'mean',
                    'sentiment_pos': 'mean',
                    'sentiment_neu': 'mean',
                    'sentiment_neg': 'mean'
                })
                .round(3)
                .reset_index()
            )
            
            summary = summary.merge(avg_scores, on=group_by, how='left')
        
        return summary.sort_values(group_by + ['sentiment_label'] if group_by else ['sentiment_label'])


def analyze_sentiment(
    df: pd.DataFrame, 
    use_transformer: bool = False,
    text_column: str = 'combined_text'
) -> pd.DataFrame:
    """Convenience function to analyze sentiment in a DataFrame."""
    analyzer = SentimentAnalyzer(use_transformer=use_transformer)
    return analyzer.analyze_dataframe(df, text_column)
