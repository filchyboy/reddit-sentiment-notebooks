"""Text cleaning, tokenization, and n-gram generation utilities."""

import re
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    console.print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    console.print("📥 Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Try to import spaCy
try:
    import spacy
    # Try to load the English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        console.print("⚠️  spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    console.print("⚠️  spaCy not installed. Falling back to NLTK for tokenization.")
    SPACY_AVAILABLE = False
    nlp = None


class TextProcessor:
    """Handles text cleaning, tokenization, and n-gram generation."""
    
    def __init__(self, use_spacy: bool = True, custom_stopwords: Optional[List[str]] = None):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Load stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
        
        # Add custom stopwords
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Add common Reddit-specific stopwords
        reddit_stopwords = {
            'reddit', 'subreddit', 'post', 'comment', 'thread', 'op', 'edit', 'update',
            'deleted', 'removed', 'http', 'https', 'www', 'com', 'org', 'net'
        }
        self.stopwords.update(reddit_stopwords)
        
        if self.use_spacy:
            console.print("✅ Using spaCy for text processing")
        else:
            console.print("✅ Using NLTK for text processing")
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, normalizing whitespace, etc."""
        # Handle pandas NA values first
        try:
            if pd.isna(text):
                return ""
        except (TypeError, ValueError):
            # Handle cases where pd.isna() fails (like with pd.NA)
            if str(text).lower() in ['na', 'nan', '<na>']:
                return ""

        # Handle None and empty strings
        if text is None:
            return ""

        # Convert to string and check if empty
        text_str = str(text).strip()
        if len(text_str) == 0:
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove username mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize text and lemmatize words."""
        if not text:
            return []
        
        if self.use_spacy:
            return self._spacy_process(text)
        else:
            return self._nltk_process(text)
    
    def _spacy_process(self, text: str) -> List[str]:
        """Process text using spaCy."""
        doc = nlp(text.lower())
        tokens = []
        
        for token in doc:
            # Skip punctuation, spaces, and stopwords
            if (not token.is_punct and 
                not token.is_space and 
                not token.is_stop and
                token.lemma_ not in self.stopwords and
                len(token.lemma_) > 2 and
                token.lemma_.isalpha()):
                tokens.append(token.lemma_)
        
        return tokens
    
    def _nltk_process(self, text: str) -> List[str]:
        """Process text using NLTK."""
        from nltk.tokenize import word_tokenize
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (token.isalpha() and 
                len(token) > 2 and 
                token not in self.stopwords):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Generate n-grams from tokens."""
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def process_text_column(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process a text column in a DataFrame."""
        df = df.copy()
        
        console.print(f"🔄 Processing {len(df)} texts from column '{text_column}'...")
        
        # Clean text
        df['clean_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize and lemmatize
        df['tokens'] = df['clean_text'].apply(self.tokenize_and_lemmatize)
        
        # Count tokens
        df['token_count'] = df['tokens'].apply(len)
        
        console.print(f"✅ Text processing complete")
        return df
    
    def calculate_ngram_frequencies(
        self, 
        df: pd.DataFrame, 
        ngram_sizes: List[int] = [1, 2], 
        top_n: int = 30
    ) -> Dict[int, pd.DataFrame]:
        """Calculate n-gram frequencies from processed DataFrame."""
        
        if 'tokens' not in df.columns:
            raise ValueError("DataFrame must have 'tokens' column. Run process_text_column first.")
        
        results = {}
        
        for n in ngram_sizes:
            console.print(f"📊 Calculating {n}-gram frequencies...")
            
            all_ngrams = []
            for tokens in df['tokens']:
                if isinstance(tokens, list) and len(tokens) >= n:
                    if n == 1:
                        all_ngrams.extend(tokens)
                    else:
                        ngrams = self.generate_ngrams(tokens, n)
                        all_ngrams.extend([' '.join(ngram) for ngram in ngrams])
            
            # Count frequencies
            ngram_counts = Counter(all_ngrams)
            top_ngrams = ngram_counts.most_common(top_n)
            
            # Create DataFrame
            ngram_df = pd.DataFrame(top_ngrams, columns=['ngram', 'frequency'])
            ngram_df['rank'] = range(1, len(ngram_df) + 1)
            ngram_df['ngram_size'] = n
            
            results[n] = ngram_df
            console.print(f"✅ Found {len(ngram_counts)} unique {n}-grams, showing top {len(ngram_df)}")
        
        return results


def combine_text_fields(df: pd.DataFrame, title_col: str = None, body_col: str = None) -> pd.DataFrame:
    """Combine title and body text fields for processing."""
    df = df.copy()
    
    # Determine text fields based on DataFrame columns
    if 'title' in df.columns and 'selftext' in df.columns:
        # Submissions DataFrame
        title_col = title_col or 'title'
        body_col = body_col or 'selftext'
    elif 'body' in df.columns:
        # Comments DataFrame
        body_col = body_col or 'body'
        title_col = None
    else:
        raise ValueError("DataFrame must have either 'title'+'selftext' or 'body' columns")
    
    # Combine text fields
    if title_col and body_col:
        df['combined_text'] = (
            df[title_col].fillna('').astype(str) + ' ' + 
            df[body_col].fillna('').astype(str)
        ).str.strip()
    elif body_col:
        df['combined_text'] = df[body_col].fillna('').astype(str)
    else:
        raise ValueError("No valid text columns found")
    
    return df


def process_dataframe_text(
    df: pd.DataFrame, 
    ngram_sizes: List[int] = [1, 2],
    top_n: int = 30
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Process text in a DataFrame and calculate n-gram frequencies."""
    
    if df.empty:
        return df, {}
    
    # Combine text fields
    df = combine_text_fields(df)
    
    # Process text
    processor = TextProcessor()
    df = processor.process_text_column(df, 'combined_text')
    
    # Calculate n-gram frequencies
    ngram_results = processor.calculate_ngram_frequencies(df, ngram_sizes, top_n)
    
    return df, ngram_results
