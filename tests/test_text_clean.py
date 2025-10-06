"""Tests for text cleaning and processing functionality."""

import pytest
import pandas as pd
from reddit_tool.text_clean import TextProcessor, combine_text_fields, process_dataframe_text


class TestTextProcessor:
    """Test the TextProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor(use_spacy=False)  # Use NLTK for consistent testing
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "This is a test with   multiple   spaces."
        cleaned = self.processor.clean_text(text)
        assert cleaned == "This is a test with multiple spaces."
    
    def test_clean_text_urls(self):
        """Test URL removal."""
        text = "Check out this link: https://www.reddit.com/r/python and this one http://example.com"
        cleaned = self.processor.clean_text(text)
        assert "https://" not in cleaned
        assert "http://" not in cleaned
        assert "reddit.com" not in cleaned
        assert "example.com" not in cleaned
    
    def test_clean_text_reddit_formatting(self):
        """Test Reddit-specific formatting removal."""
        text = "Hey /u/username check out /r/python! **bold text** and *italic* and ~~strikethrough~~"
        cleaned = self.processor.clean_text(text)
        assert "/u/username" not in cleaned
        assert "/r/python" not in cleaned
        assert "**" not in cleaned
        assert "~~" not in cleaned
        assert "bold text" in cleaned
        assert "italic" in cleaned
        assert "strikethrough" in cleaned
    
    def test_clean_text_empty_input(self):
        """Test handling of empty/null input."""
        assert self.processor.clean_text("") == ""
        assert self.processor.clean_text(None) == ""
        assert self.processor.clean_text(pd.NA) == ""
    
    def test_tokenize_and_lemmatize(self):
        """Test tokenization and lemmatization."""
        text = "The cats are running quickly through the trees"
        tokens = self.processor.tokenize_and_lemmatize(text)
        
        # Should contain lemmatized forms and filter out stopwords
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        assert all(len(token) > 2 for token in tokens)  # Should filter short tokens
        
        # Common stopwords should be removed
        stopwords_to_check = ['the', 'are', 'through']
        for stopword in stopwords_to_check:
            assert stopword not in tokens
    
    def test_tokenize_empty_text(self):
        """Test tokenization of empty text."""
        assert self.processor.tokenize_and_lemmatize("") == []
        assert self.processor.tokenize_and_lemmatize(None) == []
    
    def test_generate_ngrams(self):
        """Test n-gram generation."""
        tokens = ["this", "is", "a", "test", "sentence"]
        
        # Test bigrams
        bigrams = self.processor.generate_ngrams(tokens, 2)
        expected_bigrams = [
            ("this", "is"),
            ("is", "a"),
            ("a", "test"),
            ("test", "sentence")
        ]
        assert bigrams == expected_bigrams
        
        # Test trigrams
        trigrams = self.processor.generate_ngrams(tokens, 3)
        expected_trigrams = [
            ("this", "is", "a"),
            ("is", "a", "test"),
            ("a", "test", "sentence")
        ]
        assert trigrams == expected_trigrams
    
    def test_generate_ngrams_insufficient_tokens(self):
        """Test n-gram generation with insufficient tokens."""
        tokens = ["one"]
        
        # Should return empty list for bigrams
        bigrams = self.processor.generate_ngrams(tokens, 2)
        assert bigrams == []
        
        # Should return empty list for trigrams
        trigrams = self.processor.generate_ngrams(tokens, 3)
        assert trigrams == []
    
    def test_process_text_column(self):
        """Test processing a DataFrame text column."""
        df = pd.DataFrame({
            'text': [
                "This is a test post with https://example.com",
                "Another post with /u/user mention",
                "**Bold text** and normal text"
            ]
        })
        
        result_df = self.processor.process_text_column(df, 'text')
        
        # Check that new columns are added
        assert 'clean_text' in result_df.columns
        assert 'tokens' in result_df.columns
        assert 'token_count' in result_df.columns
        
        # Check that cleaning worked
        assert "https://example.com" not in result_df['clean_text'].iloc[0]
        assert "/u/user" not in result_df['clean_text'].iloc[1]
        assert "**" not in result_df['clean_text'].iloc[2]
        
        # Check that tokenization worked
        assert all(isinstance(tokens, list) for tokens in result_df['tokens'])
        assert all(count >= 0 for count in result_df['token_count'])
    
    def test_calculate_ngram_frequencies(self):
        """Test n-gram frequency calculation."""
        df = pd.DataFrame({
            'tokens': [
                ['test', 'word', 'frequency'],
                ['test', 'another', 'word'],
                ['word', 'frequency', 'test']
            ]
        })
        
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1, 2], top_n=10)
        
        # Check that results are returned for both n-gram sizes
        assert 1 in ngram_results
        assert 2 in ngram_results
        
        # Check unigram results
        unigrams = ngram_results[1]
        assert 'ngram' in unigrams.columns
        assert 'frequency' in unigrams.columns
        assert 'rank' in unigrams.columns
        assert 'ngram_size' in unigrams.columns
        
        # 'test' and 'word' should appear 3 times each
        test_freq = unigrams[unigrams['ngram'] == 'test']['frequency'].iloc[0]
        word_freq = unigrams[unigrams['ngram'] == 'word']['frequency'].iloc[0]
        assert test_freq == 3
        assert word_freq == 3
        
        # Check bigram results
        bigrams = ngram_results[2]
        assert len(bigrams) > 0
        assert all(bigrams['ngram_size'] == 2)


class TestTextUtilities:
    """Test utility functions for text processing."""
    
    def test_combine_text_fields_submissions(self):
        """Test combining title and selftext for submissions."""
        df = pd.DataFrame({
            'title': ['Title 1', 'Title 2'],
            'selftext': ['Body 1', 'Body 2'],
            'score': [10, 20]
        })
        
        result_df = combine_text_fields(df)
        
        assert 'combined_text' in result_df.columns
        assert result_df['combined_text'].iloc[0] == 'Title 1 Body 1'
        assert result_df['combined_text'].iloc[1] == 'Title 2 Body 2'
    
    def test_combine_text_fields_comments(self):
        """Test combining text for comments."""
        df = pd.DataFrame({
            'body': ['Comment 1', 'Comment 2'],
            'score': [5, 15]
        })
        
        result_df = combine_text_fields(df)
        
        assert 'combined_text' in result_df.columns
        assert result_df['combined_text'].iloc[0] == 'Comment 1'
        assert result_df['combined_text'].iloc[1] == 'Comment 2'
    
    def test_combine_text_fields_missing_data(self):
        """Test handling of missing data in text fields."""
        df = pd.DataFrame({
            'title': ['Title 1', None],
            'selftext': [None, 'Body 2'],
            'score': [10, 20]
        })
        
        result_df = combine_text_fields(df)
        
        assert 'combined_text' in result_df.columns
        assert result_df['combined_text'].iloc[0] == 'Title 1'
        assert result_df['combined_text'].iloc[1] == 'Body 2'
    
    def test_process_dataframe_text(self):
        """Test the complete DataFrame text processing pipeline."""
        df = pd.DataFrame({
            'title': ['Test title with https://example.com', 'Another title'],
            'selftext': ['Test body text', 'Another body with /u/user'],
            'score': [10, 20]
        })
        
        processed_df, ngram_results = process_dataframe_text(df, [1, 2], top_n=5)
        
        # Check that processing columns are added
        assert 'combined_text' in processed_df.columns
        assert 'clean_text' in processed_df.columns
        assert 'tokens' in processed_df.columns
        assert 'token_count' in processed_df.columns
        
        # Check that n-gram results are returned
        assert isinstance(ngram_results, dict)
        assert 1 in ngram_results
        assert 2 in ngram_results
        
        # Check that cleaning worked
        assert "https://example.com" not in processed_df['clean_text'].iloc[0]
        assert "/u/user" not in processed_df['clean_text'].iloc[1]
    
    def test_process_empty_dataframe(self):
        """Test processing an empty DataFrame."""
        df = pd.DataFrame()
        
        processed_df, ngram_results = process_dataframe_text(df, [1, 2])
        
        assert processed_df.empty
        assert ngram_results == {}


if __name__ == "__main__":
    pytest.main([__file__])
