"""Tests for n-gram functionality and frequency analysis."""

import pytest
import pandas as pd
from collections import Counter
from reddit_tool.text_clean import TextProcessor


class TestNgramGeneration:
    """Test n-gram generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor(use_spacy=False)
    
    def test_unigram_generation(self):
        """Test unigram (1-gram) generation."""
        tokens = ["hello", "world", "test", "hello"]
        unigrams = self.processor.generate_ngrams(tokens, 1)
        
        # For unigrams, each token should be a tuple with one element
        expected = [("hello",), ("world",), ("test",), ("hello",)]
        assert unigrams == expected
    
    def test_bigram_generation(self):
        """Test bigram (2-gram) generation."""
        tokens = ["the", "quick", "brown", "fox"]
        bigrams = self.processor.generate_ngrams(tokens, 2)
        
        expected = [
            ("the", "quick"),
            ("quick", "brown"),
            ("brown", "fox")
        ]
        assert bigrams == expected
    
    def test_trigram_generation(self):
        """Test trigram (3-gram) generation."""
        tokens = ["the", "quick", "brown", "fox", "jumps"]
        trigrams = self.processor.generate_ngrams(tokens, 3)
        
        expected = [
            ("the", "quick", "brown"),
            ("quick", "brown", "fox"),
            ("brown", "fox", "jumps")
        ]
        assert trigrams == expected
    
    def test_ngram_edge_cases(self):
        """Test n-gram generation edge cases."""
        # Empty token list
        assert self.processor.generate_ngrams([], 2) == []
        
        # Single token
        assert self.processor.generate_ngrams(["single"], 2) == []
        
        # Exact length match
        tokens = ["one", "two"]
        bigrams = self.processor.generate_ngrams(tokens, 2)
        assert bigrams == [("one", "two")]
        
        # N larger than token list
        assert self.processor.generate_ngrams(["one"], 5) == []
    
    def test_ngram_frequency_counting(self):
        """Test frequency counting of n-grams."""
        # Create test data with repeated patterns
        df = pd.DataFrame({
            'tokens': [
                ['python', 'programming', 'language'],
                ['python', 'programming', 'tutorial'],
                ['programming', 'language', 'syntax'],
                ['python', 'language', 'features']
            ]
        })
        
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1, 2], top_n=10)
        
        # Check unigram frequencies
        unigrams = ngram_results[1]
        
        # 'python' should appear 3 times
        python_row = unigrams[unigrams['ngram'] == 'python']
        assert len(python_row) == 1
        assert python_row['frequency'].iloc[0] == 3
        
        # 'programming' should appear 3 times
        programming_row = unigrams[unigrams['ngram'] == 'programming']
        assert len(programming_row) == 1
        assert programming_row['frequency'].iloc[0] == 3
        
        # 'language' should appear 3 times
        language_row = unigrams[unigrams['ngram'] == 'language']
        assert len(language_row) == 1
        assert language_row['frequency'].iloc[0] == 3
        
        # Check bigram frequencies
        bigrams = ngram_results[2]
        
        # 'python programming' should appear 2 times
        python_programming = bigrams[bigrams['ngram'] == 'python programming']
        assert len(python_programming) == 1
        assert python_programming['frequency'].iloc[0] == 2
    
    def test_ngram_ranking(self):
        """Test that n-grams are properly ranked by frequency."""
        df = pd.DataFrame({
            'tokens': [
                ['common', 'word', 'rare'],
                ['common', 'word', 'another'],
                ['common', 'different', 'word'],
                ['common', 'word', 'common']  # 'common' appears 4 times, 'word' 4 times
            ]
        })
        
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1], top_n=10)
        unigrams = ngram_results[1]
        
        # Check that results are sorted by frequency (descending)
        frequencies = unigrams['frequency'].tolist()
        assert frequencies == sorted(frequencies, reverse=True)
        
        # Check that ranks are assigned correctly
        ranks = unigrams['rank'].tolist()
        assert ranks == list(range(1, len(ranks) + 1))
        
        # Most frequent words should be 'common' and 'word'
        top_words = unigrams.head(2)['ngram'].tolist()
        assert 'common' in top_words
        assert 'word' in top_words
    
    def test_ngram_top_n_limit(self):
        """Test that top_n parameter limits results correctly."""
        # Create data with many unique tokens
        tokens = [f'word_{i}' for i in range(20)]
        df = pd.DataFrame({'tokens': [tokens]})
        
        # Request only top 5
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1], top_n=5)
        unigrams = ngram_results[1]
        
        # Should only return 5 results
        assert len(unigrams) == 5
        
        # All should have frequency 1 (each word appears once)
        assert all(unigrams['frequency'] == 1)
    
    def test_ngram_multiple_sizes(self):
        """Test generating multiple n-gram sizes simultaneously."""
        df = pd.DataFrame({
            'tokens': [
                ['machine', 'learning', 'algorithm'],
                ['deep', 'learning', 'neural'],
                ['learning', 'algorithm', 'optimization']
            ]
        })
        
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1, 2, 3], top_n=10)
        
        # Should have results for all requested sizes
        assert 1 in ngram_results
        assert 2 in ngram_results
        assert 3 in ngram_results
        
        # Each result should have the correct ngram_size
        assert all(ngram_results[1]['ngram_size'] == 1)
        assert all(ngram_results[2]['ngram_size'] == 2)
        assert all(ngram_results[3]['ngram_size'] == 3)
        
        # Unigrams should have single words
        unigram_samples = ngram_results[1]['ngram'].head(3).tolist()
        for ngram in unigram_samples:
            assert ' ' not in ngram  # Single words shouldn't contain spaces
        
        # Bigrams should have two words
        bigram_samples = ngram_results[2]['ngram'].head(3).tolist()
        for ngram in bigram_samples:
            assert len(ngram.split()) == 2
        
        # Trigrams should have three words
        if len(ngram_results[3]) > 0:
            trigram_samples = ngram_results[3]['ngram'].head(3).tolist()
            for ngram in trigram_samples:
                assert len(ngram.split()) == 3
    
    def test_ngram_empty_tokens(self):
        """Test handling of empty token lists."""
        df = pd.DataFrame({
            'tokens': [
                [],  # Empty token list
                ['single'],  # Single token (insufficient for bigrams)
                ['valid', 'tokens', 'here']  # Valid tokens
            ]
        })
        
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1, 2], top_n=10)
        
        # Should still work and return results from valid tokens
        assert 1 in ngram_results
        assert 2 in ngram_results
        
        # Unigrams should include 'single', 'valid', 'tokens', 'here'
        unigrams = ngram_results[1]
        unigram_words = unigrams['ngram'].tolist()
        assert 'single' in unigram_words
        assert 'valid' in unigram_words
        assert 'tokens' in unigram_words
        assert 'here' in unigram_words
        
        # Bigrams should only come from the third row
        bigrams = ngram_results[2]
        expected_bigrams = ['valid tokens', 'tokens here']
        bigram_phrases = bigrams['ngram'].tolist()
        for expected in expected_bigrams:
            assert expected in bigram_phrases
    
    def test_ngram_dataframe_structure(self):
        """Test the structure of returned n-gram DataFrames."""
        df = pd.DataFrame({
            'tokens': [
                ['test', 'data', 'structure'],
                ['data', 'structure', 'validation']
            ]
        })
        
        ngram_results = self.processor.calculate_ngram_frequencies(df, [1, 2], top_n=5)
        
        for n, ngram_df in ngram_results.items():
            # Check required columns
            required_columns = ['ngram', 'frequency', 'rank', 'ngram_size']
            for col in required_columns:
                assert col in ngram_df.columns, f"Missing column {col} in {n}-gram results"
            
            # Check data types
            assert ngram_df['ngram'].dtype == 'object'
            assert pd.api.types.is_integer_dtype(ngram_df['frequency'])
            assert pd.api.types.is_integer_dtype(ngram_df['rank'])
            assert pd.api.types.is_integer_dtype(ngram_df['ngram_size'])
            
            # Check that all ngram_size values are correct
            assert all(ngram_df['ngram_size'] == n)
            
            # Check that frequencies are positive
            assert all(ngram_df['frequency'] > 0)
            
            # Check that ranks start from 1 and are consecutive
            expected_ranks = list(range(1, len(ngram_df) + 1))
            assert ngram_df['rank'].tolist() == expected_ranks


if __name__ == "__main__":
    pytest.main([__file__])
