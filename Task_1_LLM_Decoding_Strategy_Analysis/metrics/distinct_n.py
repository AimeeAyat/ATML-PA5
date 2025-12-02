"""
Distinct-N metric implementation for measuring output diversity.
"""

from typing import List, Set
from collections import Counter


class DistinctNMetric:
    """Calculate Distinct-N metric for measuring lexical diversity."""

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Set[tuple]:
        """
        Extract n-grams from a list of tokens.

        Args:
            tokens: List of tokens/words
            n: N-gram size

        Returns:
            Set of n-grams as tuples
        """
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.add(ngram)
        return ngrams

    @staticmethod
    def compute_distinct_n(text: str, n: int = 1) -> float:
        """
        Compute Distinct-N score for a single text.

        Args:
            text: Input text
            n: N-gram size (1 for unigrams, 2 for bigrams, 3 for trigrams)

        Returns:
            Distinct-N score (ratio of unique n-grams to total n-grams)
        """
        # Tokenize (simple split by spaces and lowercase)
        tokens = text.lower().split()

        if len(tokens) < n:
            return 0.0

        # Count total n-grams
        total_ngrams = len(tokens) - n + 1

        # Get unique n-grams
        unique_ngrams = DistinctNMetric._get_ngrams(tokens, n)

        # Compute ratio
        distinct_score = len(unique_ngrams) / total_ngrams

        return distinct_score

    @staticmethod
    def compute_distinct_n_corpus(texts: List[str], n: int = 1) -> float:
        """
        Compute Distinct-N score across a corpus of texts.

        Aggregates all tokens from all texts and computes unique/total n-grams ratio.

        Args:
            texts: List of texts
            n: N-gram size

        Returns:
            Distinct-N score for the entire corpus
        """
        # Combine all tokens from all texts
        all_tokens = []
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)

        if len(all_tokens) < n:
            return 0.0

        # Count total n-grams in the corpus
        total_ngrams = len(all_tokens) - n + 1

        # Get unique n-grams
        unique_ngrams = DistinctNMetric._get_ngrams(all_tokens, n)

        # Compute ratio
        distinct_score = len(unique_ngrams) / total_ngrams

        return distinct_score

    @staticmethod
    def compute_all_distinct_n(texts: List[str], n_values: List[int] = [1, 2, 3]) -> dict:
        """
        Compute Distinct-N scores for multiple N values.

        Args:
            texts: List of texts
            n_values: List of N values to compute

        Returns:
            Dictionary with Distinct-N scores for each N
        """
        results = {}
        for n in n_values:
            results[f"Distinct-{n}"] = DistinctNMetric.compute_distinct_n_corpus(texts, n)
        return results
