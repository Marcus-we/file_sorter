#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Analyzer

This module provides functions for analyzing text files:
- Feature extraction
- Text embeddings
- Text preprocessing
"""

import re
import string
import os
from typing import Dict, List, Any, Union, Optional
from collections import Counter
import numpy as np


def preprocess_text(text: str) -> str:
    """
    Preprocesses text by removing punctuation, extra whitespace, and lowercasing.

    Args:
        text: Input text string

    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extracts the most frequent keywords from text.

    Args:
        text: Input text
        num_keywords: Number of keywords to extract

    Returns:
        List of keywords
    """
    # Preprocess text
    processed_text = preprocess_text(text)

    # Tokenize
    words = processed_text.split()

    # Filter out short words (likely not meaningful)
    words = [word for word in words if len(word) > 2]

    # Count word frequencies
    word_counts = Counter(words)

    # Get the most common words
    common_words = word_counts.most_common(num_keywords)

    # Extract just the words
    keywords = [word for word, _ in common_words]

    return keywords


def create_tfidf_vector(text: str, vocabulary: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """
    Creates a simple TF-IDF vector for a text document.

    Args:
        text: Input text
        vocabulary: Optional vocabulary dictionary mapping terms to indices

    Returns:
        Dictionary mapping terms to TF-IDF weights
    """
    # This is a simplified version that only computes term frequency
    # In a real implementation, you would compute proper TF-IDF with document frequencies

    # Preprocess text
    processed_text = preprocess_text(text)

    # Tokenize
    words = processed_text.split()

    # Count term frequencies
    term_freq = Counter(words)

    # Create vector
    tfidf_vector = {}
    for term, freq in term_freq.items():
        if vocabulary is None or term in vocabulary:
            # Simple TF normalization
            tfidf_vector[term] = freq / len(words)

    return tfidf_vector


def analyze_text_file(file_path: str) -> Dict[str, Any]:
    """
    Analyzes a text file and extracts features.

    Args:
        file_path: Path to the text file

    Returns:
        Dictionary with extracted features
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract basic text stats
        word_count = len(content.split())
        line_count = len(content.splitlines())
        char_count = len(content)

        # Extract features
        keywords = extract_keywords(content)
        tfidf_vector = create_tfidf_vector(content)

        return {
            'content_type': 'text',
            'file_path': file_path,
            'word_count': word_count,
            'line_count': line_count,
            'char_count': char_count,
            'keywords': keywords,
            'tfidf_vector': tfidf_vector,
            'embedding_vector': None  # Placeholder for more advanced embeddings
        }

    except Exception as e:
        print(f"Error analyzing text file {file_path}: {e}")
        return {
            'content_type': 'text',
            'file_path': file_path,
            'error': str(e)
        }


# Additional functions for advanced embeddings - placeholders for now
def create_advanced_embedding(text: str) -> np.ndarray:
    """
    Creates more advanced text embeddings using pretrained models.

    Note: This is a placeholder. In a real implementation, this would use
    sentence-transformers or similar libraries.

    Args:
        text: Input text

    Returns:
        Numpy array containing the embedding vector
    """
    # Placeholder: Return a random vector of the right dimensionality
    # In a real implementation, this would use a pretrained model

    # Example dimensionality for embeddings
    embedding_dim = 384

    # Generate a random vector (placeholder)
    return np.random.rand(embedding_dim)
