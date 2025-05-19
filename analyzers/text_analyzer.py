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

# Import for sentence embeddings
from sentence_transformers import SentenceTransformer
import torch

# Global variable for the model to avoid reloading
_embedding_model = None


def get_embedding_model():
    """
    Get or initialize the embedding model.
    Uses singleton pattern to avoid reloading the model.
    
    Returns:
        SentenceTransformer model
    """
    global _embedding_model
    if _embedding_model is None:
        # Use a small, efficient model for embeddings
        model_name = "all-MiniLM-L6-v2"
        print(f"Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


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
        
        # Create embeddings for the text content
        embedding_vector = create_advanced_embedding(content)

        return {
            'content_type': 'text',
            'file_path': file_path,
            'word_count': word_count,
            'line_count': line_count,
            'char_count': char_count,
            'keywords': keywords,
            'tfidf_vector': tfidf_vector,
            'embedding_vector': embedding_vector
        }

    except Exception as e:
        print(f"Error analyzing text file {file_path}: {e}")
        return {
            'content_type': 'text',
            'file_path': file_path,
            'error': str(e)
        }


# Implementation of advanced embedding function
def create_advanced_embedding(text: str) -> np.ndarray:
    """
    Creates text embeddings using pretrained sentence-transformers model.

    Args:
        text: Input text

    Returns:
        Numpy array containing the embedding vector
    """
    try:
        # Limit text length to avoid memory issues (most models have token limits)
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Get the model
        model = get_embedding_model()
        
        # Generate embedding
        with torch.no_grad():  # No need for gradients
            embedding = model.encode(text)
            
        return embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Fallback to random vector if embedding fails
        embedding_dim = 384  # Standard for many sentence-transformers models
        return np.random.rand(embedding_dim)
