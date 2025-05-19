#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Analyzer

This module provides functions for analyzing image files:
- Feature extraction
- Color analysis
- Image preprocessing
"""

import os
from typing import Dict, List, Any, Tuple, Union
import numpy as np
# Import the hash function from utils
from utils.file_utils import calculate_file_hash

# Import PIL for image processing
from PIL import Image, ImageStat


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Gets the dimensions of an image.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error getting image dimensions for {image_path}: {e}")
        return (0, 0)


def calculate_color_histogram(image_path: str, bins: int = 8) -> Dict[str, np.ndarray]:
    """
    Calculates color histograms for an image.

    Args:
        image_path: Path to the image file
        bins: Number of bins for each color channel

    Returns:
        Dictionary with RGB histogram arrays
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Calculate histograms for each channel
            r_hist = np.histogram(img_array[:,:,0], bins=bins, range=(0, 256))[0]
            g_hist = np.histogram(img_array[:,:,1], bins=bins, range=(0, 256))[0]
            b_hist = np.histogram(img_array[:,:,2], bins=bins, range=(0, 256))[0]
            
            # Normalize
            r_hist = r_hist / np.sum(r_hist) if np.sum(r_hist) > 0 else r_hist
            g_hist = g_hist / np.sum(g_hist) if np.sum(g_hist) > 0 else g_hist
            b_hist = b_hist / np.sum(b_hist) if np.sum(b_hist) > 0 else b_hist
            
            return {
                'red': r_hist,
                'green': g_hist,
                'blue': b_hist
            }
    except Exception as e:
        print(f"Error calculating color histogram for {image_path}: {e}")
        # Return placeholder on error
        placeholder = np.zeros(bins)
        return {
            'red': placeholder,
            'green': placeholder,
            'blue': placeholder
        }


def extract_image_features(image_path: str) -> np.ndarray:
    """
    Extracts basic features from an image.

    Args:
        image_path: Path to the image file

    Returns:
        Feature vector as a numpy array
    """
    try:
        # Get color histograms
        histograms = calculate_color_histogram(image_path)
        
        # Get image statistics
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            stats = ImageStat.Stat(img)
            mean_values = np.array(stats.mean)  # Mean color values
            std_values = np.array(stats.stddev)  # Standard deviation
            
            # Normalize
            mean_values = mean_values / 255.0
            std_values = std_values / 255.0
        
        # Combine histograms into a single feature vector
        features = np.concatenate([
            histograms['red'],
            histograms['green'],
            histograms['blue'],
            mean_values,
            std_values
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting image features for {image_path}: {e}")
        # Return placeholder on error
        return np.random.rand(8*3 + 6)  # 8 bins per channel + 6 stat values


def create_advanced_image_features(image_path: str) -> np.ndarray:
    """
    Creates more advanced image features.
    
    This implementation uses color statistics and histogram features.
    For a production system, a pre-trained CNN would be more effective.

    Args:
        image_path: Path to the image file

    Returns:
        Feature vector as a numpy array
    """
    # Currently using basic features - in a full implementation
    # this would use a pre-trained CNN for feature extraction
    return extract_image_features(image_path)


def analyze_image_file(file_path: str) -> Dict[str, Any]:
    """
    Analyzes an image file and extracts features.

    Args:
        file_path: Path to the image file

    Returns:
        Dictionary with extracted features
    """
    try:
        # Get file size
        file_size = os.path.getsize(file_path)

        # Get image dimensions
        width, height = get_image_dimensions(file_path)

        # Extract basic features
        color_features = extract_image_features(file_path)
        
        # Extract advanced features
        advanced_features = create_advanced_image_features(file_path)

        # Calculate a file hash for uniqueness checking
        file_hash = calculate_file_hash(file_path)

        return {
            'content_type': 'image',
            'file_path': file_path,
            'file_size': file_size,
            'dimensions': (width, height),
            'aspect_ratio': width / height if height != 0 else 0,
            'color_features': color_features.tolist() if isinstance(color_features, np.ndarray) else color_features,
            'file_hash': file_hash,
            'advanced_features': advanced_features.tolist() if isinstance(advanced_features, np.ndarray) else advanced_features
        }

    except Exception as e:
        print(f"Error analyzing image file {file_path}: {e}")
        return {
            'content_type': 'image',
            'file_path': file_path,
            'error': str(e)
        }
