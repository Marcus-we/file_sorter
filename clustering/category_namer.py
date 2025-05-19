#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Category Namer

This module generates meaningful category names for file clusters:
- Extract keywords from text clusters
- Identify themes in image clusters
- Generate human-readable category names
"""

import os
import re
from typing import Dict, List, Any, Tuple, Set
from collections import Counter
import numpy as np
import string


def extract_common_keywords_from_text_cluster(cluster_files: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts common keywords from a cluster of text files.

    Args:
        cluster_files: List of file data dictionaries in the cluster

    Returns:
        List of common keywords
    """
    # Collect all keywords from all files
    all_keywords = []
    for file_data in cluster_files:
        if file_data.get('content_type') == 'text' and 'keywords' in file_data:
            all_keywords.extend(file_data['keywords'])

    # Count keyword frequencies
    keyword_counter = Counter(all_keywords)

    # Get most common keywords, but filter out common stopwords
    stopwords = {'and', 'the', 'for', 'was', 'with', 'this', 'that', 'from', 'have', 'not'}
    common_keywords = [word for word, count in keyword_counter.most_common(10) 
                      if count > 1 and word not in stopwords]

    # Limit to top 5 keywords
    return common_keywords[:5]


def extract_common_words_from_filenames(cluster_files: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts common words from filenames in a cluster.

    Args:
        cluster_files: List of file data dictionaries in the cluster

    Returns:
        List of common words from filenames
    """
    # Extract words from filenames
    filename_words = []
    stopwords = {'and', 'the', 'for', 'file', 'doc', 'document', 'image', 'img', 'photo'}
    
    for file_data in cluster_files:
        if 'file_path' in file_data:
            # Get the filename without extension
            filename = os.path.basename(file_data['file_path'])
            name, _ = os.path.splitext(filename)
            
            # Clean and split the filename
            name = name.lower()
            # Replace non-alphanumeric with spaces
            name = re.sub(r'[^a-z0-9]', ' ', name)
            # Split into words
            words = [w for w in name.split() if len(w) > 2 and w not in stopwords]
            filename_words.extend(words)
    
    # Get most common words
    word_counter = Counter(filename_words)
    common_words = [word for word, count in word_counter.most_common(3) if count > 1]
    
    return common_words


def extract_common_extensions(cluster_files: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts common file extensions from a cluster.

    Args:
        cluster_files: List of file data dictionaries in the cluster

    Returns:
        List of common extensions
    """
    extensions = []
    for file_data in cluster_files:
        if 'file_path' in file_data:
            _, ext = os.path.splitext(file_data['file_path'])
            if ext:
                extensions.append(ext.lower())

    # Count extension frequencies
    ext_counter = Counter(extensions)

    # Get most common extensions
    common_extensions = [ext for ext, count in ext_counter.most_common(3)]

    return common_extensions


def extract_image_characteristics(cluster_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts common characteristics from a cluster of image files.

    Args:
        cluster_files: List of image file data dictionaries

    Returns:
        Dictionary of image characteristics
    """
    # Initialize counters
    total_width = 0
    total_height = 0
    count = 0
    aspect_ratios = []

    # Collect image dimensions
    for file_data in cluster_files:
        if file_data.get('content_type') == 'image' and 'dimensions' in file_data:
            width, height = file_data['dimensions']
            total_width += width
            total_height += height
            count += 1

            if 'aspect_ratio' in file_data:
                aspect_ratios.append(file_data['aspect_ratio'])

    # Calculate averages
    if count > 0:
        avg_width = total_width / count
        avg_height = total_height / count
    else:
        avg_width = 0
        avg_height = 0

    # Determine predominant orientation
    if aspect_ratios:
        avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
        if avg_aspect > 1.2:
            orientation = 'landscape'
        elif avg_aspect < 0.8:
            orientation = 'portrait'
        else:
            orientation = 'square'
    else:
        orientation = 'unknown'

    return {
        'avg_width': avg_width,
        'avg_height': avg_height,
        'orientation': orientation
    }


def generate_text_category_name(cluster_files: List[Dict[str, Any]]) -> str:
    """
    Generates a category name for a cluster of text files.

    Args:
        cluster_files: List of file data dictionaries in the cluster

    Returns:
        Category name string
    """
    # Extract common keywords
    keywords = extract_common_keywords_from_text_cluster(cluster_files)
    
    # If not enough keywords, try extracting from filenames
    if len(keywords) < 2:
        filename_words = extract_common_words_from_filenames(cluster_files)
        keywords.extend([w for w in filename_words if w not in keywords])
        keywords = keywords[:5]  # Limit to 5 keywords

    # Extract common extensions
    extensions = extract_common_extensions(cluster_files)

    # Generate name based on keywords and extensions
    if keywords:
        # Use the top keywords to form a name
        if len(keywords) >= 2:
            name = f"{keywords[0].capitalize()} {keywords[1].capitalize()}"
        else:
            name = keywords[0].capitalize()

        # Add document type if we have extensions
        doc_extensions = {'.doc', '.docx', '.pdf', '.txt', '.rtf', '.odt'}
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.php'}
        data_extensions = {'.csv', '.json', '.xml', '.yaml', '.yml', '.sql'}
        
        if extensions and any(ext in doc_extensions for ext in extensions):
            name += " Documents"
        elif extensions and any(ext in code_extensions for ext in extensions):
            # Try to detect programming language
            if '.py' in extensions:
                name += " Python"
            elif '.js' in extensions:
                name += " JavaScript"
            elif '.java' in extensions:
                name += " Java"
            elif '.cpp' in extensions or '.c' in extensions:
                name += " C/C++"
            elif '.html' in extensions or '.css' in extensions:
                name += " Web"
            else:
                name += " Code"
        elif extensions and any(ext in data_extensions for ext in extensions):
            name += " Data"

        return name

    # Check for content patterns if no good keywords
    word_counts = []
    for file_data in cluster_files:
        if file_data.get('content_type') == 'text' and 'word_count' in file_data:
            word_counts.append(file_data['word_count'])
    
    if word_counts:
        avg_word_count = sum(word_counts) / len(word_counts)
        if avg_word_count > 1000:
            return "Long Documents"
        elif avg_word_count < 100:
            return "Short Notes"
    
    # Fallback to extension-based naming
    if extensions:
        ext = extensions[0].replace('.', '').upper()
        return f"{ext} Files"

    # Last resort
    return "Text Files"


def generate_image_category_name(cluster_files: List[Dict[str, Any]]) -> str:
    """
    Generates a category name for a cluster of image files.

    Args:
        cluster_files: List of file data dictionaries in the cluster

    Returns:
        Category name string
    """
    # Extract image characteristics
    characteristics = extract_image_characteristics(cluster_files)

    # Extract common extensions
    extensions = extract_common_extensions(cluster_files)

    # Generate name based on characteristics
    name_parts = []

    # Add orientation if available
    if characteristics['orientation'] != 'unknown':
        name_parts.append(characteristics['orientation'].capitalize())

    # Add image type based on extension
    if extensions:
        if extensions[0] in ['.jpg', '.jpeg']:
            name_parts.append("Photos")
        elif extensions[0] == '.png':
            name_parts.append("PNG Images")
        elif extensions[0] == '.gif':
            name_parts.append("GIFs")
        elif extensions[0] in ['.svg', '.eps', '.ai']:
            name_parts.append("Vector Graphics")
        else:
            name_parts.append("Images")
    else:
        name_parts.append("Images")

    # Combine name parts
    if name_parts:
        return " ".join(name_parts)

    # Fallback
    return "Image Files"


def generate_other_category_name(cluster_files: List[Dict[str, Any]]) -> str:
    """
    Generates a category name for a cluster of miscellaneous files.

    Args:
        cluster_files: List of file data dictionaries in the cluster

    Returns:
        Category name string
    """
    # Extract common extensions
    extensions = extract_common_extensions(cluster_files)

    # Generate name based on extensions
    if extensions:
        ext = extensions[0].replace('.', '').upper()
        return f"{ext} Files"

    # Fallback
    return "Other Files"


def generate_category_names(clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
    """
    Generates meaningful category names for all clusters.

    Args:
        clusters: Dictionary mapping cluster IDs to lists of file data

    Returns:
        Dictionary mapping cluster IDs to category names
    """
    category_names = {}

    for cluster_id, cluster_files in clusters.items():
        if not cluster_files:
            category_names[cluster_id] = f"Category {cluster_id}"
            continue

        # Determine predominant content type
        text_count = sum(1 for f in cluster_files if f.get('content_type') == 'text')
        image_count = sum(1 for f in cluster_files if f.get('content_type') == 'image')
        other_count = len(cluster_files) - text_count - image_count

        # Generate name based on predominant type
        if text_count >= image_count and text_count >= other_count:
            category_names[cluster_id] = generate_text_category_name(cluster_files)
        elif image_count >= text_count and image_count >= other_count:
            category_names[cluster_id] = generate_image_category_name(cluster_files)
        else:
            category_names[cluster_id] = generate_other_category_name(cluster_files)
            
        # Add file count for context
        file_count = len(cluster_files)
        category_names[cluster_id] += f" ({file_count})"

    # Ensure no duplicate names
    ensure_unique_names(category_names)

    return category_names


def ensure_unique_names(category_names: Dict[int, str]) -> None:
    """
    Ensures all category names are unique by appending numbers if needed.

    Args:
        category_names: Dictionary mapping cluster IDs to category names

    Note: Modifies the dictionary in-place
    """
    # Count occurrences of each name
    name_counts = Counter(category_names.values())

    # Collect IDs that need to be renamed
    conflicts = {name: [] for name, count in name_counts.items() if count > 1}
    for cluster_id, name in category_names.items():
        if name in conflicts:
            conflicts[name].append(cluster_id)

    # Rename conflicting categories
    for name, cluster_ids in conflicts.items():
        for i, cluster_id in enumerate(cluster_ids):
            if i > 0:  # Leave the first occurrence as is
                category_names[cluster_id] = f"{name} {i+1}"
