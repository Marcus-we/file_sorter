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


def generate_text_category_name(files: List[Dict[str, Any]], cluster_id: int) -> str:
    """
    Generates a name for a text category based on keywords and file properties.

    Args:
        files: List of file data for this cluster
        cluster_id: ID of the cluster

    Returns:
        Category name
    """
    # Extract keywords from all files in the cluster
    all_keywords = []
    extensions = []
    word_counts = []
    line_counts = []
    
    for file_data in files:
        # Get keywords
        keywords = file_data.get('keywords', [])
        if keywords:
            all_keywords.extend(keywords)
        
        # Get file extension
        file_path = file_data.get('file_path', '')
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            if ext:
                extensions.append(ext)
        
        # Get word and line counts
        word_count = file_data.get('word_count', 0)
        if word_count:
            word_counts.append(word_count)
            
        line_count = file_data.get('line_count', 0)
        if line_count:
            line_counts.append(line_count)
    
    # Count occurrences of each keyword and extension
    keyword_counts = Counter(all_keywords)
    extension_counts = Counter(extensions)
    
    # Get average word and line count
    avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
    avg_line_count = sum(line_counts) / len(line_counts) if line_counts else 0
    
    # Determine most common extension
    most_common_ext = extension_counts.most_common(1)[0][0] if extension_counts else ""
    if most_common_ext.startswith('.'):
        most_common_ext = most_common_ext[1:].upper()
    
    # Get common keywords (exclude very generic terms)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'to', 'of', 'in', 'on', 'this', 'that', 'these', 'those', 'it',
                'function', 'class', 'method', 'return', 'true', 'false', 'null',
                'none', 'undefined', 'import', 'export', 'as', 'from', 'try', 'catch'}
    
    filtered_keywords = [kw for kw, count in keyword_counts.items() 
                      if kw.lower() not in stopwords and len(kw) > 2]
    
    # Determine content type descriptor based on extension and keywords
    content_descriptor = ""
    code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.go', '.rb', '.swift'}
    doc_extensions = {'.txt', '.md', '.doc', '.docx', '.rtf', '.pdf', '.odt'}
    data_extensions = {'.csv', '.json', '.xml', '.yaml', '.yml', '.ini', '.conf', '.cfg'}
    
    if most_common_ext:
        if any(ext in code_extensions for ext in extension_counts):
            # Code files
            code_keywords = {'function', 'class', 'method', 'import', 'variable', 'const', 'return', 'if', 'for', 'while'}
            if any(kw.lower() in code_keywords for kw in all_keywords):
                content_descriptor = f"{most_common_ext} Code"
            else:
                content_descriptor = f"{most_common_ext} Files"
        elif any(ext in doc_extensions for ext in extension_counts):
            # Text documents
            if avg_word_count > 500:
                content_descriptor = "Documents"
            else:
                content_descriptor = "Notes"
        elif any(ext in data_extensions for ext in extension_counts):
            content_descriptor = "Data Files"
        else:
            content_descriptor = f"{most_common_ext} Files"
    else:
        content_descriptor = "Text Files"
    
    # Generate the final name
    if filtered_keywords:
        # Use top keywords for naming
        top_keywords = [kw.title() for kw in filtered_keywords[:2]]
        section_name = " ".join(top_keywords)
        return f"{section_name} {content_descriptor} ({len(files)})"
    else:
        # Fallback to generic section name
        return f"Section {cluster_id} {content_descriptor} ({len(files)})"


def generate_image_category_name(files: List[Dict[str, Any]], cluster_id: int) -> str:
    """
    Generates a name for an image category based on supervised classification results.

    Args:
        files: List of file data for this cluster
        cluster_id: ID of the cluster

    Returns:
        Category name
    """
    # Collect all classifications from files in this cluster
    categories = []
    labels = []
    
    for file_data in files:
        # Get classification data
        classification = file_data.get('classification', {})
        if not classification or 'error' in classification:
            continue
            
        # Get category and top label
        category = classification.get('category', '')
        primary_label = classification.get('primary_label', '')
        
        if category:
            categories.append(category)
        if primary_label:
            labels.append(primary_label)
    
    # Count occurrences of each category and label
    category_counts = Counter(categories)
    label_counts = Counter(labels)
    
    # Get the most common category and label
    most_common_category = category_counts.most_common(1)[0][0] if category_counts else "Unknown"
    most_common_labels = [label for label, count in label_counts.most_common(2)]
    
    # Determine image shape characteristics
    aspect_ratios = [f.get('aspect_ratio', 1.0) for f in files if 'aspect_ratio' in f]
    avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 1.0
    
    # Determine if images are likely landscape or portrait
    shape_descriptor = ""
    if avg_aspect_ratio > 1.2:
        shape_descriptor = "Landscape"
    elif avg_aspect_ratio < 0.8:
        shape_descriptor = "Portrait"
    else:
        shape_descriptor = "Square"
    
    # Get file extensions
    extensions = [os.path.splitext(f.get('file_path', ''))[1].lower() for f in files]
    extension_counts = Counter(extensions)
    most_common_ext = extension_counts.most_common(1)[0][0] if extension_counts else ""
    
    # Remove the dot from extension
    if most_common_ext.startswith('.'):
        most_common_ext = most_common_ext[1:].upper()
    
    # Combine information into a meaningful name
    if most_common_category.lower() != "unknown":
        # If we have a good category, use it
        if most_common_labels and most_common_labels[0].lower() not in most_common_category.lower():
            # Include the most common label if it adds information
            return f"{most_common_category.title()} {most_common_labels[0].title()} {most_common_ext} Images ({len(files)})"
        else:
            return f"{most_common_category.title()} {most_common_ext} Images ({len(files)})"
    elif most_common_labels:
        # Fall back to labels if no good category
        return f"{most_common_labels[0].title()} {most_common_ext} Images ({len(files)})"
    else:
        # Last resort
        return f"{shape_descriptor} {most_common_ext} Images ({len(files)})"


def generate_category_names(clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
    """
    Generates human-readable names for file clusters.

    Args:
        clusters: Dictionary mapping cluster IDs to lists of file data

    Returns:
        Dictionary mapping cluster IDs to category names
    """
    category_names = {}
    
    for cluster_id, files in clusters.items():
        if not files:
            continue
            
        # Determine if this is a text or image cluster
        content_type = files[0].get('content_type', 'unknown')
        
        if content_type == 'text':
            # For text clusters, use existing keyword-based naming
            category_names[cluster_id] = generate_text_category_name(files, cluster_id)
        elif content_type == 'image':
            # For image clusters, use the classification results
            category_names[cluster_id] = generate_image_category_name(files, cluster_id)
        else:
            # Fallback for other content types
            category_names[cluster_id] = f"Other Files ({len(files)})"
    
    return category_names


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
