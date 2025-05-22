#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cluster Engine

This module implements clustering algorithms for file organization:
- Hierarchical agglomerative clustering 
- Parameter estimation for clustering
- File-to-cluster assignment
"""

import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def prepare_feature_vectors(file_features: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
    """
    Prepares feature vectors for clustering from file analysis results.

    Args:
        file_features: List of file feature dictionaries from analyzers

    Returns:
        Tuple of (file_paths, feature_matrix)
    """
    file_paths = []
    feature_vectors = []

    for file_data in file_features:
        file_path = file_data.get('file_path', '')

        # Skip files with errors
        if 'error' in file_data:
            continue

        # Get feature vector based on content type
        if file_data.get('content_type') == 'text':
            # For text files, use embeddings by default (preferred), fallback to TF-IDF
            if file_data.get('embedding_vector') is not None:
                features = file_data['embedding_vector']
            elif file_data.get('tfidf_vector'):
                # Convert dict to vector (very simplified)
                features = list(file_data['tfidf_vector'].values())
            else:
                # Skip if no features available
                continue

        elif file_data.get('content_type') == 'image':
            # For image files, use advanced features by default, fallback to color features
            if file_data.get('advanced_features') is not None:
                features = file_data['advanced_features']
            elif file_data.get('color_features') is not None:
                features = file_data['color_features']
            else:
                # Skip if no features available
                continue
        else:
            # Skip unknown content types
            continue

        # Append data
        file_paths.append(file_path)
        feature_vectors.append(features)

    # Convert to numpy array for clustering
    if feature_vectors:
        # Handle potential issues with different length vectors (shouldn't happen but just in case)
        max_length = max(len(fv) for fv in feature_vectors)
        padded_vectors = []
        for fv in feature_vectors:
            if len(fv) < max_length:
                # Pad with zeros if needed
                padded = np.zeros(max_length)
                padded[:len(fv)] = fv
                padded_vectors.append(padded)
            else:
                padded_vectors.append(fv)
        
        feature_matrix = np.array(padded_vectors)
        # Normalize features
        feature_matrix = normalize_features(feature_matrix)
        return file_paths, feature_matrix
    else:
        # Return empty arrays if no valid features
        return [], np.array([])


def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize features for better clustering.

    Args:
        feature_matrix: Matrix of feature vectors

    Returns:
        Normalized feature matrix
    """
    # Use standard scaling if we have enough samples
    if feature_matrix.shape[0] > 1:
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            return scaled_features
        except Exception as e:
            print(f"Error during feature scaling: {e}")
            # Fall back to simple normalization
    
    # Simple normalization: scale each feature to have unit norm
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    return feature_matrix / norms


def estimate_cluster_count(feature_matrix: np.ndarray, is_image_data: bool = False) -> int:
    """
    Estimates an appropriate number of clusters for hierarchical clustering.
    
    Args:
        feature_matrix: Matrix of feature vectors
        is_image_data: Whether this is image data (needs different cluster counts)
        
    Returns:
        Estimated number of clusters
    """
    n_samples = feature_matrix.shape[0]
    
    # For very small datasets, use simple heuristic
    if n_samples < 10:
        return max(2, min(4, n_samples // 2))
    
    try:
        # Use the elbow method to determine a good number of clusters
        if feature_matrix.shape[1] > 50:
            # Reduce dimensionality for more stable calculation if needed
            pca = PCA(n_components=min(50, n_samples - 1))
            feature_matrix = pca.fit_transform(feature_matrix)
        
        # Calculate distances between points using Euclidean distance instead of cosine
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(feature_matrix, metric='euclidean')
        
        # Calculate within-cluster sum of squares for different k values
        wcss = []
        max_k = min(10, n_samples // 2)  # Don't try too many clusters
        k_values = range(1, max_k + 1)
        
        for k in k_values:
            # Create temporary clustering
            clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')  # Ward linkage works well with Euclidean
            labels = clustering.fit_predict(feature_matrix)
            
            # Calculate within-cluster sum of squares
            wss = 0
            for i in range(k):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 1:
                    # Calculate pairwise distances within this cluster
                    cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
                    wss += np.sum(cluster_distances) / (2 * len(cluster_indices))
            
            wcss.append(wss)
        
        # Find the elbow point (if there are enough clusters to analyze)
        if len(wcss) > 2:
            # Calculate the rate of decrease
            decreases = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
            
            # Find where the rate of decrease slows down significantly
            elbow_idx = 0
            for i in range(1, len(decreases)):
                if decreases[i] / decreases[0] < 0.2:  # Threshold for significant slowdown
                    elbow_idx = i
                    break
                    
            # Return the optimal k value
            optimal_k = k_values[elbow_idx]
        else:
            # Default to a reasonable number of clusters
            optimal_k = max(2, min(n_samples // 5, 5))
        
        # Adjust based on whether it's image data (image data usually has fewer natural clusters)
        if is_image_data:
            optimal_k = max(2, min(optimal_k, n_samples // 5))
        else:
            # Allow more clusters for text files by reducing the divisor
            optimal_k = max(2, min(optimal_k, n_samples // 2))
            
        return optimal_k
            
    except Exception as e:
        print(f"Error estimating cluster count: {e}")
        # Fall back to a simple heuristic
        if is_image_data:
            return max(2, min(4, n_samples // 5))
        else:
            # Allow more clusters for text files by reducing the divisor
            return max(2, min(10, n_samples // 3))


def perform_hierarchical_clustering(feature_matrix: np.ndarray, n_clusters: int = None, is_image_data: bool = False) -> np.ndarray:
    """
    Performs hierarchical clustering on feature vectors.
    
    Args:
        feature_matrix: Matrix of feature vectors
        n_clusters: Number of clusters to form (if None, estimated automatically)
        is_image_data: Whether this is image data
        
    Returns:
        Array of cluster assignments for each sample
    """
    n_samples = feature_matrix.shape[0]
    
    # Special cases for very small datasets
    if n_samples <= 1:
        return np.zeros(n_samples)
    elif n_samples <= 3:
        # For very small datasets, put everything in one cluster
        return np.zeros(n_samples)
    
    # Estimate cluster count if not provided
    if n_clusters is None:
        n_clusters = estimate_cluster_count(feature_matrix, is_image_data)
    
    # Ensure a reasonable number of clusters, but allow for more granularity with text files
    if is_image_data:
        n_clusters = min(n_clusters, max(2, n_samples // 2))
    else:
        # For text files, allow more clusters by using a smaller divisor
        # Set a higher upper limit to allow for more logical grouping
        n_clusters = min(n_clusters, max(2, min(20, n_samples // 2)))
    
    print(f"Performing hierarchical clustering with {n_clusters} clusters")
    
    try:
        # Use Euclidean distance instead of cosine similarity to prevent text files from being grouped too closely
        print("Using Euclidean distance for hierarchical clustering")
        
        # For text files, Euclidean distance often works better than cosine similarity
        # as it prevents documents with similar word distributions but different lengths
        # from being grouped together
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # Ward linkage works well with Euclidean distance
        )
        cluster_assignments = clustering.fit_predict(feature_matrix)
        
        print(f"Hierarchical clustering found {len(np.unique(cluster_assignments))} clusters")
        return cluster_assignments
        
    except Exception as e:
        print(f"Error during hierarchical clustering: {e}")
        # Ultimate fallback - all in one cluster
        return np.zeros(n_samples)


def cluster_files_by_content(file_features: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Clusters files based on their content features.

    Args:
        file_features: List of file feature dictionaries from analyzers

    Returns:
        Dictionary mapping cluster IDs to lists of file paths
    """
    # Prepare feature vectors
    file_paths, feature_matrix = prepare_feature_vectors(file_features)

    # Check if we have enough data to cluster
    if len(file_paths) < 2:
        # Not enough data for meaningful clustering
        # Put everything in one cluster
        return {0: file_paths}

    # Perform hierarchical clustering
    cluster_assignments = perform_hierarchical_clustering(feature_matrix)

    # Group files by cluster
    clusters = defaultdict(list)
    for i, file_path in enumerate(file_paths):
        cluster_id = int(cluster_assignments[i])
        clusters[cluster_id].append(file_path)

    return dict(clusters)


def perform_clustering(file_features: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Main clustering function that organizes files into groups.
    For text files, uses hierarchical clustering.
    For images, uses supervised classification from ResNet50.

    Args:
        file_features: List of file feature dictionaries from analyzers

    Returns:
        Dictionary mapping cluster IDs to lists of file data
    """
    # First, separate files by content type
    text_files = [f for f in file_features if f.get(
        'content_type') == 'text' and 'error' not in f]
    image_files = [f for f in file_features if f.get(
        'content_type') == 'image' and 'error' not in f]
    other_files = [f for f in file_features if f.get(
        'content_type') not in ('text', 'image') or 'error' in f]

    # Cluster text files using hierarchical clustering
    text_clusters = {}
    if text_files:
        text_file_paths, text_features = prepare_feature_vectors(text_files)
        if len(text_file_paths) > 1:
            # Use hierarchical clustering for text files
            text_assignments = perform_hierarchical_clustering(text_features, is_image_data=False)

            # Map back to original file data
            text_path_to_data = {f['file_path']: f for f in text_files}
            for i, path in enumerate(text_file_paths):
                cluster_id = int(text_assignments[i])
                if cluster_id not in text_clusters:
                    text_clusters[cluster_id] = []
                text_clusters[cluster_id].append(text_path_to_data[path])

    # For image files, use supervised classification instead of clustering
    image_clusters = {}
    if image_files:
        print("Using supervised classification for images...")
        
        # Group images by their predicted categories
        for img_file in image_files:
            # Skip files with errors
            if 'error' in img_file:
                continue
                
            # Get classification data
            classification = img_file.get('classification', {})
            
            # Determine category ID based on predicted category
            category = classification.get('category', 'unknown')
            if not category or category == 'unknown' or 'error' in classification:
                # Use primary label if no category mapping was found
                category = classification.get('primary_label', 'unknown')
            
            # Create a consistent ID for this category
            # Use hash of category name to create a numeric ID starting at 100
            # This ensures the same category always gets the same ID
            import hashlib
            category_hash = int(hashlib.md5(category.encode()).hexdigest(), 16) % 1000
            cluster_id = 100 + category_hash % 900  # Range 100-999
            
            # Add to the appropriate cluster
            if cluster_id not in image_clusters:
                image_clusters[cluster_id] = []
            image_clusters[cluster_id].append(img_file)
        
        print(f"Classified images into {len(image_clusters)} categories")

    # Combine results
    all_clusters = {}
    all_clusters.update(text_clusters)
    all_clusters.update(image_clusters)

    # Add other files as a separate cluster if any exist
    if other_files:
        other_cluster_id = max(all_clusters.keys()) + \
            1 if all_clusters else 999
        all_clusters[other_cluster_id] = other_files

    return all_clusters
