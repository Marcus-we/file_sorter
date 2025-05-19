#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cluster Engine

This module implements clustering algorithms for file organization:
- K-means clustering
- Optimal cluster number determination
- File-to-cluster assignment
"""

import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def determine_optimal_k(feature_matrix: np.ndarray, max_k: int = 10) -> int:
    """
    Determines the optimal number of clusters using the Elbow Method.

    Args:
        feature_matrix: Matrix of feature vectors
        max_k: Maximum number of clusters to test

    Returns:
        Optimal number of clusters
    """
    n_samples = feature_matrix.shape[0]
    max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples
    
    # For very small datasets, use a simple approach
    if n_samples < 10:
        return max(2, min(3, n_samples // 2))
    
    # For larger datasets, use the elbow method
    try:
        # Reduce dimensionality if needed for faster processing
        if feature_matrix.shape[1] > 50:
            pca = PCA(n_components=min(50, n_samples - 1))
            feature_matrix = pca.fit_transform(feature_matrix)
        
        # Calculate inertia for different k values
        inertias = []
        k_values = range(2, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
            kmeans.fit(feature_matrix)
            inertias.append(kmeans.inertia_)
        
        # Find the elbow point
        if len(inertias) > 2:
            # Calculate the rate of decrease
            decreases = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            
            # Find where the rate of decrease slows down significantly
            elbow_idx = 0
            for i in range(1, len(decreases)):
                if decreases[i] / decreases[0] < 0.2:  # Threshold for significant slowdown
                    elbow_idx = i
                    break
            
            # Return the optimal k value, ensuring at least 2 clusters
            return max(2, k_values[elbow_idx])
    
    except Exception as e:
        print(f"Error determining optimal k: {e}")
    
    # Fallback to heuristic
    if n_samples < 100:
        return max(2, min(n_samples // 10, max_k))
    else:
        return min(n_samples // 20, max_k)


def perform_kmeans_clustering(feature_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Performs K-means clustering on feature vectors using scikit-learn.

    Args:
        feature_matrix: Matrix of feature vectors
        n_clusters: Number of clusters

    Returns:
        Array of cluster assignments for each sample
    """
    try:
        # Use scikit-learn KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(feature_matrix)
        return cluster_assignments
    
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        # Fallback to simple clustering
        n_samples = feature_matrix.shape[0]
        # Assign to evenly distributed clusters as a fallback
        return np.array([i % n_clusters for i in range(n_samples)])


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

    # Determine optimal number of clusters
    n_clusters = determine_optimal_k(feature_matrix)
    # Ensure we don't have too many clusters
    n_clusters = min(n_clusters, len(file_paths) // 2 + 1)
    n_clusters = max(2, n_clusters)  # Ensure at least 2 clusters

    # Perform clustering
    cluster_assignments = perform_kmeans_clustering(feature_matrix, n_clusters)

    # Group files by cluster
    clusters = defaultdict(list)
    for i, file_path in enumerate(file_paths):
        cluster_id = int(cluster_assignments[i])
        clusters[cluster_id].append(file_path)

    return dict(clusters)


def perform_clustering(file_features: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Main clustering function that organizes files into groups.

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

    # Cluster text files
    text_clusters = {}
    if text_files:
        text_file_paths, text_features = prepare_feature_vectors(text_files)
        if len(text_file_paths) > 1:
            n_clusters = determine_optimal_k(text_features)
            text_assignments = perform_kmeans_clustering(
                text_features, n_clusters)

            # Map back to original file data
            text_path_to_data = {f['file_path']: f for f in text_files}
            for i, path in enumerate(text_file_paths):
                cluster_id = int(text_assignments[i])
                if cluster_id not in text_clusters:
                    text_clusters[cluster_id] = []
                text_clusters[cluster_id].append(text_path_to_data[path])

    # Cluster image files
    image_clusters = {}
    if image_files:
        image_file_paths, image_features = prepare_feature_vectors(image_files)
        if len(image_file_paths) > 1:
            n_clusters = determine_optimal_k(image_features)
            image_assignments = perform_kmeans_clustering(
                image_features, n_clusters)

            # Map back to original file data
            image_path_to_data = {f['file_path']: f for f in image_files}
            for i, path in enumerate(image_file_paths):
                # Offset for image clusters
                cluster_id = 100 + int(image_assignments[i])
                if cluster_id not in image_clusters:
                    image_clusters[cluster_id] = []
                image_clusters[cluster_id].append(image_path_to_data[path])

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
