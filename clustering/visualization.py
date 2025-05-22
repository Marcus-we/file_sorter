#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cluster Visualization

This module provides functions for visualizing clusters:
- Dimensionality reduction for visualization using t-SNE
- Plotting files in 2D space colored by cluster
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


def prepare_visualization_data(file_features: List[Dict[str, Any]], clusters: Dict[int, List[Dict[str, Any]]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for visualization by extracting features and cluster assignments.

    Args:
        file_features: List of file feature dictionaries
        clusters: Dictionary mapping cluster IDs to lists of file data

    Returns:
        Tuple of (feature_matrix, cluster_labels, filenames)
    """
    # Extract feature vectors
    feature_vectors = []
    cluster_labels = []
    filenames = []
    
    # Create a mapping from file path to cluster ID
    file_to_cluster = {}
    for cluster_id, files in clusters.items():
        for file_data in files:
            if 'file_path' in file_data:
                file_to_cluster[file_data['file_path']] = cluster_id
    
    # Collect features with corresponding cluster labels
    for file_data in file_features:
        file_path = file_data.get('file_path', '')
        if not file_path or file_path not in file_to_cluster:
            continue
        
        # Get features based on content type
        features = None
        if file_data.get('content_type') == 'text':
            if file_data.get('embedding_vector') is not None:
                features = file_data['embedding_vector']
            elif file_data.get('tfidf_vector'):
                features = list(file_data['tfidf_vector'].values())
        elif file_data.get('content_type') == 'image':
            if file_data.get('advanced_features') is not None:
                features = file_data['advanced_features']
            elif file_data.get('color_features') is not None:
                features = file_data['color_features']
        
        if features is not None:
            feature_vectors.append(features)
            cluster_labels.append(file_to_cluster[file_path])
            filenames.append(os.path.basename(file_path))
    
    # Convert to numpy arrays
    if not feature_vectors:
        return np.array([]), np.array([]), []
    
    # Pad vectors to the same length if needed
    max_length = max(len(fv) for fv in feature_vectors)
    padded_vectors = []
    for fv in feature_vectors:
        if len(fv) < max_length:
            padded = np.zeros(max_length)
            padded[:len(fv)] = fv
            padded_vectors.append(padded)
        else:
            padded_vectors.append(fv)
    
    return np.array(padded_vectors), np.array(cluster_labels), filenames


def reduce_dimensions(feature_matrix: np.ndarray, method: str = 'tsne') -> np.ndarray:
    """
    Reduce high-dimensional features to 2D for visualization.

    Args:
        feature_matrix: Matrix of feature vectors
        method: Dimensionality reduction method ('pca' or 'tsne')

    Returns:
        2D array for visualization
    """
    if feature_matrix.shape[0] < 2:
        return np.zeros((0, 2))
    
    try:
        if method.lower() == 'pca':
            # Use PCA for dimensionality reduction (faster and more stable)
            print("Using PCA for dimensionality reduction (faster and more stable)")
            pca = PCA(n_components=2)
            return pca.fit_transform(feature_matrix)
        else:
            # Use t-SNE for better visualization (better at preserving cluster structure)
            print("Using t-SNE for dimensionality reduction (better at preserving cluster structure)")
            # Adjust perplexity based on dataset size
            perplexity = min(30, max(5, feature_matrix.shape[0] // 10))
            # Increase iterations for better convergence
            n_iter = 2000  # Increased from 1000
            tsne = TSNE(
                n_components=2, 
                perplexity=perplexity,
                random_state=42, 
                n_iter=n_iter, 
                learning_rate=200
            )
            return tsne.fit_transform(feature_matrix)
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        return np.zeros((feature_matrix.shape[0], 2))


def visualize_clusters(file_features: List[Dict[str, Any]], clusters: Dict[int, List[Dict[str, Any]]],
                      category_names: Dict[int, str], output_path: str = None,
                      show_plot: bool = True, method: str = 'tsne') -> str:
    """
    Visualize clusters of files in 2D space.

    Args:
        file_features: List of file feature dictionaries
        clusters: Dictionary mapping cluster IDs to lists of file data
        category_names: Dictionary mapping cluster IDs to category names
        output_path: Path to save the visualization (if None, uses temp file)
        show_plot: Whether to display the plot on screen
        method: Dimensionality reduction method ('pca' or 'tsne')

    Returns:
        Path to the saved visualization image
    """
    # Prepare data
    feature_matrix, cluster_labels, filenames = prepare_visualization_data(file_features, clusters)
    
    if feature_matrix.shape[0] < 2:
        print("Not enough data for visualization.")
        return None
    
    # Reduce dimensions for visualization
    reduced_data = reduce_dimensions(feature_matrix, method)
    
    # Define a colormap with distinct colors for clusters
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot each cluster with a different color
    unique_clusters = np.unique(cluster_labels)
    
    # Create legend handles
    legend_handles = []
    
    for i, cluster_id in enumerate(unique_clusters):
        # Get points for this cluster
        indices = cluster_labels == cluster_id
        cluster_data = reduced_data[indices]
        
        # Get category name
        category_name = category_names.get(cluster_id, f"Cluster {cluster_id}")
        
        # Plot points
        color = colors[i % len(colors)]
        scatter = plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                     c=[color], label=category_name,
                     alpha=0.7, edgecolors='w', s=100)
        
        legend_handles.append(scatter)
    
    # Add cluster labels  
    for i, (x, y) in enumerate(reduced_data):
        cluster_id = cluster_labels[i]
        plt.annotate(filenames[i], (x, y), fontsize=8, alpha=0.7,
                    textcoords="offset points", xytext=(0, 5),
                    ha='center')
    
    title = 't-SNE Visualization of File Clusters'
    plt.title(title)
    plt.legend(handles=legend_handles, title="Categories", loc="best", fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        # Create a temporary file
        import tempfile
        output_path = os.path.join(tempfile.gettempdir(), "cluster_visualization.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path 