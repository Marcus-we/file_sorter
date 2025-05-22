import numpy as np
from sklearn.metrics import pairwise_distances

def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_similarity(x, y):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Cosine similarity
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def calculate_pairwise_distances(feature_matrix, metric='euclidean'):
    """
    Calculate pairwise distances between all vectors in a feature matrix.
    
    Args:
        feature_matrix: Matrix of feature vectors (n_samples, n_features)
        metric: Distance metric to use ('euclidean' or 'cosine')
        
    Returns:
        Distance matrix (n_samples, n_samples)
    """
    if metric == 'cosine':
        # Convert cosine similarity to distance
        similarity_matrix = pairwise_distances(feature_matrix, metric='cosine')
        return similarity_matrix
    else:
        # Use Euclidean distance
        return pairwise_distances(feature_matrix, metric='euclidean') 