#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Cluster Engine

This module implements an improved clustering approach with:
- Content-first clustering (prioritizes semantic similarity over file extensions)
- Adaptive cluster count determination
- Enhanced feature engineering
- Semantic-aware clustering strategies
"""

import os
import re
import ast
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import hdbscan


class ContentFirstDomainDetector:
    """Detects the domain/category of files based primarily on content, with file type as secondary."""
    
    def __init__(self):
        self.academic_keywords = {
            'abstract', 'methodology', 'conclusion', 'references', 'hypothesis',
            'research', 'study', 'analysis', 'experiment', 'results', 'discussion',
            'literature', 'review', 'journal', 'publication', 'citation', 'theory',
            'empirical', 'statistical', 'correlation', 'significance', 'sample'
        }
        
        self.business_keywords = {
            'contract', 'invoice', 'proposal', 'memo', 'meeting', 'budget',
            'revenue', 'profit', 'client', 'customer', 'stakeholder', 'project',
            'deadline', 'deliverable', 'requirement', 'specification', 'strategy',
            'management', 'corporate', 'business', 'financial', 'quarterly'
        }
        
        self.technical_keywords = {
            'function', 'class', 'method', 'variable', 'algorithm', 'implementation',
            'documentation', 'api', 'framework', 'library', 'module', 'package',
            'configuration', 'setup', 'installation', 'deployment', 'tutorial',
            'example', 'guide', 'manual', 'reference', 'specification'
        }
        
        self.programming_keywords = {
            'function', 'class', 'method', 'variable', 'import', 'export', 'return',
            'if', 'else', 'for', 'while', 'try', 'catch', 'throw', 'async', 'await',
            'const', 'let', 'var', 'def', 'lambda', 'yield', 'generator'
        }
        
        # Only use extensions as hints, not primary categorization
        self.extension_hints = {
            'code': {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.go', 
                    '.rb', '.swift', '.kt', '.rs', '.scala', '.r', '.m', '.sh', '.bat'},
            'data': {'.csv', '.json', '.xml', '.yaml', '.yml', '.tsv', '.parquet', 
                    '.xlsx', '.xls', '.sqlite', '.db'},
            'config': {'.ini', '.conf', '.cfg', '.config', '.properties', '.env'},
            'document': {'.md', '.txt', '.doc', '.docx', '.pdf', '.rtf', '.odt', '.tex'}
        }

    def detect_domain(self, file_data: Dict[str, Any]) -> str:
        """Detect the domain of a file based primarily on content, with file type as secondary hint."""
        file_path = file_data.get('file_path', '')
        keywords = file_data.get('keywords', [])
        content_type = file_data.get('content_type', 'unknown')
        
        # Get file extension as a hint only
        _, ext = os.path.splitext(file_path.lower())
        
        # Analyze content keywords first
        keyword_set = set(kw.lower() for kw in keywords)
        
        # Calculate content-based scores
        academic_score = len(keyword_set & self.academic_keywords)
        business_score = len(keyword_set & self.business_keywords)
        technical_score = len(keyword_set & self.technical_keywords)
        programming_score = len(keyword_set & self.programming_keywords)
        
        # Content-first classification - require stronger signal
        max_score = max(academic_score, business_score, technical_score, programming_score)
        
        # Require at least 2 keyword matches to classify into a domain
        if max_score >= 2:
            if academic_score == max_score:
                return 'academic'
            elif business_score == max_score:
                return 'business'
            elif programming_score == max_score:
                return 'programming'
            elif technical_score == max_score:
                return 'technical'
        
        # If no clear content signal, use extension as a hint
        for hint_category, extensions in self.extension_hints.items():
            if ext in extensions:
                # But still prefer content-based classification if there's any signal
                if hint_category == 'code' and programming_score > 0:
                    return 'programming'
                elif hint_category == 'document':
                    # For documents, return the best content match or generic document
                    if max_score > 0:
                        if academic_score == max_score:
                            return 'academic'
                        elif business_score == max_score:
                            return 'business'
                        elif technical_score == max_score:
                            return 'technical'
                    return 'document'
                else:
                    return hint_category
        
        return 'other'

    def categorize_files(self, file_features: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize files primarily by content domain."""
        categories = defaultdict(list)
        
        for file_data in file_features:
            if 'error' in file_data:
                categories['error'].append(file_data)
                continue
            
            content_type = file_data.get('content_type', 'unknown')
            
            # Handle images separately - they should use ResNet50 classification
            if content_type == 'image':
                categories['image'].append(file_data)
                continue
                
            domain = self.detect_domain(file_data)
            # Don't split by content_type anymore - focus on semantic content
            categories[domain].append(file_data)
        
        return dict(categories)


class ContentFirstFeatureExtractor:
    """Enhanced feature extraction that prioritizes semantic content over file structure."""
    
    def extract_semantic_features(self, content: str) -> np.ndarray:
        """Extract semantic features that work across file types."""
        if not content:
            return np.zeros(15)
        
        lines = content.split('\n')
        words = content.split()
        
        features = [
            # Content complexity
            len(words) / max(len(lines), 1),  # Words per line
            len(set(words)) / max(len(words), 1),  # Lexical diversity
            len([w for w in words if len(w) > 6]) / max(len(words), 1),  # Complex words ratio
            
            # Structural patterns (language-agnostic)
            len([l for l in lines if l.strip().startswith('#')]) / max(len(lines), 1),  # Header-like lines
            len([l for l in lines if re.match(r'^\s*[-*+]\s', l)]) / max(len(lines), 1),  # List-like lines
            len([l for l in lines if ':' in l]) / max(len(lines), 1),  # Key-value like lines
            
            # Semantic indicators
            len(re.findall(r'\b[A-Z][a-z]+\b', content)) / max(len(words), 1),  # Title case ratio
            len(re.findall(r'\b[A-Z]{2,}\b', content)) / max(len(words), 1),  # Acronym ratio
            len(re.findall(r'\d+', content)) / max(len(words), 1),  # Number ratio
            len(re.findall(r'[.!?]', content)) / max(len(content), 1),  # Punctuation density
            
            # Technical indicators (work across languages)
            len(re.findall(r'[{}()\[\]]', content)) / max(len(content), 1),  # Bracket density
            len(re.findall(r'[=<>!]', content)) / max(len(content), 1),  # Operator density
            len(re.findall(r'http[s]?://\S+', content)) / max(len(content), 1),  # URL density
            len(re.findall(r'\w+@\w+\.\w+', content)) / max(len(content), 1),  # Email density
            
            # Indentation patterns (indicates structure regardless of language)
            len([l for l in lines if l.startswith('\t') or l.startswith('    ')]) / max(len(lines), 1),
        ]
        
        return np.array(features, dtype=float)
    
    def combine_features(self, file_data: Dict[str, Any]) -> np.ndarray:
        """Combine semantic embeddings with content-agnostic features."""
        # Fixed embedding size (all-MiniLM-L6-v2 produces 384-dimensional vectors)
        EMBEDDING_SIZE = 384
        
        # Initialize with fixed-size embedding
        embedding_features = np.zeros(EMBEDDING_SIZE)
        
        # Prioritize semantic embeddings (most important for content similarity)
        if 'embedding_vector' in file_data and file_data['embedding_vector'] is not None:
            embedding = file_data['embedding_vector']
            if isinstance(embedding, (list, np.ndarray)):
                embedding_array = np.array(embedding).flatten()
                if len(embedding_array) == EMBEDDING_SIZE:
                    embedding_features = embedding_array * 1.5  # Moderate weight for semantic content
                else:
                    # Handle unexpected embedding sizes
                    if len(embedding_array) > EMBEDDING_SIZE:
                        embedding_features = embedding_array[:EMBEDDING_SIZE] * 1.5
                    else:
                        embedding_features[:len(embedding_array)] = embedding_array * 1.5
        
        # Add content-agnostic semantic features
        file_path = file_data.get('file_path', '')
        content_type = file_data.get('content_type', 'unknown')
        
        if content_type == 'text' and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                semantic_features = self.extract_semantic_features(content)
                    
            except Exception as e:
                print(f"Error extracting features from {file_path}: {e}")
                semantic_features = np.zeros(15)
        else:
            semantic_features = np.zeros(15)
        
        # Add minimal metadata (de-emphasized)
        metadata_features = np.array([
            file_data.get('word_count', 0) / 10000.0,  # Normalized word count (reduced weight)
            file_data.get('line_count', 0) / 1000.0,   # Normalized line count (reduced weight)
        ])
        
        # Combine all features into a single vector with consistent size
        all_features = np.concatenate([
            embedding_features,      # 384 features
            semantic_features,       # 15 features  
            metadata_features        # 2 features
        ])
        
        return all_features.astype(float)


class AdaptiveClusterValidator:
    """Validates and determines optimal cluster counts using multiple metrics."""
    
    def determine_optimal_clusters(self, features: np.ndarray, min_k: int = 2, max_k: int = None) -> int:
        """Determine optimal number of clusters using multiple validation metrics."""
        n_samples = features.shape[0]
        
        if n_samples < 2:
            return 1
        elif n_samples < 4:
            return 2
        
        if max_k is None:
            max_k = min(4, max(2, n_samples // 6))  # Very conservative upper bound
        
        max_k = min(max_k, n_samples - 1)
        
        if min_k >= max_k:
            return min_k
        
        scores = {}
        
        for k in range(min_k, max_k + 1):
            try:
                clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = clustering.fit_predict(features)
                
                # Skip if all points in one cluster or each point is its own cluster
                unique_labels = len(set(labels))
                if unique_labels <= 1 or unique_labels == n_samples:
                    continue
                
                # Calculate multiple metrics
                silhouette = silhouette_score(features, labels)
                calinski = calinski_harabasz_score(features, labels)
                davies = davies_bouldin_score(features, labels)
                
                # Combined score (higher is better)
                # Normalize davies_bouldin (lower is better) by taking negative
                combined_score = silhouette + (calinski / 1000.0) - davies
                
                scores[k] = {
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'davies': davies,
                    'combined': combined_score
                }
                
            except Exception as e:
                print(f"Error evaluating k={k}: {e}")
                continue
        
        if not scores:
            return min_k
        
        # Find k with best combined score
        best_k = max(scores.keys(), key=lambda k: scores[k]['combined'])
        
        print(f"Cluster validation scores: {scores}")
        print(f"Selected optimal k: {best_k}")
        
        return best_k


class ContentFirstClusterer:
    """Content-first clustering strategies that ignore file extensions."""
    
    def __init__(self):
        self.feature_extractor = ContentFirstFeatureExtractor()
        self.validator = AdaptiveClusterValidator()
    
    def cluster_images_by_classification(self, files: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster images based on ResNet50 classification results."""
        if len(files) < 2:
            return {0: files}
        
        # Group images by their broad category from ResNet50 classification
        category_groups = defaultdict(list)
        
        for file_data in files:
            classification = file_data.get('classification', {})
            
            if 'category' in classification:
                category = classification['category']
                category_groups[category].append(file_data)
            elif 'primary_label' in classification:
                # Fallback to primary label if no broad category
                primary_label = classification['primary_label']
                category_groups[primary_label].append(file_data)
            else:
                # Fallback for images without classification
                category_groups['unclassified'].append(file_data)
        
        # Convert to numbered clusters
        clusters = {}
        cluster_id = 0
        
        for category, category_files in category_groups.items():
            if len(category_files) == 1:
                # Single image in category
                clusters[cluster_id] = category_files
                cluster_id += 1
            elif len(category_files) <= 5:
                # Small group - keep together
                clusters[cluster_id] = category_files
                cluster_id += 1
            else:
                # Large group - potentially sub-cluster by confidence or specific labels
                sub_clusters = self._sub_cluster_images_by_confidence(category_files)
                for sub_cluster_files in sub_clusters.values():
                    clusters[cluster_id] = sub_cluster_files
                    cluster_id += 1
        
        return clusters
    
    def _sub_cluster_images_by_confidence(self, files: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Sub-cluster images within a category by confidence and specific labels."""
        # Group by primary label (more specific than broad category)
        label_groups = defaultdict(list)
        
        for file_data in files:
            classification = file_data.get('classification', {})
            primary_label = classification.get('primary_label', 'unknown')
            label_groups[primary_label].append(file_data)
        
        # Convert to numbered sub-clusters
        sub_clusters = {}
        sub_cluster_id = 0
        
        for label, label_files in label_groups.items():
            if len(label_files) <= 3:
                # Small group - keep together
                sub_clusters[sub_cluster_id] = label_files
                sub_cluster_id += 1
            else:
                # Split by confidence levels
                high_conf = []
                low_conf = []
                
                for file_data in label_files:
                    classification = file_data.get('classification', {})
                    confidence = classification.get('confidence', 0.0)
                    
                    if confidence > 0.5:  # High confidence threshold
                        high_conf.append(file_data)
                    else:
                        low_conf.append(file_data)
                
                if high_conf:
                    sub_clusters[sub_cluster_id] = high_conf
                    sub_cluster_id += 1
                
                if low_conf:
                    sub_clusters[sub_cluster_id] = low_conf
                    sub_cluster_id += 1
        
        return sub_clusters
    
    def cluster_by_content_similarity(self, files: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster files purely by content similarity, ignoring file types."""
        if len(files) < 2:
            return {0: files}
        
        # For small groups, don't cluster at all - keep them together
        if len(files) <= 5:
            return {0: files}
        
        try:
            # Extract content-focused features
            features = []
            for file_data in files:
                feature_vector = self.feature_extractor.combine_features(file_data)
                features.append(feature_vector)
            
            # Convert to numpy array and check for consistency
            features = np.array(features)
            
            # Verify all feature vectors have the same length
            if features.ndim != 2:
                print(f"Warning: Inconsistent feature shapes detected. Creating single cluster.")
                return {0: files}
            
            features = self._normalize_features(features)
            
            # Use very conservative clustering - only split if really necessary
            min_cluster_size = max(4, len(files) // 3)  # Very conservative cluster sizes
            
            # If min_cluster_size is too large, just keep everything together
            if min_cluster_size >= len(files) - 1:
                return {0: files}
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3,  # Require at least 3 samples to form a cluster
                metric='euclidean',
                cluster_selection_epsilon=0.5  # Require high similarity to form clusters
            )
            
            labels = clusterer.fit_predict(features)
            
            # If HDBSCAN creates too many clusters, fall back to single cluster
            unique_labels = set(labels)
            num_real_clusters = len([l for l in unique_labels if l != -1])
            
            if num_real_clusters > 3:  # Limit to max 3 clusters per domain
                return {0: files}
            
            # Handle noise points (label -1) - put them in the largest cluster
            if -1 in unique_labels:
                noise_indices = [i for i in range(len(labels)) if labels[i] == -1]
                if noise_indices:
                    if num_real_clusters > 0:
                        # Find the largest cluster and add noise points to it
                        cluster_sizes = Counter([l for l in labels if l != -1])
                        if cluster_sizes:
                            largest_cluster = cluster_sizes.most_common(1)[0][0]
                            for noise_idx in noise_indices:
                                labels[noise_idx] = largest_cluster
                    else:
                        # All points are noise, create single cluster
                        return {0: files}
            
            return self._organize_clusters(files, labels)
            
        except Exception as e:
            print(f"Error in HDBSCAN clustering: {e}")
            print("Creating single cluster as fallback...")
            return {0: files}
    
    def _cluster_with_agglomerative(self, files: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Helper method for agglomerative clustering with content focus."""
        # For small groups, don't cluster at all
        if len(files) <= 6:
            return {0: files}
            
        try:
            features = []
            for file_data in files:
                feature_vector = self.feature_extractor.combine_features(file_data)
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Verify feature consistency
            if features.ndim != 2:
                print(f"Warning: Feature shape inconsistency in agglomerative clustering. Creating single cluster.")
                return {0: files}
            
            features = self._normalize_features(features)
            
            # Very conservative cluster count - maximum 3 clusters
            max_clusters = min(3, len(files) // 4)
            if max_clusters < 2:
                return {0: files}
            
            optimal_k = self.validator.determine_optimal_clusters(
                features, 
                min_k=2, 
                max_k=max_clusters
            )
            
            clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
            labels = clustering.fit_predict(features)
            
            return self._organize_clusters(files, labels)
            
        except Exception as e:
            print(f"Error in agglomerative clustering: {e}")
            print("Creating single cluster as fallback...")
            return {0: files}
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for better clustering."""
        if features.shape[0] <= 1:
            return features
        
        try:
            # Handle NaN and infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Standard scaling
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # L2 normalization
            normalized_features = normalize(scaled_features, norm='l2')
            
            return normalized_features
        except Exception as e:
            print(f"Error normalizing features: {e}")
            return features
    
    def _organize_clusters(self, files: List[Dict[str, Any]], labels: np.ndarray) -> Dict[int, List[Dict[str, Any]]]:
        """Organize files into clusters based on labels."""
        clusters = defaultdict(list)
        for i, file_data in enumerate(files):
            cluster_id = int(labels[i])
            clusters[cluster_id].append(file_data)
        return dict(clusters)


class ContentFirstFileClustering:
    """Main clustering class that prioritizes content over file types."""
    
    def __init__(self):
        self.domain_detector = ContentFirstDomainDetector()
        self.content_clusterer = ContentFirstClusterer()
    
    def cluster_files(self, file_features: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Main clustering method implementing content-first approach."""
        print("Starting content-first clustering...")
        
        # Stage 1: Light categorization by content domain (not file type)
        categorized = self.domain_detector.categorize_files(file_features)
        print(f"Categorized files into {len(categorized)} content domains: {list(categorized.keys())}")
        
        # Stage 2: Cluster within domains by content similarity
        all_clusters = {}
        cluster_id_offset = 0
        
        for domain, files in categorized.items():
            if not files:
                continue
            
            if domain == 'image':
                print(f"Clustering {len(files)} images using ResNet50 classification...")
                
                if len(files) == 1:
                    # Single image
                    all_clusters[cluster_id_offset] = files
                    cluster_id_offset += 1
                    continue
                
                # Use ResNet50-based image clustering
                domain_clusters = self.content_clusterer.cluster_images_by_classification(files)
                
                # Merge with global clusters
                for local_id, cluster_files in domain_clusters.items():
                    all_clusters[cluster_id_offset + local_id] = cluster_files
                
                cluster_id_offset += len(domain_clusters)
                
            else:
                print(f"Clustering {len(files)} files in domain '{domain}' by content similarity...")
                
                if len(files) == 1:
                    # Single file clusters
                    all_clusters[cluster_id_offset] = files
                    cluster_id_offset += 1
                    continue
                
                # Cluster by content similarity regardless of file type
                domain_clusters = self.content_clusterer.cluster_by_content_similarity(files)
                
                # Merge with global clusters
                for local_id, cluster_files in domain_clusters.items():
                    all_clusters[cluster_id_offset + local_id] = cluster_files
                
                cluster_id_offset += len(domain_clusters)
        
        print(f"Content-first clustering complete. Created {len(all_clusters)} clusters.")
        return all_clusters


# Main function to replace the original clustering
def perform_enhanced_clustering(file_features: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Content-first clustering function that prioritizes semantic similarity over file extensions.
    
    Args:
        file_features: List of file feature dictionaries from analyzers
    
    Returns:
        Dictionary mapping cluster IDs to lists of file data
    """
    clusterer = ContentFirstFileClustering()
    return clusterer.cluster_files(file_features) 