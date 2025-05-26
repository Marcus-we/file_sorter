#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Category Namer

This module generates meaningful category names for the content-first clustering system:
- Content-first naming strategies (prioritizes semantic content over file extensions)
- Better keyword extraction and analysis
- More descriptive and intuitive category names based on actual content
"""

import os
import re
from typing import Dict, List, Any, Set
from collections import Counter, defaultdict


class ContentFirstCategoryNamer:
    """Content-first category naming that prioritizes semantic content over file types."""
    
    def __init__(self):
        self.content_descriptors = {
            'academic': {
                'biology': 'Biology Research',
                'computer': 'Computer Science',
                'physics': 'Physics Studies',
                'chemistry': 'Chemistry Research',
                'psychology': 'Psychology Papers',
                'economics': 'Economics Analysis',
                'literature': 'Literature Studies',
                'mathematics': 'Mathematical Research',
                'engineering': 'Engineering Papers',
                'medicine': 'Medical Research',
                'data': 'Data Science',
                'machine': 'Machine Learning',
                'artificial': 'AI Research',
                'statistical': 'Statistical Analysis'
            },
            'business': {
                'contract': 'Contracts & Agreements',
                'invoice': 'Financial Documents',
                'proposal': 'Proposals & Plans',
                'memo': 'Communications',
                'report': 'Business Reports',
                'meeting': 'Meeting Documents',
                'budget': 'Budget & Finance',
                'project': 'Project Management',
                'strategy': 'Strategic Planning',
                'marketing': 'Marketing Materials',
                'sales': 'Sales Documents'
            },
            'programming': {
                'tutorial': 'Programming Tutorials',
                'example': 'Code Examples',
                'demo': 'Demonstrations',
                'test': 'Test Code',
                'documentation': 'Code Documentation',
                'api': 'API Documentation',
                'framework': 'Framework Code',
                'library': 'Library Code',
                'algorithm': 'Algorithms',
                'data': 'Data Processing',
                'web': 'Web Development',
                'mobile': 'Mobile Development'
            },
            'technical': {
                'manual': 'Technical Manuals',
                'guide': 'Technical Guides',
                'documentation': 'Technical Documentation',
                'specification': 'Technical Specifications',
                'installation': 'Installation Guides',
                'configuration': 'Configuration Docs',
                'troubleshooting': 'Troubleshooting Guides'
            }
        }
        
        self.semantic_indicators = {
            'tutorial': 'Tutorials',
            'example': 'Examples',
            'demo': 'Demonstrations',
            'test': 'Tests',
            'documentation': 'Documentation',
            'readme': 'Documentation',
            'guide': 'Guides',
            'manual': 'Manuals',
            'reference': 'Reference',
            'specification': 'Specifications',
            'analysis': 'Analysis',
            'report': 'Reports',
            'study': 'Studies',
            'research': 'Research',
            'review': 'Reviews'
        }
        
        # De-emphasize file extensions - only use as last resort
        self.extension_fallbacks = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.go': 'Go',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.csv': 'Data',
            '.json': 'Data',
            '.xml': 'Data',
            '.html': 'Web',
            '.css': 'Stylesheets'
        }

    def generate_category_names(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
        """Generate content-first category names for all clusters."""
        category_names = {}
        
        for cluster_id, files in clusters.items():
            if not files:
                category_names[cluster_id] = f"Empty Cluster {cluster_id}"
                continue
            
            # Analyze the cluster to determine the best content-based naming strategy
            cluster_analysis = self._analyze_cluster_content(files)
            category_name = self._generate_content_based_name(cluster_analysis, cluster_id, len(files))
            
            category_names[cluster_id] = category_name
        
        return category_names

    def _analyze_cluster_content(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a cluster focusing on content rather than file types."""
        analysis = {
            'all_keywords': [],
            'filenames': [],
            'content_themes': [],
            'semantic_patterns': [],
            'extensions': [],  # Keep but de-emphasize
            'content_type': None,
            'image_categories': [],
            'image_labels': []
        }
        
        # Check if this is an image cluster
        first_file = files[0] if files else {}
        content_type = first_file.get('content_type', 'unknown')
        analysis['content_type'] = content_type
        
        if content_type == 'image':
            # Handle image clusters differently
            return self._analyze_image_cluster(files)
        
        for file_data in files:
            file_path = file_data.get('file_path', '')
            
            # Extract filename for pattern analysis (but don't prioritize it)
            filename = os.path.basename(file_path).lower()
            analysis['filenames'].append(filename)
            
            # Extract file extension (as fallback only)
            _, ext = os.path.splitext(file_path.lower())
            if ext:
                analysis['extensions'].append(ext)
            
            # Extract and prioritize content keywords
            keywords = file_data.get('keywords', [])
            analysis['all_keywords'].extend([kw.lower() for kw in keywords])
            
            # Look for semantic patterns in content
            for keyword in keywords:
                kw_lower = keyword.lower()
                for indicator, theme in self.semantic_indicators.items():
                    if indicator in kw_lower:
                        analysis['semantic_patterns'].append(theme)
                        break
        
        # Count occurrences for analysis
        analysis['keyword_counts'] = Counter(analysis['all_keywords'])
        analysis['semantic_counts'] = Counter(analysis['semantic_patterns'])
        analysis['extension_counts'] = Counter(analysis['extensions'])
        
        # Identify content themes
        analysis['content_themes'] = self._identify_content_themes(analysis['keyword_counts'])
        
        return analysis
    
    def _analyze_image_cluster(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze an image cluster using ResNet50 classification results."""
        analysis = {
            'content_type': 'image',
            'image_categories': [],
            'image_labels': [],
            'confidence_levels': [],
            'filenames': []
        }
        
        for file_data in files:
            file_path = file_data.get('file_path', '')
            filename = os.path.basename(file_path).lower()
            analysis['filenames'].append(filename)
            
            classification = file_data.get('classification', {})
            
            # Extract broad category
            if 'category' in classification:
                analysis['image_categories'].append(classification['category'])
            
            # Extract primary label
            if 'primary_label' in classification:
                analysis['image_labels'].append(classification['primary_label'])
            
            # Extract confidence
            if 'confidence' in classification:
                analysis['confidence_levels'].append(classification['confidence'])
        
        # Count occurrences
        analysis['category_counts'] = Counter(analysis['image_categories'])
        analysis['label_counts'] = Counter(analysis['image_labels'])
        
        return analysis

    def _identify_content_themes(self, keyword_counts: Counter) -> List[str]:
        """Identify content themes from keywords."""
        themes = []
        
        # Check for academic themes
        academic_keywords = {'research', 'study', 'analysis', 'paper', 'journal', 'academic', 'theory', 'empirical'}
        if any(kw in keyword_counts for kw in academic_keywords):
            themes.append('academic')
        
        # Check for business themes
        business_keywords = {'business', 'corporate', 'financial', 'management', 'strategy', 'project', 'client'}
        if any(kw in keyword_counts for kw in business_keywords):
            themes.append('business')
        
        # Check for programming themes
        programming_keywords = {'code', 'function', 'class', 'programming', 'development', 'software', 'algorithm'}
        if any(kw in keyword_counts for kw in programming_keywords):
            themes.append('programming')
        
        # Check for technical themes
        technical_keywords = {'technical', 'documentation', 'manual', 'guide', 'specification', 'installation'}
        if any(kw in keyword_counts for kw in technical_keywords):
            themes.append('technical')
        
        return themes

    def _generate_content_based_name(self, analysis: Dict[str, Any], cluster_id: int, file_count: int) -> str:
        """Generate a descriptive name based primarily on content analysis."""
        
        # Handle image clusters using ResNet50 classification
        if analysis.get('content_type') == 'image':
            return self._generate_image_based_name(analysis, file_count)
        
        # Strategy 1: Use semantic patterns (highest priority)
        if analysis.get('semantic_counts'):
            most_common_semantic = analysis['semantic_counts'].most_common(1)[0][0]
            return f"{most_common_semantic} ({file_count})"
        
        # Strategy 2: Use content themes with specific descriptors
        if analysis.get('content_themes'):
            primary_theme = analysis['content_themes'][0]
            return self._generate_theme_based_name(primary_theme, analysis, file_count)
        
        # Strategy 3: Use top keywords for content-based naming
        top_keywords = analysis.get('keyword_counts', Counter()).most_common(5)
        if top_keywords:
            return self._generate_keyword_based_name(top_keywords, file_count)
        
        # Strategy 4: Filename pattern analysis (secondary)
        filename_pattern = self._analyze_filename_patterns(analysis['filenames'])
        if filename_pattern:
            return f"{filename_pattern} ({file_count})"
        
        # Strategy 5: Extension fallback (last resort)
        if analysis.get('extension_counts'):
            most_common_ext = analysis['extension_counts'].most_common(1)[0][0]
            if most_common_ext in self.extension_fallbacks:
                ext_name = self.extension_fallbacks[most_common_ext]
                return f"{ext_name} Files ({file_count})"
        
        # Final fallback
        return f"Mixed Content {cluster_id} ({file_count})"
    
    def _generate_image_based_name(self, analysis: Dict[str, Any], file_count: int) -> str:
        """Generate names for image clusters based on ResNet50 classification."""
        
        # Strategy 1: Use broad category from ResNet50
        if analysis.get('category_counts'):
            most_common_category = analysis['category_counts'].most_common(1)[0][0]
            category_name = most_common_category.title()
            
            # Make category names more descriptive
            category_mappings = {
                'Vehicle': 'Vehicle Images',
                'Animal': 'Animal Photos',
                'Person': 'People Photos',
                'Sport': 'Sports Images',
                'Food': 'Food Photos',
                'Landscape': 'Landscape Photos',
                'Building': 'Architecture Images',
                'Object': 'Object Photos'
            }
            
            descriptive_name = category_mappings.get(category_name, f"{category_name} Images")
            return f"{descriptive_name} ({file_count})"
        
        # Strategy 2: Use specific labels if no broad category
        if analysis.get('label_counts'):
            most_common_label = analysis['label_counts'].most_common(1)[0][0]
            # Clean up the label name
            clean_label = most_common_label.replace('_', ' ').title()
            return f"{clean_label} Images ({file_count})"
        
        # Strategy 3: Check confidence levels
        if analysis.get('confidence_levels'):
            avg_confidence = sum(analysis['confidence_levels']) / len(analysis['confidence_levels'])
            if avg_confidence > 0.8:
                confidence_desc = "High Confidence"
            elif avg_confidence > 0.5:
                confidence_desc = "Medium Confidence"
            else:
                confidence_desc = "Low Confidence"
            return f"{confidence_desc} Images ({file_count})"
        
        # Final fallback for images
        return f"Image Collection ({file_count})"

    def _generate_theme_based_name(self, theme: str, analysis: Dict[str, Any], file_count: int) -> str:
        """Generate name based on identified content theme."""
        
        keywords = [kw for kw, count in analysis['keyword_counts'].most_common(10)]
        
        if theme == 'academic':
            # Look for specific academic fields
            for field, descriptor in self.content_descriptors['academic'].items():
                if any(field in kw for kw in keywords):
                    return f"{descriptor} ({file_count})"
            return f"Academic Papers ({file_count})"
        
        elif theme == 'business':
            # Look for specific business document types
            for doc_type, descriptor in self.content_descriptors['business'].items():
                if any(doc_type in kw for kw in keywords):
                    return f"{descriptor} ({file_count})"
            return f"Business Documents ({file_count})"
        
        elif theme == 'programming':
            # Look for specific programming contexts
            for context, descriptor in self.content_descriptors['programming'].items():
                if any(context in kw for kw in keywords):
                    return f"{descriptor} ({file_count})"
            return f"Programming Content ({file_count})"
        
        elif theme == 'technical':
            # Look for specific technical document types
            for doc_type, descriptor in self.content_descriptors['technical'].items():
                if any(doc_type in kw for kw in keywords):
                    return f"{descriptor} ({file_count})"
            return f"Technical Documentation ({file_count})"
        
        return f"{theme.title()} Content ({file_count})"

    def _generate_keyword_based_name(self, top_keywords: List[tuple], file_count: int) -> str:
        """Generate name based on content keywords, filtering out noise."""
        
        # Filter out common stopwords and noise
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'should', 'could',
            'can', 'may', 'might', 'must', 'shall', 'file', 'files', 'document', 'documents'
        }
        
        meaningful_keywords = []
        for keyword, count in top_keywords:
            if (len(keyword) > 2 and 
                keyword not in stopwords and 
                not keyword.isdigit() and
                count > 1):
                meaningful_keywords.append(keyword.title())
        
        if meaningful_keywords:
            # Check for specific content indicators first
            for indicator, descriptor in self.semantic_indicators.items():
                if any(indicator.lower() in kw.lower() for kw in meaningful_keywords):
                    return f"{descriptor} ({file_count})"
            
            # Use top meaningful keywords for naming
            if len(meaningful_keywords) >= 2:
                return f"{meaningful_keywords[0]} {meaningful_keywords[1]} Content ({file_count})"
            else:
                return f"{meaningful_keywords[0]} Content ({file_count})"
        
        return f"Text Content ({file_count})"

    def _analyze_filename_patterns(self, filenames: List[str]) -> str:
        """Analyze filename patterns as secondary naming strategy."""
        
        # Look for common filename patterns
        patterns = {
            'readme': 'Documentation',
            'tutorial': 'Tutorials',
            'example': 'Examples',
            'demo': 'Demos',
            'test': 'Tests',
            'config': 'Configuration',
            'setup': 'Setup Files',
            'install': 'Installation',
            'guide': 'Guides',
            'manual': 'Manuals'
        }
        
        pattern_counts = Counter()
        for filename in filenames:
            for pattern, description in patterns.items():
                if pattern in filename:
                    pattern_counts[description] += 1
        
        if pattern_counts:
            return pattern_counts.most_common(1)[0][0]
        
        return None


# Main function to replace the original category naming
def generate_enhanced_category_names(clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
    """
    Content-first category naming function that prioritizes semantic content over file extensions.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of file data
    
    Returns:
        Dictionary mapping cluster IDs to category names
    """
    namer = ContentFirstCategoryNamer()
    return namer.generate_category_names(clusters) 