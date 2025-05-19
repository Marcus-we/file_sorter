#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Policy Generator

This module creates organization policies based on clustering results:
- Defines target directory structure
- Maps files to their destination folders
- Handles file naming conflicts
"""

import os
import json
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict


def generate_category_path(category_name: str) -> str:
    """
    Generates a suitable filesystem path from a category name.

    Args:
        category_name: The category name

    Returns:
        Filesystem-safe path name
    """
    # Replace spaces with underscores and remove special characters
    safe_name = ''.join(c if c.isalnum() or c in [
                        ' ', '_', '-'] else '_' for c in category_name)
    safe_name = safe_name.replace(' ', '_')

    # Ensure the name is not too long for filesystems
    if len(safe_name) > 64:
        safe_name = safe_name[:61] + '...'

    return safe_name


def create_target_structure(categories: Dict[int, str], base_path: str) -> Dict[int, str]:
    """
    Creates the target directory structure based on categories.

    Args:
        categories: Dictionary mapping cluster IDs to category names
        base_path: Base directory for the organized files

    Returns:
        Dictionary mapping cluster IDs to target directory paths
    """
    target_dirs = {}

    # Keep track of used directory names to avoid conflicts
    used_names = set()

    for cluster_id, category_name in categories.items():
        # Create a safe directory name
        dir_name = generate_category_path(category_name)

        # Handle name conflicts by appending numbers
        original_name = dir_name
        counter = 1
        while dir_name in used_names:
            dir_name = f"{original_name}_{counter}"
            counter += 1

        # Add to used names
        used_names.add(dir_name)

        # Create the full path
        target_path = os.path.join(base_path, dir_name)

        # Store in the mapping
        target_dirs[cluster_id] = target_path

    return target_dirs


def resolve_file_conflicts(files_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Resolves filename conflicts in the target directories.

    Args:
        files_mapping: Dictionary mapping source file paths to target file paths

    Returns:
        Updated dictionary with conflicts resolved
    """
    # Group files by target path
    target_groups = defaultdict(list)
    for src_path, target_path in files_mapping.items():
        target_dir, filename = os.path.split(target_path)
        target_groups[(target_dir, filename)].append(src_path)

    # Resolve conflicts
    resolved_mapping = {}
    for (target_dir, filename), src_paths in target_groups.items():
        if len(src_paths) == 1:
            # No conflict
            resolved_mapping[src_paths[0]] = os.path.join(target_dir, filename)
        else:
            # Conflict - add counter to filenames
            name, ext = os.path.splitext(filename)
            for i, src_path in enumerate(src_paths):
                new_filename = f"{name}_{i+1}{ext}"
                resolved_mapping[src_path] = os.path.join(
                    target_dir, new_filename)

    return resolved_mapping


def generate_organization_policy(
    clusters: Dict[int, List[Dict[str, Any]]],
    category_names: Dict[int, str],
    output_dir: str = None,
    copy_files: bool = False
) -> Dict[str, Any]:
    """
    Generates a complete organization policy based on clustering results.

    Args:
        clusters: Dictionary mapping cluster IDs to lists of file data
        category_names: Dictionary mapping cluster IDs to category names
        output_dir: Output directory for organized files (if None, creates a subdirectory)
        copy_files: Whether to copy files instead of moving them

    Returns:
        Dictionary containing the complete organization policy
    """
    # Determine the base directory for organization
    if output_dir:
        base_dir = output_dir
    else:
        # Extract the first file path to get the parent directory
        for cluster_files in clusters.values():
            if cluster_files:
                parent_dir = os.path.dirname(
                    os.path.dirname(cluster_files[0]['file_path']))
                base_dir = os.path.join(parent_dir, "organized_files")
                break
        else:
            # Fallback if no files
            base_dir = "organized_files"

    # Create target directory mapping
    target_dirs = create_target_structure(category_names, base_dir)

    # Build file mapping
    files_mapping = {}
    moved_files_count = 0

    for cluster_id, cluster_files in clusters.items():
        # Skip empty clusters
        if not cluster_files:
            continue

        # Get target directory
        target_dir = target_dirs.get(
            cluster_id, os.path.join(base_dir, "uncategorized"))

        # Map each file to its destination
        for file_data in cluster_files:
            if 'file_path' in file_data and 'error' not in file_data:
                src_path = file_data['file_path']
                filename = os.path.basename(src_path)
                target_path = os.path.join(target_dir, filename)
                files_mapping[src_path] = target_path
                moved_files_count += 1

    # Resolve filename conflicts
    files_mapping = resolve_file_conflicts(files_mapping)

    # Create the policy
    policy = {
        'base_directory': base_dir,
        'categories': {str(cid): {
            'name': name,
            'directory': target_dirs[cid],
            'file_count': len(clusters.get(cid, []))
        } for cid, name in category_names.items() if cid in clusters},
        'files_mapping': files_mapping,
        'total_files': moved_files_count,
        'copy_files': copy_files
    }

    return policy


def save_policy_to_file(policy: Dict[str, Any], output_file: str) -> None:
    """
    Saves the organization policy to a JSON file.

    Args:
        policy: The organization policy dictionary
        output_file: Path to the output file
    """
    # Convert file paths to strings for JSON serialization
    serializable_policy = dict(policy)

    # Convert non-string dictionary keys to strings
    if 'categories' in serializable_policy:
        serializable_policy['categories'] = {
            str(key): value for key, value in serializable_policy['categories'].items()
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_policy, f, indent=2, ensure_ascii=False)


def load_policy_from_file(input_file: str) -> Dict[str, Any]:
    """
    Loads an organization policy from a JSON file.

    Args:
        input_file: Path to the input file

    Returns:
        The organization policy dictionary
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        policy = json.load(f)

    return policy
