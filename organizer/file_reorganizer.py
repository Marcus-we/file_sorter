#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Reorganizer

This module implements file organization based on the generated policy:
- Creates necessary directory structure
- Moves or copies files to their target locations
- Handles errors and logs operations
"""

import os
import shutil
from typing import Dict, List, Any, Callable
import logging


def setup_directory_structure(policy: Dict[str, Any]) -> None:
    """
    Creates the target directory structure based on the policy.

    Args:
        policy: The organization policy dictionary
    """
    # Get base directory
    base_dir = policy.get('base_directory', 'organized_files')

    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create category directories
    categories = policy.get('categories', {})
    for category_id, category_info in categories.items():
        target_dir = category_info.get('directory')
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir)


def move_file(src_path: str, dst_path: str) -> bool:
    """
    Moves a file from source to destination.

    Args:
        src_path: Source file path
        dst_path: Destination file path

    Returns:
        Boolean indicating success
    """
    try:
        # Ensure the target directory exists
        dst_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Move the file
        shutil.move(src_path, dst_path)
        return True
    except Exception as e:
        logging.error(f"Error moving file {src_path} to {dst_path}: {e}")
        return False


def copy_file(src_path: str, dst_path: str) -> bool:
    """
    Copies a file from source to destination.

    Args:
        src_path: Source file path
        dst_path: Destination file path

    Returns:
        Boolean indicating success
    """
    try:
        # Ensure the target directory exists
        dst_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Copy the file
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        logging.error(f"Error copying file {src_path} to {dst_path}: {e}")
        return False


def transfer_file(src_path: str, dst_path: str, copy_files: bool = False) -> bool:
    """
    Transfers a file from source to destination (copy or move).

    Args:
        src_path: Source file path
        dst_path: Destination file path
        copy_files: Whether to copy files instead of moving them

    Returns:
        Boolean indicating success
    """
    if copy_files:
        return copy_file(src_path, dst_path)
    else:
        return move_file(src_path, dst_path)


def reorganize_files(policy: Dict[str, Any], progress_callback: Callable = None) -> Dict[str, Any]:
    """
    Reorganizes files according to the policy.

    Args:
        policy: The organization policy dictionary
        progress_callback: Optional callback function to report progress

    Returns:
        Dictionary with results
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(policy.get(
            'base_directory', ''), 'reorganization.log')
    )

    # Create the directory structure
    setup_directory_structure(policy)

    # Get file mapping and copy flag
    files_mapping = policy.get('files_mapping', {})
    copy_files = policy.get('copy_files', False)

    # Initialize counters
    total_files = len(files_mapping)
    success_count = 0
    error_count = 0
    skipped_count = 0

    # Process each file
    for i, (src_path, dst_path) in enumerate(files_mapping.items()):
        # Skip non-existent files
        if not os.path.exists(src_path):
            logging.warning(f"Source file does not exist: {src_path}")
            skipped_count += 1
            continue

        # Skip if destination already exists and we're copying
        if os.path.exists(dst_path) and copy_files:
            logging.warning(f"Destination file already exists: {dst_path}")
            skipped_count += 1
            continue

        # Transfer the file
        success = transfer_file(src_path, dst_path, copy_files)

        if success:
            success_count += 1
            logging.info(
                f"Successfully {'copied' if copy_files else 'moved'} {src_path} to {dst_path}")
        else:
            error_count += 1

        # Report progress if callback provided
        if progress_callback and total_files > 0:
            progress_percent = (i + 1) / total_files * 100
            progress_callback(progress_percent, success_count,
                              error_count, skipped_count)

    # Create results summary
    results = {
        'total_files': total_files,
        'successful': success_count,
        'errors': error_count,
        'skipped': skipped_count,
        'operation': 'copy' if copy_files else 'move'
    }

    return results


def verify_organization(policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies that the organization was performed correctly.

    Args:
        policy: The organization policy dictionary

    Returns:
        Dictionary with verification results
    """
    files_mapping = policy.get('files_mapping', {})
    copy_files = policy.get('copy_files', False)

    verified_count = 0
    missing_sources = 0
    missing_targets = 0

    # Check each mapping
    for src_path, dst_path in files_mapping.items():
        # Check if source exists (should always exist for copy, might not for move)
        source_exists = os.path.exists(src_path)
        if not source_exists and copy_files:
            missing_sources += 1

        # Check if target exists (should always exist)
        target_exists = os.path.exists(dst_path)
        if not target_exists:
            missing_targets += 1

        # Count as verified if target exists
        if target_exists:
            verified_count += 1

    # Create verification results
    verification = {
        'total_files': len(files_mapping),
        'verified': verified_count,
        'missing_sources': missing_sources,
        'missing_targets': missing_targets,
        'success_rate': verified_count / len(files_mapping) if files_mapping else 0
    }

    return verification


def undo_reorganization(policy: Dict[str, Any], progress_callback: Callable = None) -> Dict[str, Any]:
    """
    Undoes a file reorganization.

    Args:
        policy: The organization policy dictionary
        progress_callback: Optional callback function to report progress

    Returns:
        Dictionary with results
    """
    # Get file mapping and copy flag
    files_mapping = policy.get('files_mapping', {})
    copy_files = policy.get('copy_files', False)

    # Skip undo for copy operations
    if copy_files:
        return {
            'total_files': len(files_mapping),
            'undone': 0,
            'skipped': len(files_mapping),
            'errors': 0,
            'reason': 'Copy operation - undo not required'
        }

    # Initialize counters
    total_files = len(files_mapping)
    undone_count = 0
    error_count = 0
    skipped_count = 0

    # Process each file in reverse (target to source)
    for i, (src_path, dst_path) in enumerate(files_mapping.items()):
        # Skip if source already exists
        if os.path.exists(src_path):
            logging.warning(
                f"Source file already exists, skipping undo: {src_path}")
            skipped_count += 1
            continue

        # Skip if destination doesn't exist
        if not os.path.exists(dst_path):
            logging.warning(
                f"Destination file doesn't exist, skipping undo: {dst_path}")
            skipped_count += 1
            continue

        # Move file back to original location
        success = move_file(dst_path, src_path)

        if success:
            undone_count += 1
            logging.info(f"Successfully moved back {dst_path} to {src_path}")
        else:
            error_count += 1

        # Report progress if callback provided
        if progress_callback and total_files > 0:
            progress_percent = (i + 1) / total_files * 100
            progress_callback(progress_percent, undone_count,
                              error_count, skipped_count)

    # Create results summary
    results = {
        'total_files': total_files,
        'undone': undone_count,
        'skipped': skipped_count,
        'errors': error_count
    }

    return results
