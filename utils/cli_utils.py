#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI Utilities

This module provides functions for command-line interface:
- Argument parsing
- User interaction
- Display functions
"""

import os
import argparse
import sys
from typing import Dict, List, Any, Optional


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Organize files based on content using local AI clustering."
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Directory to organize",
        nargs="?",
        default=os.getcwd()
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without user interaction"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for organized files (default: creates subdirectory)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a visualization of the clusters"
    )
    
    parser.add_argument(
        "--vis-method",
        type=str,
        choices=["tsne", "pca"],
        default="tsne",
        help="Method to use for dimensionality reduction in visualization (default: tsne)"
    )
    
    parser.add_argument(
        "--vis-output",
        type=str,
        help="Path to save the visualization image (default: temp file)"
    )

    return parser.parse_args()


def display_organization_plan(organization_policy: Dict[str, Any]) -> None:
    """
    Display the organization plan to the user.

    Args:
        organization_policy: Dictionary containing the organization policy
    """
    print("\n===== File Organization Plan =====\n")

    # Check if we have a valid policy
    if not organization_policy or 'categories' not in organization_policy:
        print("No organization plan available.")
        return

    # Get base directory
    base_dir = organization_policy.get('base_directory', 'organized_files')
    print(f"Base directory: {base_dir}")
    
    # Get operation type
    operation = "Copying" if organization_policy.get('copy_files', False) else "Moving"
    total_files = organization_policy.get('total_files', 0)
    print(f"{operation} {total_files} files into the following categories:\n")

    # Display categories and sample files
    categories = organization_policy.get('categories', {})
    files_mapping = organization_policy.get('files_mapping', {})
    
    # Group files by category
    category_files = {}
    for src_path, dst_path in files_mapping.items():
        # Find which category this file belongs to
        for cat_id, cat_info in categories.items():
            cat_dir = cat_info.get('directory', '')
            if dst_path.startswith(cat_dir):
                if cat_id not in category_files:
                    category_files[cat_id] = []
                category_files[cat_id].append((src_path, dst_path))
                break
    
    # Display each category with sample files
    for cat_id, cat_info in categories.items():
        cat_name = cat_info.get('name', f"Category {cat_id}")
        file_count = cat_info.get('file_count', 0)
        print(f"{cat_name} ({file_count} files):")
        
        # Show sample files (up to 5)
        if cat_id in category_files:
            sample_files = category_files[cat_id][:5]
            for src_path, _ in sample_files:
                print(f"  - {os.path.basename(src_path)}")
            
            # Indicate if there are more files
            if len(category_files[cat_id]) > 5:
                print(f"  - ... and {len(category_files[cat_id]) - 5} more files")
        
        print()

    print("==================================\n")


def get_user_confirmation() -> bool:
    """
    Get confirmation from the user to proceed with organization.

    Returns:
        Boolean indicating whether the user confirmed
    """
    while True:
        response = input(
            "Do you want to proceed with this organization? (y/n): ").strip().lower()

        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '',
                 decimals: int = 1, length: int = 50, fill: str = '█') -> None:
    """
    Display a command-line progress bar.

    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places for percentage
        length: Character length of bar
        fill: Bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

    # Print new line on complete
    if iteration == total:
        print()
