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

    return parser.parse_args()


def display_organization_plan(organization_policy: Dict[str, Any]) -> None:
    """
    Display the organization plan to the user.

    Args:
        organization_policy: Dictionary containing the organization policy
    """
    print("\n===== File Organization Plan =====\n")

    # This is a placeholder - will be implemented with actual policy data
    print("The following categories will be created:")

    # Placeholder example categories
    example_categories = {
        "Documents": ["doc1.txt", "note.md"],
        "Images": ["photo.jpg", "diagram.png"],
    }

    for category, files in example_categories.items():
        print(f"\n{category}:")
        for file in files:
            print(f"  - {file}")

    print("\n==================================\n")


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
