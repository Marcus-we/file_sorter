#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Organization Program

This program analyzes a folder structure and organizes files based on their content
using clustering techniques.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Any

# Import local modules
from utils.file_utils import traverse_directory, get_file_content, detect_file_type
from utils.cli_utils import parse_args, display_organization_plan, get_user_confirmation, progress_bar

# Uncommented these imports
from analyzers.text_analyzer import analyze_text_file
from analyzers.image_analyzer import analyze_image_file
from clustering.cluster_engine import perform_clustering
from clustering.category_namer import generate_category_names
from organizer.policy_generator import generate_organization_policy, save_policy_to_file
from organizer.file_reorganizer import reorganize_files, verify_organization


def analyze_file(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Analyze a file based on its type.
    
    Args:
        file_path: Path to the file
        file_type: Type of the file ('text', 'image', or 'unknown')
        
    Returns:
        Dictionary with analysis results
    """
    if file_type == 'text':
        return analyze_text_file(file_path)
    elif file_type == 'image':
        return analyze_image_file(file_path)
    else:
        # Basic metadata for unknown file types
        return {
            'content_type': 'unknown',
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1].lower()
        }


def main():
    """Main entry point for the file organization program."""
    # Parse command-line arguments
    args = parse_args()

    print(
        f"Starting file organization in {'headless' if args.headless else 'interactive'} mode.")
    print(f"Target directory: {args.directory}")
    
    # Verify directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist or is not accessible.")
        return 1

    try:
        # Step 1: Traverse directory and collect files
        print("Scanning directory...")
        files_data = traverse_directory(args.directory)
        
        if not files_data:
            print("No files found in the specified directory.")
            return 0
            
        print(f"Found {len(files_data)} files.")

        # Step 2: Analyze files
        print("Analyzing file content...")
        file_features = []
        total_files = len(files_data)
        
        start_time = time.time()
        for i, (file_path, metadata) in enumerate(files_data.items()):
            file_type = detect_file_type(file_path)
            # Show progress
            progress_bar(i + 1, total_files, prefix='Progress:', suffix=f'Processing {os.path.basename(file_path)}')
            # Analyze the file
            feature = analyze_file(file_path, file_type)
            file_features.append(feature)
        
        analysis_time = time.time() - start_time
        print(f"\nAnalysis complete in {analysis_time:.2f} seconds.")
        
        # Check if we have enough files to proceed
        if len(file_features) < 2:
            print("Not enough files with extractable features to perform clustering.")
            return 0

        # Step 3: Cluster files
        print("Clustering files...")
        start_time = time.time()
        clusters = perform_clustering(file_features)
        
        if not clusters:
            print("Clustering failed or no meaningful clusters were found.")
            return 1
            
        category_names = generate_category_names(clusters)
        clustering_time = time.time() - start_time
        print(f"Clustering complete in {clustering_time:.2f} seconds. Found {len(clusters)} categories.")

        # Step 4: Generate organization policy
        print("Generating organization policy...")
        organization_policy = generate_organization_policy(
            clusters, 
            category_names, 
            output_dir=args.output_dir,
            copy_files=args.dry_run
        )
        
        # Save the policy for reference
        policy_file = os.path.join(organization_policy['base_directory'], 'organization_policy.json')
        try:
            # Create the base directory if it doesn't exist
            os.makedirs(os.path.dirname(policy_file), exist_ok=True)
            save_policy_to_file(organization_policy, policy_file)
            print(f"Organization policy saved to {policy_file}")
        except Exception as e:
            print(f"Note: Could not save policy file: {e}")

        # Step 5: Get user confirmation (if not headless)
        should_proceed = True
        if not args.headless:
            print("Displaying organization plan...")
            display_organization_plan(organization_policy)
            should_proceed = get_user_confirmation()

        # Step 6: Apply organization (if confirmed)
        if should_proceed:
            print("Applying organization...")
            start_time = time.time()
            result = reorganize_files(organization_policy, progress_callback=progress_bar)
            organization_time = time.time() - start_time
            
            # Verify the results
            verification = verify_organization(organization_policy)
            
            print(f"Organization complete in {organization_time:.2f} seconds!")
            print(f"Summary: {result['successful']} files organized, {result['errors']} errors, {result['skipped']} skipped.")
            
            if verification['success_rate'] < 1.0:
                print(f"Warning: Only {verification['success_rate']*100:.1f}% of files were successfully verified.")
                if verification['missing_targets'] > 0:
                    print(f"  - {verification['missing_targets']} files did not reach their destination.")
                    
            print(f"Files organized according to their content into {len(clusters)} categories.")
            print(f"Organization complete in directory: {organization_policy['base_directory']}")
        else:
            print("Organization cancelled by user.")

        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
