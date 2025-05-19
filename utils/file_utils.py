#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Utilities

This module provides functions for file system operations:
- Directory traversal
- File type detection
- File content reading
- Metadata extraction
"""

import os
import mimetypes
import hashlib
from typing import Dict, List, Any, Tuple


def traverse_directory(directory_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Recursively traverses a directory and collects file information.

    Args:
        directory_path: Path to the directory to traverse

    Returns:
        Dictionary with file paths as keys and metadata as values
    """
    files_data = {}

    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Skip hidden files
            if os.path.basename(file_path).startswith('.'):
                continue

            # Get basic file metadata
            try:
                file_stat = os.stat(file_path)
                file_size = file_stat.st_size
                file_mtime = file_stat.st_mtime

                # Store file metadata
                files_data[file_path] = {
                    'size': file_size,
                    'modified_time': file_mtime,
                    'filename': filename,
                    'extension': os.path.splitext(filename)[1].lower(),
                    'relative_path': os.path.relpath(file_path, directory_path)
                }
            except (PermissionError, FileNotFoundError) as e:
                print(f"Warning: Could not access file {file_path}: {e}")
                continue

    return files_data


def detect_file_type(file_path: str) -> str:
    """
    Detects the type of a file based on MIME type.

    Args:
        file_path: Path to the file

    Returns:
        String indicating the file type ('text', 'image', or 'unknown')
    """
    # Initialize mimetypes if not already done
    if not mimetypes.inited:
        mimetypes.init()

    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type is None:
        return 'unknown'

    if mime_type.startswith('text/'):
        return 'text'
    elif mime_type.startswith('image/'):
        return 'image'
    else:
        return 'unknown'


def get_file_content(file_path: str, file_type: str) -> Any:
    """
    Reads the content of a file based on its type.

    Args:
        file_path: Path to the file
        file_type: Type of the file ('text', 'image', or 'unknown')

    Returns:
        File content in appropriate format based on file type
    """
    try:
        if file_type == 'text':
            # For text files, read as string
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif file_type == 'image':
            # For images, just return the path for now
            # Later, this would use something like PIL to read the image
            return file_path
        else:
            # For unknown types, just return the file path
            return file_path
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return None


def calculate_file_hash(file_path: str) -> str:
    """
    Calculates MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash as hexadecimal string
    """
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # Read in 64k chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Warning: Could not calculate hash for {file_path}: {e}")
        return ''
