"""
Utility functions for the retrieval pipeline.

Provides helpers for path manipulation, file I/O, and text processing.
"""

import ast
import json
import logging
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import torch

def extract_module_path(file_path: str, dataset_root: str = "dataset/") -> str:
    """
    Extract full module path relative to dataset root.

    Args:
        file_path: Full file path.
        dataset_root: Root directory string to look for in path.

    Returns:
        Extracted module path or 'unknown'.
    """
    try:
        file_path = os.path.normpath(file_path)
        dataset_root = os.path.normpath(dataset_root)
        parts = file_path.split(os.sep)
        
        try:
            root_idx = parts.index(dataset_root)
        except ValueError:
            return os.path.dirname(file_path)
            
        module_parts = parts[root_idx+1:-1]  # Exclude filename
        return os.path.join(*module_parts) if module_parts else "root"
    except Exception as e:
        logging.warning(f"Path parsing error {file_path}: {e}")
        return "unknown"

def load_bug_report(bug_report_path: Path) -> str:
    """
    Load bug report content.

    Args:
        bug_report_path: Path to the bug report file.

    Returns:
        String content of the bug report.
    """
    try:
        with open(bug_report_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error loading bug report: {str(e)}")

def save_json(data: Any, file_path: Path) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to serialize.
        file_path: Destination path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: Path) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Deserialized data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize(text: str) -> List[str]:
    """
    Basic tokenizer for code search.

    Args:
        text: Input text string.

    Returns:
        List of alphanumeric tokens.
    """
    return re.findall(r'\b\w+[\w\d_]*\b', text.lower())

def get_torch_device(preferred: str = "cuda") -> str:
    """
    Return a safe torch device string.

    Falls back to CPU when CUDA appears available but cannot be initialized,
    which happens on systems with incompatible or outdated NVIDIA drivers.
    """
    if preferred != "cuda":
        return preferred

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if not torch.cuda.is_available():
                return "cpu"
            torch.empty(1, device="cuda")
        return "cuda"
    except Exception as exc:
        logging.warning("CUDA unavailable during initialization, falling back to CPU: %s", exc)
        return "cpu"
