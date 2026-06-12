"""
Module for analyzing and organizing retrieved code snippets.

This module helps in grouping code snippets by their module structure,
making it easier to construct context for the LLM.
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import os
from .utils import extract_module_path
from repgen_logging import get_logger, trace

logger = get_logger(__name__, component="retrieval.module_analyzer")

class ModuleAnalyzer:
    """
    Analyzes code location map and groups snippets by module.
    """
    def __init__(self, config):
        self.config = config
        self.code_dir = Path(config.CODE_DIR)

    def _make_relative(self, file_path: str) -> str:
        """
        Convert absolute path to relative path from the code directory.

        Args:
            file_path: Absolute path string or Path object.

        Returns:
            Relative path string with forward slashes.
        """
        try:
            path = Path(file_path)
            # Handle both str and Path inputs
            if not isinstance(file_path, (str, Path)):
                return file_path
            
            # Make relative to the code directory
            relative_path = os.path.relpath(str(path), start=str(self.code_dir))
            return relative_path.replace('\\', '/')  # Normalize to forward slashes
        except (TypeError, ValueError):
            return file_path  # Return original if conversion fails

    def analyze_modules(self, relevant_code: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group snippets by module and file using relative paths.

        Args:
            relevant_code: List of code snippet dictionaries from search.

        Returns:
            List of module dictionaries containing files and their snippets.
        """
        module_map = defaultdict(lambda: defaultdict(list))
        
        for snippet in relevant_code:
            file_path = snippet['metadata']['source']
            relative_path = self._make_relative(file_path)
            module_path = extract_module_path(relative_path)
            
            module_map[module_path][relative_path].append({
                'start_line': snippet['metadata'].get('start_line'),
                'end_line': snippet['metadata'].get('end_line'),
                'code': snippet['page_content']
            })
        
        modules = [
            {
                'module': module,
                'files': [
                    {
                        'path': file,
                        'snippets': [
                            {
                                'lines': f"{s['start_line']}-{s['end_line']}",
                                'code': s['code']
                            }
                            for s in snippets
                        ]
                    }
                    for file, snippets in files.items()
                ]
            }
            for module, files in module_map.items()
        ]
        trace(logger, "Module analysis completed", stage="retrieval", action="analyze_modules", status="ok", details={"modules": len(modules), "snippets": len(relevant_code)})
        return modules
