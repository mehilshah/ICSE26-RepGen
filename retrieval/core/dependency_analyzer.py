"""
Module for analyzing file dependencies.

This module provides the `DependencyAnalyzer` class to identify external dependencies
(imports and function calls) within the codebase to understand component interactions.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import logging
from repgen_logging import get_logger, trace

logger = get_logger(__name__, component="retrieval.dependency_analyzer")

class DependencyAnalyzer:
    """
    Analyzes Python files to extract external dependencies.
    """
    def __init__(self, config):
        self.config = config
        self.code_dir = Path(config.CODE_DIR).resolve()
        self.ignore_modules = {'os', 'sys', 're', 'math', 'tensorflow', 'pytorch', 'numpy', 'pandas'}

    def _make_relative(self, path: Path) -> str:
        """
        Convert path to relative path from the code directory.

        Args:
            path: Absolute Path object.

        Returns:
            Relative path string.
        """
        try:
            return str(path.relative_to(self.code_dir))
        except ValueError:
            return str(path)

    def _get_function_calls(self, node: ast.AST) -> Set[Tuple[str, str]]:
        """
        Extract function calls from AST nodes.

        Args:
            node: AST node to traverse.

        Returns:
            Set of tuples (module_variable, function_name).
        """
        calls = set()
        
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # Handle method calls like x.y()
                if isinstance(node.func.value, ast.Name):
                    calls.add((node.func.value.id, node.func.attr))
            elif isinstance(node.func, ast.Name):
                # Handle direct function calls like x()
                calls.add(('', node.func.id))
        
        return calls

    def analyze_file_dependencies(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Analyze a file and return its external dependencies.

        Parses imports and function calls to identify which external modules
        and functions are being used.

        Args:
            file_path: Path to the file to analyze.

        Returns:
            List of dictionaries identifying module and function dependencies.
        """
        if self.config.AB_NO_DEPENDENCY_EXTRACTION:
            return [] # Skip all dependency analysis
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                tree = ast.parse(f.read())
        except Exception as e:
            logger.debug(f"Could not parse {file_path}: {e}")
            return []

        imports = {}
        dependencies = set()

        # First pass: collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in self.ignore_modules:
                        imports[alias.asname or alias.name.split('.')[0]] = module
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # Only absolute imports
                    module = node.module.split('.')[0]
                    if module not in self.ignore_modules:
                        for alias in node.names:
                            imports[alias.asname or alias.name] = f"{module}.{alias.name}"

        # Second pass: find function calls
        for node in ast.walk(tree):
            for (module, func) in self._get_function_calls(node):
                if module in imports:
                    # Resolve the actual module from imports
                    actual_module = imports[module].split('.')[0]
                    dependencies.add((actual_module, func))
                elif not module:  # Direct function call
                    if func in imports:
                        full_path = imports[func].split('.')
                        if len(full_path) > 1:
                            dependencies.add((full_path[0], full_path[1]))

        # Convert to the desired output format
        return [{"module": m, "function": f} for m, f in sorted(dependencies)]

    def analyze_training_dependencies(self, training_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dependencies for all detected training files.

        Args:
            training_report: Output from TrainingCodeDetector.

        Returns:
            Dictionary containing bug report and dependency analysis for each file.
        """
        report = []
        trace(logger, "Analyzing training file dependencies", stage="retrieval", action="analyze_dependencies", status="start", details={"training_files": len(training_report['training_files'])})
        
        for entry in training_report['training_files']:
            try:
                file_path = (self.code_dir / entry['file']).resolve()
                if not file_path.exists():
                    continue

                external_deps = self.analyze_file_dependencies(file_path)
                
                report.append({
                    "file": self._make_relative(file_path),
                    "score": entry['score'],
                    "contains_training": entry['contains_training'],
                    "external_dependencies": external_deps
                })

            except Exception as e:
                logger.warning(f"Error processing {entry.get('file')}: {e}")
                continue
        
        trace(logger, "Dependency analysis completed", stage="retrieval", action="analyze_dependencies", status="ok", details={"dependency_reports": len(report)})
        return {
            "bug_report": training_report["bug_report"],
            "dependencies": sorted(report, key=lambda x: x["score"], reverse=True)
        }
