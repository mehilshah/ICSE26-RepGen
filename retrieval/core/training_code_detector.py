"""
Module for detecting and ranking training code files.

This module provides the `TrainingCodeDetector` class, which identifies files likely containing
model training loops using AST analysis and ranks them by relevance to the bug report.
"""

import ast
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from sentence_transformers import CrossEncoder
import os
import logging
from .utils import get_torch_device
from repgen_logging import get_logger, trace

logger = get_logger(__name__, component="retrieval.training_detector")

import ast

class TrainingLoopVisitor(ast.NodeVisitor):
    """
    AST visitor to detect training loops in:
    - TensorFlow 1.x (Session.run in loops)
    - TensorFlow 2.x (Keras APIs, GradientTape, and optimizer methods)
    - PyTorch (optimizer steps in loops, DataLoader usage)
    - PyTorch Lightning (training_step method)
    """
    
    def __init__(self):
        # Framework flags
        self.found_training = False
        
        # Context tracking
        self.loop_depth = 0               # Track nested loops
        self.gradient_tape_depth = 0       # Track GradientTape contexts
        self.current_class = None          # Track class context

    # Context management ------------------------------------------------------
    def visit_ClassDef(self, node):
        """Track class context for LightningModule detection"""
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_For(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_With(self, node):
        """Handle GradientTape and PyTorch profiler contexts"""
        # Check for GradientTape first
        if any(self._is_gradient_tape(item.context_expr) for item in node.items):
            self.gradient_tape_depth += 1
            self.generic_visit(node)
            self.gradient_tape_depth -= 1
        else:
            self.generic_visit(node)

    # Core detection logic -----------------------------------------------------
    def visit_Call(self, node):
        """Analyze method calls for framework patterns"""
        self._check_tf1_patterns(node)
        self._check_tf2_patterns(node)
        self._check_pytorch_patterns(node)
        self.generic_visit(node)

    def _check_tf1_patterns(self, node):
        """TensorFlow 1.x: Session.run in loops with training ops"""
        if self.loop_depth > 0:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "run":
                # Basic check for Session-like objects
                if isinstance(node.func.value, (ast.Name, ast.Attribute)):
                    self.found_training = True

    def _check_tf2_patterns(self, node):
        """TensorFlow 2.x detection"""
        if isinstance(node.func, ast.Attribute):
            # High-level APIs
            if node.func.attr in {"fit", "fit_generator", "train_on_batch"}:
                self.found_training = True
            
            # Optimizer methods
            if node.func.attr in {"apply_gradients", "minimize"}:
                self.found_training = True
            
            # GradientTape context patterns
            if self.gradient_tape_depth > 0:
                if node.func.attr in {"gradient", "watch"}:
                    self.found_training = True

    def _check_pytorch_patterns(self, node):
        """PyTorch detection: Training steps in loops"""
        if self.loop_depth > 0 and isinstance(node.func, ast.Attribute):
            # Core training methods
            if node.func.attr in {"backward", "step", "zero_grad"}:
                self.found_training = True
                
            # Loss calculation patterns
            if node.func.attr in {"item", "backward"} and \
                self._is_loss_node(node.func.value):
                self.found_training = True

            # DataLoader patterns
            if node.func.attr in {"to", "cuda"} and \
                self._is_dataloader_node(node.func.value):
                self.found_training = True

    # Helper methods -----------------------------------------------------------
    def _is_gradient_tape(self, node):
        """Identify GradientTape usage (direct or via tf.GradientTape)"""
        if isinstance(node, ast.Call):
            func = node.func
            return (
                (isinstance(func, ast.Attribute) and func.attr == "GradientTape") or
                (isinstance(func, ast.Name) and func.id == "GradientTape")
            )
        return False

    def _is_loss_node(self, node):
        """Heuristic to identify loss nodes (e.g., 'loss' in names)"""
        if isinstance(node, ast.Name):
            return "loss" in node.id.lower()
        elif isinstance(node, ast.Attribute):
            return "loss" in node.attr.lower()
        return False

    def _is_dataloader_node(self, node):
        """Heuristic to identify DataLoader instances"""
        if isinstance(node, ast.Name):
            return "loader" in node.id.lower() or "dataloader" in node.id.lower()
        elif isinstance(node, ast.Attribute):
            return "loader" in node.attr.lower() or "dataloader" in node.attr.lower()
        return False

    # PyTorch Lightning support -----------------------------------------------
    def visit_FunctionDef(self, node):
        """Check for LightningModule training_step hooks"""
        if self.current_class and node.name == "training_step":
            self.found_training = True
        self.generic_visit(node)

class TrainingCodeDetector:
    """
    Detects and ranks training-related code files.

    Combines AST-based pattern matching (via TrainingLoopVisitor) with
    semantic re-ranking to identify files relevant to the bug report
    that also contain training logic.
    """
    def __init__(self, config):
        self.config = config
        self.device = get_torch_device()
        self.reranker = CrossEncoder(config.RERANKER_MODEL, device=self.device)
        trace(logger, "Training detector initialized", stage="retrieval", action="init_training_detector", status="ok", details={"device": self.device, "model": config.RERANKER_MODEL})

    def contains_training(self, file_path: Path) -> bool:
        """
        Check if a file contains training-related code using AST analysis.

        Args:
            file_path: Path to the Python file.

        Returns:
            True if training patterns are detected, False otherwise.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            tree = ast.parse(code)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return False

        visitor = TrainingLoopVisitor()
        visitor.visit(tree)
        return visitor.found_training

    def _get_all_files(self, module_report: Dict[str, Any]) -> List[Path]:
        """
        Get all Python files from modules in the report.

        Args:
            module_report: Dictionary containing module structure and files.

        Returns:
            List of Path objects for all found Python files.
        """
        # --- MODIFICATION: Ablation: No Module-Centric Partitioning ---
        files = set() # Use set to avoid duplicates

        if self.config.AB_NO_MODULE_PARTITIONING:
            # Ablation: Only use files from which snippets were retrieved
            for module in module_report['modules']:
                for file_info in module['files']:
                    file_path = self.config.CODE_DIR / file_info['path']
                    if file_path.exists() and file_path.suffix == '.py':
                        files.add(file_path)
        else:
            # Original: Walk all files in the retrieved modules' directories
            for module in module_report['modules']:
                module_path = self.config.CODE_DIR / module['module']
                if not module_path.exists():
                    continue
                    
                for root, _, filenames in os.walk(module_path):
                    for filename in filenames:
                        if filename.endswith('.py'):
                            files.add(Path(root) / filename)
        
        return list(files)

    def _rank_files(self, files: List[Path], bug_report: str) -> List[Tuple[Path, float]]:
        """
        Rank files by relevance to the bug report using a CrossEncoder.

        Args:
            files: List of files to rank.
            bug_report: Text of the bug report.

        Returns:
            List of tuples (file_path, relevance_score), sorted by score descending.
        """
        if not files:
            trace(logger, "No training candidates available for ranking", stage="retrieval", action="rank_training_files", status="empty")
            return []

        pairs = [
            [bug_report[:100000], file.read_text(encoding='utf-8')[:100000]]
            for file in files
        ]
        scores = np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False
            )
        ).reshape(-1)
        
        # Normalize scores
        if len(scores) > 1:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        ranked = sorted(zip(files, scores), key=lambda x: x[1], reverse=True)
        trace(logger, "Training file ranking completed", stage="retrieval", action="rank_training_files", status="ok", details={"candidates": len(files), "ranked": len(ranked)})
        return ranked

    def detect_training_code(self, module_report: Dict[str, Any], bug_report_path: Path) -> Dict[str, Any]:
        """
        Detect and rank training-related code.

        Orchestrates the process of finding files, checking for training loops
        (if enabled), and ranking them against the bug report.

        Args:
            module_report: Output from ModuleAnalyzer.
            bug_report_path: Path to the bug report file.

        Returns:
            Dictionary containing the bug report and list of ranked training files.
        """
        bug_report = bug_report_path.read_text(encoding='utf-8')
        all_files = self._get_all_files(module_report)
        trace(logger, "Collected training-code candidates", stage="retrieval", action="collect_training_candidates", status="ok", details={"candidates": len(all_files)})
        
        training_files = []
        if self.config.AB_NO_TRAINING_LOOP_EXTRACTION:
            # Skip AST-based extraction, just use all files found
            training_files = all_files
        else:
            for file in all_files:
                if self.contains_training(file):
                    training_files.append(file)
        trace(logger, "Training loop detection completed", stage="retrieval", action="detect_training_code", status="ok", details={"detected": len(training_files), "ablation_skip_extraction": self.config.AB_NO_TRAINING_LOOP_EXTRACTION})
        
        if self.config.AB_NO_TRAINING_LOOP_RANKING:
            # Skip ranking, return all found training files with a dummy score
            ranked_files = [(file, 1.0) for file in training_files]
        else:
            # Original ranking logic
            ranked_files = self._rank_files(training_files, bug_report)
        trace(logger, "Training code report ready", stage="retrieval", action="detect_training_code", status="ok", details={"ranked_files": len(ranked_files), "ablation_skip_ranking": self.config.AB_NO_TRAINING_LOOP_RANKING})
        # --- END MODIFICATION ---
        
        return {
            'bug_report': module_report['bug_report'],
            'training_files': [
                {
                    'file': str(file.relative_to(self.config.CODE_DIR)),
                    'score': float(score),
                    'contains_training': True
                }
                for file, score in ranked_files
            ]
        }
