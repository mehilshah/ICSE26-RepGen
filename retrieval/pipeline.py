"""
Main pipeline for the Retrieval-Augmented Generation (RAG) process.

This module orchestrates the entire retrieval workflow:
1. Indexing the codebase.
2. Finding relevant code snippets (Hybrid Search).
3. Analyzing module structure.
4. Detecting training loops.
5. Analyzing dependencies.
6. Generating context files for the LLM.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from .config import Config
from .core.code_indexer import CodeIndexer
from .core.module_analyzer import ModuleAnalyzer
from .core.training_code_detector import TrainingCodeDetector
from .core.dependency_analyzer import DependencyAnalyzer
from .core.utils import load_bug_report, save_json
import json
import logging
from repgen_logging import get_logger, trace

logger = get_logger(__name__, component="retrieval.pipeline")

class RetrievalPipeline:
    """
    Orchestrates the retrieval and context generation pipeline for a specific bug.
    """
    def __init__(self, bug_id: str = "001", config: Config = None, ablation_config: Dict[str, Any] = None, retrieval_ablation_name: str = "full_system", generation_ablation_name: str = "all_steps", dataset_dir: str = None):
        """
        Initialize the pipeline.

        Args:
            bug_id: The ID of the bug to process.
            config: Optional pre-configured Config object.
            ablation_config: Dictionary of ablation flags.
            retrieval_ablation_name: Name of the retrieval ablation study.
            generation_ablation_name: Name of the generation ablation study.
            dataset_dir: Custom path to the dataset directory.
        """
        self.config = config or Config(bug_id=bug_id, ablation_config=ablation_config, retrieval_ablation_name=retrieval_ablation_name, generation_ablation_name=generation_ablation_name, dataset_dir=dataset_dir)
        self.bug_id = bug_id
        self.code_indexer = CodeIndexer(self.config)
        self.module_analyzer = ModuleAnalyzer(self.config)
        self.training_detector = TrainingCodeDetector(self.config)
        self.dependency_analyzer = DependencyAnalyzer(self.config)

    def run_pipeline(self, bug_id: str) -> Optional[Dict[str, Any]]:
        """
        Run the full retrieval pipeline for a given bug ID.

        Args:
            bug_id: The ID of the bug (redundant if set in __init__, but kept for consistency).

        Returns:
            A dictionary containing the status and output directory path, or None if failed.
        """
        try:
            bound_logger = logger.bind(bug_id=bug_id)
            # 1. Setup paths
            bug_report_path = self.config.BUG_REPORTS_DIR / f"{bug_id}.txt"
            code_dir = self.config.CODE_DIR
            
            if not bug_report_path.exists():
                raise FileNotFoundError(f"Bug report not found: {bug_report_path}")
            if not code_dir.exists():
                raise FileNotFoundError(f"Code directory not found: {code_dir}")

            # 2. Index codebase
            trace(bound_logger, "Indexing codebase", stage="retrieval", action="index_codebase", status="start")
            hybrid_index = self.code_indexer.index_codebase(code_dir)

            # 3. Find relevant code
            trace(bound_logger, "Finding relevant code", stage="retrieval", action="retrieve_chunks", status="start")
            bug_report = load_bug_report(bug_report_path)
            relevant_code = self.code_indexer.find_relevant_code(bug_report, hybrid_index)
            if not relevant_code:
                trace(bound_logger, "No relevant code found", level=logging.WARNING, stage="retrieval", action="retrieve_chunks", status="empty")
                return None

            # 4. Analyze modules
            trace(bound_logger, "Analyzing modules", stage="retrieval", action="analyze_modules", status="start", details={"snippets": len(relevant_code)})
            module_report = {
                'bug_report': bug_id,
                'modules': self.module_analyzer.analyze_modules(relevant_code)
            }
            # 5. Detect training code
            trace(bound_logger, "Detecting training code", stage="retrieval", action="detect_training_code", status="start", details={"modules": len(module_report["modules"])})
            training_report = self.training_detector.detect_training_code(module_report, bug_report_path)
  
            # # 6. Analyze dependencies
            trace(bound_logger, "Analyzing dependencies", stage="retrieval", action="analyze_dependencies", status="start", details={"training_files": len(training_report["training_files"])})
            dependency_report = self.dependency_analyzer.analyze_training_dependencies(training_report)

            # 7. Generate final context
            trace(bound_logger, "Generating final context", stage="retrieval", action="create_context_files", status="start", details={"dependencies": len(dependency_report.get("dependencies", []))})
            self.create_context_files(bug_id, module_report, dependency_report)
            return {"status": "success", "context_dir": str(self.config.CONTEXT_DIR_OUT)}
        except Exception as e:
            trace(bound_logger, f"Pipeline failed: {str(e)}", level=logging.ERROR, stage="retrieval", action="run_pipeline", status="error")
            raise

    def create_context_files(self, bug_id, module_report, dependency_report):
        """
        Generate the final context JSON files for the LLM.

        Constructs context files based on the analyzed dependencies and modules.
        If dependencies are found, creates a context file for each top dependency.
        Otherwise, falls back to creating context files for each module.

        Args:
            bug_id: The bug ID.
            module_report: Result of module analysis.
            dependency_report: Result of dependency analysis.
        """
        bug_report = dependency_report.get('bug_report')

        # Create the context directory if it doesn't exist
        context_dir = self.config.CONTEXT_DIR_OUT

        dependencies = dependency_report.get('dependencies', [])
        if not dependencies:
            for i, module in enumerate(module_report['modules'], 1):
                context = {
                    "bug_report": bug_report,
                    "module": module,
                    "module_snippets": [],
                }

                # Find relevant module snippets for this file
                for file in module['files']:
                    context['module_snippets'].append({
                        "file": file['path'],
                        "snippets": file['snippets']
                    })

                # Create the output file path
                output_filename = f"{bug_id}_module_{i}.json"
                filename = context_dir / output_filename

                # Write to file
                with open(filename, 'w') as f:
                    json.dump(context, f, indent=2)

        else:
            for i, dep in enumerate(dependencies[:5], 1):
                training_file_path = self.config.CODE_DIR / dep['file']
                try:
                    with open(training_file_path, 'r') as f:
                        training_file_content = f.read()
                except FileNotFoundError:
                    print(f"Warning: File not found - {training_file_path}")
                    training_file_content = f"Content not available for {training_file_path}"
                
                # Create the context structure
                context = {
                    "bug_report": bug_report,
                    "rank": i,
                    "score": dep['score'],
                    "main_file": {
                        "path": dep['file'],
                        "content": training_file_content
                    },
                    "module_snippets": [],
                    "dependencies": dep['external_dependencies']
                }
                
                # Find relevant module snippets for this file
                for module in module_report['modules']:
                    for file in module['files']:
                        if file['path'] == dep['file']:
                            context['module_snippets'].append({
                                "file": file['path'],
                                "snippets": file['snippets']
                            })
                        else:
                            # Check if this module file is a dependency
                            for ext_dep in dep['external_dependencies']:
                                if ext_dep['module'].replace('/', '_') in file['path']:
                                    context['module_snippets'].append({
                                        "file": file['path'],
                                        "snippets": file['snippets']
                                    })
                
                # Create the output file path
                output_filename = f"{bug_id}_{i}.json"
                filename = context_dir / output_filename

                # Write to file
                with open(filename, 'w') as f:
                    json.dump(context, f, indent=2)
