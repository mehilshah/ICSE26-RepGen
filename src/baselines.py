"""
Script for generating bug reproduction code using various baselines.

Supports Zero-Shot, Few-Shot, and Chain-of-Thought prompting techniques
with multiple LLM backends (Ollama, OpenAI, Groq, DeepSeek).
"""

import argparse
import os
import logging
import subprocess
import time
import requests
from pathlib import Path
import json
from repgen_logging import setup_logging, trace

logger = setup_logging(logging.INFO).bind(component="baseline.generator")

class CodeGenerator:
    """
    Handles code generation using advanced prompting techniques with configurable backends.
    """
    
    def __init__(self, backend: str, model_name: str):
        self.backend = backend.lower()
        self.model_name = model_name
        self.examples = self._get_high_quality_examples()
        
        # Pre-check for API keys to fail fast
        if self.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable is missing!")
        elif self.backend == "groq" and not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY environment variable is missing!")
        elif self.backend == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
            logger.warning("DEEPSEEK_API_KEY environment variable is missing!")

    def generate(self, bug_report: str, technique: str = "zero_shot", n_examples: int = 3) -> str:
        """Generate reproduction code using specified technique and backend"""
        prompt = self._build_prompt(bug_report, technique, n_examples)
        trace(logger, "Baseline generation starting", stage="generation", action="baseline_generate", status="start", backend=self.backend, model=self.model_name, details={"technique": technique, "examples": n_examples, "prompt_chars": len(prompt)})

        if self.backend == "ollama":
            return self._ollama_local(prompt)
        elif self.backend == "deepseek":
            return self._deepseek_api(prompt)
        elif self.backend == "groq":
            return self._groq_api(prompt)
        elif self.backend == "openai":
            return self._openai_api(prompt)
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            return ""
    
    def _build_prompt(self, bug_report: str, technique: str, n_examples: int = 3) -> str:
        """
        Build optimized prompt based on the selected technique.

        Args:
            bug_report: The bug report text.
            technique: One of 'zero_shot', 'few_shot', 'cot'.
            n_examples: Number of examples for few-shot learning.

        Returns:
            Formatted prompt string.
        """
        if technique == "zero_shot":
            return self._zero_shot_prompt(bug_report)
        elif technique == "few_shot":
            return self._few_shot_prompt(bug_report, n_examples)
        elif technique == "cot":
            return self._cot_prompt(bug_report)
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def _zero_shot_prompt(self, bug_report: str) -> str:
        """Generate a zero-shot prompt requesting a minimal reproduction script."""
        return f"""Generate a minimal, self-contained Python script to reproduce this bug:

Bug Report:
{bug_report}

Requirements:
1. Complete runnable code with all necessary imports
2. Minimal setup to reproduce the issue
3. No explanations or comments
4. Code wrapped in ```python``` block"""

    def _few_shot_prompt(self, bug_report: str, n_examples: int = 3) -> str:
        """Generate a few-shot prompt with high-quality examples."""
        examples = "\n\n".join(
            f"===== EXAMPLE {i+1} =====\n"
            f"Bug Type: {ex['type']}\n"
            f"Bug Description: {ex['description']}\n"
            f"Reproduction Code:\n{ex['code']}"
            for i, ex in enumerate(self.examples[:n_examples])
        )
        
        return f"""Below are examples of high-quality bug reproduction code.

{examples}

Now generate reproduction code for this new bug:

Bug Report:
{bug_report}

Guidelines:
1. Follow the same format as the examples
2. Keep the code minimal but complete
3. Include all necessary imports
4. Wrap the final code in ```python```"""

    def _cot_prompt(self, bug_report: str) -> str:
        """Generate a Chain-of-Thought prompt encouraging step-by-step reasoning."""
        return f"""Let's analyze and reproduce this bug step by step:

Bug Report:
{bug_report}

Thinking Process:
1. What are the key symptoms of this bug?
2. What Python components are needed to reproduce it?
3. What minimal setup is required?
4. What specific conditions trigger the bug?
5. How can we isolate the core issue?

Now generate the reproduction code:
1. Start with necessary imports
2. Set up minimal environment
3. Add triggering conditions
4. Wrap final code in ```python```"""

    def _get_high_quality_examples(self) -> list:
        """Retrieve a list of curated bug reproduction examples."""
        # Full list of examples from your original file
        return [
            {
                "type": "Gradient Vanishing",
                "description": "Gradients become too small in deep networks with sigmoid activation",
                "code": "import torch\nimport torch.nn as nn\n\nclass DeepNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.Sequential(\n            nn.Linear(784, 512),\n            nn.Sigmoid(),\n            nn.Linear(512, 512),\n            nn.Sigmoid(),\n            nn.Linear(512, 512),\n            nn.Sigmoid(),\n            nn.Linear(512, 10)\n        )\n    \n    def forward(self, x):\n        return self.layers(x)\n\nmodel = DeepNet()\noptimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n\nfor name, param in model.named_parameters():\n    if 'weight' in name:\n        print(f\"{name} gradient norm: {param.grad.norm() if param.grad is not None else 0}\")"
            },
            {
                "type": "Shape Mismatch in CNN",
                "description": "Incorrect tensor shapes in CNN architecture",
                "code": "import torch\nimport torch.nn as nn\n\nclass BadCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)\n        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)\n        self.fc1 = nn.Linear(32*28*28, 10)\n    \n    def forward(self, x):\n        x = torch.relu(self.conv1(x))\n        x = torch.relu(self.conv2(x))\n        x = x.view(x.size(0), -1)\n        return self.fc1(x)\n\nmodel = BadCNN()\nx = torch.randn(4, 3, 32, 32)\noutput = model(x)"
            },
            {
                "type": "CUDA Memory Leak",
                "description": "Memory accumulation due to retained computational graph",
                "code": "import torch\nimport torch.nn as nn\n\nmodel = nn.Linear(1000, 1000).cuda()\nlosses = []\n\nfor i in range(1000):\n    x = torch.randn(100, 1000).cuda()\n    y = torch.randn(100, 1000).cuda()\n    output = model(x)\n    loss = nn.MSELoss()(output, y)\n    losses.append(loss)\n    if i % 100 == 0:\n        print(f\"GPU Memory: {torch.cuda.memory_allocated()}\")"
            }
        ]

    def _extract_code(self, text: str) -> str:
        """
        Extract Python code block from model response.

        Args:
            text: Raw model output.

        Returns:
            Extracted code string, stripped of markdown fences.
        """
        start_marker = "```python"
        end_marker = "```"
        
        start_idx = text.find(start_marker)
        if start_idx == -1:
            start_idx = text.find(end_marker)
            if start_idx != -1:
                start_idx += len(end_marker)
        else:
            start_idx += len(start_marker)
        
        if start_idx == -1:
            return text.strip()
        
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            return text[start_idx:].strip()
        
        code = text[start_idx:end_idx].strip()
        return code.replace(start_marker, '').replace(end_marker, '')

    def _ollama_local(self, prompt: str) -> str:
        """Generate code using local Ollama models via subprocess."""
        cmd = ["ollama", "run", self.model_name, prompt]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=True
            )
            return self._extract_code(result.stdout)
        except subprocess.TimeoutExpired:
            logger.error(f"{self.model_name} timed out after 5 minutes")
            return ""
        except subprocess.CalledProcessError as e:
            logger.error(f"{self.model_name} failed: {e.stderr}")
            return ""
    
    def _openai_api(self, prompt: str) -> str:
        """Generate code using OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4096
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_code(content)
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return ""

    def _groq_api(self, prompt: str) -> str:
        """Generate code using Groq Cloud API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        try:
            time.sleep(1) # Gentle rate limiting
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_code(content)
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return ""

    def _deepseek_api(self, prompt: str) -> str:
        """Generate code using DeepSeek API."""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_code(content)
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            return ""

def main():
    """
    Main function for running baseline code generation.

    Parses arguments for bug ID, backend, model, technique, and dataset path,
    then executes the generation process.
    """
    parser = argparse.ArgumentParser(description="Advanced bug reproduction code generation")
    parser.add_argument("--bug_id", required=True, help="Bug ID to reproduce")
    
    # New Configurable Arguments
    parser.add_argument("--backend", default="ollama", 
                       choices=["ollama", "openai", "groq", "deepseek"],
                       help="The backend provider to use (default: ollama)")
    
    parser.add_argument("--model", default="llama3-8b", 
                       help="Specific model name (e.g. gpt-4.1, deepseek-reasoner, llama3.3-70b-versatile)")
    
    parser.add_argument("--technique", default="zero_shot", 
                       choices=["zero_shot", "few_shot", "cot"],
                       help="Prompting technique")
    
    parser.add_argument("--examples", type=int, default=3, 
                       help="Number of examples for few-shot learning")
    
    parser.add_argument("--dataset_path", default="./dataset", 
                        help="Root path of the dataset. Must contain '{bug_id}/bug_report/{bug_id}.txt'.")

    args = parser.parse_args()
    
    # Construct paths using the configurable dataset path
    dataset_root = Path(args.dataset_path)
    bug_report_path = dataset_root / args.bug_id / "bug_report" / f"{args.bug_id}.txt"
    output_dir = dataset_root / args.bug_id / "reproduction_code"
    
    if not bug_report_path.exists():
        logger.error(f"Bug report not found at: {bug_report_path}")
        return

    try:
        with open(bug_report_path, "r", encoding="utf-8") as f:
            bug_report = f.read()
    except Exception as e:
        logger.error(f"Failed to read bug report: {e}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating for {args.bug_id} using {args.backend} : {args.model}")
    generator = CodeGenerator(args.backend, args.model)
    
    start_time = time.time()
    code = generator.generate(bug_report, args.technique, args.examples)
    elapsed = time.time() - start_time
    
    if code:
        # Save file with a safe name based on model and technique
        safe_model_name = args.model.replace(":", "-").replace("/", "-")
        output_path = output_dir / f"{safe_model_name}_{args.technique}.py"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Generated code in {elapsed:.1f}s → {output_path}")
    else:
        logger.error("Failed to generate code (empty output)")

if __name__ == "__main__":
    main()
