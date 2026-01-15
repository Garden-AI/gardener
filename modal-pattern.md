# Modal App Pattern for Garden-AI

**Referenced from:** SKILL.md (Phase 7: Generate Code)

Load this file when generating Modal apps for Garden-AI publication.

## CRITICAL Requirements

**Every Modal app MUST have:**

1. ✅ `import modal` and `app = modal.App(name)`
2. ✅ `image` defined with ALL dependencies
3. ✅ All functions use `@app.function(image=image, ...)` decorator
4. ✅ **ALL imports INSIDE functions** (not at module level) - this is REQUIRED for remote execution
5. ✅ `@app.local_entrypoint()` for testing (NOT `if __name__ == "__main__"`)
6. ✅ Type hints on all parameters and returns
7. ✅ Comprehensive docstrings with paper references

## How Users Call Modal Functions

**IMPORTANT:** When functions are published on Garden-AI and accessed via the Python SDK:
- Users call functions **WITHOUT** `.remote()`: `garden.my_function(args)`
- For classes: `garden.MyClass.method_name(args)` (NO `.remote()`)
- Backend handles the remote execution automatically
- Large data (>10MB) automatically uses S3 blob storage

**During development/testing:**
- In your `@app.local_entrypoint()`: Call with `.remote()`: `my_function.remote(args)`
- This tests the function locally before publishing

## Complete Pattern Template

```python
"""
[Model/App Name] - Brief scientific description.

[Longer description of what this does scientifically, referencing
the paper and explaining the use case.]

Published on Garden-AI for citable scientific inference.
"""

import modal

app = modal.App("descriptive-app-name")

# Define container image with ALL dependencies
# Use EXACT versions from repository's requirements.txt
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "package1==1.2.3",
        "package2==4.5.6",
        "deep-learning-framework==x.y.z",
    )
)


# ============================================================================
# Helper Functions (no decorators - called by remote functions)
# ============================================================================

def _helper_function(data):
    """
    Helper that processes data.

    Put imports INSIDE helper functions too if they're only used here.
    """
    # Imports can go inside helpers if framework-specific
    from some_package import specific_tool

    result = specific_tool(data)
    return result


# ============================================================================
# Remote Functions (all decorated with @app.function)
# ============================================================================

@app.function(image=image, cpu=1.0, memory=2048)
def scientific_task_name(inputs: list[str], param: float = 1.0) -> list[dict]:
    """
    One-line description of what this does scientifically.

    Longer description explaining the scientific meaning, when to use
    this function, and what the results mean. Reference the paper here.

    Args:
        inputs: Description using domain terminology
                Example: ["SMILES strings for molecules"]
        param: Scientific meaning of this parameter
                Example: "Temperature in Kelvin for simulation"

    Returns:
        List of dicts with:
        - input: Original input for traceability
        - prediction: Main scientific result
        - metadata: Additional context (confidence, features, etc.)

    Reference:
        Paper Title, Authors, DOI or arXiv ID
    """
    # ALL imports must be inside the function
    import numpy as np
    from domain_package import SpecificTool

    # Initialize any models or tools
    tool = SpecificTool()

    results = []
    for item in inputs:
        try:
            # Process using helper or direct computation
            processed = _helper_function(item)

            # Main computation
            prediction = tool.predict(processed, param)

            results.append({
                "input": item,
                "prediction": float(prediction),
                "metadata": {"param_used": param},
                "status": "success",
            })

        except Exception as e:
            # Graceful error handling
            results.append({
                "input": item,
                "status": "error",
                "error": str(e),
            })

    return results


# Add more @app.function() decorated functions as needed
# Each one follows the same pattern: imports inside, type hints, docstring


# ============================================================================
# Local Test Entrypoint
# ============================================================================

@app.local_entrypoint()
def main():
    """
    Test the Modal app locally before deploying.

    Use realistic examples from the paper or repository.
    """
    # Real example from paper (cite figure/table if possible)
    test_inputs = ["example from paper abstract or methods"]

    print("=" * 70)
    print("TESTING [MODEL NAME] ON MODAL")
    print("=" * 70)
    print(f"Test inputs: {test_inputs}\n")

    # Call the remote function
    results = scientific_task_name.remote(test_inputs, param=1.0)

    print("Results:")
    for r in results:
        print(f"  {r}")

    print("\n" + "=" * 70)
    print("Expected: [What outputs should look like from paper]")
    print("=" * 70)
```

## Key Modal Patterns

### 1. App Definition
```python
import modal

# Use descriptive name (lowercase, hyphens)
app = modal.App("protein-folding-predictor")
```

### 2. Image with Dependencies
```python
# Start with base image
image = modal.Image.debian_slim(python_version="3.11")

# Add dependencies from requirements.txt
image = image.pip_install(
    "torch==2.0.1",
    "numpy==1.24.0",
    # ... all dependencies with EXACT versions
)

# Can chain other setup if needed
image = image.run_commands(
    "apt-get update",
    "apt-get install -y some-system-package",
)
```

### 3. Function Decorator
```python
@app.function(
    image=image,        # Required
    gpu="any",          # If GPU needed: "any", "t4", "a100", etc.
    cpu=1.0,            # Number of CPUs
    memory=4096,        # Memory in MB
    timeout=300,        # Timeout in seconds
)
def my_function(args):
    # Imports go HERE
    import torch
    # ...
```

### 4. Imports Inside Functions
```python
# ❌ WRONG - imports at module level
import torch
import numpy as np

@app.function(image=image)
def predict(data):
    # These imports won't work in remote execution
    model = torch.load("model.pt")
    return model(data)

# ✅ CORRECT - imports inside function
@app.function(image=image)
def predict(data):
    # Import inside the function
    import torch
    import numpy as np

    # Now they work in remote execution
    model = torch.load("model.pt")
    return np.array(model(data))
```

### 5. Local Entrypoint
```python
# ✅ CORRECT - Modal pattern
@app.local_entrypoint()
def main():
    result = my_function.remote(["test"])
    print(result)

# ❌ WRONG - Don't use if __name__ == "__main__"
if __name__ == "__main__":
    result = my_function.remote(["test"])
    print(result)
```

## Common Mistakes

| Mistake | Why It Fails | Fix |
|---------|--------------|-----|
| Module-level imports | Remote execution can't access them | Move ALL imports inside functions |
| `if __name__ == "__main__"` | Not the Modal pattern | Use `@app.local_entrypoint()` |
| Missing `image` param | Function doesn't know environment | Add `image=image` to decorator |
| `.remote()` inside `@app.function()` | Wrong execution context | Only call `.remote()` from entrypoint |
| No type hints | Garden-AI needs them | Add types to all params and returns |
| Vague function names | Users don't understand purpose | Use domain-specific names |
| Missing docstring references | No context for users | Cite paper in docstrings |
| Relative imports | Don't work in Modal | Use absolute imports or inline code |
| Classes for simple functions | Over-complexity | Use simple functions unless state needed |
| No error handling | Failures kill entire batch | Try/except around each item |

## When to Use @app.cls vs @app.function

**Use `@app.cls` when:**
- Need to load large model once and reuse (avoid reloading per call)
- Model has expensive initialization
- Want to maintain state across calls

**Use `@app.function` when:**
- Stateless computation
- Small/fast initialization
- Simpler and recommended for most cases

**Example with @app.cls:**
```python
@app.cls(image=image, gpu="any")
class ModelPredictor:
    def __init__(self):
        """Called once per container startup."""
        import torch
        self.model = torch.load("large_model.pt")
        self.model.eval()

    @modal.method
    def predict(self, inputs: list[str]) -> list[float]:
        """Called for each request."""
        import torch
        # self.model is already loaded
        results = []
        for inp in inputs:
            tensor = self.preprocess(inp)
            with torch.no_grad():
                pred = self.model(tensor)
            results.append(float(pred))
        return results

    def preprocess(self, data: str):
        """Helper method."""
        # preprocessing logic
        pass
```

## Verification Checklist

Before claiming Modal app is ready:

**Structure:**
- [ ] `import modal` at top
- [ ] `app = modal.App("name")` defined
- [ ] `image` defined with dependencies
- [ ] ALL dependencies have version numbers
- [ ] Version numbers match repository requirements

**Functions:**
- [ ] All functions have `@app.function(image=image, ...)` decorator
- [ ] CPU/memory/GPU settings appropriate for workload
- [ ] Imports are INSIDE functions (not at module level)
- [ ] Type hints on all parameters and return types
- [ ] Comprehensive docstrings with paper reference
- [ ] Function names describe scientific task
- [ ] Error handling for batch processing

**Testing:**
- [ ] `@app.local_entrypoint()` defined (not `if __name__`)
- [ ] Test uses realistic examples from paper/repo
- [ ] Test shows expected output format
- [ ] `.remote()` called only from entrypoint

**Documentation:**
- [ ] Module docstring explains scientific purpose
- [ ] Functions reference the paper (DOI/arXiv)
- [ ] Input/output formats clearly documented
- [ ] Scientific meaning of parameters explained

## Advanced Modal Patterns

Use these patterns for common challenges with large models and authenticated services.

### Caching Model Weights with Volumes

**When to use:** Model checkpoint is >100MB or takes >30s to download

**Pattern:**
```python
import modal

app = modal.App("my-model")

# Create persistent volume for model storage
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.0.1",
    "huggingface-hub==0.19.4",
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},  # Mount volume at /models
    timeout=3600,
)
def predict(inputs: list[str]) -> list[dict]:
    import torch
    from pathlib import Path

    model_path = Path("/models/my_checkpoint.pt")

    # Download once, reuse on subsequent runs
    if not model_path.exists():
        # Download logic here
        import requests
        response = requests.get("https://example.com/checkpoint.pt")
        model_path.write_bytes(response.content)
        volume.commit()  # Persist to volume

    # Load from volume (fast on subsequent runs)
    model = torch.load(model_path)
    # ... rest of inference
```

**Performance impact:**
- First run: Downloads and caches (~30-60s)
- Subsequent runs: Loads from volume (~2-5s)
- Volume persists across function invocations

**Common mistake:**
```python
# ❌ DON'T: Download on every invocation
@app.function(...)
def predict(inputs):
    model = download_model()  # Downloads every time! Slow and wasteful
```

### Using Secrets for API Keys

**When to use:** Model needs HuggingFace, Weights & Biases, or other authenticated services

**Pattern:**
```python
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict(inputs: list[str]) -> list[dict]:
    import os
    from transformers import AutoModel

    # Secret is available as environment variable
    hf_token = os.environ["HF_TOKEN"]

    model = AutoModel.from_pretrained(
        "gated-model/checkpoint",
        token=hf_token,
    )
    # ...
```

**Setup instructions for user:**
```bash
# User must create secret first (one-time setup):
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxx
```

**In your code docstring, tell users:**
```python
"""
Setup required:
    modal secret create huggingface-secret HF_TOKEN=your_token

Get token from: https://huggingface.co/settings/tokens
"""
```

### GPU Configuration Guide

Choose GPU based on model size:

| Model Size | GPU Type | Memory | Relative Cost |
|------------|----------|--------|---------------|
| <1B params | `"T4"` | 16GB | $ |
| 1-7B params | `"A10G"` | 24GB | $$ |
| 7-13B params | `"A100"` | 40GB | $$$ |
| 13-70B params | `"A100-80GB"` | 80GB | $$$$ |

**Pattern:**
```python
@app.function(
    image=image,
    gpu="A10G",  # Adjust based on model size
    timeout=600,  # Increase for large models
    container_idle_timeout=300,  # Keep warm for repeated calls
)
def predict(inputs: list[str]) -> list[dict]:
    # ...
```

**For CPU-only models:**
```python
@app.function(
    image=image,
    # No gpu parameter = CPU only
    cpu=2.0,  # Number of CPUs
    memory=4096,  # MB of RAM
)
```

### Batch Processing Optimization

**Problem:** Processing items one-by-one is slow
**Solution:** Batch inputs internally

```python
@app.function(image=image, gpu="A10G")
def predict(inputs: list[str], batch_size: int = 32) -> list[dict]:
    """Process inputs in batches for better GPU utilization."""
    import torch

    # Load model once
    model = load_model()

    results = []
    # Process in batches
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]

        # Batch inference is MUCH faster than one-by-one
        batch_results = model(batch)

        results.extend(batch_results)

    return results
```

**Performance:**
- One-by-one: 100 items × 0.5s = 50s
- Batched (32): 4 batches × 2s = 8s (6× faster!)

---

## Real-World Testing

**Test locally before publishing:**
```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Run locally (uses Modal's infrastructure)
modal run your_app.py
```

**Common errors and fixes:**
- `ImportError`: Move imports inside functions
- `AttributeError on .remote()`: Add `@app.local_entrypoint()` decorator
- `Image not found`: Define `image` variable before using in decorator
- `Module not found`: Add dependency to `image.pip_install()`

## Complete Working Example

See the drug discovery example in the Garden-AI repository:
`garden_ai/agent/agent.py` lines 8-202

This shows proper:
- Import placement (inside functions)
- Multiple related functions
- Error handling per item
- Rich output structure
- Test harness with realistic data

## When Modal App is Ready

Proceed to Phase 8 in workflow-phases.md to create test harness and publication guidance.
