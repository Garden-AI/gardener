# Modal App Pattern for Garden-AI

**Referenced from:** SKILL.md (Phase 7: Generate Code)

Load this file when generating Modal apps for Garden-AI publication.

**For complete working examples**, see modal-examples.md

---

## CRITICAL Requirements

**Every Modal app MUST have:**

1. ✅ `import modal` and `app = modal.App(name)`
2. ✅ `image` defined with ALL dependencies
3. ✅ All functions use `@app.function(image=image, ...)` decorator
4. ✅ **ALL imports INSIDE functions** (not at module level) - this is REQUIRED for remote execution
5. ✅ `@app.local_entrypoint()` for testing (NOT `if __name__ == "__main__"`)
6. ✅ Type hints on all parameters and returns
7. ✅ Comprehensive docstrings referencing scientific source

---

## How Users Call Modal Functions

**IMPORTANT:** When functions are published on Garden-AI and accessed via the Python SDK:

```python
# ✅ Correct: Call WITHOUT .remote()
result = garden.my_function(args)
result = garden.MyClass.method_name(args)

# ❌ Wrong: Don't use .remote() in user code
result = garden.my_function.remote(args)  # Error!
```

**During development/testing in your code:**

```python
@app.local_entrypoint()
def main():
    # ✅ Use .remote() when testing before publishing
    result = my_function.remote(test_data)
```

**Garden-AI backend handles:**
- Remote execution routing
- Large data (>10MB) via S3 blob storage
- Authentication and access control

---

## Complete Pattern Template

```python
"""[Brief scientific description of what this computes]"""

import modal

app = modal.App("descriptive-app-name")

# Define container image with dependencies from repository
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "package1==1.2.3",  # Use exact versions from repo requirements
    "package2==4.5.6",
)


@app.function(image=image, gpu="T4", cpu=2.0, memory=4096)
def compute_batch(inputs: list[str], param: float = 1.0) -> dict:
    """
    [One-line description of scientific computation]

    Args:
        inputs: [Domain-appropriate input description]
        param: [Scientific meaning of parameter]

    Returns:
        {"results": [...], "summary": {...}}

    Reference:
        [Paper citation or DOI]
    """
    # ALL imports INSIDE the function
    import relevant_package

    # Load model/tool
    tool = relevant_package.load_model()

    # Process each input with error handling
    results = []
    succeeded = 0

    for idx, item in enumerate(inputs):
        result = {"index": idx, "input": item, "success": False}
        try:
            # Computation
            output = tool.process(item, param)
            result.update({"success": True, "output": output})
            succeeded += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(inputs), "succeeded": succeeded}}


@app.local_entrypoint()
def main():
    """Test with realistic data from paper/repo."""
    test_data = ["example_from_paper"]
    result = compute_batch.remote(test_data)
    print(f"Completed: {result['summary']['succeeded']}/{result['summary']['total']}")
```

**See modal-examples.md for complete working examples across different domains.**

---

## Critical Modal-Specific Patterns

### 1. Imports MUST Be Inside Functions

```python
# ❌ WRONG - Modal won't execute correctly
import torch

@app.function(image=image)
def predict(data):
    return torch.process(data)  # Fails on remote execution

# ✅ CORRECT - Import inside
@app.function(image=image)
def predict(data):
    import torch  # Import HERE
    return torch.process(data)
```

**Why:** Modal serializes functions and executes them in containers. Module-level imports aren't available in the remote context.

### 2. Use `@app.local_entrypoint()` Not `if __name__`

```python
# ✅ Correct
@app.local_entrypoint()
def main():
    result = my_function.remote(test_data)

# ❌ Wrong
if __name__ == "__main__":
    my_function.remote(test_data)
```

### 3. Resource Specifications

```python
@app.function(
    image=image,
    gpu="T4",          # "T4", "A10G", "A100", or "A100-80GB"
    cpu=2.0,           # Number of CPUs
    memory=4096,       # MB of RAM
    timeout=600,       # Seconds before timeout
)
```

**GPU sizing:**
- <1B params: `gpu="T4"` (cheapest)
- 1-7B params: `gpu="A10G"`
- 7B+ params: `gpu="A100"` or `"A100-80GB"`

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Module-level imports | Move ALL imports inside functions |
| `if __name__ == "__main__"` | Use `@app.local_entrypoint()` |
| Missing `image` in decorator | Add `image=image` to `@app.function()` |
| No type hints | Add to all parameters and returns |
| No error handling | Use try/except per batch item |

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

---

## When to Use Classes vs Functions

**Use `@app.function()` for most cases:**
- Stateless computation
- Quick model initialization (<5s)
- Recommended default

**Use `@app.cls()` when:**
- Model loading is expensive (>5s, large weights)
- Need to load model once and reuse across calls
- See modal-examples.md for class pattern

---

## Advanced Patterns

For advanced use cases, see modal-examples.md:
- **Volumes**: Cache large model weights (>100MB)
- **Secrets**: Handle API keys for HuggingFace, etc.
- **Batch optimization**: Process inputs in batches for GPU efficiency

---

## Verification Checklist

Before claiming Modal app is ready:

**Required:**
- [ ] `import modal` and `app = modal.App("name")`
- [ ] `image` defined with dependencies (exact versions)
- [ ] All imports INSIDE functions (not module level)
- [ ] `@app.function(image=image)` on all functions
- [ ] Type hints on parameters and returns
- [ ] `@app.local_entrypoint()` for testing
- [ ] Batch processing with per-item error handling
- [ ] Output format: `{"results": [...], "summary": {...}}`

**Testing:**
```bash
modal run your_app.py
```

Common errors:
- `ImportError` → Move imports inside functions
- `Module not found` → Add to `image.pip_install()`

---

## Next Steps

Proceed to Phase 8 in workflow-phases.md for testing and publication guidance.
