# Modal App Pattern for Garden-AI

**Referenced from:** SKILL.md (Phase 7: Generate Code)

Load this file when generating Modal apps for Garden-AI publication.

**For complete working examples**, see modal-examples.md

---

## CRITICAL Requirements

**Every Modal app MUST have:**

1. ✅ `import modal` and `app = modal.App(name)`
2. ✅ `image` defined with ALL dependencies (Python AND system packages)
3. ✅ **Check repository for system dependencies** - Use `apt_install()` for non-Python packages
4. ✅ All functions use `@app.function(image=image, ...)` decorator
5. ✅ **ALL imports INSIDE functions** (not at module level) - this is REQUIRED for remote execution
6. ✅ `@app.local_entrypoint()` for testing (NOT `if __name__ == "__main__"`)
7. ✅ **For classes: Use `@modal.enter()` for initialization** (NOT `__init__` for model loading)
8. ✅ **For classes: Use `@modal.method()` on all public methods**
9. ✅ Type hints on all parameters and returns
10. ✅ Comprehensive docstrings referencing scientific source

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
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("package-name")  # System packages (if needed)
    .pip_install(
        "package1==1.2.3",  # Use exact versions from repo requirements
        "package2==4.5.6",
    )
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

**Note on Classes:** If model loading is expensive (>5 seconds), use `@app.cls` with `@modal.enter()` for initialization instead of `@app.function`. See the "When to Use @app.cls vs @app.function" section below for the complete class pattern with `@modal.enter()`.

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

### 3. Use `@modal.enter()` for Class Initialization (NOT `__init__`)

**CRITICAL: For `@app.cls`, use `@modal.enter()` for model loading, NOT `__init__`**

```python
# ✅ CORRECT - Use @modal.enter() for setup
@app.cls(image=image, gpu="T4")
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Called once when container starts on Modal infrastructure."""
        import torch
        self.model = torch.load("model.pt")
        self.model.eval()
        print("Model loaded!")

    @modal.method()
    def predict(self, data: list[str]) -> list[float]:
        """Called for each request - model already loaded."""
        # Use self.model here
        return self.model.predict(data)

# ❌ WRONG - Don't use __init__ for model loading
@app.cls(image=image, gpu="T4")
class ModelInference:
    def __init__(self):
        """This runs LOCALLY during serialization, NOT on Modal!"""
        import torch
        self.model = torch.load("model.pt")  # ❌ Fails - file not on local machine
```

**Why this matters:**
- `__init__()` runs **locally** when Python serializes your class (before sending to Modal)
- `@modal.enter()` runs **remotely** once per container startup on Modal's infrastructure
- Model files, GPUs, and dependencies are only available in the remote container
- Using `__init__` will fail because the model file isn't on your local machine

**Optional cleanup with `@modal.exit()`:**
```python
@app.cls(image=image)
class ModelInference:
    @modal.enter()
    def setup(self):
        self.model = load_large_model()
        self.temp_files = []

    @modal.exit()
    def cleanup(self):
        """Called when container shuts down. Optional."""
        # Clean up temporary files, close connections, etc.
        for f in self.temp_files:
            f.unlink()
```

**Key rules:**
- ✅ Use `@modal.enter()` for: Loading models, initializing resources, allocating memory
- ✅ Use `@modal.method()` for: Public methods users will call
- ❌ Don't use `__init__` for: Model loading, GPU operations, accessing remote files
- ⚠️ Use `__init__` only for: Simple attribute initialization with literals/constants (rare)

### 4. Installing System Packages with apt_install()

**IMPORTANT: Many ML models require system libraries (non-Python packages)**

```python
# ✅ CORRECT - Install system dependencies with apt_install()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",        # OpenCV needs OpenGL
        "libglib2.0-0",  # Often needed with OpenCV
        "ffmpeg",        # Audio/video processing
    )
    .pip_install(
        "opencv-python==4.8.0",
        "torch==2.0.1",
    )
)

# ❌ WRONG - Missing system dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "opencv-python==4.8.0",  # Will fail: ImportError: libGL.so.1
)
```

**Common system packages needed by ML models:**

| Python Package | Requires System Package | Error Without It |
|----------------|-------------------------|------------------|
| `opencv-python` | `libgl1`, `libglib2.0-0` | `ImportError: libGL.so.1` |
| `rdkit` | `libxrender1`, `libxext6` | Rendering errors |
| Audio processing | `ffmpeg`, `libsndfile1` | Cannot load audio files |
| `matplotlib` (headless) | `libgomp1` | Import errors |
| `gdal` | `gdal-bin`, `libgdal-dev` | Geospatial processing fails |
| Molecular viz | `libxrender1` | Visualization errors |

**How to find what system packages you need:**

1. **Check repository documentation** - Look for Dockerfile, apt-get commands, or system requirements
2. **Look for import errors** - When testing, errors like `libXYZ.so not found` tell you what's missing
3. **Check Python package docs** - Many packages list system dependencies in their installation guides
4. **Common patterns**:
   - Computer vision (OpenCV, Pillow with special features) → OpenGL/X11 libraries
   - Audio/video (librosa, torchaudio) → ffmpeg, libsndfile
   - Scientific computing (some scipy features) → BLAS/LAPACK libraries
   - Molecular modeling → Rendering and chemistry libraries

**Order matters - apt_install() before pip_install():**

```python
# ✅ CORRECT order
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1")      # System packages FIRST
    .pip_install("opencv-python")  # Then Python packages
)

# ❌ Wrong - Python package installed before system dependency
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python")  # Might work but not recommended
    .apt_install("libgl1")
)
```

**Checking repository for system dependencies:**

```python
# Look for these in the repository:
# - Dockerfile: RUN apt-get install <packages>
# - environment.yml: Often has comments about system deps
# - README.md: Installation section mentioning apt/brew packages
# - setup.py or pyproject.toml: Sometimes documents system requirements
```

**groundhog_hpc note:** groundhog does NOT support apt_install(). If a model requires system packages that aren't already on the HPC system, you must work with the HPC admin to install them or choose Modal instead.

### 5. Resource Specifications

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
| Using `__init__` for model loading in classes | Use `@modal.enter()` for initialization |
| Missing system dependencies | Check repo for system packages, use `apt_install()` |
| `apt_install()` after `pip_install()` | Always put `apt_install()` before `pip_install()` |
| Missing `image` in decorator | Add `image=image` to `@app.function()` |
| Forgetting `@modal.method()` on class methods | Add decorator to all public methods |
| No type hints | Add to all parameters and returns |
| No error handling | Use try/except per batch item |

## When to Use @app.cls vs @app.function

**Use `@app.cls` when:**
- Need to load large model once and reuse (avoid reloading per call)
- Model has expensive initialization (>5 seconds)
- Want to maintain state across calls within the same container
- Use `@modal.enter()` for one-time setup like model loading

**Use `@app.function` when:**
- Stateless computation
- Small/fast initialization (<5 seconds)
- Simpler and recommended for most cases
- No need for persistent state

**Example with @app.cls:**
```python
@app.cls(image=image, gpu="any")
class ModelPredictor:
    @modal.enter()
    def setup(self):
        """Called ONCE when container starts. Use for model loading."""
        import torch
        print("Loading model (happens once per container)...")
        self.model = torch.load("large_model.pt")
        self.model.eval()
        print("Model loaded and ready!")

    @modal.method()
    def predict(self, inputs: list[str]) -> list[float]:
        """Called for EACH request."""
        import torch
        # self.model is already loaded from setup()
        results = []
        for inp in inputs:
            tensor = self.preprocess(inp)
            with torch.no_grad():
                pred = self.model(tensor)
            results.append(float(pred))
        return results

    def preprocess(self, data: str):
        """Helper method (not decorated, only used internally)."""
        # preprocessing logic
        pass

    @modal.exit()
    def cleanup(self):
        """Called when container shuts down. Optional cleanup."""
        print("Container shutting down, cleaning up...")
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
- [ ] System packages with `apt_install()` if needed (check repo for system deps)
- [ ] `apt_install()` called BEFORE `pip_install()` in image definition
- [ ] All imports INSIDE functions (not module level)
- [ ] `@app.function(image=image)` on all functions
- [ ] For classes: `@modal.enter()` for initialization (NOT `__init__`)
- [ ] For classes: `@modal.method()` on all public methods
- [ ] Type hints on parameters and returns
- [ ] `@app.local_entrypoint()` for testing
- [ ] Batch processing with per-item error handling
- [ ] Output format: `{"results": [...], "summary": {...}}`

**Testing:**
```bash
# Run local entrypoint with dependencies
uv run modal run your_app.py

# This ensures modal and all dependencies are available
# Without 'uv run', you'd need modal installed globally
```

Common errors:
- `ImportError: libGL.so.1` → Missing system package, add `apt_install("libgl1")` to image
- `ImportError: libXYZ.so` → Missing system library, check repo for system dependencies
- `ImportError` in function → Move imports inside functions
- `ModuleNotFoundError` → Add Python package to `image.pip_install()`
- `AttributeError` in class methods → Use `@modal.enter()` not `__init__` for model loading
- Model file not found → `__init__` runs locally; use `@modal.enter()` which runs on Modal

---

## Next Steps

Proceed to Phase 8 in workflow-phases.md for testing and publication guidance.
