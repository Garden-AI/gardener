# Groundhog HPC Pattern for Garden-AI

**Referenced from:** SKILL.md (Phase 7: Generate Code)

Load this file when generating groundhog HPC scripts for Garden-AI publication.

**For complete working examples**, see hpc-examples.md

---

## CRITICAL Requirements

**Every groundhog HPC script MUST have:**

1. ✅ PEP 723 `# /// script` metadata at top with `requires-python` field
2. ✅ `[tool.hog.endpoint_name]` configurations for each endpoint
3. ✅ `import groundhog_hpc as hog` (exactly this form, at module level)
4. ✅ All compute functions use `@hog.function()` decorator
5. ✅ Test functions use `@hog.harness()` decorator
6. ✅ **All other imports INSIDE functions** (recommended for Garden-AI consistency)
7. ✅ Type hints and comprehensive docstrings

---

## How Users Call groundhog Functions

**IMPORTANT:** When functions are published on Garden-AI and accessed via the Python SDK:

```python
# ✅ Correct: MUST use .remote() with endpoint
result = garden.my_function.remote(args, endpoint="anvil", account="abc123")
result = garden.MyClass.method_name.remote(args, endpoint="polaris")

# ✅ Async execution
future = garden.my_function.submit(args, endpoint="anvil")
result = future.result()  # Get result later

# ❌ Wrong: Can't call directly
result = garden.my_function(args)  # Error!
```

**Required parameters:**
- `endpoint`: Which HPC system ("anvil", "polaris", "perlmutter", etc.)
- `account`: User's HPC allocation (usually required)

**Optional parameters:**
- `walltime`: Override default walltime
- `user_endpoint_config`: Custom SLURM options

**Key difference from Modal:**
- groundhog: **REQUIRES** `.remote(endpoint="name")`
- Modal: Call directly **WITHOUT** `.remote()`

---

## Complete Pattern Template

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "package1==1.2.3",  # Use exact versions from repo
#     "package2==4.5.6",
# ]
# [tool.hog.anvil]
# endpoint = "uuid-from-globus-compute-endpoint-list"
# ///

import groundhog_hpc as hog


@hog.function()
def compute_batch(inputs: list[str], param: float = 1.0) -> dict:
    """
    [One-line description of HPC computation]

    Args:
        inputs: [Domain-appropriate input description]
        param: [Scientific meaning of parameter]

    Returns:
        {"results": [...], "summary": {...}}

    Reference:
        [Paper citation or DOI]

    Example:
        import garden_ai
        garden = garden_ai.get_garden("10.26311/doi")
        result = garden.compute_batch.remote(
            data, endpoint="anvil", account="allocation"
        )
    """
    # Imports can be at module level or here
    import relevant_package

    # Process each input with error handling
    results = []
    succeeded = 0

    for idx, item in enumerate(inputs):
        result = {"index": idx, "input": item, "success": False}
        try:
            # Computation
            output = relevant_package.process(item, param)
            result.update({"success": True, "output": output})
            succeeded += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(inputs), "succeeded": succeeded}}


@hog.harness()
def test_anvil():
    """Test with realistic data from paper/repo."""
    test_data = ["example_from_paper"]
    result = compute_batch.remote(
        test_data,
        endpoint="anvil",
        account="your-account",  # User replaces
    )
    print(f"Completed: {result['summary']['succeeded']}/{result['summary']['total']}")
```

**See hpc-examples.md for complete working examples across different domains.**

---

## PEP 723 Metadata

**Required at top of file:**

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["package==version", ...]
# [tool.hog.endpoint_name]
# endpoint = "uuid-from-globus-compute"
# ///
```

**Optional endpoint configuration:**

```python
# [tool.hog.anvil]
# endpoint = "uuid-here"
# qos = "gpu"                              # SLURM QOS
# partition = "gpu-debug"                   # SLURM partition
# scheduler_options = "#SBATCH --gpus-per-node=1"
# walltime = "01:00:00"
# worker_init = "module load conda"         # Init script
```

**Finding endpoint UUIDs:**
```bash
globus-compute-endpoint list
```

---

## Critical groundhog-Specific Patterns

### 1. Import Placement

**RECOMMENDED: Put imports inside functions (except `groundhog_hpc as hog`)**

```python
# ✅ Recommended for Garden-AI (works for both Modal and groundhog)
import groundhog_hpc as hog  # Module-level is required

@hog.function()
def compute(data):
    # All other imports inside the function
    import numpy as np
    from ase import Atoms

    atoms = Atoms(data)
    return np.array(atoms.positions)
```

**Why this pattern?**
- Consistent with Modal requirements (easier to port code between platforms)
- Dependencies only loaded when function executes
- Cleaner separation of concerns
- Matches all Garden-AI examples

**Note:** Module-level imports (other than `groundhog_hpc as hog`) are *technically* allowed by groundhog, but NOT recommended for Garden-AI:

```python
# ⚠️ This works but NOT recommended for Garden-AI:
import groundhog_hpc as hog
import numpy as np  # Avoid putting other imports at module level
from ase import Atoms

@hog.function()
def compute(data):
    # Can use module-level imports, but not recommended
    atoms = Atoms(data)
    return np.array(atoms.positions)
```

### 2. Must Use .remote() with Endpoint

```python
# ✅ Correct
result = garden.my_function.remote(
    data,
    endpoint="anvil",      # Required
    account="allocation",  # Usually required
)

# ✅ Async
future = garden.my_function.submit(data, endpoint="polaris")
result = future.result()

# ❌ Wrong
result = garden.my_function(data)  # Error!
```

### 3. Test Harness Pattern

```python
@hog.harness()
def test_anvil():
    """Test on Anvil endpoint."""
    result = my_function.remote(
        test_data,
        endpoint="anvil",  # Matches [tool.hog.anvil]
        account="allocation",
    )
    print(result)
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| No PEP 723 metadata | Add `# /// script` block at top |
| Missing `requires-python` field | Add `# requires-python = ">=3.10"` to metadata |
| `from groundhog_hpc import hog` | Use `import groundhog_hpc as hog` (exactly) |
| Module-level imports (other than hog) | Put all other imports inside functions |
| Missing `@hog.function()` | Decorate all compute functions |
| No `@hog.harness()` | Add test harness |
| Using `@app.function()` | Use `@hog.function()` (not Modal's decorator) |
| Calling without `.remote()` | MUST use `.remote(endpoint="name")` |
| No type hints | Add to all parameters and returns |

---

## When to Use Classes vs Functions

**Use `@hog.function()` for most cases:**
- Standalone computations
- Recommended default

**Use class with `@hog.method()` when:**
- Multiple related methods
- Shared configuration
- See hpc-examples.md for class pattern

**Key difference:** `@hog.method()` creates static methods (no `self` parameter)

---

## Advanced Patterns

For advanced use cases, see hpc-examples.md:
- **Async execution**: Use `.submit()` for non-blocking calls
- **Multi-GPU**: Configure via `user_endpoint_config`
- **Endpoint-specific setup**: Polaris, Anvil, Perlmutter configurations

---

## Verification Checklist

Before claiming HPC script is ready:

**Metadata:**
- [ ] PEP 723 `# /// script` block at top
- [ ] `requires-python = ">=3.10"` field present
- [ ] All dependencies with versions listed
- [ ] `[tool.hog.endpoint_name]` section with UUID

**Code Structure:**
- [ ] `import groundhog_hpc as hog` (NOT `from groundhog_hpc import hog`)
- [ ] All other imports INSIDE functions (not module-level)
- [ ] `@hog.function()` on all compute functions
- [ ] `@hog.harness()` test function showing `.remote()` usage
- [ ] Type hints on all parameters and returns

**Functionality:**
- [ ] Batch processing with per-item error handling
- [ ] Output format: `{"results": [...], "summary": {...}}`

**Testing:**
```bash
# Run harness test with dependencies
uv run hog run your_script.py

# This ensures groundhog-hpc and all dependencies are available
# Without 'uv run', you'd need groundhog-hpc installed globally
```

Common errors:
- `ModuleNotFoundError` → Add to PEP 723 dependencies
- `Endpoint not found` → Check UUID in `[tool.hog.endpoint_name]`

---

## Next Steps

Proceed to Phase 8 in workflow-phases.md for testing and publication guidance.
