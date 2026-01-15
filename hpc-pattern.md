# Groundhog HPC Pattern for Garden-AI

**Referenced from:** SKILL.md (Phase 7: Generate Code)

Load this file when generating groundhog HPC scripts for Garden-AI publication.

## CRITICAL Requirements

**Every groundhog HPC script MUST have:**

1. ✅ PEP 723 `# /// script` metadata at top
2. ✅ `[tool.hog.endpoint_name]` configurations for each endpoint
3. ✅ `import groundhog_hpc as hog`
4. ✅ All compute functions use `@hog.function()` decorator
5. ✅ Test functions use `@hog.harness()` decorator
6. ✅ Imports can be at **module level** (unlike Modal - this is OK for groundhog!)
7. ✅ Shows `.remote(endpoint="name")` usage in docstrings and harness
8. ✅ Type hints and comprehensive docstrings

## How Users Call groundhog Functions

**IMPORTANT:** When functions are published on Garden-AI and accessed via the Python SDK:
- Users **MUST use `.remote()`**: `garden.my_function.remote(args, endpoint="anvil", account="abc123")`
- Or async with `.submit()`: `future = garden.my_function.submit(args, endpoint="anvil")`
- For classes: `garden.MyClass.method_name.remote(args, endpoint="anvil")`
- **Must specify `endpoint` parameter** - which HPC system to run on
- Optional: `account` (HPC allocation), `walltime`, `user_endpoint_config` (SLURM options)

**Key difference from Modal:**
- groundhog REQUIRES `.remote()` or `.submit()` - direct calls are not supported
- Modal functions are called directly WITHOUT `.remote()` via Garden SDK

## Complete Pattern Template

```python
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy==1.24.0",
#     "ase==3.22.1",
#     "torch==2.0.1",
#     # ... ALL dependencies with EXACT versions from repo
# ]
#
# [tool.hog.anvil]
# endpoint = "user-endpoint-uuid-here"
# qos = "gpu"
# partition = "gpu-debug"
# scheduler_options = "#SBATCH --gpus-per-node=1"
#
# [tool.hog.polaris]
# endpoint = "another-endpoint-uuid"
# worker_init = "
# module load conda
# module load cuda
# export PATH=$HOME/.local/bin:$PATH
# "
# ///

import groundhog_hpc as hog

# Module-level imports are OK for groundhog (unlike Modal)
import os
from pathlib import Path
import numpy as np
from ase.io import read
from io import StringIO


@hog.function()
def scientific_computation(
    input_data: str,
    param: float = 1.0,
    max_iterations: int = 100,
) -> dict:
    """
    One-line description of the HPC computation.

    Longer description explaining what this computes, when to use it,
    and what the results mean scientifically. Reference the paper.

    Args:
        input_data: Description using domain terminology
                    Example: "XYZ format atomic structure as string"
        param: Scientific meaning of parameter
                Example: "Convergence threshold in eV/Angstrom"
        max_iterations: Maximum computation steps

    Returns:
        Dict with:
        - result: Main computational output
        - metadata: Convergence info, timing, etc.
        - status: "completed" or "failed"

    Reference:
        Paper Title, Authors, DOI

    Example:
        # Via Garden SDK (after publishing):
        import garden_ai
        garden = garden_ai.get_garden("10.26311/doi")
        result = garden.scientific_computation.remote(
            xyz_structure,
            endpoint="anvil",
            account="abc123",
        )
    """

    # Helper function can be defined inside main function
    def parse_input(data_str):
        # Parsing logic
        return parsed

    # Main computation logic
    try:
        # Parse input
        parsed_data = parse_input(input_data)

        # Run computation
        result = compute_something(parsed_data, param, max_iterations)

        # Format output
        return {
            "result": result,
            "metadata": {
                "param_used": param,
                "iterations": max_iterations,
            },
            "status": "completed",
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }


@hog.harness()
def test_anvil():
    """
    Test harness for Anvil HPC endpoint.

    Uses realistic example from paper or repository.
    """
    # Real example from paper's methods or benchmark section
    test_input = """2
    Lattice="3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0"
    H 0.0 0.0 0.0
    H 1.5 1.5 1.5
    """

    print("Testing on Anvil...")
    result = scientific_computation.remote(
        test_input,
        endpoint="anvil",
        account="your-account-here",  # User fills in
        param=0.01,
    )

    print(f"Result: {result}")
    print("Expected: [Description from paper]")


@hog.harness()
def test_polaris():
    """Test harness for Polaris HPC endpoint."""
    test_input = """Example from paper"""

    print("Testing on Polaris...")
    result = scientific_computation.remote(
        test_input,
        endpoint="polaris",
        account="your-project",
        param=0.01,
    )

    print(f"Result: {result}")
```

## PEP 723 Metadata Requirements

### Basic Structure
```python
# /// script
# requires-python = "==3.12.*"  # Specify Python version
# dependencies = [
#     "package1==version",
#     "package2==version",
#     # ... ALL dependencies
# ]
# ///
```

### Endpoint Configuration
```python
# [tool.hog.endpoint_name]
# endpoint = "uuid-from-globus-compute"
# qos = "gpu"                    # SLURM QOS
# partition = "gpu-debug"         # SLURM partition
# scheduler_options = "#SBATCH --gpus-per-node=1"  # Extra SLURM directives
# walltime = "01:00:00"           # Optional walltime
# worker_init = "                 # Optional initialization script
# module load software
# export PATH=$HOME/.local/bin:$PATH
# "
```

### Finding Endpoint UUIDs
Users get these from:
```bash
globus-compute-endpoint list
```
Or from their HPC system administrator.

## Key HPC Patterns

### 1. Import Statement
```python
# Always import as 'hog'
import groundhog_hpc as hog
```

### 2. Function Decorator
```python
@hog.function()
def compute(data: str) -> dict:
    # ALL imports inside
    import numpy as np
    from ase import Atoms

    # Computation logic
    return result
```

### 3. Import Patterns (Flexible)

**groundhog_hpc allows imports at module level** (unlike Modal):

```python
# ✅ Option 1: Module-level imports (WORKS for groundhog!)
import numpy as np
from ase import Atoms

@hog.function()
def compute(data):
    # Can use module-level imports
    atoms = Atoms(data)
    return np.array(atoms.positions)

# ✅ Option 2: Imports inside function (also works)
@hog.function()
def compute(data):
    import numpy as np
    from ase import Atoms

    atoms = Atoms(data)
    return np.array(atoms.positions)
```

**Recommendation:** Use module-level imports for groundhog (cleaner code).
**Note:** Modal REQUIRES imports inside functions - don't confuse the two!

### 4. Harness Functions
```python
@hog.harness()
def test_endpoint_name():
    """
    Test function showing usage.

    NOT deployed - only for local testing and documentation.
    """
    result = my_function.remote(
        "test data",
        endpoint="endpoint_name",  # Matches [tool.hog.endpoint_name]
        account="project-account",  # User's HPC account
    )
    print(result)
```

### 5. Remote Execution
```python
# Call .remote() with endpoint specified
result = compute_function.remote(
    input_data,
    endpoint="anvil",           # Which HPC system
    account="abc123-gpu",       # User's allocation
    walltime="02:00:00",        # Optional override
    user_endpoint_config={      # Optional additional config
        "qos": "high-priority",
    },
)
```

## Common Mistakes

| Mistake | Why It Fails | Fix |
|---------|--------------|-----|
| No PEP 723 metadata | Dependencies unclear | Add `# /// script` block at top |
| Missing endpoint configs | Can't deploy to HPC | Add `[tool.hog.endpoint_name]` sections |
| No `@hog.function()` | Not executable on HPC | Decorate all compute functions |
| No `@hog.harness()` | No usage examples | Create test harness for each endpoint |
| No `import groundhog_hpc` | Can't use decorators | Add `import groundhog_hpc as hog` |
| Confusing with Modal | Wrong import rules/calling | groundhog: imports OK at module level, MUST use `.remote()` |
| Using `@app.function()` | Wrong decorator | Use `@hog.function()` for groundhog |
| Calling without `.remote()` | Not how groundhog works | Always call with `.remote(endpoint="name")` |
| No type hints | Garden-AI needs them | Add types to all parameters/returns |
| Vague function names | Users confused | Use domain-specific names |
| No endpoint in `.remote()` | Don't know where to run | Always specify `endpoint="name"` |

## When to Use Multiple Functions

**Create separate `@hog.function()` functions when:**
- Different computational tasks (relaxation vs property calculation)
- Different resource requirements (CPU vs GPU intensive)
- Distinct scientific purposes

**Example:**
```python
@hog.function()
def relax_structure(xyz: str) -> dict:
    """Geometry optimization."""
    # ...

@hog.function()
def calculate_properties(xyz: str) -> dict:
    """Single-point energy and properties."""
    # ...

@hog.function()
def run_molecular_dynamics(xyz: str, steps: int) -> dict:
    """MD simulation."""
    # ...
```

## Using Classes with groundhog_hpc

**When to use classes:**
- Multiple related methods that share state or configuration
- Complex workflows with multiple steps
- Models that need initialization with parameters

**Pattern:**
```python
# /// script
# [PEP 723 metadata...]
# ///

import groundhog_hpc as hog


class ModelAnalysis:
    """Analysis tools using a specific model."""

    @hog.method()
    def predict_batch(sequences: list[str]) -> dict:
        """
        Batch prediction using the model.

        Note: @hog.method() makes this a static method - no self parameter.

        Args:
            sequences: Input sequences to analyze

        Returns:
            {"results": [...], "summary": {...}}

        Example:
            # Via Garden SDK:
            import garden_ai
            garden = garden_ai.get_garden("10.26311/doi")
            result = garden.ModelAnalysis.predict_batch.remote(
                sequences=["ACGT..."],
                endpoint="anvil",
                account="bio-project"
            )
        """
        # Imports at module level or here
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained("model-name")
        model.eval()

        results = []
        for seq in sequences:
            # Processing logic
            results.append({"sequence": seq, "prediction": pred})

        return {"results": results, "summary": {"total": len(sequences)}}

    @hog.method()
    def analyze_structures(structures: list[dict]) -> dict:
        """Another method in the same class."""
        # ...
```

**Key points:**
- Use `@hog.method()` for class methods (not `@hog.function()`)
- Methods are implicitly static - no `self` parameter
- Call via Garden SDK: `garden.ClassName.method_name.remote(args, endpoint="name")`

## Handling Large Data

For large inputs/outputs (>10MB), use file-based patterns:

```python
@hog.function()
def process_large_dataset(file_content: str) -> str:
    """
    Process large dataset, return result file content.

    For very large data, Garden-AI handles chunking automatically.
    """
    # Write to temp file
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(file_content)
        temp_path = Path(f.name)

    try:
        # Process file
        result = process_file(temp_path)

        # Read result
        with open(result_path, 'r') as f:
            result_content = f.read()

        return result_content

    finally:
        # Cleanup
        temp_path.unlink()
```

## Endpoint-Specific Patterns

### Polaris (ALCF)
```python
# [tool.hog.polaris]
# endpoint = "uuid-here"
# worker_init = "
# module use /soft/modulefiles
# module load conda
# module load openmpi/5.0.3
# export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
# export HTTPS_PROXY=https://proxy.alcf.anl.gov:3128
# "
```

### Anvil (ACCESS)
```python
# [tool.hog.anvil]
# endpoint = "uuid-here"
# qos = "gpu"
# partition = "gpu-debug"
# scheduler_options = "#SBATCH --gpus-per-node=1"
```

### Perlmutter (NERSC)
```python
# [tool.hog.perlmutter]
# endpoint = "uuid-here"
# scheduler_options = "#SBATCH --constraint=gpu"
```

## Verification Checklist

Before claiming HPC script is ready:

**Metadata:**
- [ ] PEP 723 `# /// script` block at top of file
- [ ] `requires-python` specifies version
- [ ] ALL dependencies listed with EXACT versions
- [ ] Dependencies match repository requirements
- [ ] At least one `[tool.hog.endpoint_name]` section
- [ ] Endpoint UUID provided (or placeholder with comment)
- [ ] Closing `# ///` after metadata

**Code Structure:**
- [ ] `import groundhog_hpc as hog` present
- [ ] All compute functions use `@hog.function()` decorator
- [ ] At least one `@hog.harness()` test function per endpoint
- [ ] ALL imports inside functions (not at module level)
- [ ] Type hints on all parameters and returns
- [ ] Comprehensive docstrings with paper reference

**Functions:**
- [ ] Function names describe scientific computation
- [ ] Docstrings explain scientific purpose
- [ ] Example usage shown in docstring
- [ ] Error handling returns status dict
- [ ] Returns structured dict with results + metadata

**Test Harness:**
- [ ] Shows `.remote(endpoint="name")` usage
- [ ] Uses realistic examples from paper/repo
- [ ] Includes `account` parameter (user fills in)
- [ ] Prints expected outputs from paper
- [ ] One harness per configured endpoint

## Real-World Testing

**Test locally:**
```bash
# Install dependencies
uv sync  # or pip install -r requirements.txt

# Run test harness
uv run your_script.py  # or python your_script.py
```

**Common errors and fixes:**
- `ModuleNotFoundError`: Add dependency to PEP 723 block
- `No module named groundhog_hpc`: User needs to install it
- `ImportError on remote`: Move imports inside `@hog.function()`
- `Endpoint not found`: Check UUID in `[tool.hog.endpoint_name]`
- `SLURM submission failed`: Check qos/partition settings

## Real-World Example: Matbench Discovery Benchmark

The Garden-AI benchmarks show production-quality groundhog patterns:

**File:** `garden_ai/benchmarks/matbench_discovery/tasks.py`

**Key patterns demonstrated:**
- PEP 723 metadata with all dependencies pinned
- Multiple HPC endpoint configurations (anvil, sophia, polaris)
- Module-level imports for clarity
- Model factory pattern for flexible model loading
- Multi-GPU support via `user_endpoint_config`
- Comprehensive error handling and metrics
- Batch processing with progress tracking

**Usage example:**
```python
from garden_ai.benchmarks.matbench_discovery import MatbenchDiscovery

def create_model(device):
    from fairchem.core import FAIRChemCalculator
    return FAIRChemCalculator.from_model_checkpoint("esen_30m_omat")

# Call via .remote() with endpoint specified
results = MatbenchDiscovery.IS2RE.remote(
    endpoint="anvil",
    model_factory=create_model,
    user_endpoint_config={
        "qos": "gpu",
        "account": "cis250461-gpu",
        "partition": "gpu",
        "walltime": "08:00:00",
        "mem_per_node": 32,
    },
    model_packages=["fairchem-core"],
)
```

This shows:
- ✅ Function called with `.remote()` and `endpoint` parameter
- ✅ Advanced SLURM configuration via `user_endpoint_config`
- ✅ Model passed as factory function (not instance)
- ✅ Dynamic package installation with `model_packages`

## When HPC Script is Ready

Proceed to Phase 8 in workflow-phases.md to complete test harness and publication guidance.
