# groundhog_hpc Examples for Garden-AI

**Referenced from:** hpc-pattern.md, SKILL.md

This file contains complete, working groundhog_hpc examples across different scientific domains. Use these as reference implementations when generating HPC scripts.

---

## Example 1: Materials Science - Structure Relaxation

**Domain:** Computational materials science / catalysis
**Model:** OC20 (Open Catalyst) for structure optimization
**Input:** Atomic structures (ASE format)
**Output:** Relaxed geometries with energies and forces
**Compute:** GPU-accelerated, minutes to hours per structure

### The Published Function

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["ase>=3.22", "torch>=2.0", "fairchem-core>=1.1.0"]
# [tool.hog.polaris]
# endpoint = "4b116d3c-1703-4f8f-9f6f-39921e5864df"
# ///

"""Structure relaxation using OC20 model for catalysis research."""

import groundhog_hpc as hog


@hog.function()
def relax_structures_batch(
    structures: list[dict],
    fmax: float = 0.05,
    max_steps: int = 200,
) -> dict:
    """
    Batch structure relaxation for catalyst screening.

    Args:
        structures: List of ASE Atoms dicts (from atoms.todict())
        fmax: Force convergence threshold (eV/Angstrom)
        max_steps: Maximum optimization steps per structure

    Returns:
        {"results": [...], "summary": {...}}
    """
    from ase import Atoms
    from ase.optimize import LBFGS
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit("oc20", device="cuda")
    calc = FAIRChemCalculator(predictor)

    results = []
    succeeded = converged = 0

    for idx, structure_dict in enumerate(structures):
        result = {"index": idx, "success": False}
        try:
            atoms = Atoms.fromdict(structure_dict)
            atoms.calc = calc
            initial_energy = atoms.get_potential_energy()

            opt = LBFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=max_steps)

            final_energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            final_fmax = (forces**2).sum(axis=1).max()**0.5

            result.update({
                "success": True,
                "optimized_structure": atoms.todict(),
                "initial_energy": float(initial_energy),
                "final_energy": float(final_energy),
                "converged": final_fmax < fmax,
                "steps": opt.get_number_of_steps(),
                "final_fmax": float(final_fmax),
            })
            succeeded += 1
            if result["converged"]:
                converged += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(structures), "succeeded": succeeded, "converged": converged}}
```

### Using via Garden SDK

```python
import garden_ai
from ase.build import fcc111, add_adsorbate

garden = garden_ai.get_garden("10.26311/garden-abcd-1234")

# Prepare structures
candidates = [fcc111("Pt", size=(2, 2, 3), vacuum=10.0).todict()]

# Call groundhog function with .remote()
result = garden.relax_structures_batch.remote(
    structures=candidates,
    fmax=0.05,
    endpoint="polaris",
    account="my-allocation"
)

print(f"Converged: {result['summary']['converged']}/{result['summary']['total']}")
```

---

## Example 2: Biology - Protein Structure Prediction

**Domain:** Structural biology
**Model:** ESM-2 transformer for protein folding
**Input:** Amino acid sequences
**Output:** 3D coordinates and confidence scores
**Compute:** GPU-accelerated, minutes per protein

### The Published Class

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.0", "esm>=2.0.0"]
# [tool.hog.anvil]
# endpoint = "8c45d12f-2b3a-4e5c-9d1e-2f3b4c5d6e7f"
# ///

"""Protein structure and function prediction using ESM-2."""

import groundhog_hpc as hog


class ProteinAnalysis:
    """Protein analysis tools using ESM-2 transformer."""

    @hog.method()
    def predict_structure_batch(sequences: list[str]) -> dict:
        """
        Predict 3D structure from amino acid sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            {"results": [...], "summary": {...}}
        """
        import torch
        import esm

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()

        results = []
        succeeded = 0

        for idx, seq in enumerate(sequences):
            result = {"index": idx, "sequence": seq, "success": False}
            try:
                with torch.no_grad():
                    batch_tokens = alphabet.get_batch_converter()([(f"protein_{idx}", seq)])[2]
                    predictions = model(batch_tokens)

                result.update({
                    "success": True,
                    "coordinates": predictions["positions"].tolist(),
                    "confidence": float(predictions["plddt"].mean()),
                })
                succeeded += 1
            except Exception as e:
                result["error"] = str(e)

            results.append(result)

        return {"results": results, "summary": {"total": len(sequences), "succeeded": succeeded}}
```

### Using via Garden SDK

```python
import garden_ai

garden = garden_ai.get_garden("10.26311/garden-efgh-5678")

sequences = ["MKTAYIAKQRQISFVK..."]

# Call groundhog class method with .remote()
result = garden.ProteinAnalysis.predict_structure_batch.remote(
    sequences=sequences,
    endpoint="anvil",
    account="bio-project"
)

for r in result['results']:
    if r['success']:
        print(f"Confidence: {r['confidence']:.2f}")
```

---

## Example 3: Quantum Chemistry - DFT Calculations

**Domain:** Quantum chemistry
**Model:** Custom DFT calculator
**Input:** Molecular structures
**Output:** Electronic properties (energy, orbitals, charges)
**Compute:** CPU-intensive, hours to days per structure

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["ase>=3.22", "gpaw>=22.8.0", "numpy>=1.24"]
# [tool.hog.perlmutter]
# endpoint = "9b23c45d-3e4f-5a6b-8c9d-0e1f2a3b4c5d"
# scheduler_options = "#SBATCH --constraint=cpu"
# ///

"""DFT calculations using GPAW for electronic structure."""

import groundhog_hpc as hog


@hog.function()
def compute_electronic_structure_batch(
    structures: list[str],
    functional: str = "PBE",
    basis_set: str = "dzp",
) -> dict:
    """
    Batch DFT calculations for electronic properties.

    Args:
        structures: List of XYZ format structures
        functional: Exchange-correlation functional
        basis_set: Basis set for calculations

    Returns:
        {"results": [...], "summary": {...}}
    """
    from ase.io import read
    from gpaw import GPAW, PW
    from io import StringIO
    import numpy as np

    results = []
    succeeded = 0

    for idx, xyz_str in enumerate(structures):
        result = {"index": idx, "success": False}
        try:
            # Parse structure
            atoms = read(StringIO(xyz_str), format="xyz")

            # Setup calculator
            calc = GPAW(
                mode=PW(500),
                xc=functional,
                basis=basis_set,
                txt=None  # Suppress output
            )
            atoms.calc = calc

            # Run calculation
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Get electronic properties
            calc.get_pseudo_density()
            homo, lumo = calc.get_homo_lumo()
            charges = calc.get_all_electron_density()

            result.update({
                "success": True,
                "energy": float(energy),
                "forces": forces.tolist(),
                "homo": float(homo),
                "lumo": float(lumo),
                "gap": float(lumo - homo),
                "optimized_structure": xyz_str,  # Return structure
            })
            succeeded += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(structures), "succeeded": succeeded}}


@hog.harness()
def test_perlmutter():
    """Test on Perlmutter HPC."""
    test_structure = """2
Lattice="2.5 0 0 0 2.5 0 0 0 2.5"
H 0.0 0.0 0.0
H 1.25 1.25 1.25
"""

    print("Running DFT on Perlmutter...")
    result = compute_electronic_structure_batch.remote(
        structures=[test_structure],
        endpoint="perlmutter",
        account="m1234",
        walltime="01:00:00"
    )

    print(f"Completed: {result['summary']['succeeded']}/{result['summary']['total']}")
    for r in result['results']:
        if r['success']:
            print(f"  HOMO-LUMO gap: {r['gap']:.3f} eV")
```

---

## Example 4: Async Execution Pattern

**Use case:** Submit many long-running jobs without blocking

```python
import garden_ai

garden = garden_ai.get_garden("10.26311/garden-doi")

# Submit multiple jobs asynchronously
futures = []
for batch in large_dataset_batches:
    future = garden.relax_structures_batch.submit(  # .submit() not .remote()
        structures=batch,
        endpoint="polaris",
        account="my-allocation"
    )
    futures.append(future)

# Continue doing other work...
print(f"Submitted {len(futures)} jobs")

# Collect results later
results = [future.result() for future in futures]  # Blocks until all complete
print(f"All jobs finished!")
```

---

## Common Patterns Across Examples

### 1. PEP 723 Metadata Block

Every groundhog script needs this at the top:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "package1>=1.0",
#     "package2>=2.0",
# ]
# [tool.hog.endpoint_name]
# endpoint = "uuid-here"
# ///
```

### 2. Batch Processing with Summary

All examples return structured results with summary:

```python
results = []
succeeded = 0

for idx, item in enumerate(items):
    result = {"index": idx, "success": False}
    try:
        # Processing
        output = process(item)
        result.update({"success": True, "output": output})
        succeeded += 1
    except Exception as e:
        result["error"] = str(e)

    results.append(result)

return {"results": results, "summary": {"total": len(items), "succeeded": succeeded}}
```

### 3. Remote Execution with Endpoint

Users MUST specify endpoint:

```python
# Blocking call
result = garden.function_name.remote(
    args,
    endpoint="anvil",        # Required
    account="allocation",    # Usually required
    walltime="02:00:00",    # Optional
)

# Async call
future = garden.function_name.submit(
    args,
    endpoint="anvil",
    account="allocation",
)
result = future.result()  # Get result later
```

### 4. Test Harnesses

Include `@hog.harness()` for testing:

```python
@hog.harness()
def test_anvil():
    """Test on Anvil endpoint."""
    result = my_function.remote(
        test_data,
        endpoint="anvil",
        account="test-account",  # User replaces
    )
    print(f"Result: {result}")
```

### 5. Import Placement

**RECOMMENDED: Put imports inside functions (except `groundhog_hpc as hog`)**

```python
# ✅ Recommended for Garden-AI
import groundhog_hpc as hog  # Module-level required

@hog.function()
def compute(data):
    # All other imports inside the function
    import numpy as np
    from ase import Atoms

    atoms = Atoms(data)
    return np.array(atoms.positions)
```

**Note:** All three working examples above follow this pattern - imports are inside functions. This provides consistency with Modal and is the recommended Garden-AI pattern.

Module-level imports (other than `groundhog_hpc as hog`) are *technically* allowed by groundhog but not recommended for Garden-AI code.

---

## When to Use Which Pattern

**Use simple `@hog.function()` when:**
- Standalone computation
- No shared state needed
- Single scientific task

**Use class with `@hog.method()` when:**
- Multiple related computations
- Shared configuration
- Complex workflows

**Choose endpoint based on requirements:**
- **Polaris (ALCF)**: Large-scale GPU, multi-node
- **Anvil (ACCESS)**: General-purpose GPU
- **Perlmutter (NERSC)**: CPU or GPU, large memory
- **Custom endpoint**: User's institutional HPC

---

## Advanced: Multi-GPU Configuration

```python
# Via user_endpoint_config in the call
result = garden.my_function.remote(
    data,
    endpoint="polaris",
    account="project",
    user_endpoint_config={
        "scheduler_options": "#SBATCH --gpus-per-node=4\n#SBATCH --nodes=2",
        "worker_init": "module load conda; module load cuda",
    }
)
```

---

## Key Differences from Modal

| Aspect | groundhog_hpc (shown here) | Modal |
|--------|----------------------------|-------|
| User calls | `garden.func.remote(args, endpoint="name")` - MUST use .remote() | `garden.func(args)` - NO .remote() |
| Imports | Inside functions (recommended for Garden-AI) | INSIDE functions only (required) |
| Metadata | PEP 723 `# /// script` | `modal.App()` + `image` |
| Testing | `@hog.harness()` | `@app.local_entrypoint()` |
| Decorators | `@hog.function()`, `@hog.method()` | `@app.function()`, `@app.cls()` |
| Async | `.submit()` returns future | Not directly supported |

---

## Usage After Publishing

Once published to Garden-AI, users access via the Garden SDK:

```python
import garden_ai

# Get the garden by DOI
garden = garden_ai.get_garden("10.26311/your-doi-here")

# MUST use .remote() and specify endpoint
result = garden.your_function_name.remote(
    your_args,
    endpoint="anvil",      # Which HPC system
    account="allocation",  # User's HPC account
)

# For classes
result = garden.YourClassName.method_name.remote(
    your_args,
    endpoint="perlmutter",
    account="m1234",
)

# Async execution
future = garden.your_function_name.submit(
    your_args,
    endpoint="polaris",
)
# Do other work...
result = future.result()  # Blocks until complete
```

No groundhog_hpc CLI or Globus Compute setup required for end users - Garden-AI handles endpoint routing.
