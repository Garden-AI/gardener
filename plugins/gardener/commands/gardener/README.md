# Gardener Skill - Garden-AI Model Publishing Assistant

## Overview

The Gardener skill helps researchers publish scientific ML models on Garden-AI, a FAIR AI/ML model publishing platform. It guides users through a 9-phase workflow from research paper to deployed, citable functions.

## Key Updates (January 2026)

This skill has been updated with accurate information from the Garden-AI codebase, including:

- **Correct calling conventions** for Modal vs groundhog_hpc functions
- **Accurate import patterns** (Modal requires imports inside functions, groundhog allows module-level)
- **Real-world examples** from Matbench Discovery benchmarks
- **Clear decision criteria** for choosing Modal vs HPC deployment
- **Production patterns** for classes, batch processing, and error handling

## File Structure

| File | Purpose |
|------|---------|
| `SKILL.md` | Main entry point with overview and workflow reference |
| `workflow-phases.md` | Detailed 9-phase publication workflow (1400+ lines) |
| `modal-pattern.md` | Complete Modal app patterns with examples |
| `hpc-pattern.md` | Complete groundhog_hpc patterns with examples |

## Critical Information for the Gardener

### Platform Differences

| Aspect | Modal | groundhog_hpc |
|--------|-------|---------------|
| **Imports** | MUST be inside functions | Can be at module level |
| **Calling** | Direct: `garden.func(args)` | `.remote()`: `garden.func.remote(args, endpoint="anvil")` |
| **Metadata** | `modal.App()` + `image` definition | PEP 723 `# /// script` block |
| **Decorators** | `@app.function()`, `@app.cls()` | `@hog.function()`, `@hog.method()` |
| **Testing** | `@app.local_entrypoint()` | `@hog.harness()` |
| **Execution** | Backend proxy (automatic) | Must specify `endpoint` parameter |
| **Classes** | `@modal.method()` in `@app.cls()` | `@hog.method()` in regular class |

### Calling Convention Examples

**Modal (via Garden SDK):**
```python
import garden_ai
garden = garden_ai.get_garden("10.26311/doi")

# Functions - NO .remote()
result = garden.my_function(args)

# Class methods - NO .remote()
result = garden.MyClass.method_name(args)
```

**groundhog_hpc (via Garden SDK):**
```python
import garden_ai
garden = garden_ai.get_garden("10.26311/doi")

# Functions - MUST use .remote() with endpoint
result = garden.my_function.remote(
    args,
    endpoint="anvil",
    account="abc123",
)

# Class methods - MUST use .remote() with endpoint
result = garden.MyClass.method_name.remote(
    args,
    endpoint="polaris",
    account="project-123",
)

# Async execution - use .submit()
future = garden.my_function.submit(args, endpoint="anvil")
result = future.result()  # blocks until complete
```

### When to Use Each Platform

**Use Modal when:**
- Inference < 5 minutes per batch
- Single GPU or CPU-only
- Standard Python packages
- Need auto-scaling
- Examples: Image classification, SMILES property prediction, protein function prediction

**Use groundhog_hpc when:**
- Long computations (hours)
- Multi-GPU or multi-node required
- HPC-specific libraries (MPI, SLURM)
- User has HPC allocations
- Examples: DFT calculations, MD simulations, large-scale structure optimization

## The 9-Phase Workflow

1. **Gather Artifacts** - Get paper PDF + code repository
2. **Analyze Paper** - Understand model purpose, inputs, outputs (CHECKPOINT 1)
3. **Explore Repository** - Find inference code, dependencies
4. **Understand Model** - Synthesize paper + code (CHECKPOINT 2)
5. **Design API** - Create domain-appropriate function signatures (CHECKPOINT 3)
6. **Choose Deployment** - Modal vs HPC decision (CHECKPOINT 4)
7. **Generate Code** - Write Modal app or groundhog script (CHECKPOINT 5)
8. **Test & Validate** - Run code, debug, iterate with user feedback
9. **Guide Publication** - Provide upload instructions

**All checkpoints are MANDATORY** - use AskUserQuestion to get user validation before proceeding.

## Common Pitfalls

❌ **Mixing up import patterns:**
- Modal REQUIRES imports inside functions
- groundhog allows imports at module level

❌ **Wrong calling convention:**
```python
# Modal - DON'T use .remote() via Garden SDK
result = garden.modal_func.remote(args)  # ❌ Wrong

# groundhog - MUST use .remote() with endpoint
result = garden.hpc_func(args)  # ❌ Wrong - will error
```

❌ **Skipping checkpoints:**
- Each checkpoint catches errors early
- Getting user validation prevents wasted work
- All 5 checkpoints are mandatory

❌ **Not testing in Phase 8:**
- Code MUST be run and validated
- Fix errors iteratively with user
- Don't move to Phase 9 until working

## Real-World Patterns

### Batch Processing (CRITICAL)

**Always design for batch processing first:**
```python
def process_batch(inputs: list[Any]) -> dict:
    """Process a batch of inputs with per-item error handling."""
    results = []
    succeeded = 0

    for idx, item in enumerate(inputs):
        result = {"index": idx, "input": item, "success": False}
        try:
            output = process_single(item)
            result.update({"success": True, "output": output})
            succeeded += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {
        "results": results,
        "summary": {"total": len(inputs), "succeeded": succeeded}
    }
```

**Why batch-first:**
- Real use cases are screening campaigns (100s-1000s of items)
- One HPC job vs 1000 separate invocations
- Continue after individual failures
- Summary statistics for assessment

### Model Factory Pattern (HPC)

For HPC functions that need to load models, use factory pattern:
```python
def create_model(device):
    """Factory function that creates model instance."""
    from fairchem.core import FAIRChemCalculator
    return FAIRChemCalculator.from_model_checkpoint("model-name")

# Pass factory, not instance
result = garden.compute_task.remote(
    model_factory=create_model,
    endpoint="anvil",
)
```

### Dependency Specifications

**Pin what you import, not transitive dependencies:**
```python
# ✅ Good - direct dependencies only
# dependencies = [
#     "fairchem-core>=1.1.0",  # You import this
#     "torch>=2.0",             # You import this
#     "ase>=3.26",              # You import this
# ]

# ❌ Bad - including transitive dependencies
# dependencies = [
#     "fairchem-core>=1.1.0",
#     "e3nn==0.5.1",            # Transitive - fairchem needs it
#     "huggingface-hub==0.27",  # Transitive
#     # ... 15 more you don't import
# ]
```

## Usage

1. Read `SKILL.md` for overview
2. Follow `workflow-phases.md` phases 1-6
3. Load pattern file (`modal-pattern.md` or `hpc-pattern.md`) for phase 7
4. Complete phases 8-9 in `workflow-phases.md`

## Sources

Information compiled from:
- `garden_ai/modal/functions.py` - Modal function implementation
- `garden_ai/hpc/functions.py` - HPC function implementation
- `garden_ai/benchmarks/matbench_discovery/` - Real-world benchmark examples
- `garden_ai/schemas/modal.py` and `hpc.py` - API schemas
- Garden-AI frontend and backend documentation (CLAUDE.md files)
