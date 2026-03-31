---
description: Detailed 10-phase Garden-AI publication workflow with checkpoints and validation steps
---

# Garden-AI Publication Workflow Phases

**Referenced from:** SKILL.md

This file contains detailed instructions for the 10-phase publication workflow. Load this file when working through phases 1-8.

**For CLI deployment and publication (Phases 9-10):** Also load the `gardener:cli-reference` skill.

---

## Garden-AI Platform Overview

**What is Garden-AI?**

Garden-AI is a platform for publishing scientific ML models as **citable, reusable functions**. Functions get DOIs and can be referenced in papers just like datasets or code repositories.

**Key Goals:**
- Make ML models **FAIR** (Findable, Accessible, Interoperable, Reusable)
- Provide **high-level interfaces** for domain scientists (not ML experts)
- Enable **reproducible science** through versioned, documented functions
- Support **diverse compute**: from cloud GPUs to HPC clusters

**Two Deployment Options:**

### Modal (Serverless Cloud)
- **What**: Serverless GPU/CPU compute on Modal's infrastructure
- **When**: Fast inference (<5 min), single GPU, standard dependencies
- **How**: Write Python with `@app.function()` decorators
- **Benefits**: Zero infrastructure setup, automatic scaling, pay-per-use
- **Pattern skill**: `gardener:modal-pattern`

### groundhog_hpc (HPC Clusters)
- **What**: Run functions on existing HPC systems via Globus Compute
- **When**: Long computations (hours), multi-GPU, special libraries (MPI, SLURM)
- **How**: Write Python with `@hog.function()` decorators
- **Benefits**: Use institution's HPC resources, large memory, specialized hardware
- **Pattern skill**: `gardener:hpc-pattern`

**What "Publishing to Garden" Means:**

You're creating **scientific APIs** that wrap ML models:
1. Researcher publishes paper with model
2. You design domain-appropriate API (e.g., `relax_structures_batch()`)
3. Function gets deployed to Modal or HPC
4. Other scientists discover and use it via Garden-AI
5. Function has citable DOI for attribution

**Your Role:**

Transform research code → production-ready scientific API. Bridge gap between:
- **ML researchers** (who built the model)
- **Domain scientists** (who will use it)

---

## Code Patterns and Examples

**For complete working examples across different scientific domains**, see the example files in the commands/gardener directory.

**Generic pattern structure**:

```python
# Function that processes batch of inputs
def compute_batch(inputs: list[DomainType], param: float = default) -> dict:
    """[Scientific task description]"""

    # Load model/tool
    tool = load_domain_tool()

    # Process each input with error handling
    results = []
    succeeded = 0

    for idx, item in enumerate(inputs):
        result = {"index": idx, "input": item, "success": False}
        try:
            # Domain-specific computation
            output = tool.process(item, param)
            result.update({"success": True, "output": output})
            succeeded += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(inputs), "succeeded": succeeded}}
```

**Key principles** (applies to all domains):
- Batch processing (lists of inputs)
- Per-item error handling
- Structured output: `{"results": [...], "summary": {...}}`
- Type hints and comprehensive docstrings

---

## Critical Platform Differences

**Calling Convention:**

| Platform | Type | Calling Pattern | Example |
|----------|------|-----------------|---------|
| groundhog | function | `.remote()` (blocking) | `garden.func.remote(args)` |
| groundhog | function | `.submit()` (async) | `future = garden.func.submit(args)` |
| groundhog | class method | `.remote()` (blocking) | `garden.Class.method.remote(args)` |
| groundhog | class method | `.submit()` (async) | `future = garden.Class.method.submit(args)` |
| Modal | function | NO `.remote()` | `garden.func(args)` |
| Modal | class method | NO `.remote()` | `garden.Class.method(args)` |

**groundhog_hpc:**
- PEP 723 inline metadata (`# /// script`)
- `@hog.function()` for standalone functions
- `@hog.method()` for class methods
- Imports can be at module level (unlike Modal)
- **Execution modes:**
  - `.remote(endpoint="name")` - Blocking call, waits for result
  - `.submit(endpoint="name")` - Returns GroundhogFuture, non-blocking
- Must specify `endpoint` parameter (e.g., "anvil", "polaris")
- Optional parameters: `account`, `walltime`, `user_endpoint_config`

**Modal:**
- `modal.App()` and `image` definition
- `@app.function()` for standalone functions
- `@app.cls()` for classes with `@modal.method()` methods
- ALL imports **inside** functions/methods
- `@app.local_entrypoint()` for testing
- Call WITHOUT `.remote()`

**Garden SDK:**
- Load garden: `garden = garden_ai.get_garden("10.26311/doi")`
- Gardens are namespaces containing functions/classes
- Results are native Python types (dicts, lists, floats)

---

## Phase Summary Table

| Phase | Don't Proceed Until | Checkpoint |
|-------|---------------------|------------|
| 1. Gather artifacts | Both paper PDF and repo are accessible | - |
| 2. Analyze paper | User validates your understanding | CHECKPOINT 1 |
| 3. Explore repository | You know how to load the model | - |
| 4. Understand model | User confirms synthesized understanding | CHECKPOINT 2 |
| 5. Design API | User approves API design | CHECKPOINT 3 |
| 6. Choose deployment | User confirms deployment target | CHECKPOINT 4 |
| 7. Generate code | User reviews and approves code | CHECKPOINT 5 |
| 8. Test & refine | Code runs successfully with correct output | - |
| 9. Deploy function | Function deployed via CLI, user confirms | CHECKPOINT 6 |
| 10. Create garden | Garden created with DOI, user has usage instructions | - |

**IMPORTANT:** All checkpoints are MANDATORY. Use AskUserQuestion at each checkpoint to get user validation before proceeding.

---

## Phase 1: Gather Artifacts

**Goal:** Collect paper and code repository.

**Actions:**
1. Ask for paper PDF path or URL
2. Ask for repository path or URL
3. If repository is URL, clone it locally with `git clone --depth 1`
4. Confirm both artifacts are accessible using Read tool

**Don't proceed** without both paper and code.

---

## Phase 2: Analyze Paper

**Goal:** Understand what the model does scientifically.

**Actions:**
1. Use Read tool on PDF to extract:
   - **Model purpose** and scientific domain
   - **Input** data types and formats
   - **Output** predictions or results
   - **Key algorithms** or architectures mentioned
   - **Computational requirements** (GPU, memory, typical runtime)
   - **Dependencies** and frameworks used

### CHECKPOINT 1: Validate Paper Understanding (MANDATORY)

**STOP HERE.** Do not proceed to Phase 3 without user validation.

Use AskUserQuestion to validate your understanding summary.

---

## Phase 3: Explore Repository

**Goal:** Find the inference code and understand dependencies.

**Actions:**
1. Find dependency files (requirements.txt, pyproject.toml, etc.)
2. Check for system package requirements (Dockerfile, README)
3. Find code structure and key functions
4. Read critical files (README, main model file, inference functions)

---

## Phase 4: Understand Model

**Goal:** Synthesize paper + code into clear model behavior.

Create explicit understanding of: Domain, Input Format, Interface, Output Format, Performance.

### CHECKPOINT 2: Validate Model Understanding (MANDATORY)

**STOP HERE.** Present your synthesized understanding and validate with user.

---

## Phase 5: Design API

**Goal:** Create user-friendly function signatures that domain scientists understand.

**Key principles:**
- Design for scientific workflows, not model internals
- Batch-first API design (lists of inputs)
- Simple, serializable inputs (strings, lists, dicts)
- Rich structured outputs

### CHECKPOINT 3: Approve API Design (MANDATORY)

**STOP HERE.** Present your proposed API and get user approval.

---

## Phase 6: Choose Deployment Target

### Use Modal When:
- Inference takes <5 minutes per batch
- Single GPU sufficient
- Standard pip dependencies
- Stateless inference

### Use HPC (groundhog) When:
- Long-running computations (>5 min to hours)
- Multi-GPU or multi-node required
- Special HPC libraries (MPI, SLURM)
- User has HPC endpoints configured

### CHECKPOINT 4: Confirm Deployment Target (MANDATORY)

**STOP HERE.** Present your recommendation and get user confirmation.

---

## Phase 7: Generate Code

**At this phase**, load the appropriate pattern skill:
- **For Modal:** Load `gardener:modal-pattern` skill
- **For HPC:** Load `gardener:hpc-pattern` skill

Follow those patterns exactly for code generation.

### CHECKPOINT 5: Review Generated Code (MANDATORY)

**STOP HERE.** Present code and get user approval before testing.

---

## Phase 8: Test, Validate, and Refine

**Goal:** Validate the generated code actually works through iterative testing.

**CRITICAL SUCCESS CRITERION:**
- **Modal**: `uv run modal run modal_app.py` completes successfully
- **HPC**: `uv run hog run hpc_script.py` completes successfully

**This is an ACTIVE DEBUGGING CYCLE:**
1. RUN the code
2. ENCOUNTER errors
3. FIX the errors
4. REPEAT until it works
5. VERIFY output with user

**DO NOT move to Phase 9 without actual successful test run.**

---

## Phase 9: Deploy Function via CLI

**PREREQUISITE: Code must be tested and working (Phase 8 complete).**

**For detailed CLI reference, load:** `gardener:cli-reference` skill

**For Modal:**
```bash
garden-ai function modal app deploy my_app.py \
  -t "Title" -a "Authors" --tags "tag1,tag2"
```

**For HPC:**
```bash
garden-ai function hpc deploy my_script.py \
  -t "Title" -e "endpoint-id" -a "Authors"
```

### CHECKPOINT 6: Confirm Deployment (MANDATORY)

Present deployment results and confirm before creating garden.

---

## Phase 10: Create Garden

**Goal:** Create a Garden containing the deployed function(s) and obtain citable DOI.

```bash
garden-ai garden create \
  -t "Garden Title" \
  -a "Authors" \
  -d "Description" \
  -m "modal-func-id"
```

Provide user with:
- Garden DOI
- URL
- Python SDK usage example
- Citation format

---

## Summary

Follow these phases in order. Don't skip ahead.

**Supporting skills to load:**
- Code generation patterns: `gardener:modal-pattern` or `gardener:hpc-pattern`
- CLI deployment and publication: `gardener:cli-reference`
