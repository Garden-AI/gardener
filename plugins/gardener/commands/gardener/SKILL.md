---
name: gardener
description: Use when helping researchers publish scientific ML models on Garden-AI, especially when analyzing research papers and code repositories to design and generate Modal apps or groundhog_hpc scripts for model inference and computational workflows
argument-hint: "[paper/repo URL or description of what to publish]"
---

# Gardener: End-to-End Garden-AI Publication

## User's Request

$ARGUMENTS

---

## MANDATORY: Follow This Workflow

**STOP. You MUST follow this workflow. Do NOT skip ahead to writing code.**

**Your first action** must be to load the workflow skill using the Skill tool:
```
Load skill: gardener:workflow-phases
```

Then follow these phases IN ORDER:

1. **Phase 1-5: Understand** - Gather artifacts, analyze paper, explore repo, synthesize, design API
2. **Phase 6: Choose platform** - Modal vs groundhog_hpc decision
3. **Phase 7: Generate code** - Load the pattern skill FIRST:
   - Modal: Load skill `gardener:modal-pattern`
   - HPC: Load skill `gardener:hpc-pattern`
4. **Phase 8: Test** - Actually RUN the code with `uv run modal run` or `uv run hog run`
5. **Phase 9-10: Deploy & Publish** - Load skill `gardener:cli-reference` for CLI commands

**DO NOT:**
- Skip loading the workflow-phases skill
- Write code before completing phases 1-6
- Generate Modal/HPC code without loading the pattern skill first
- Claim "done" without running the code

**DO:**
- Load gardener:workflow-phases skill NOW before proceeding
- Follow each phase and checkpoint
- Ask the user questions at each checkpoint
- Load the appropriate pattern skill before generating code

---

## Overview

Guides researchers from published paper to deployed Garden-AI function. Core principle: **Understand the science first, design thoughtful APIs, then generate correct Garden-AI code.**

This is a complete workflow from research artifacts (paper PDF + code repo) to publishable Garden-AI functions.

## What is Garden-AI?

Garden-AI is a FAIR (Findable, Accessible, Interoperable, Reusable) AI/ML model publishing platform that enables:

- **Citable Scientific Functions**: Each published function/model gets a DOI for academic citation
- **Remote Execution**: Run models on cloud (Modal) or HPC systems without local setup
- **Simple Python SDK**: Domain scientists use models via clean Python APIs
- **Diverse Compute**: From serverless GPUs to multi-node HPC clusters

**Two Deployment Platforms:**
1. **Modal** - Serverless cloud (fast inference, auto-scaling, pay-per-use)
2. **groundhog_hpc** - HPC clusters via Globus Compute (long jobs, multi-GPU, specialized hardware)

## When to Use

Use when:
- Researcher wants to publish ML model on Garden-AI
- User has paper + code but doesn't know Modal/groundhog
- Converting research code to Garden-AI functions
- User says "publish my model on Garden" or "make this Garden-AI ready"
- Need to design API for scientific model deployment
- Helping with Matbench Discovery or other Garden benchmarks

Don't use for:
- Standard Python scripts (not for Garden-AI)
- Non-scientific computing workloads
- Code already in Garden-AI format
- General CLI tool development

## Key Implementation Facts

**CRITICAL: Calling Conventions Differ by Platform**

| Platform | Type | How Users Call It | Example |
|----------|------|-------------------|---------|
| Modal | function | Direct call (NO `.remote()`) | `garden.func(args)` |
| Modal | class | Direct call (NO `.remote()`) | `garden.Class.method(args)` |
| groundhog | function | `.remote()` (blocking) | `garden.func.remote(args, endpoint="anvil")` |
| groundhog | function | `.submit()` (async) | `future = garden.func.submit(args, endpoint="anvil")` |
| groundhog | class | `.remote()` on method | `garden.Class.method.remote(args, endpoint="anvil")` |

**Other Key Differences:**

| Aspect | Modal | groundhog_hpc |
|--------|-------|---------------|
| Imports | INSIDE functions only | Module level OK |
| Metadata | `modal.App()` + `image` | PEP 723 `# /// script` |
| Testing | `@app.local_entrypoint()` | `@hog.harness()` |
| Endpoint | Auto-handled by backend | Must specify `endpoint="name"` |
| Classes | `@app.cls()` + `@modal.method()` | Class + `@hog.method()` |

## Workflow Overview

```dot
digraph publication_workflow {
    "Gather artifacts" [shape=box];
    "Analyze paper" [shape=box];
    "Explore repository" [shape=box];
    "Understand model" [shape=box];
    "Design API" [shape=box];
    "Choose deployment?" [shape=diamond];
    "Generate code" [shape=box];
    "Test & refine" [shape=box];
    "Deploy function" [shape=box];
    "Create garden" [shape=box];

    "Gather artifacts" -> "Analyze paper";
    "Analyze paper" -> "Explore repository";
    "Explore repository" -> "Understand model";
    "Understand model" -> "Design API";
    "Design API" -> "Choose deployment?";
    "Choose deployment?" -> "Generate code" [label="Modal or HPC"];
    "Generate code" -> "Test & refine";
    "Test & refine" -> "Deploy function" [label="CLI deploy"];
    "Deploy function" -> "Create garden" [label="CLI create"];
}
```

**Quick Reference:**

| Phase | Key Actions | Supporting Skill |
|-------|-------------|------------------|
| 1-5: Understand | Gather, analyze, explore, synthesize, design API | `gardener:workflow-phases` |
| 6: Choose | Modal vs HPC decision | `gardener:workflow-phases` |
| 7: Generate | Write Modal app or groundhog script following patterns | `gardener:modal-pattern` / `gardener:hpc-pattern` |
| 8: Test & Refine | **ACTIVELY RUN** code, debug errors, fix, repeat until working | `gardener:workflow-phases` |
| 9: Deploy | Deploy function via CLI: `garden-ai function modal deploy` or `hpc deploy` | `gardener:cli-reference` |
| 10: Publish | Create garden via CLI: `garden-ai garden create` | `gardener:cli-reference` |

**Phase 8 is MANDATORY and ACTIVE:**
- Must execute: `uv run modal run` or `uv run hog run`
- If it works → Ready for CLI deployment (Phase 9)
- If it fails → Debug and fix until it works
- Cannot skip or assume it works

**Phases 9-10: CLI-Driven Publication:**
- Phase 9: Deploy function with `garden-ai function modal deploy` or `garden-ai function hpc deploy`
- Phase 10: Create garden with `garden-ai garden create` including the deployed function IDs

## How to Use This Skill

1. **Start here** - Read this file to understand the workflow
2. **Follow phases 1-6** - Load `gardener:workflow-phases` skill for detailed guidance
3. **Choose deployment**:
   - For Modal: Load `gardener:modal-pattern` skill
   - For HPC: Load `gardener:hpc-pattern` skill
4. **Test code (Phase 8)** - Follow testing instructions from workflow-phases
5. **Deploy & Publish (Phases 9-10)** - Load `gardener:cli-reference` skill for CLI commands

**IMPORTANT:** You don't need all skills in context at once. Load them as needed using the Skill tool:
- Always load: This file (SKILL.md)
- Phases 1-6: `gardener:workflow-phases`
- Modal generation: `gardener:modal-pattern`
- HPC generation: `gardener:hpc-pattern`
- CLI deployment/publishing: `gardener:cli-reference`

## Red Flags - STOP and Follow Workflow

**Checkpoint violations (check gardener:workflow-phases skill):**
- "I'll just move to the next phase"
- "User understands, no need to confirm"
- "This is obviously correct"
- "Asking will slow us down"
- "I'll check with user if there are problems"
- "Checkpoints are optional for simple models"

**Workflow shortcuts (check gardener:workflow-phases skill):**
- "I can skip reading the paper"
- "I know what this model does"
- "Standard ML API will work"
- "User can figure out the API"
- "Just make it work quickly"

**Code violations (check pattern skills):**
- "Imports at top are cleaner"
- "User will add metadata later"
- "This works without decorators"
- "Time is tight, skip the boilerplate"
- "Ready to publish" (before verification)

**Testing violations (check Phase 8 in gardener:workflow-phases skill):**
- "I wrote tests but won't run them"
- "It should work without testing"
- "User can test it themselves"
- "Small errors are probably fine"
- "Testing takes too long"
- Moving to Phase 9 without running code

**All of these mean: Stop. Load the appropriate supporting file and follow it exactly. All checkpoints are MANDATORY.**

## Verification Checklist

Before claiming done, verify against checklists in:
- `gardener:workflow-phases` skill (phases completed)
- `gardener:modal-pattern` or `gardener:hpc-pattern` skill (code patterns)

## Real-World Impact

**Following this workflow:**
- Researchers publish scientifically accurate models
- Domain scientists can easily use the functions
- APIs match field conventions
- Code runs reliably on Garden-AI platform
- Models become citable and reusable

**Skipping steps:**
- APIs that don't match domain conventions
- Functions that misinterpret the model
- Missing dependencies or preprocessing
- Scientific errors in outputs
- Code fails on remote execution

## Quick Decision Guide

**Choose Modal when:**
- Fast inference (<5 min per batch)
- Single GPU or CPU-only
- Standard Python packages
- Auto-scaling needed
- Example: Image classification, molecule property prediction

**Choose groundhog_hpc when:**
- Long computations (hours)
- Multi-GPU or multi-node
- HPC-specific libraries (MPI, SLURM tools)
- User has HPC allocations
- Example: DFT calculations, MD simulations, structure optimization

## Supporting Skills

Load these skills using the Skill tool as needed:

- **gardener:workflow-phases** - Detailed publication workflow phases 1-10 with checkpoints
- **gardener:modal-pattern** - Complete Modal app pattern, imports inside functions, @app.function decorators
- **gardener:hpc-pattern** - Complete HPC/groundhog pattern, PEP 723 metadata, @hog.function decorators
- **gardener:cli-reference** - Garden-AI CLI commands for deployment and publication

## Common Pitfalls

❌ **Mixing up Modal and groundhog patterns:**
- Modal: Imports INSIDE functions, NO `.remote()` in user code
- groundhog: Imports at module level OK, MUST use `.remote()` with `endpoint`

❌ **Wrong calling convention:**
```python
# Modal (via Garden SDK):
result = garden.my_modal_function(args)  # ✅ Correct
result = garden.my_modal_function.remote(args)  # ❌ Wrong

# groundhog (via Garden SDK):
result = garden.my_hpc_function.remote(args, endpoint="anvil")  # ✅ Correct
result = garden.my_hpc_function(args)  # ❌ Wrong - will error
```

❌ **Using wrong decorators:**
- Modal: `@app.function()` and `@app.cls()` / `@modal.method()`
- groundhog: `@hog.function()` and `@hog.method()`

❌ **Not testing in Phase 8:**
- Phase 8 is MANDATORY - you MUST run the code
- Execute: `uv run modal run` or `uv run hog run`
- If errors occur: Fix them and re-run (iterate until success)
- Success criterion: Command completes without errors + output looks correct
- Don't move to Phase 9 until code actually runs successfully
