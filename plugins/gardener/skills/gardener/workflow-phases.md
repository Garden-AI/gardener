# Garden-AI Publication Workflow Phases

**Referenced from:** SKILL.md

This file contains detailed instructions for the 10-phase publication workflow. Load this file when working through phases 1-8.

**For CLI deployment and publication (Phases 9-10):** Also load cli-reference.md for complete command reference.

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
- **Pattern file**: modal-pattern.md

### groundhog_hpc (HPC Clusters)
- **What**: Run functions on existing HPC systems via Globus Compute
- **When**: Long computations (hours), multi-GPU, special libraries (MPI, SLURM)
- **How**: Write Python with `@hog.function()` decorators
- **Benefits**: Use institution's HPC resources, large memory, specialized hardware
- **Pattern file**: hpc-pattern.md

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

**For complete working examples across different scientific domains**, see:
- **modal-examples.md** - Chemistry, computer vision, NLP examples
- **hpc-examples.md** - Materials science, biology, quantum chemistry examples

**Generic pattern structure** (see example files for complete implementations):

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

**Async execution example:**
```python
import garden_ai

garden = garden_ai.get_garden("10.26311/garden-doi")

# Submit multiple jobs without blocking
futures = []
for batch in batches:
    future = garden.relax_structures_batch.submit(
        structures=batch,
        endpoint="polaris",
        account="my-allocation"
    )
    futures.append(future)

# Collect results
results = [future.result() for future in futures]
```

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

**Both platforms:**
- ✅ Batch processing with `list[...]` inputs
- ✅ Output: `{"results": [...], "summary": {...}}`
- ✅ Per-item error handling
- ✅ Type hints and comprehensive docstrings

---

## Technical Best Practices

**Follow these throughout all phases:**

### Repository Handling
```bash
# Always use shallow clone
git clone --depth 1 <repo-url>
```
- Avoid downloading entire git history (faster, less disk space)
- Use Glob/Grep tools to explore code, not manual browsing

### Data Serialization (CRITICAL)
**All inputs/outputs to remote functions MUST be easily serializable.**

✅ **Use:** strings, ints, floats, lists, dicts, bool
❌ **Never pass:** NumPy arrays, PIL images, custom objects, file handles
⚠️ **Large data:** Pass URLs or paths, not raw data

```python
# ✅ Good: Serializable types
def predict(smiles: list[str]) -> list[dict]:
    return [{"smiles": s, "score": 0.95} for s in smiles]

# ❌ Bad: NumPy arrays not serializable
def predict(features: np.ndarray) -> np.ndarray:
    return model(features)  # Won't work in remote function

# ✅ Fix: Convert to lists
def predict(features: list[list[float]]) -> list[float]:
    import numpy as np
    features_array = np.array(features)  # Convert inside function
    result = model(features_array)
    return result.tolist()  # Convert back to list
```

### File Handling
- Use Read tool for PDFs (supports PDF reading)
- Don't commit large files (model weights >50MB)
- Document download locations in docstrings
- Use Modal volumes for large model checkpoints (see modal-pattern.md)

### Dependency Management

**Philosophy: Pin what you import, specify versions conservatively**

✅ **Good dependency specifications:**
```python
# For groundhog (PEP 723):
# dependencies = [
#     "fairchem-core>=1.1.0",  # Direct: you import from fairchem
#     "torch>=2.0",            # Direct: you import torch
#     "ase>=3.26",             # Direct: you import from ase
# ]

# For Modal:
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fairchem-core>=1.1.0",
    "torch>=2.0",
    "ase>=3.26",
)
```

❌ **Don't pin transitive dependencies:**
```python
# ❌ Wrong: Including packages you don't import
# dependencies = [
#     "fairchem-core>=1.1.0",
#     "e3nn==0.5.1",            # Transitive (fairchem needs this)
#     "huggingface-hub==0.27",  # Transitive
#     "lmdb==1.7.3",            # Transitive
#     # ... Package manager handles these automatically
# ]
```

**How to determine:**
1. **Direct dependency:** Any package you `import` in your code
2. **Transitive dependency:** Packages your direct deps need (auto-resolved)
3. **Don't inspect repo dependencies:** Just list what YOU import

**Version pinning strategies:**
- ✅ **Prefer ranges:** `>=2.0,<3.0` allows compatible updates
- ⚠️ **Pin exact if needed:** `==2.0.1` only for known bugs
- ✅ **Major version constraints:** `>=2.0,<3.0` prevents breaking changes
- ❌ **Don't over-constrain:** Let package manager resolve compatible versions

**Real-world example from Matbench benchmarks:**
```python
# dependencies = [
#     "groundhog-hpc",      # Garden framework
#     "ase",                # You import ase
#     "numpy",              # You import numpy
#     "torch",              # You import torch
#     "matbench-discovery", # You import matbench_discovery
# ]
# That's it! Package manager handles: pandas, scipy, pymatgen, etc.
```

### Complete Results Pattern (CRITICAL)
**Return the full computational output, not just derived metrics.**

✅ **Good:** Return complete results for downstream analysis
```python
{
    "processed_output": complete_result,  # Full result (structure, sequence, image, etc.)
    "initial_input": original_input,
    "computed_property": value,
    "metadata": {...},
    # Users can do any downstream analysis, visualization, or further computation
}
```

❌ **Bad:** Return only summary metrics
```python
{
    "computed_property": value,
    # Lost the full output! Users can't visualize, analyze, or process further
}
```

**Why this matters:** Users need complete computational outputs for:
- Visualization and interpretation
- Downstream analysis and processing
- Further computation or refinement
- Comparison with initial inputs

**Domain examples:**
- Molecular/materials: Return optimized structure + energy (not just energy)
- Image generation: Return generated image + metadata (not just quality score)
- Sequence models: Return generated sequence + embeddings (not just perplexity)

**Red flags:**
- "I'll clone the full git history" (use --depth 1)
- "I'll pass non-serializable objects to remote functions" (NumPy arrays, PIL images, etc.)
- "Return just the computed metrics" (return complete outputs)
- "Latest versions should work" (pin versions from repo)
- "I'll include large files in repo" (>50MB checkpoints)
- "I'll pin all transitive dependencies" (only pin what you import)

---

## Phase Summary Table

| Phase | Don't Proceed Until | Checkpoint |
|-------|---------------------|------------|
| 1. Gather artifacts | Both paper PDF and repo are accessible | - |
| 2. Analyze paper | User validates your understanding | ✅ CHECKPOINT 1 |
| 3. Explore repository | You know how to load the model | - |
| 4. Understand model | User confirms synthesized understanding | ✅ CHECKPOINT 2 |
| 5. Design API | User approves API design | ✅ CHECKPOINT 3 |
| 6. Choose deployment | User confirms deployment target | ✅ CHECKPOINT 4 |
| 7. Generate code | User reviews and approves code | ✅ CHECKPOINT 5 |
| 8. Test & refine | Code runs successfully with correct output | - |
| 9. Deploy function | Function deployed via CLI, user confirms | ✅ CHECKPOINT 6 |
| 10. Create garden | Garden created with DOI, user has usage instructions | - |

**IMPORTANT:** All checkpoints are MANDATORY. Use AskUserQuestion at each checkpoint to get user validation before proceeding.

---

## Phase 1: Gather Artifacts

**Goal:** Collect paper and code repository.

**Actions:**
1. Ask for paper PDF path or URL
2. Ask for repository path or URL
3. If repository is URL, clone it locally with `git clone`
4. Confirm both artifacts are accessible using Read tool

**Don't proceed** without both paper and code.

**Red flags:**
- "User will provide them later"
- "I can work with what I have"
- "Assume it's a standard setup"

---

## Phase 2: Analyze Paper

**Goal:** Understand what the model does scientifically.

**Actions:**
1. Use Read tool on PDF to extract:
   - **Model purpose** and scientific domain
   - **Input** data types and formats (e.g., SMILES, XYZ, images, sequences)
   - **Output** predictions or results
   - **Key algorithms** or architectures mentioned
   - **Computational requirements** (GPU, memory, typical runtime)
   - **Dependencies** and frameworks used (PyTorch, TensorFlow, RDKit, ASE, etc.)
2. Take notes on scientific terminology specific to this domain
3. Identify the "core inference task"

**Look for sections:**
- **Abstract** - High-level purpose in 2-3 sentences
- **Methods** - How the model works, architecture details
- **Results** - What outputs look like, example predictions
- **Figures** - Visual examples of input/output
- **Tables** - Benchmark data, hyperparameters, performance metrics

**Red flags:**
- "I'll skip the paper and go straight to code"
- "It's probably like other models I've seen"
- "The abstract is enough"
- "I understand the domain already"

### CHECKPOINT 1: Validate Paper Understanding (MANDATORY)

**STOP HERE.** Do not proceed to Phase 3 without user validation.

**Summarize your understanding:**
```
I've analyzed the paper. Here's what I understand:

Scientific Domain: [e.g., computational chemistry, protein engineering, materials science]

Model Purpose: [1-2 sentence description of what the model does scientifically]

Input Data: [What format? e.g., SMILES strings, protein sequences, XYZ coordinates]

Output Data: [What does it predict? e.g., binding affinity scores, 3D structures, energy values]

Key Dependencies: [Major frameworks mentioned, e.g., PyTorch, RDKit, ASE]

Computational Requirements: [GPU needed? Memory? Typical runtime?]
```

**Use AskUserQuestion to validate:**
- Present your understanding summary
- Ask: "Is this understanding correct? Are there any important aspects I've missed or misunderstood?"
- If user corrects you, update your understanding before proceeding

**Why this matters:** False assumptions about the scientific purpose will cascade through all remaining phases. Catch them now.

---

## Phase 3: Explore Repository

**Goal:** Find the inference code and understand dependencies.

**Helper:** For complex repos or unfamiliar ML frameworks, load `repository-patterns.md` for guidance on finding model weights, preprocessing logic, and framework-specific patterns.

**Actions:**

### 1. Find Dependency Files
Use Glob to locate:
```
- requirements.txt
- setup.py
- pyproject.toml
- environment.yml
- Pipfile
```

Read these to identify:
- **All Python dependencies** with versions
- **Python version** requirements
- **Special system dependencies** (CUDA, MPI, etc.)

### 1b. Check for System Package Requirements (Modal Only)

**IMPORTANT: Many ML models need non-Python system packages**

Look for system dependencies in:
- **Dockerfile**: `RUN apt-get install <packages>` or `apt install <packages>`
- **README.md**: Installation sections mentioning apt/brew/yum packages
- **environment.yml**: Comments about system requirements
- **.github/workflows/**: CI files often show system package installs
- **docs/**: Installation documentation

**Common indicators you need system packages:**
- Repository has a Dockerfile
- README mentions "install libXYZ" or "requires system libraries"
- Uses computer vision (OpenCV) → likely needs `libgl1`, `libglib2.0-0`
- Uses audio/video processing → likely needs `ffmpeg`, `libsndfile1`
- Uses molecular visualization → might need rendering libraries
- Has complex dependencies like GDAL, cartopy → geospatial system libs

**Action for Modal apps:**
If you find system dependencies, you'll use `image.apt_install("package-name")` when generating the Modal code. Note these down during exploration.

**Example findings to note:**
```
Python deps: opencv-python==4.8.0, torch==2.0.1
System deps: libgl1, libglib2.0-0 (found in Dockerfile line 5)
→ Will need: image.apt_install("libgl1", "libglib2.0-0")
```

**groundhog_hpc note:** groundhog does NOT support installing system packages. If system deps are required, either:
- Verify they're already on the HPC system
- Recommend Modal instead
- Note this limitation for the user

### 2. Find Code Structure
Use Glob to find Python files:
```
**/*.py in src/, model/, inference/ directories
```

Look for:
- Main model files (model.py, network.py, etc.)
- Inference/prediction scripts
- Example scripts or notebooks
- Test files

### 3. Search for Key Functions
Use Grep to find:
```
- Model class: class.*Model
- Inference functions: def.*(predict|inference|forward)
- Data loading: def.*(load|preprocess)
- Model loading: def.*load_model
```

### 4. Read Critical Files
Use Read tool on:
- **README.md** - Installation and usage instructions
- **Main model file** - Model architecture
- **Inference functions** - How to run predictions
- **Example scripts** - End-to-end usage patterns

**Understand:**
- How to load the model (checkpoint path, HuggingFace model ID, URLs)
- What inputs it expects (data types, shapes, preprocessing needed)
- What outputs it returns (format, structure, interpretation)
- Critical dependency versions (especially deep learning frameworks)
- GPU requirements and memory needs

**Red flags:**
- "I don't need to read the README"
- "Standard model loading will work"
- "I'll figure out dependencies as I go"
- "Version numbers don't matter much"

---

## Phase 4: Understand Model

**Goal:** Synthesize paper + code into clear model behavior.

**Create explicit understanding:**

```
Domain: [What scientific field?]
- materials science, drug discovery, climate modeling, genomics, etc.

Input Format: [What does user provide?]
- SMILES strings for molecules
- XYZ coordinates for atomic structures
- Image files or arrays
- Protein/DNA sequences
- Time series data
- etc.

Interface: [How do users interact with the model?]
- Does it provide a calculator/pipeline/predictor class?
- What high-level methods are available? (e.g., .predict(), .get_forces())
- Are there preprocessing helpers, or is it automatic?

**Note:** Don't explain internal processing (embeddings, attention, graph construction) unless the user specifically asks. Focus on the interface and usage patterns.

Output Format: [What does user get back?]
- Single values (energy, score, probability)
- Structures (coordinates, graphs)
- Sequences or text
- Distributions or uncertainties

Performance: [Computational characteristics?]
- Inference time per input
- GPU memory requirements
- Batch size capabilities
- Scaling behavior
```

**Write this down explicitly** in your reasoning before proceeding.

**If unclear:** Ask the user questions with AskUserQuestion tool.

**Red flags:**
- "It's obviously a transformer model"
- "Standard preprocessing will work"
- "I can infer the details"

### CHECKPOINT 2: Validate Model Understanding (MANDATORY)

**STOP HERE.** Do not proceed to Phase 5 (Design API) without user validation.

**Present your synthesized understanding:**

Show the user the complete understanding template you filled out:
- Domain
- Input Format
- Preprocessing
- Model Processing
- Output Format
- Performance characteristics

**Use AskUserQuestion to validate:**
- Present the filled template
- Ask: "Does this accurately capture how the model works? Have I misunderstood any preprocessing steps, input/output formats, or computational requirements?"
- If user corrects you, update before proceeding to API design

**Why this matters:** API design depends on correct understanding of inputs/outputs. If you misunderstood the preprocessing or data formats, your API will be wrong.

---

## Phase 5: Design API

**Goal:** Create user-friendly function signatures that domain scientists understand.

---

### API Design Philosophy (CRITICAL)

**Design for scientific workflows, not model internals.**

Ask yourself:
1. ✅ What does a domain scientist want to **accomplish**? (relax structure, predict property, screen candidates)
2. ❌ NOT: What does the model **compute**? (embeddings, attention weights, hidden states)

**Examples:**

✅ **Workflow-focused (Good):**
```python
def process_scientific_data(data: list[str], method: str = "standard") -> list[dict]:
    """Process data using specified scientific method."""
```
User thinks: "I need to process my data for analysis"

❌ **Internals-focused (Bad):**
```python
def compute_model_embeddings(
    data: list[str],
    embedding_dim: int = 512,
    num_layers: int = 6,
    attention_heads: int = 8,
) -> list[dict]:
    """Compute neural network embeddings."""
```
User thinks: "What's an embedding? Why do I care about attention heads?"

**Expose scientifically meaningful parameters:**
- ✅ Domain-specific methods/settings (scientists understand these)
- ✅ Convergence thresholds, confidence levels (standard in the field)
- ✅ Physical/experimental parameters (temperature, pressure, etc.)
- ❌ Model architecture choices (`hidden_dim`, `num_layers`)
- ❌ Implementation details (`batch_size`, `learning_rate`)
- ❌ Internal preprocessing parameters (cutoffs, normalizations)

**Red flags you're exposing internals:**
- Parameters are model hyperparameters
- Parameters are architecture choices
- Function name describes computation, not scientific task
- User needs to understand how the model works to use the function

---

### Batch-First API Design (CRITICAL for HPC)

**Default to batch processing. Real workflows are screening campaigns, not single items.**

❌ **Wrong: Start with single-item API**
```python
def process_item(item: dict, method: str) -> dict:
    """Process a single item."""
    # User has to call this 1000x for screening workflow
    # Expensive invocation overhead
```

✅ **Correct: Start with batch API**
```python
def process_batch(
    items: list[dict],
    method: str,
    fail_fast: bool = False
) -> dict:
    """Batch processing for screening workflows."""
    return {
        "results": [
            {"index": i, "success": True, "output": ..., ...}
            for i in range(len(items))
        ],
        "summary": {"total": 100, "succeeded": 98, "failed": 2}
    }
```

**Why batch-first:**
1. **Real use case:** Scientists run screening campaigns (100s-1000s of inputs)
2. **Efficiency:** One HPC job vs 1000 separate invocations
3. **Error handling:** Continue processing after individual failures
4. **Statistics:** Summary metrics for campaign assessment

**Batch output structure:**
```python
{
    "results": [
        {
            "index": int,           # Original position
            "success": bool,        # Did this item succeed?
            "output": {...},        # Main result if success=True
            "error": Optional[str]  # Error message if success=False
        },
        # ... one per input
    ],
    "summary": {
        "total": int,
        "succeeded": int,
        "failed": int,
        # ... domain-specific stats
    }
}
```

---

**Design Principles:**

1. **Simple inputs** - Accept common formats for the domain
   - Chemistry: SMILES strings, not RDKit mol objects
   - Materials: XYZ/CIF strings or ASE dicts (serializable)
   - Biology: FASTA sequences, not Bio SeqRecord objects

2. **Batch processing** - ALWAYS design for batches first
   ```python
   def predict(molecules: list[str]) -> dict:  # ✅ Batch with summary
   def predict(molecule: str) -> dict:  # ❌ Single-item API
   ```

3. **Rich outputs** - Return structured results with metadata
   ```python
   {
       "input": original_input,
       "prediction": main_result,
       "confidence": score,
       "metadata": {...}
   }
   ```

4. **Sensible defaults** - Use paper's recommended parameters
   ```python
   def predict(
       inputs: list[str],
       model_variant: str = "base",  # From paper
       batch_size: int = 32,         # Tested value
   ) -> list[dict]:
   ```

5. **Clear names** - Function names describe the scientific task
   - ✅ `predict_binding_affinity`
   - ✅ `relax_structure`
   - ✅ `classify_protein_function`
   - ❌ `run_model`
   - ❌ `inference`
   - ❌ `predict`

**Multiple Functions When:**
- Model does multiple distinct tasks (prediction + optimization)
- Different input/output types for different use cases
- Clear separation of concerns

**Single Function When:**
- Model has one primary task
- All outputs naturally go together
- Additional options fit as optional parameters

**API Design Template:**
```python
def task_specific_name(
    domain_appropriate_input: list[CommonType],
    model_config: str = "recommended_from_paper",
    performance_tuning: int = 32,
    optional_outputs: bool = False,
) -> list[dict]:
    """
    One-line description of scientific task.

    Longer explanation referencing the paper, explaining
    the scientific meaning of inputs and outputs, and
    providing context about when to use this function.

    Args:
        domain_appropriate_input: Explanation using domain terms
        model_config: Which variant/checkpoint
        performance_tuning: Batch size or similar
        optional_outputs: Whether to return extra info

    Returns:
        List of dicts with:
        - main_result: The primary prediction
        - metadata: Input info, confidence, etc.

    Reference:
        Paper title, authors, DOI/arXiv
    """
```

**Ask yourself:**
- Would a domain scientist instantly understand this?
- Are input formats standard in this field?
- Do outputs provide actionable scientific information?
- Can users easily integrate this into their workflows?

**Red flags:**
- "Generic predict() function is fine"
- "Users will know what the outputs mean"
- "We don't need defaults"
- "Just copy the original function signature"

### CHECKPOINT 3: Approve API Design (MANDATORY)

**STOP HERE.** Do not proceed to Phase 6 without user approval of API design.

**Present your proposed API:**

Show the complete function signature(s) with:
- Function name(s)
- Parameter types and names
- Default values
- Return type structure
- Brief docstring

**Include your reasoning:**
- Why this function name? (connects to scientific task)
- Why these input types? (standard in the domain)
- Why these defaults? (from paper recommendations)
- Why this output structure? (what users need)

**Use AskUserQuestion to get approval:**
- Present the API design with reasoning
- Ask: "Does this API make sense for domain scientists? Should I change any function names, parameter types, or output structure?"
- Offer alternatives if there are multiple reasonable approaches
- If user suggests changes, revise before proceeding

**Why this matters:** Once you generate code, changing the API is expensive. Get it right now before writing hundreds of lines.

---

## Phase 6: Choose Deployment Target

**Decision criteria:**

### Use Modal When:
- Inference takes **<5 minutes** per batch
- **Single GPU** sufficient (or CPU-only)
- Model weights **<10GB**
- Standard dependencies available via **pip**
- **No special HPC environment** needed (no MPI, no SLURM-specific code)
- **Stateless inference** (no checkpointing between calls)
- Rapid prototyping and iteration

**Examples:**
- Image classification with ResNet/ViT
- Molecule property prediction with GNNs
- Protein function prediction with transformers
- Fast structure prediction

### Use HPC (groundhog) When:
- Long-running computations (**>5 min to hours**)
- **Multi-GPU** or **multi-node** parallelism required
- Large memory requirements (**>64GB**)
- Special HPC libraries (MPI, SLURM-specific tools)
- **Iterative simulations** needing checkpoints (MD, DFT convergence)
- User has **specific HPC endpoints** already configured
- Need access to specialized hardware (A100s, large memory nodes)

**Examples:**
- DFT calculations (VASP, Quantum ESPRESSO)
- Molecular dynamics simulations
- Large-scale structure relaxation
- Multi-GPU training or inference

### Use Both When:
- Model has **fast prediction** + **expensive optimization** modes
- **Example:** Modal for quick structure prediction, HPC for DFT refinement
- **Example:** Modal for screening thousands of candidates, HPC for detailed simulation of top hits

### Decision Process:
1. Check computational requirements from paper (Phase 2)
2. Check typical runtime from repo examples (Phase 3)
3. Consider user's available resources
4. **When unsure:** Ask user with AskUserQuestion:
   - "Do you have HPC endpoints configured?"
   - "What's your typical batch size and acceptable wait time?"
   - "Do you need multi-node parallelism?"

**Red flags:**
- "Modal is easier, let's use that"
- "HPC is more professional"
- "User can migrate later if needed"

### CHECKPOINT 4: Confirm Deployment Target (MANDATORY)

**STOP HERE.** Do not proceed to Phase 7 (code generation) without user confirmation.

**Present your deployment recommendation:**

```
Based on my analysis:

Computational Requirements:
- Runtime: [estimate from paper/code]
- GPU needs: [single/multi-GPU/none]
- Memory: [estimate]
- Parallelism: [single node vs multi-node]

Recommendation: [Modal / HPC / Both]

Reasoning:
- [Why this choice fits the requirements]
- [What tradeoffs are we making]
- [What alternatives exist and why not choosing them]
```

**Use AskUserQuestion to confirm:**
- Present recommendation with reasoning
- If recommending Modal: "This model seems suitable for Modal (fast inference, standard dependencies). Do you agree, or do you have HPC endpoints you'd prefer to use?"
- If recommending HPC: "This model needs HPC resources (long runtime / multi-GPU / special requirements). Do you have HPC endpoints configured? Should we use groundhog_hpc?"
- If recommending both: "I see both fast and slow modes. Should we create Modal for quick predictions and HPC for detailed computations?"

**Why this matters:** Modal and HPC use completely different code patterns. Choose wrong and you'll need to rewrite everything.

---

## Phase 7: Generate Code

**At this phase**, load the appropriate pattern file:
- **For Modal:** Read modal-pattern.md
- **For HPC:** Read hpc-pattern.md

Follow those files exactly for code generation.

**Critical reminders:**
- Use **exact dependency versions** from repo's requirements (Phase 3)
- Copy **preprocessing logic** verbatim from source code
- Test **model loading** matches the repo's pattern
- Preserve **scientific accuracy** over code elegance
- Include **comprehensive docstrings** referencing the paper

### CHECKPOINT 5: Review Generated Code (MANDATORY)

**STOP HERE.** Do not proceed to Phase 8 (test harness) without user review.

**Present your generated code:**

1. **Show the complete code** you've generated
2. **Walk through key sections:**
   - Dependencies: "I'm using [versions] from the repo's requirements"
   - Model loading: "The model loads from [checkpoint/URL/HuggingFace ID]"
   - Preprocessing: "Input preprocessing follows [specific function from repo]"
   - Main inference: "The core prediction logic does [scientific task]"
   - Output formatting: "Results are returned as [structure] with [metadata]"

3. **Highlight any deviations or assumptions:**
   - "I assumed [X] because [reasoning]"
   - "I couldn't find [Y] in the repo, so I [approach]"
   - "I chose [Z] over [alternative] because [rationale]"

**Use AskUserQuestion to get approval:**
- Present code with walkthrough
- Ask: "Does this implementation look correct? Should I adjust any dependency versions, preprocessing logic, or output formatting?"
- If user identifies issues, fix before proceeding to test harness

**Why this matters:** Test harness and publication guidance depend on correct code. If there are fundamental issues, catch them now before investing more tokens in testing infrastructure.

---

## Phase 8: Test, Validate, and Refine

**Goal:** Validate the generated code actually works through iterative testing and refinement.

**CRITICAL SUCCESS CRITERION:**
- **Modal**: If `uv run modal run modal_app.py` completes successfully with correct output → **READY FOR GARDEN PUBLICATION**
- **HPC**: If `uv run hog run hpc_script.py` completes successfully with correct output → **READY FOR GARDEN PUBLICATION**

**This is NOT a "write tests and move on" phase. This is an ACTIVE DEBUGGING CYCLE:**
1. ✅ **RUN** the code using `uv run modal run` or `uv run hog run`
2. ❌ **ENCOUNTER** errors or issues (they WILL happen)
3. 🔧 **FIX** the errors by editing the code
4. 🔄 **REPEAT** steps 1-3 until it works
5. ✅ **VERIFY** output with user before moving to Phase 9

**You MUST actually execute the code. Do NOT:**
- Just write test code without running it
- Assume it will work without testing
- Move to Phase 9 with untested code
- Skip testing because "it looks correct"

**WORKFLOW FOR PHASE 8:**
```
Write test → Run code → Fix errors → Run again → Verify output → Get approval → Phase 9
          ↑___________________________________________|
                    (Repeat until success)
```

**You are in active debugging mode. Keep running and fixing until it works.**

---

### Step 1: Generate Test Harness

**Requirements:**
1. Use **example data from paper or repo**
   - Check paper for example inputs
   - Look in repo for test data or example scripts
   - Use actual scientific examples, not dummy data

2. Show **complete workflow**
   - Load/initialize
   - Run prediction with realistic inputs
   - Display outputs in interpretable format

3. Include **expected output description**
   - What the numbers mean scientifically
   - How to interpret results
   - What "good" output looks like

4. Make it **locally runnable** for validation

**For Modal (already included in pattern):**
```python
@app.local_entrypoint()
def main():
    # Real example from paper (e.g., from abstract or results)
    test_molecule = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen from paper Fig. 2

    result = predict_property.remote([test_molecule])

    print(f"Prediction for ibuprofen: {result}")
    print(f"Expected: LogP ~3.5, MW ~206 (from paper Table 1)")
```

**For HPC (in @hog.harness() functions):**
```python
@hog.harness()
def test_anvil():
    # Example structure from paper's benchmark set
    test_structure = """2
    Lattice="2.5 0 0 0 2.5 0 0 0 2.5"
    C 0.0 0.0 0.0
    C 1.25 1.25 1.25
    """  # Diamond structure from paper Methods section

    print("Running on Anvil HPC...")
    result = relax_structure.remote(
        test_structure,
        endpoint="anvil",
        account="your-account",  # User will fill in
    )

    print(f"Relaxed energy: {result['energy']} eV")
    print("Expected: ~-10.5 eV (paper Table 2)")
```

**Red flags:**
- "Test with random data"
- "User can write their own tests"
- "Simple example is enough"

---

### Step 2: Run and Validate

**MANDATORY: Actually run the code. Don't skip to publication guidance.**

**This is the moment of truth - your code must run successfully in Modal's/groundhog's container.**

**For Modal:**
```bash
# Run local entrypoint with dependencies
uv run modal run your_modal_app.py

# What happens:
# 1. Modal builds the container with your image definition
# 2. Installs all apt and pip packages
# 3. Runs your @app.local_entrypoint() in the container
# 4. If this succeeds with correct output → READY FOR GARDEN PUBLICATION
```

**For groundhog_hpc:**
```bash
# Run harness test with dependencies
uv run hog run your_hpc_script.py

# What happens:
# 1. groundhog sets up the function environment
# 2. Runs your @hog.harness() test function
# 3. If this succeeds with correct output → READY FOR GARDEN PUBLICATION
```

**What success looks like:**
- ✅ Command exits with code 0 (no errors)
- ✅ Prints expected output from your test data
- ✅ Output format matches the documented structure
- ✅ Values are scientifically reasonable
- ✅ No warnings about missing dependencies

**If it runs successfully → You're done with testing! Move to Phase 9 after user verification.**

**What to check when it runs:**
1. ✅ **Syntax errors**: Does Python parse the file?
2. ✅ **Import errors**: Are all dependencies available (Python AND system)?
3. ✅ **Model loading**: Can it load the model/weights?
4. ✅ **Test execution**: Does the test harness run without errors?
5. ✅ **Output format**: Does output match expected structure?
6. ✅ **Scientific sanity**: Do values fall in reasonable ranges?

**Common execution flow:**
```
$ uv run modal run your_app.py
✓ Created objects.
├── 🔨 Created mount /Users/.../your_app.py
├── 🔨 Created image (apt packages, pip packages)
└── 🔨 Created function compute_batch
✓ Running local entrypoint...
[Your test output here]
✓ App completed successfully.

→ SUCCESS! Ready for Garden-AI publication.
```

---

### Step 3: Debug and Fix Issues (Iterate Until It Works)

**Errors WILL occur on first run - this is normal. Your job is to fix them iteratively.**

**The debugging cycle:**
```
1. Run: uv run modal run app.py
2. Error: Read the error message
3. Fix: Edit the code to address the error
4. Run again: uv run modal run app.py
5. Repeat until success
```

**DO NOT give up after one error. Keep iterating until the code works.**

#### 1. Read Error Messages Carefully

Look for:
- **Error type**: ImportError, AttributeError, ValueError, RuntimeError
- **Error location**: Which line/file triggered it
- **Stack trace**: Full path of the error
- **Modal's container logs**: Often show build vs runtime errors

**Example error analysis:**
```
ImportError: libGL.so.1: cannot open shared object file
→ Missing system package
→ Fix: Add apt_install("libgl1") to image definition
→ Re-run: Should work now
```

#### 2. Common Issues and Fixes

| Error Type | Likely Cause | Fix | Re-run? |
|------------|--------------|-----|---------|
| `ImportError: libGL.so.1` | Missing system library | Add `apt_install("libgl1", "libglib2.0-0")` | ✅ Yes |
| `ImportError: No module named 'X'` | Missing Python package | Add `"X==version"` to `pip_install()` | ✅ Yes |
| `ModuleNotFoundError: transformers` | Missing dependency | Add to pip_install, check repo versions | ✅ Yes |
| `AttributeError: 'NoneType'` in model loading | Model failed to load | Check HF model ID, checkpoint URL, or path | ✅ Yes |
| `FileNotFoundError: model.pt` | Wrong checkpoint path | Update to correct path or add download logic | ✅ Yes |
| `RuntimeError: CUDA out of memory` | Model too large for GPU | Change to larger GPU: `gpu="A100"` | ✅ Yes |
| `RuntimeError: Expected tensor on cuda` | Model on wrong device | Add `.to(device)` or check device handling | ✅ Yes |
| `ValueError: Invalid task` | Typo in task validation | Fix task name in code | ✅ Yes |
| `TypeError: 'dict' object is not callable` | Wrong calling syntax | Check if using Modal vs groundhog correctly | ✅ Yes |
| Import works locally but fails on Modal | Imports at module level | Move ALL imports inside functions (Modal rule) | ✅ Yes |

#### 3. Fix the Code and Re-run

**Process:**
1. **Edit the file** - Use the Edit tool to fix the issue
2. **Re-run immediately** - `uv run modal run app.py` or `uv run hog run script.py`
3. **Check if fixed** - Did this error go away? Is there a new error?
4. **Iterate** - Keep fixing until no errors remain

**Example iteration:**
```
Run 1: ImportError: libGL.so.1
→ Fix: Added apt_install("libgl1")
Run 2: ImportError: libglib2.0.so.0
→ Fix: Added apt_install("libglib2.0-0")
Run 3: ModuleNotFoundError: cv2
→ Fix: Added opencv-python to pip_install
Run 4: Success! ✅
```

#### 4. When You're Stuck

If you've tried multiple fixes and still hitting errors:
- **Re-read the repository** - Check if there's setup you missed
- **Check versions** - Ensure dependency versions match the repo exactly
- **Look for Dockerfile** - Repository's Dockerfile often has the solution
- **Ask the user** - Use AskUserQuestion to get guidance on persistent issues

**But keep trying! Most errors are fixable with careful attention to:**
- System dependencies (apt_install)
- Python dependencies (pip_install)
- Import locations (inside functions for Modal)
- Model paths and URLs
- GPU sizes and memory

#### 5. Document Workarounds

If you had to deviate from paper/repo:
- Add comments explaining why
- Note limitations in docstrings
- Document assumptions made

---

### Step 4: Verify Success with User

**After the code runs successfully, STOP and present results to user.**

**IMPORTANT: You must show the ACTUAL output from running the code, not hypothetical output.**

#### If Successful (Code Ran Without Errors)

Present the actual test results:

```
✅ Code runs successfully!

Command executed:
uv run modal run your_app.py

Output from test run:
[PASTE THE ACTUAL OUTPUT FROM THE TERMINAL HERE]
[Include all printed results, summaries, and completion messages]

Scientific validation check:
- Input: [What you tested with - e.g., "Ibuprofen SMILES from paper"]
- Output: [What was computed - e.g., "LogP = 3.47, MW = 206.3"]
- Expected: [From paper - e.g., "LogP ~3.5, MW ~206 (Table 1)"]
- Status: [✅ Matches expected range / ⚠️ Close but verify / ❌ Doesn't match]

The code is now ready for Garden-AI publication. Does this output look scientifically correct? Should I proceed to Phase 9 (publication guidance)?
```

**DO NOT move to Phase 9 without:**
- ✅ Showing actual terminal output from successful run
- ✅ Comparing results to paper's expected values
- ✅ Getting user confirmation

#### If There Are Persistent Errors

**Only present to user after you've tried multiple fixes yourself:**

```
⚠️ Encountered persistent error during testing:

Error message:
[PASTE EXACT ERROR FROM TERMINAL]

What I tried (iterations):
1. [First fix attempt] - Result: [New error or same error]
2. [Second fix attempt] - Result: [New error or same error]
3. [Third fix attempt] - Result: [Current state]

My analysis:
This error likely means: [Your hypothesis]

Possible solutions:
1. [Option 1: What you can try next]
2. [Option 2: What user might need to provide]
3. [Option 3: Alternative approach]

Should I try option [X], or do you have other suggestions?
```

**Use AskUserQuestion when:**
- Code works but output values seem unusual
- You need clarification on expected scientific ranges
- Persistent errors after 3+ fix attempts
- Missing credentials/access (HuggingFace tokens, model access)
- Repository structure is unclear after thorough exploration

---

### Step 5: Move to Phase 9 Only When Ready

**Prerequisites for Phase 9 (ALL must be true):**

✅ **Code executes successfully**
- `uv run modal run app.py` completes with exit code 0 (no errors)
- OR `uv run hog run script.py` completes with exit code 0

✅ **Output is correct**
- Printed results match expected format: `{"results": [...], "summary": {...}}`
- Values fall within scientific reasonable ranges
- Matches expectations from paper (within tolerance)

✅ **User confirmed**
- You showed the ACTUAL terminal output
- User verified it looks scientifically correct
- User approved moving to publication

✅ **All dependencies included**
- System packages (apt_install) if needed
- Python packages (pip_install) with versions
- No missing imports or modules

✅ **Assumptions documented**
- Any deviations from paper/repo are noted in comments
- Known limitations are in docstrings

**The iteration cycle (repeat until above conditions met):**
```
1. Run: uv run modal run app.py
2. Error/Issue: Read what went wrong
3. Fix: Edit the code
4. Run again: uv run modal run app.py
5. Success: Show output to user
6. Confirm: Get user approval
7. Proceed to Phase 9
```

**Red flags that mean you're NOT ready:**
- ❌ "I'll let the user test it themselves"
- ❌ "It should work, moving to publication" (without running)
- ❌ "Small errors are probably fine"
- ❌ "User can debug after publishing"
- ❌ "Testing would take too long"
- ❌ "I wrote tests but didn't run them"

**Remember:**
- If `uv run modal run` succeeds → It's ready for Garden
- If it fails → Keep debugging until it works
- Don't move forward with broken code

---

## Phase 9: Deploy Function via CLI

**Goal:** Deploy the tested function to Garden-AI infrastructure using the CLI.

**PREREQUISITE: Code must be tested and working (Phase 8 complete).**

**For detailed CLI reference, load:** cli-reference.md

---

### Step 1: Verify Authentication

```bash
garden-ai whoami
```

If not logged in:
```bash
garden-ai login
```

---

### Step 2: Deploy the Function

**For Modal apps:**
```bash
garden-ai function modal deploy <FILE> [OPTIONS]
```

**Example:**
```bash
garden-ai function modal deploy binding_affinity_app.py \
  -t "Protein-Ligand Binding Affinity Prediction" \
  -a "Jane Doe, John Smith" \
  --tags "drug-discovery,binding-affinity,virtual-screening"
```

**For HPC/Groundhog functions:**
```bash
garden-ai function hpc deploy <FILE> [OPTIONS]
```

**Example:**
```bash
garden-ai function hpc deploy structure_relaxation.py \
  -t "Structure Relaxation with MACE" \
  -e "polaris-endpoint-id,anvil-endpoint-id" \
  -a "Jane Doe, John Smith" \
  --tags "materials-science,MACE,relaxation"
```

---

### Step 3: Capture Function ID

The deploy command outputs a function ID. **Save this for Phase 10.**

```
✓ Function deployed successfully
Function ID: modal-func-abc123
```

---

### Step 4: Verify Deployment

```bash
# For Modal
garden-ai function modal show <APP_ID>

# For HPC
garden-ai function hpc show <FUNCTION_ID>
```

Verify:
- Function name matches expected
- Metadata (title, authors, tags) is correct
- No deployment errors

---

### CHECKPOINT 6: Confirm Deployment (MANDATORY)

**Present deployment results:**
```
Function deployed successfully:
- Function ID: [ID]
- Title: [Title]
- Authors: [Authors]
- Tags: [Tags]

The function is now deployed and ready to be added to a Garden.
Should I proceed to create a Garden with this function?
```

**Use AskUserQuestion to confirm** before proceeding to Phase 10.

---

## Phase 10: Create Garden

**Goal:** Create a Garden containing the deployed function(s) and obtain citable DOI.

---

### Step 1: Create the Garden

```bash
garden-ai garden create [OPTIONS]
```

**Required options:**
- `-t, --title TEXT` - Garden title
- `-a, --authors TEXT` - Comma-separated authors

**Include function IDs:**
- `-m, --modal-function-ids TEXT` - For Modal functions
- `-g, --hpc-function-ids TEXT` - For HPC functions

**Example:**
```bash
garden-ai garden create \
  -t "Drug Discovery ML Models (Smith et al. 2024)" \
  -a "Jane Doe, John Smith" \
  -d "Machine learning models for drug discovery workflows based on Smith et al., Nature 2024. DOI: 10.1038/s41586-024-xxxxx" \
  --tags "drug-discovery,binding-affinity,virtual-screening" \
  -m "modal-func-abc123" \
  --year 2025
```

---

### Step 2: Capture Garden DOI

The create command outputs a DOI:
```
✓ Garden created successfully
Garden DOI: 10.26311/garden-xyz789
```

---

### Step 3: Verify Garden

```bash
garden-ai garden show "10.26311/garden-xyz789"
```

Verify:
- All functions are listed
- Metadata is correct
- DOI is valid

---

### Step 4: Provide Usage Instructions

**Present the final result to user:**

```
✅ Garden published successfully!

Garden DOI: 10.26311/garden-xyz789
URL: https://thegardens.ai/gardens/10.26311/garden-xyz789

To use this garden in Python:

from garden_ai import GardenClient

client = GardenClient()
garden = client.get_garden("10.26311/garden-xyz789")

# Call your function
result = garden.predict_binding_affinity(molecules=["CCO", "CCCO"])

Citation:
Jane Doe, John Smith. (2025). Drug Discovery ML Models (Smith et al. 2024).
Garden-AI. DOI: 10.26311/garden-xyz789
```

---

### Optional: Update Garden Later

If you need to add more functions or update metadata:

```bash
# Add functions to existing garden
garden-ai garden add-functions "10.26311/garden-xyz789" \
  -m "new-modal-func-id"

# Update metadata
garden-ai garden update "10.26311/garden-xyz789" \
  -d "Updated description with more details"
```

---

**Red flags:**
- "Just upload it" (use the CLI workflow)
- "User knows what to do" (guide them through)
- "Test it after publishing" (testing happens in Phase 8)
- Skipping deployment verification
- Not providing usage instructions

---

## Verification Checklist

Before claiming workflow is complete:

**Workflow completion:**
- [ ] Read and understood the paper (can explain model purpose)
- [ ] Explored repository structure and key files
- [ ] Identified model loading and inference code
- [ ] Listed all critical dependencies with versions
- [ ] Designed API that domain scientists will understand
- [ ] Chose appropriate deployment target (Modal vs HPC)
- [ ] API matches what paper/code actually does
- [ ] Test harness uses realistic examples from paper/repo
- [ ] Code tested successfully with `uv run modal run` or `uv run hog run`
- [ ] Function deployed via CLI (`garden-ai function modal/hpc deploy`)
- [ ] Garden created via CLI (`garden-ai garden create`)
- [ ] User has DOI and usage instructions

**Pattern compliance:**
- [ ] Checked against modal-pattern.md OR hpc-pattern.md
- [ ] All code patterns followed correctly
- [ ] Dependencies match repository
- [ ] Preprocessing matches source code
- [ ] Outputs are scientifically accurate

**CLI deployment compliance:**
- [ ] Checked against cli-reference.md
- [ ] Authentication verified (`garden-ai whoami`)
- [ ] Function deployed with correct metadata
- [ ] Garden created with function IDs
- [ ] DOI captured and shared with user

---

## Summary

Follow these phases in order. Don't skip ahead. Each phase builds on the previous one to ensure scientifically accurate, usable Garden-AI functions.

**Supporting files to load:**
- Code generation patterns: modal-pattern.md or hpc-pattern.md
- CLI deployment and publication: cli-reference.md

**The full agent-driven publication flow:**
1. Understand the science (Phases 1-4)
2. Design the API (Phase 5)
3. Choose and generate code (Phases 6-7)
4. Test until it works (Phase 8)
5. Deploy function via CLI (Phase 9)
6. Create garden via CLI (Phase 10)

The result: A citable DOI for the published function that scientists can discover and use via the Garden-AI SDK.
