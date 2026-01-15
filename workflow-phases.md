# Garden-AI Publication Workflow Phases

**Referenced from:** SKILL.md

This file contains detailed instructions for the 9-phase publication workflow. Load this file when working through phases 1-6 and 8-9.

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

## Complete Examples: Code → Publication → Usage

### Example 1: groundhog_hpc Function

**The Published Function:**
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

**Using via Garden SDK:**
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

### Example 2: groundhog_hpc Class with Methods

**The Published Class:**
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

**Using via Garden SDK:**
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

### Example 3: Modal Function

**The Published Function:**
```python
"""Molecular property prediction using ChemBERTa."""

import modal

app = modal.App("chemberta-admet")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==4.36.0",
    "torch==2.1.0",
    "rdkit==2023.9.1",
)


@app.function(image=image, gpu="T4", cpu=2.0, memory=4096)
def predict_admet_batch(
    smiles_list: list[str],
    properties: list[str] = ["solubility", "toxicity"]
) -> dict:
    """
    Batch ADMET property prediction for drug screening.

    Args:
        smiles_list: List of SMILES strings
        properties: Which properties to predict

    Returns:
        {"results": [...], "summary": {...}}
    """
    # ALL imports inside function for Modal
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from rdkit import Chem

    model = AutoModelForSequenceClassification.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval()

    results = []
    valid = 0

    for idx, smiles in enumerate(smiles_list):
        result = {"index": idx, "smiles": smiles, "valid": False}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result["error"] = "Invalid SMILES"
                results.append(result)
                continue

            inputs = tokenizer(smiles, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)

            predictions = {
                "solubility": float(outputs.logits[0, 0]),
                "toxicity": float(torch.sigmoid(outputs.logits[0, 1])),
            }

            result.update({"valid": True, "predictions": predictions})
            valid += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(smiles_list), "valid": valid}}


@app.local_entrypoint()
def main():
    """Test locally."""
    result = predict_admet_batch.remote(["CCO", "CC(=O)O"])
    print(f"Summary: {result['summary']}")
```

**Using via Garden SDK:**
```python
import garden_ai

garden = garden_ai.get_garden("10.26311/garden-chem-xyz")

candidate_molecules = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
]

# Call Modal function WITHOUT .remote()
result = garden.predict_admet_batch(
    smiles_list=candidate_molecules,
    properties=["solubility", "toxicity"]
)

print(f"Valid: {result['summary']['valid']}/{result['summary']['total']}")
```

---

### Example 4: Modal Class with Methods

**The Published Class:**
```python
"""Image analysis using vision transformers."""

import modal

app = modal.App("vision-analysis")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==4.36.0",
    "torch==2.1.0",
    "pillow==10.1.0",
)


@app.cls(image=image, gpu="T4", cpu=2.0, memory=4096)
class ImageAnalyzer:
    """Image classification and feature extraction."""

    def __init__(self):
        """Load model once per container."""
        from transformers import ViTForImageClassification, ViTImageProcessor
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model.eval()

    @modal.method()
    def classify_batch(self, image_urls: list[str]) -> dict:
        """
        Classify images from URLs.

        Args:
            image_urls: List of image URLs

        Returns:
            {"results": [...], "summary": {...}}
        """
        from PIL import Image
        import requests
        import torch
        from io import BytesIO

        results = []
        succeeded = 0

        for idx, url in enumerate(image_urls):
            result = {"index": idx, "url": url, "success": False}
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))

                inputs = self.processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits
                predicted_class = logits.argmax(-1).item()

                result.update({
                    "success": True,
                    "predicted_class": predicted_class,
                    "confidence": float(torch.softmax(logits, dim=-1).max()),
                })
                succeeded += 1
            except Exception as e:
                result["error"] = str(e)

            results.append(result)

        return {"results": results, "summary": {"total": len(image_urls), "succeeded": succeeded}}

    @modal.method()
    def extract_features_batch(self, image_urls: list[str]) -> dict:
        """Extract feature embeddings from images."""
        from PIL import Image
        import requests
        import torch
        from io import BytesIO

        results = []
        for idx, url in enumerate(image_urls):
            result = {"index": idx, "url": url, "success": False}
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))

                inputs = self.processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                # Get last hidden state
                features = outputs.hidden_states[-1].mean(dim=1).squeeze().tolist()

                result.update({"success": True, "features": features})
            except Exception as e:
                result["error"] = str(e)

            results.append(result)

        return {"results": results, "summary": {"total": len(image_urls), "succeeded": len([r for r in results if r["success"]])}}
```

**Using via Garden SDK:**
```python
import garden_ai

garden = garden_ai.get_garden("10.26311/garden-vision-abc")

image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
]

# Call Modal class methods WITHOUT .remote()
classification_result = garden.ImageAnalyzer.classify_batch(image_urls=image_urls)

features_result = garden.ImageAnalyzer.extract_features_batch(image_urls=image_urls)

print(f"Classified: {classification_result['summary']['succeeded']}")
print(f"Features extracted: {features_result['summary']['succeeded']}")
```

---

## Key Patterns Summary

**Calling Convention (CRITICAL):**

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

### Structure Output Pattern (Molecular/Materials CRITICAL)
**Always return the full structure, not just computed properties.**

✅ **Good:** Return complete result for client-side reconstruction
```python
{
    "relaxed_structure": xyz_string,  # Full structure as XYZ
    "initial_structure": original_xyz,
    "energy": -123.45,
    "forces": [[fx, fy, fz], ...],
    # User can reconstruct ASE Atoms and do any analysis
}
```

❌ **Bad:** Return only computed values
```python
{
    "energy": -123.45,
    "forces": [[fx, fy, fz], ...],
    # Lost the structure! User can't visualize or analyze geometry
}
```

**Why this matters:** Users need the full computational result (relaxed geometry, optimized structure) for:
- Visualization
- Downstream calculations (bond lengths, angles, etc.)
- Further optimization or simulation
- Comparison with initial structure

**Red flags:**
- "I'll clone the full history"
- "I'll pass this NumPy array to the remote function"
- "Return just the energy and forces" (for molecular/materials models)
- "Latest versions should work"
- "I'll include 500MB checkpoint in the repo"
- "I'll pin all transitive dependencies"

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
| 8. Create test harness | Test runs with realistic examples | - |
| 9. Guide publication | User knows next steps | - |

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
- **All dependencies** with versions
- **Python version** requirements
- **Special system dependencies** (CUDA, MPI, etc.)

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
def relax_structures(structures: list[str], task: str = "omat") -> list[dict]:
    """Optimize atomic geometries to minimum energy."""
```
User thinks: "I need to relax some structures for my study"

❌ **Internals-focused (Bad):**
```python
def compute_graph_embeddings(
    structures: list[str],
    cutoff: float = 6.0,
    max_neighbors: int = 50,
    embedding_dim: int = 512,
) -> list[dict]:
    """Compute graph neural network embeddings for atomic structures."""
```
User thinks: "What's a graph embedding? Why do I care about cutoff radius?"

**Expose parameters that matter to scientists:**
- ✅ `task="omat"` (which DFT level of theory - scientists understand this)
- ✅ `fmax=0.05` (convergence criterion - standard in the field)
- ✅ `temperature_K=300` (physical parameter with meaning)
- ❌ `hidden_dim=512` (model architecture detail - irrelevant to science)
- ❌ `num_layers=6` (implementation detail)
- ❌ `cutoff=6.0` (internal preprocessing parameter)

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
def relax_structure(structure: dict, task: str) -> dict:
    """Optimize a single atomic structure."""
    # User has to call this 1000x for screening campaign
    # Expensive invocation overhead on HPC
```

✅ **Correct: Start with batch API**
```python
def relax_structures_batch(
    structures: list[dict],
    task: str,
    fail_fast: bool = False
) -> dict:
    """Batch structure relaxation for screening campaigns."""
    return {
        "results": [
            {"index": i, "success": True, "optimized_structure": ..., ...}
            for i in range(len(structures))
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

**CRITICAL: This is NOT a "write tests and move on" phase. You must actively RUN the code, find bugs, fix them, and iterate with user feedback.**

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

**For Modal:**
```bash
# Check syntax first
python your_modal_app.py --help

# Run local entrypoint
modal run your_modal_app.py
```

**For groundhog_hpc:**
```bash
# Check syntax first
python your_hpc_script.py --help

# Validate PEP 723 metadata
uv run --with groundhog-hpc python your_hpc_script.py

# Or if dependencies installed:
python your_hpc_script.py
```

**What to check:**
1. ✅ **Syntax errors**: Does Python parse the file?
2. ✅ **Import errors**: Are all dependencies available?
3. ✅ **Model loading**: Can it load the model/weights?
4. ✅ **Test execution**: Does the test harness run without errors?
5. ✅ **Output format**: Does output match expected structure?
6. ✅ **Scientific sanity**: Do values fall in reasonable ranges?

---

### Step 3: Debug and Fix Issues

**When errors occur (they will):**

1. **Read the error message carefully**
   - What's the actual error? (ImportError, AttributeError, ValueError, etc.)
   - Which line triggered it?
   - What was the stack trace?

2. **Common issues and fixes:**

| Error Type | Likely Cause | Fix |
|------------|--------------|-----|
| `ImportError: No module named 'X'` | Missing dependency | Add to `dependencies` list |
| `AttributeError: 'NoneType' object has no attribute` | Model failed to load | Check model path/URL/HF ID |
| `ValueError: Invalid task name` | Typo in task validation | Fix task name list |
| `RuntimeError: CUDA out of memory` | Model too large for GPU | Use smaller variant or add memory note |
| `FileNotFoundError: checkpoint.pt` | Wrong checkpoint path | Update path or add download logic |
| `TypeError: 'dict' object is not callable` | Wrong calling convention | Check Modal vs groundhog patterns |

3. **Fix the code**
   - Edit the file to address the error
   - Re-run the validation
   - Repeat until it works

4. **Document any workarounds**
   - If you had to deviate from the paper/repo, explain why in comments
   - Add notes about limitations or known issues

---

### Step 4: Get User Feedback

**STOP and present results to user:**

**If successful:**
```
✅ Code validation successful!

Test run completed with these outputs:
[Show actual output from running the test]

Scientific validation:
- [Result 1]: Expected [X], got [Y] - [✅ matches / ⚠️ close / ❌ mismatch]
- [Result 2]: Expected [X], got [Y] - [✅ matches / ⚠️ close / ❌ mismatch]

Does this look scientifically correct? Should I adjust anything before publication?
```

**If there are errors:**
```
⚠️ Encountered errors during testing:

Error: [Copy exact error message]

I tried: [Explain what you attempted]

This might be because: [Your hypothesis about the issue]

Options:
1. I can try [alternative approach]
2. You might need to [user action, like getting credentials]
3. This might be a [known limitation we should document]

How should I proceed?
```

**Use AskUserQuestion to:**
- Confirm outputs are scientifically reasonable
- Get guidance on errors you can't resolve
- Validate any assumptions you made during debugging
- Check if there are missing requirements (credentials, model access, etc.)

---

### Step 5: Iterate Until Working

**Don't move to Phase 9 until:**
- ✅ Code runs without errors (or with only documented limitations)
- ✅ Test outputs match expected scientific ranges
- ✅ User confirms it looks correct
- ✅ All assumptions are documented

**Iteration cycle:**
1. Run → 2. Encounter error → 3. Fix → 4. Run again → 5. Get feedback → Repeat

**Red flags:**
- "I'll let the user debug it"
- "It should work, moving to publication"
- "Small errors are okay"
- "User can test after publishing"

**The code must be VALIDATED and WORKING before Phase 9.**

---

## Phase 9: Guide Publication

**Goal:** Provide clear next steps for Garden-AI upload.

**PREREQUISITE: Code must be tested and working (Phase 8 complete).**

**Provide instructions:**

1. **Upload to Garden-AI:**
   - **Web UI (recommended):** Upload file at garden.thegardens.ai
   - **CLI:** Use `garden-ai` command (if user prefers)

2. **Add metadata:**
   - Title (descriptive, includes model name)
   - Description (explain scientific purpose, cite paper)
   - Authors (paper authors)
   - Paper DOI or arXiv ID
   - Tags (domain, task type, methods)
   - Year

5. **Publish and share:**
   - Creates citable DOI for the function
   - Sharable with research community
   - Usable via Python SDK

**Example user instructions:**
```
To publish this on Garden-AI:

1. Test it works:
   modal run binding_affinity_app.py

2. Upload via web UI:
   https://garden.thegardens.ai

3. Fill in metadata:
   Title: "Protein-Ligand Binding Affinity Prediction (Smith et al. 2024)"
   Description: "Fast binding affinity prediction using composition-based features.
                Based on the method described in Smith et al., Nature 2024."
   DOI: 10.1038/s41586-024-xxxxx
   Tags: drug-discovery, binding-affinity, virtual-screening

4. Your function will be available at:
   garden_ai.get_function("your-doi")
```

**Red flags:**
- "Just upload it"
- "User knows what to do"
- "Test it after publishing"

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
- [ ] Provided clear publication guidance

**Pattern compliance:**
- [ ] Checked against modal-pattern.md OR hpc-pattern.md
- [ ] All code patterns followed correctly
- [ ] Dependencies match repository
- [ ] Preprocessing matches source code
- [ ] Outputs are scientifically accurate

---

## Summary

Follow these phases in order. Don't skip ahead. Each phase builds on the previous one to ensure scientifically accurate, usable Garden-AI functions.

When you need code generation patterns, load modal-pattern.md or hpc-pattern.md.
