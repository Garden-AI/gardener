# Repository Patterns for Model Analysis

**Referenced from:** workflow-phases.md (Phase 3: Explore Repository)

Load this file when you need guidance on analyzing unfamiliar repositories and ML frameworks.

---

## CRITICAL: Use Library Abstractions, Don't Reimplement

**Most important rule: If the library provides a high-level interface, USE IT.**

### Common Interface Patterns

**Atomistic ML Models → ASE Calculator Pattern:**
```python
# ✅ Correct: Use the calculator interface
from fairchem.core import FAIRChemCalculator

atoms.calc = FAIRChemCalculator(model, task_name="omat")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

# ❌ Wrong: Manual preprocessing
graph = build_neighbor_list(atoms, cutoff=6.0)
embeddings = compute_composition_embedding(atoms)
output = model.forward(graph, embeddings)
energy = output["energy"]  # Don't do this!
```

**Transformers → Pipeline Pattern:**
```python
# ✅ Correct: Use the pipeline
from transformers import pipeline

classifier = pipeline("text-classification", model="model-name")
results = classifier(["text1", "text2"])

# ❌ Wrong: Manual tokenization
tokenizer = AutoTokenizer.from_pretrained("model-name")
tokens = tokenizer(text, return_tensors="pt", padding=True)
output = model(**tokens)
logits = output.logits  # Don't do this!
```

**Image Models → Transform Pattern:**
```python
# ✅ Correct: Use provided transforms
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()
prediction = model(preprocess(image))

# ❌ Wrong: Manual preprocessing
image = image.resize((224, 224))
image = np.array(image) / 255.0
image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Don't!
```

### How to Find the Interface

**Search for these patterns in the repository:**
```bash
# Look for calculator/pipeline/interface classes
rg "class.*Calculator|class.*Pipeline|class.*Interface"

# Look for example usage
rg "get_potential_energy|get_forces|pipeline\(|\.transform\("

# Check for preprocessing functions that are PUBLIC
rg "def (preprocess|transform|prepare).*:" --type py
```

**Priority for learning the interface:**
1. **README.md** - Often shows quickstart usage
2. **examples/ directory** - Real end-to-end examples
3. **tutorials/ or docs/** - Step-by-step guides
4. **tests/ directory** - Shows expected usage patterns

### Red Flags You're Doing It Wrong

**If you find yourself:**
- Explaining internal model processing (embeddings, attention, routing)
- Writing code to build graphs, compute neighbors, tokenize
- Manually normalizing inputs or postprocessing outputs
- Copying internal functions from the model implementation

**STOP. Find and use the high-level interface instead.**

### What to Do in Your Code

**For API functions:**
```python
@hog.function()  # or @app.function()
def predict(inputs: list[str]) -> list[dict]:
    """User-facing function using high-level interface."""
    # Import the interface
    from library import Calculator, Model

    # Use the interface, don't reimplement
    calc = Calculator(model)
    results = calc.compute(inputs)

    # Convert to serializable format
    return [{"input": inp, "result": res} for inp, res in zip(inputs, results)]
```

**Don't expose interface details to users:**
- User shouldn't need to know about preprocessing steps
- User shouldn't need to provide cutoff radii, max_neighbors, etc.
- User provides domain-natural inputs (SMILES, XYZ files, sentences)

---

## Finding Model Weights

### PyTorch Models

**Look for:**
```
*.pt, *.pth, *.ckpt files
model_checkpoints/
weights/
pretrained/
```

**Common loading patterns:**
```python
# Direct checkpoint
model.load_state_dict(torch.load("checkpoint.pt"))

# From training script
checkpoint = torch.load("model.ckpt")
model.load_state_dict(checkpoint["model_state_dict"])

# PyTorch Lightning
model = MyModel.load_from_checkpoint("checkpoint.ckpt")
```

**If weights not in repo:**
- Check README for download links
- Look for HuggingFace model IDs in code
- Check model download functions in scripts

### HuggingFace Transformers

**Look for:**
```python
from transformers import AutoModel, AutoTokenizer

# Model ID patterns:
model = AutoModel.from_pretrained("org/model-name")
model = AutoModel.from_pretrained("./local_path")
```

**Generate code:**
```python
@app.function(...)
def predict(texts: list[str]) -> list[dict]:
    from transformers import AutoModel, AutoTokenizer

    # HuggingFace downloads automatically to cache
    model = AutoModel.from_pretrained(
        "org/model-name",
        cache_dir="/models",  # With Modal volume
    )
    tokenizer = AutoTokenizer.from_pretrained("org/model-name")
    # ...
```

### TensorFlow/Keras

**Look for:**
```
*.h5, *.keras files
saved_model/ directories
```

**Loading pattern:**
```python
import tensorflow as tf
model = tf.keras.models.load_model("model.h5")
```

## Finding Preprocessing Logic

**Priority order:**
1. **Example scripts** - Look for `examples/`, `scripts/`, `demo.py`
2. **Test files** - `tests/` often show end-to-end usage
3. **Dataset classes** - `dataset.py`, `data_loader.py`
4. **Main inference** - `predict.py`, `inference.py`

**Use Grep to find:**
```
pattern: "def preprocess|def transform|def prepare"
pattern: "Normalize|StandardScaler|tokenize"
```

**Copy preprocessing verbatim** - Don't "improve" it:
```python
# ✅ DO: Copy exactly from source
def preprocess_molecule(smiles: str):
    # Copied from repo's data.py:145-160
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)
    # ... exact logic from source
```

## Dependency Version Strategy

### Priority 1: Exact versions from repo
```python
# ✅ Read requirements.txt and use exact versions
.pip_install(
    "torch==2.0.1",
    "rdkit==2023.3.2",
)
```

### Priority 2: Compatible versions if exact fails
```python
# ⚠️ If exact version unavailable on Modal's architecture
.pip_install(
    "torch>=2.0,<2.1",  # Pin major.minor
)
```

### Priority 3: Document uncertainty
```python
# ❌ If repo has no requirements file
.pip_install(
    "torch",  # Latest version - user should test
)
# TODO in docstring: "Using latest torch, may need pinning"
```

## Framework-Specific Input/Output Handling

### Chemistry (RDKit, SMILES)

**Input:** String SMILES
**Processing:** Convert to RDKit mol object internally
**Output:** Structured dict with results

```python
def predict_property(smiles_list: list[str]) -> list[dict]:
    """
    Predict molecular properties from SMILES strings.

    Args:
        smiles_list: List of SMILES strings (e.g., ["CCO", "CC(=O)O"])

    Returns:
        List of dicts with structure:
        {
            "smiles": original input,
            "property": predicted value,
            "valid": whether SMILES was valid,
        }
    """
    from rdkit import Chem

    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)  # Internal conversion
        if mol is None:
            results.append({"smiles": smiles, "valid": False})
            continue

        # ... prediction logic
        results.append({
            "smiles": smiles,
            "property": prediction,
            "valid": True,
        })

    return results  # Native Python types (serializable)
```

### Materials (ASE, XYZ files)

**Input:** String XYZ format or CIF
**Processing:** Parse to atoms object internally
**Output:** Dict with results (not ASE objects)

```python
def relax_structure(xyz_string: str) -> dict:
    """
    Relax atomic structure to minimum energy.

    Args:
        xyz_string: Structure in XYZ format:
            "2
            Lattice=\"5.0 0 0 0 5.0 0 0 0 5.0\"
            C 0.0 0.0 0.0
            C 1.25 1.25 1.25"

    Returns:
        {
            "energy": final energy in eV,
            "positions": list of [x, y, z] coordinates,
            "forces": list of force vectors,
        }
    """
    from ase.io import read
    from io import StringIO

    # Parse internally
    atoms = read(StringIO(xyz_string), format="xyz")

    # ... relaxation logic

    # Return serializable data
    return {
        "energy": float(final_energy),
        "positions": atoms.positions.tolist(),  # NumPy → list
        "forces": forces.tolist(),
    }
```

### Images (PIL, NumPy)

**Input:** Base64 encoded image string OR URL
**Processing:** Decode/download internally
**Output:** Dict with predictions

```python
def classify_image(image_data: str, source: str = "base64") -> dict:
    """
    Classify image content.

    Args:
        image_data: Base64 string or URL
        source: "base64" or "url"

    Returns:
        {
            "top_class": class name,
            "confidence": probability,
            "top_5": list of (class, prob) tuples,
        }
    """
    from PIL import Image
    import base64
    from io import BytesIO
    import requests

    # Handle input
    if source == "base64":
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
    else:  # URL
        response = requests.get(image_data)
        img = Image.open(BytesIO(response.content))

    # ... classification

    return {
        "top_class": "cat",
        "confidence": 0.95,
        "top_5": [("cat", 0.95), ("dog", 0.03), ...],
    }
```

## Common Repository Structures

### Structure 1: Simple Script Repo
```
repo/
├── model.py          # Model definition
├── train.py          # Training (ignore)
├── inference.py      # KEY: Copy from here
├── requirements.txt  # KEY: Use these versions
└── checkpoint.pt     # Weights
```

**Strategy:** Read inference.py and copy its logic

### Structure 2: Package Repo
```
repo/
├── src/
│   └── package_name/
│       ├── models/
│       │   └── model.py     # Model definition
│       ├── data/
│       │   └── dataset.py   # Preprocessing
│       └── inference/
│           └── predict.py   # KEY: Copy from here
├── setup.py
└── requirements.txt
```

**Strategy:** Install package if needed, or extract relevant code

### Structure 3: Research Code Repo
```
repo/
├── notebooks/
│   └── demo.ipynb           # KEY: Best end-to-end example
├── configs/
│   └── model_config.yaml    # Hyperparameters
├── src/
│   └── [scattered files]
└── README.md                # KEY: Read first
```

**Strategy:** Read notebook first to understand workflow, then find code

## Red Flags in Repositories

**Stop and ask user if you see:**
- [ ] No clear inference/prediction code (only training)
- [ ] Multiple different model architectures (which one?)
- [ ] Weights not in repo and no download link
- [ ] Dependencies with known incompatibilities
- [ ] Code hasn't been updated in >3 years (dependencies might be outdated)
- [ ] Multiple conflicting requirements files
- [ ] Proprietary dependencies you can't install

**Document and proceed with caution if:**
- [ ] Repo has minimal documentation
- [ ] No example usage found
- [ ] Preprocessing is implicit/scattered
- [ ] Version numbers missing from dependencies
