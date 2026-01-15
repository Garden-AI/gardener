# Repository Analysis Patterns

**Referenced from:** workflow-phases.md (Phase 3: Explore Repository)

Load this file when you need guidance on analyzing unfamiliar repositories and ML frameworks.

---

## Core Principle: Find and Use High-Level Interfaces

**Most important rule: If the library provides a high-level interface, USE IT.**

### How to Identify High-Level Interfaces

Most ML libraries provide abstractions that hide complexity. Your job is to find and use them.

**Look for these patterns:**

1. **Classes named:** Calculator, Pipeline, Predictor, Interface, Engine, Wrapper
2. **Methods that return results:** `.predict()`, `.compute()`, `.generate()`, `.process()`
3. **Example scripts:** `examples/`, `demo.py`, README quickstart sections

**Don't:**
- Manually build internal representations (graphs, embeddings, tokens)
- Call low-level model methods directly
- Reimplement preprocessing logic

### Interface Pattern Examples

These are examples of the principle above. The specific classes/methods vary by framework - always check the repository's documentation and examples.

**Example 1: Model with Calculator/Predictor Interface**
```python
# ✅ Use the high-level interface
from library import Calculator

calc = Calculator(model)
result = calc.compute(input_data)

# ❌ Don't manually preprocess
preprocessed = manual_preprocessing(input_data)
output = model.forward(preprocessed)
```

**Example 2: NLP Library with Pipeline**
```python
# ✅ Use the pipeline
from transformers import pipeline

classifier = pipeline("task-name", model="model-name")
results = classifier(texts)

# ❌ Don't manually tokenize
tokenizer = AutoTokenizer.from_pretrained("model-name")
tokens = tokenizer(text)
output = model(**tokens)
```

**Example 3: Vision Library with Transforms**
```python
# ✅ Use provided transforms
from library.models import ModelClass

weights = ModelClass.DEFAULT_WEIGHTS
model = ModelClass(weights=weights)
preprocess = weights.transforms()
result = model(preprocess(image))

# ❌ Don't manually normalize
normalized = (image - mean) / std
```

### How to Find the Interface

**Priority order for finding interfaces:**

1. **README.md** - Usually shows quickstart/basic usage
2. **examples/ directory** - Complete working examples
3. **tutorials/ or docs/** - Step-by-step guides
4. **tests/ directory** - Shows expected usage patterns

**Search patterns:**
```bash
# Find interface classes
rg "class.*(Calculator|Pipeline|Predictor|Interface|Engine)"

# Find example usage in code
rg "\.predict\(|\.compute\(|\.process\(|\.generate\("
```

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

---

## General Input/Output Patterns

**Use standard serializable types for inputs/outputs:**

✅ **Serializable:** strings, ints, floats, lists, dicts, bools
❌ **Not serializable:** NumPy arrays, PIL images, custom objects, file handles

**General pattern:**
```python
def process_batch(inputs: list[str]) -> list[dict]:
    """Process domain-specific inputs."""
    # Import domain library
    from domain_library import DomainObject

    results = []
    for item in inputs:
        # Convert string input to domain object internally
        obj = DomainObject.from_string(item)

        # Process
        output = obj.compute_property()

        # Convert back to serializable types
        results.append({
            "input": item,  # Original string
            "output": serialize(output),  # Back to basic types
            "metadata": {...},
        })

    return results  # All basic Python types
```

**Key principle:** Accept and return serializable types. Do conversions (strings ↔ objects, lists ↔ arrays) internally.

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
