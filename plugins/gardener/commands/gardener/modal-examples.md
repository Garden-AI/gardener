# Modal Examples for Garden-AI

**Referenced from:** modal-pattern.md, SKILL.md

This file contains complete, working Modal app examples across different scientific domains. Use these as reference implementations when generating Modal apps.

---

## Example 1: Chemistry - Molecular Property Prediction

**Domain:** Computational chemistry / drug discovery
**Model:** ChemBERTa transformer for ADMET properties
**Input:** SMILES strings
**Output:** Molecular properties (solubility, toxicity)

### The Published Function

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

### Using via Garden SDK

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

## Example 2: Computer Vision - Image Classification

**Domain:** Computer vision
**Model:** Vision Transformer (ViT)
**Input:** Image URLs
**Output:** Classifications and feature embeddings

### The Published Class

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

### Using via Garden SDK

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

## Example 3: Text Processing - Sentiment Analysis

**Domain:** Natural language processing
**Model:** BERT-based sentiment classifier
**Input:** Text strings
**Output:** Sentiment scores

```python
"""Sentiment analysis using BERT."""

import modal

app = modal.App("sentiment-analysis")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==4.36.0",
    "torch==2.1.0",
)


@app.function(image=image, gpu="T4", cpu=2.0, memory=4096)
def analyze_sentiment_batch(texts: list[str]) -> dict:
    """
    Batch sentiment analysis for text screening.

    Args:
        texts: List of text strings to analyze

    Returns:
        {"results": [...], "summary": {...}}
    """
    from transformers import pipeline
    import torch

    # Use pipeline for simplicity
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )

    results = []
    succeeded = 0

    for idx, text in enumerate(texts):
        result = {"index": idx, "text": text[:100], "success": False}  # Truncate text in output
        try:
            prediction = sentiment_pipeline(text)[0]

            result.update({
                "success": True,
                "sentiment": prediction["label"],
                "confidence": float(prediction["score"]),
            })
            succeeded += 1
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return {"results": results, "summary": {"total": len(texts), "succeeded": succeeded}}


@app.local_entrypoint()
def main():
    """Test locally."""
    test_texts = [
        "This is a wonderful product!",
        "I'm very disappointed with this purchase.",
        "It works okay, nothing special.",
    ]

    result = analyze_sentiment_batch.remote(test_texts)
    print(f"Analyzed: {result['summary']['succeeded']}/{result['summary']['total']}")
    for r in result['results']:
        if r['success']:
            print(f"  '{r['text']}...' → {r['sentiment']} ({r['confidence']:.2f})")
```

---

## Common Patterns Across Examples

### 1. Batch Processing with Error Handling

All examples process lists of inputs and handle errors per-item:

```python
results = []
succeeded = 0

for idx, input_item in enumerate(input_list):
    result = {"index": idx, "input": input_item, "success": False}
    try:
        # Processing logic
        output = process(input_item)
        result.update({"success": True, "output": output})
        succeeded += 1
    except Exception as e:
        result["error"] = str(e)

    results.append(result)

return {"results": results, "summary": {"total": len(input_list), "succeeded": succeeded}}
```

### 2. Imports Inside Functions

Modal requires ALL imports inside the function:

```python
@app.function(image=image)
def my_function(data):
    # Imports go HERE, not at module level
    import torch
    from transformers import AutoModel
    # ... rest of function
```

### 3. Resource Specifications

Specify GPU, CPU, and memory based on model requirements:

```python
@app.function(
    image=image,
    gpu="T4",        # or "A10G", "A100", etc.
    cpu=2.0,         # Number of CPUs
    memory=4096,     # Memory in MB
    timeout=600,     # Timeout in seconds
)
```

### 4. Testing with Local Entrypoint

Use `@app.local_entrypoint()` for testing:

```python
@app.local_entrypoint()
def main():
    # Test with realistic data
    result = my_function.remote(test_data)
    print(result)
```

### 5. Classes for Expensive Initialization

Use `@app.cls()` when model loading is expensive:

```python
@app.cls(image=image, gpu="T4")
class MyPredictor:
    def __init__(self):
        # Load model ONCE per container
        self.model = load_expensive_model()

    @modal.method()
    def predict(self, inputs: list[str]) -> dict:
        # Use self.model (already loaded)
        return self.model(inputs)
```

---

## When to Use Which Pattern

**Use simple `@app.function()` when:**
- Model loads quickly (<5 seconds)
- Stateless computation
- Simple use case

**Use `@app.cls()` with methods when:**
- Model loading is expensive (>5 seconds, large weights)
- Need to maintain state across calls
- Multiple related prediction methods

**Choose GPU type based on model size:**
- <1B parameters: `gpu="T4"` (cheapest)
- 1-7B parameters: `gpu="A10G"`
- 7-13B parameters: `gpu="A100"`
- 13B+ parameters: `gpu="A100-80GB"`

---

## Key Differences from groundhog_hpc

| Aspect | Modal (shown here) | groundhog_hpc |
|--------|-------------------|---------------|
| User calls | `garden.func(args)` - NO .remote() | `garden.func.remote(args, endpoint="name")` - MUST use .remote() |
| Imports | INSIDE functions only | Module level OK |
| Metadata | `modal.App()` + `image` | PEP 723 `# /// script` |
| Testing | `@app.local_entrypoint()` | `@hog.harness()` |
| Decorators | `@app.function()`, `@app.cls()` | `@hog.function()`, `@hog.method()` |

---

## Usage After Publishing

Once published to Garden-AI, users access via the Garden SDK:

```python
import garden_ai

# Get the garden by DOI
garden = garden_ai.get_garden("10.26311/your-doi-here")

# Call functions directly (Garden backend handles Modal execution)
result = garden.your_function_name(your_args)

# For classes
result = garden.YourClassName.method_name(your_args)
```

No Modal CLI or setup required for end users - Garden-AI handles everything.
