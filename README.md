# Gardener - Garden-AI Publishing Assistant for Claude Code

A Claude Code skill that helps researchers publish scientific machine learning models on [Garden-AI](https://thegardens.ai).

## What is this?

Gardener is a specialized skill for [Claude Code](https://claude.ai/code) that guides you from a research paper and code repository to a fully deployed, citable function on Garden-AI. Think of it as a pair programmer that:

- Analyzes your research paper to understand your model's purpose
- Explores your code repository to find the inference logic
- Designs domain-appropriate Python APIs for your model
- Generates deployment code for Modal (cloud) or HPC systems
- Tests the generated code and helps you debug it
- Guides you through publishing to Garden-AI

## What is Garden-AI?

[Garden-AI](https://thegardens.ai) is a FAIR AI/ML model publishing platform where:

- Published models get DOIs for academic citation
- Users run your models remotely via simple Python APIs
- Models can execute on serverless cloud (Modal) or HPC clusters (via Globus Compute)
- No installation or environment setup required for users

## Installation

### Prerequisites

- [Claude Code CLI](https://claude.ai/code) installed and authenticated

### Install Gardener

```bash
# Start Claude Code
claude

# In Claude Code:
/plugin marketplace add Garden-AI/gardener
/plugin install gardener
```

## Usage

Once installed, use the `/gardener` command to start the workflow, or simply mention "help me publish on Garden-AI" in your conversation.

### Example Session

```
You: /gardener

     I want to publish my protein structure prediction model on Garden-AI.
     Here's the paper PDF: paper.pdf
     And the GitHub repo: https://github.com/user/protein-model

Claude: I'll help you publish this model on Garden-AI. Let me start by
        analyzing your paper to understand the model's purpose...

        [Gardener guides you through 9 phases:]
        1. Gather paper + code
        2. Analyze paper (CHECKPOINT: validates understanding)
        3. Explore repository
        4. Synthesize understanding (CHECKPOINT: confirms model details)
        5. Design API (CHECKPOINT: approves function signatures)
        6. Choose deployment platform (CHECKPOINT: Modal vs HPC)
        7. Generate deployment code (CHECKPOINT: reviews generated code)
        8. Test & debug iteratively
        9. Guide publication to Garden-AI

You: Looks good, let's test it!

Claude: I'll run the code now and we can fix any issues together...
```

### What Makes a Good Use Case?

**Great for Gardener:**
- You have a paper + code repository for an ML model
- The model performs inference/prediction (not training)
- You want domain scientists to easily use your model
- You want to publish on Garden-AI with a DOI

**Not ideal:**
- Pure data processing scripts (not ML models)
- Training code (Garden-AI is for inference)
- Code already in Garden-AI format

## Key Features

### Scientifically-Aware API Design

Gardener designs APIs based on your domain, not ML implementation details:

```python
# ✅ Gardener designs this
def predict_stability(protein_sequences: list[str]) -> list[float]:
    """Predict thermodynamic stability for protein sequences."""

# ❌ Not this
def run_model(inputs: np.ndarray, batch_size: int = 32, hidden_dim: int = 512):
    """Run the neural network."""
```

### Platform-Appropriate Code Generation

Gardener generates correct code for your deployment target:

- **Modal** - For fast inference, single GPU, standard packages
- **groundhog_hpc** - For long computations, multi-GPU, HPC libraries

Each platform has different requirements (imports, decorators, calling conventions), and Gardener handles these correctly.

### Checkpoint-Driven Workflow

Gardener validates its understanding with you at 5 critical checkpoints before generating code. This ensures the final API matches your scientific intent.

### Iterative Testing

Gardener doesn't just write code and declare victory. It runs the code, debugs errors with you, and iterates until it works.

## Documentation

- **Garden-AI Documentation**: [garden-ai.readthedocs.io](https://garden-ai.readthedocs.io/en/latest/)
- **Skill Technical Docs**: See `skills/gardener/README.md` for developer documentation
- **Claude Code Help**: Run `/help` in Claude Code

## Contributing

For developers working on the skill itself:

- `CLAUDE.md` - Development guide for this repository
- `skills/gardener/README.md` - Technical documentation for the skill
- `skills/gardener/SKILL.md` - Skill entry point and overview

## Support

- **Issues**: Report bugs or request features at [github.com/Garden-AI/gardener/issues](https://github.com/Garden-AI/gardener/issues)
- **Garden-AI Platform**: [thegardens.ai](https://thegardens.ai)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
