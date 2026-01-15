# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains the **Gardener skill** - a specialized Claude Code skill that helps researchers publish scientific ML models on Garden-AI. The skill guides users through analyzing research papers and code repositories to generate deployable Modal apps or groundhog_hpc scripts.

Garden-AI is a FAIR AI/ML model publishing platform where models get DOIs and can be executed remotely on cloud or HPC infrastructure.

## Repository Structure

This is a **documentation-only repository** consisting of markdown files that define the skill's behavior:

```
gardener/
├── README.md                          # User-facing documentation for GitHub
├── CLAUDE.md                          # This file - developer guidance
├── LICENSE
├── .claude-plugin/
│   └── marketplace.json              # Marketplace plugin definition
└── skills/
    └── gardener/
        ├── .claude-plugin/
        │   └── plugin.json           # Skill plugin definition (includes /gardener command)
        ├── SKILL.md                  # Main entry point and overview
        ├── workflow-phases.md        # Detailed 9-phase workflow instructions
        ├── modal-pattern.md          # Patterns for generating Modal apps
        ├── modal-examples.md         # Real-world Modal examples
        ├── hpc-pattern.md            # Patterns for generating groundhog_hpc scripts
        ├── hpc-examples.md           # Real-world HPC examples
        ├── repository-patterns.md    # Guidance for analyzing ML repositories
        ├── test-baseline-checkpoints.md
        ├── test-with-checkpoints.md
        └── README.md                 # Technical skill documentation
```

### Key Files

- **README.md** (root) - User-facing installation and usage guide for GitHub visitors
- **CLAUDE.md** (root) - This file, developer guidance for working with the repository
- **skills/gardener/README.md** - Technical documentation for the skill itself
- **skills/gardener/SKILL.md** - The skill's main entry point loaded by Claude Code
- **skills/gardener/plugin.json** - Defines the `/gardener` command users can invoke

### Two README Files

This repository has **two README files** with different purposes:

1. **Root `README.md`** - User-facing, explains what Gardener does and how to install it
   - Target audience: Researchers who want to use the skill
   - Focus: Installation, usage examples, key features
   - Tone: Welcoming and accessible

2. **`skills/gardener/README.md`** - Technical documentation for the skill
   - Target audience: Developers working on the skill, Claude Code when loaded
   - Focus: Platform differences, calling conventions, implementation patterns
   - Tone: Precise technical reference

## Working with This Repository

### Installation for Users

Users install this skill via the Claude Code plugin marketplace:

```bash
/plugin marketplace add Garden-AI/gardener
/plugin install gardener
```

Once installed, they can invoke the skill with `/gardener` or by mentioning "help me publish on Garden-AI".

### No Build/Test Commands

This repository has no code to compile, test, or run. The "deliverable" is the skill definition itself (markdown files in `skills/gardener/`).

### Making Changes to the Skill

When modifying skill files in `skills/gardener/`:

1. **Consider file dependencies** - Files reference each other (e.g., `SKILL.md` references the pattern files)
2. **Maintain consistency** - If you update patterns in one file, check if other files need updates
3. **Keep examples realistic** - Examples should reflect actual Garden-AI usage patterns
4. **Verify cross-references** - Ensure file references and phase numbers are correct

### File Loading Strategy

The skill files are comprehensive (1400+ lines total). When the skill is invoked (via `/gardener` command or natural language):
- It should load files **contextually** based on what phase/task it's working on
- Don't load everything at once - files reference when to load supporting documentation
- Follow the loading guidance in `skills/gardener/SKILL.md`

## Architecture Principles

### Two Deployment Platforms

The skill generates code for two distinct platforms:

1. **Modal** - Serverless cloud compute (fast inference, auto-scaling)
2. **groundhog_hpc** - HPC clusters via Globus Compute (long jobs, multi-GPU)

These platforms have **different technical requirements**:
- Different import patterns
- Different calling conventions
- Different metadata formats
- Different decorators

The skill must generate platform-appropriate code, not mix patterns between them.

### Workflow-Based Approach

The skill follows a **multi-phase workflow** from paper analysis to code generation to testing. Key characteristics:

- **Checkpoint-driven** - Must validate understanding with user at specific points
- **Contextual file loading** - Load supporting docs as needed per phase
- **Iterative testing** - Generated code must be run and debugged, not just written

### API Design Philosophy

The skill designs **domain-scientist-facing APIs**, not ML-engineer-facing ones. This means:
- Function names describe scientific tasks, not model internals
- Parameters are scientifically meaningful, not architecture details
- Always batch-first design for real screening workflows
- Use standard domain formats for inputs/outputs

## Development Workflow

### Adding New Patterns

When adding examples or patterns to the skill files:

1. Base them on **real Garden-AI code** from the platform's repositories
2. Include both the **implementation** (what to generate) and **usage** (how users call it)
3. Show **both platforms** if the pattern applies to Modal and groundhog_hpc
4. Highlight **critical differences** between platforms where they diverge

### Updating Documentation

When updating workflow or pattern files:

1. **Test the instructions** - Can you follow them to generate valid code?
2. **Check cross-references** - Do phase numbers and file references still match?
3. **Maintain consistency** - Do all examples follow the same conventions?
4. **Keep synchronized** - If you change calling conventions in one place, update all examples
5. **Use correct URLs**:
   - Garden-AI website: `https://thegardens.ai`
   - Garden-AI docs: `https://garden-ai.readthedocs.io/en/latest/`

### Common Patterns

- Use `rg` instead of `grep` for searching (preference from global CLAUDE.md)
- Use `fd` instead of `find` for file searching
- Use `pushd` instead of `cd` for directory changes

## Verification

When the skill claims to be "done" with a user's request:

- Has it actually **tested the generated code** (not just written it)?
- Did it **validate at all checkpoints** with AskUserQuestion?
- Does the generated code follow the patterns in `skills/gardener/modal-pattern.md` or `skills/gardener/hpc-pattern.md`?
- Are the calling conventions correct for the platform?

The skill should produce **working, tested code** that follows Garden-AI conventions, not just syntactically correct Python.

## Key Success Criteria

The skill is successful when:

1. Generated code matches the appropriate platform's patterns exactly
2. APIs are designed for domain scientists (not ML implementation details)
3. Code has been actually tested and debugged (not just written)
4. User validated understanding at each checkpoint
5. Dependencies match the source repository's versions

The skill fails when:

1. Modal and groundhog patterns are mixed
2. Checkpoints are skipped without user validation
3. APIs expose model internals instead of scientific tasks
4. Code is "ready to publish" without being run
5. Imports or calling conventions don't match platform requirements
