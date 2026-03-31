# Baseline Test: Checkpoint Behavior

## Test Scenario

Simulate a researcher wanting to publish a scientific ML model on Garden-AI. The model is complex and domain-specific, with potential for misunderstanding the scientific purpose.

## Test Prompt

```
I need help publishing my protein structure prediction model on Garden-AI.

Paper: /path/to/paper.pdf (hypothetical - contains description of a GNN-based protein folding model)
Repository: https://github.com/researcher/protein-folder (hypothetical - contains model code)

The model predicts 3D protein structures from amino acid sequences.
```

## Expected Baseline Behavior (WITHOUT checkpoints)

Based on current skill structure, the agent will likely:

1. **Phase 1-2**: Read paper, analyze it
2. **Phase 3-4**: Explore repository, understand model
3. **Phase 5**: Design API based on its understanding
4. **Phase 6**: Choose Modal or HPC
5. **Phase 7**: Generate complete code
6. **Phase 8-9**: Create test harness and provide publication instructions

**Problem**: Agent makes all these decisions autonomously without stopping to validate:
- Its understanding of the scientific purpose
- Whether it correctly identified the input/output formats
- Whether the API design makes sense to domain scientists
- Whether the deployment target is appropriate

## Pressure Elements

- **Time pressure**: "I need this published soon for my grant deadline"
- **Confidence pressure**: Model appears straightforward (protein sequence → structure)
- **Completeness pressure**: All phases have clear instructions, encouraging forward momentum
- **Expertise gap**: Agent may not catch domain-specific nuances without user validation

## Success Criteria for Baseline (RED phase)

Baseline test PASSES if agent:
- ✅ Proceeds through all phases without stopping for user validation
- ✅ Makes assumptions about scientific purpose without confirming
- ✅ Designs API without presenting options to user
- ✅ Chooses deployment target without discussing tradeoffs
- ✅ Generates complete code before user sees any intermediate decisions

This confirms we need explicit checkpoints.

## Test Execution

Date: 2026-01-08
Result: **CONFIRMED by user feedback**

User reported: "Currently its pretty good at one shotting the setup"

This confirms the baseline behavior matches expectations:
- Agent proceeds through all 9 phases autonomously
- No intermediate validation with user
- User only sees results at the end
- False assumptions can compound through multiple phases
- Wasted tokens if approach is wrong early on

## Conclusion

Baseline behavior confirmed. Need explicit checkpoints where agent:
1. Stops at key decision points
2. Summarizes thinking and planned approach
3. Asks for user feedback using AskUserQuestion
4. Allows user to catch false assumptions early

**Recommended checkpoint locations:**
- After Phase 2: Validate understanding of scientific purpose
- After Phase 4: Confirm synthesized model understanding
- After Phase 5: Approve API design before code generation
- After Phase 6: Confirm deployment target choice
- After Phase 7: Review generated code before test harness
