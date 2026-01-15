# GREEN Phase Test: Verify Checkpoints

## Changes Made

Added 5 MANDATORY checkpoints to workflow-phases.md:

### Checkpoint 1 (After Phase 2: Analyze Paper)
- **Location**: End of Phase 2
- **Stops for**: Validation of paper understanding
- **Presents**: Summary of scientific domain, model purpose, input/output data, dependencies, computational requirements
- **Asks**: "Is this understanding correct? Are there any important aspects I've missed or misunderstood?"

### Checkpoint 2 (After Phase 4: Understand Model)
- **Location**: End of Phase 4
- **Stops for**: Validation of synthesized model understanding
- **Presents**: Complete filled template (domain, input format, preprocessing, model processing, output format, performance)
- **Asks**: "Does this accurately capture how the model works? Have I misunderstood any preprocessing steps, input/output formats, or computational requirements?"

### Checkpoint 3 (After Phase 5: Design API)
- **Location**: End of Phase 5
- **Stops for**: Approval of API design before code generation
- **Presents**: Complete function signatures with reasoning for names, types, defaults, and structure
- **Asks**: "Does this API make sense for domain scientists? Should I change any function names, parameter types, or output structure?"

### Checkpoint 4 (After Phase 6: Choose Deployment)
- **Location**: End of Phase 6
- **Stops for**: Confirmation of deployment target
- **Presents**: Deployment recommendation with computational requirements analysis and reasoning
- **Asks**: Modal/HPC-specific confirmation questions based on recommendation

### Checkpoint 5 (After Phase 7: Generate Code)
- **Location**: End of Phase 7
- **Stops for**: Review of generated code
- **Presents**: Complete code with walkthrough of key sections and any assumptions/deviations
- **Asks**: "Does this implementation look correct? Should I adjust any dependency versions, preprocessing logic, or output formatting?"

## Updated Elements

1. **Phase Summary Table**: Added Checkpoint column showing which phases require validation
2. **Red Flags in SKILL.md**: Added checkpoint violation red flags
3. **Mandatory Language**: All checkpoints use "MANDATORY" and "STOP HERE" language
4. **Justifications**: Each checkpoint includes "Why this matters" section

## Verification Approach

Two options for testing:

### Option 1: User Testing (Recommended)
- User runs `/gardener` on a real research paper + code
- Observes whether agent stops at each checkpoint
- Provides feedback on:
  - Did agent stop at all 5 checkpoints?
  - Were the summaries clear and useful?
  - Could user provide meaningful feedback?
  - Any rationalizations used to skip checkpoints?

### Option 2: Subagent Testing
- Dispatch subagent with modified skill
- Provide test research paper + code
- Monitor for checkpoint adherence
- Document any attempts to skip or rationalize around checkpoints

## Expected Behavior

With checkpoints in place, agent should:
- ✅ Stop after Phase 2 to validate paper understanding
- ✅ Stop after Phase 4 to confirm model synthesis
- ✅ Stop after Phase 5 to get API approval
- ✅ Stop after Phase 6 to confirm deployment choice
- ✅ Stop after Phase 7 to review generated code
- ✅ Use AskUserQuestion at each checkpoint
- ✅ Wait for user response before proceeding
- ✅ Adjust based on user feedback

## Next Steps

If agent tries to skip checkpoints, document rationalizations for REFACTOR phase.
