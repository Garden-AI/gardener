---
name: cli-reference
description: Garden-AI CLI commands for deploying functions and creating gardens - garden-ai function modal deploy, garden-ai garden create, authentication
---

# Garden-AI CLI Reference

**Referenced from:** SKILL.md, workflow-phases.md

This file contains the complete CLI reference for publishing and managing gardens and functions. Use this when guiding users through the CLI-based publication workflow.

---

## Overview

The `garden-ai` CLI enables agent-driven publication workflows:

```bash
garden-ai [OPTIONS] COMMAND [ARGS]...
```

**Main command groups:**
- `whoami`, `login`, `logout` - Authentication
- `garden` - Manage Gardens (create, list, search, update, delete)
- `function` - Manage Functions (Modal and HPC)
- `mcp` - MCP server commands

---

## Authentication

### Check Current User

```bash
garden-ai whoami
```

Prints the email of the currently logged in user. If logged out, attempts a login.

### Login

```bash
garden-ai login
```

Initiates Globus authentication flow. Opens browser for OAuth.

### Logout

```bash
garden-ai logout
```

Logs out the current user and clears credentials.

---

## Garden Management

Gardens are collections of related functions that get a citable DOI.

### Create a Garden

```bash
garden-ai garden create [OPTIONS]
```

**Required options:**
| Option | Description |
|--------|-------------|
| `-t, --title TEXT` | Title of the garden |

**Optional options:**
| Option | Description |
|--------|-------------|
| `-a, --authors TEXT` | Comma-separated list of authors. If not provided, uses the current user. |
| `-c, --contributors TEXT` | Comma-separated list of contributors |
| `-d, --description TEXT` | Description of the garden |
| `--tags TEXT` | Comma-separated list of tags |
| `-m, --modal-function-ids TEXT` | Comma-separated Modal function IDs to include |
| `-g, --hpc-function-ids TEXT` | Comma-separated HPC function IDs to include |
| `--year TEXT` | Publication year (default: current year) |
| `--version TEXT` | Garden version (default: 0.0.1) |

**Example - Create garden with deployed functions:**
```bash
garden-ai garden create \
  -t "MACE Structure Relaxation Models" \
  -a "Author One, Author Two" \
  -d "Structure relaxation using MACE neural network potentials" \
  --tags "materials-science,structure-relaxation,MACE" \
  -m "modal-func-id-1,modal-func-id-2" \
  --year 2025
```

### List Gardens

```bash
garden-ai garden list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--all` | List all published gardens (not just yours) |
| `--tags TEXT` | Filter by comma-separated tags |
| `--authors TEXT` | Filter by comma-separated authors |
| `--year TEXT` | Filter by year |
| `-n, --limit INTEGER` | Maximum results to show (default: 20) |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON output |

**Examples:**
```bash
# List your gardens
garden-ai garden list

# List all gardens with specific tags
garden-ai garden list --all --tags "materials-science"

# Get JSON output for programmatic use
garden-ai garden list --json --pretty
```

### Search Gardens

```bash
garden-ai garden search [OPTIONS] QUERY
```

Full-text search across all published gardens.

**Arguments:**
- `QUERY` - Search query (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --limit INTEGER` | Maximum results (default: 10) |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

**Example:**
```bash
garden-ai garden search "protein structure prediction" --limit 5
```

### Show Garden Details

```bash
garden-ai garden show [OPTIONS] DOI
```

Display detailed information about a specific garden.

**Arguments:**
- `DOI` - The DOI of the garden (required)

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

**Example:**
```bash
garden-ai garden show "10.26311/my-garden-doi"

# Get JSON for programmatic use
garden-ai garden show "10.26311/my-garden-doi" --json --pretty
```

### Update Garden Metadata

```bash
garden-ai garden update [OPTIONS] DOI
```

Update an existing garden's metadata.

**Note:** For list fields (authors, contributors, tags, function IDs), the provided values will REPLACE the entire existing list. To add or remove individual items, first retrieve the current values with `garden show`, modify as needed, then provide the complete new list.

**Arguments:**
- `DOI` - DOI of the garden to update (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-t, --title TEXT` | New title |
| `-a, --authors TEXT` | Comma-separated list of authors. Replaces the entire authors list. |
| `-c, --contributors TEXT` | Comma-separated list of contributors. Replaces the entire contributors list. |
| `-d, --description TEXT` | New description |
| `--tags TEXT` | Comma-separated list of tags. Replaces the entire tags list. |
| `--version TEXT` | New version |
| `-m, --modal-function-ids TEXT` | Comma-separated Modal function IDs. Replaces the entire list. |
| `-g, --hpc-function-ids TEXT` | Comma-separated HPC function IDs. Replaces the entire list. |

**Example:**
```bash
garden-ai garden update "10.26311/my-garden" \
  -d "Updated description with more details" \
  --tags "materials-science,MACE,relaxation,new-tag"
```

### Add Functions to Garden

```bash
garden-ai garden add-functions [OPTIONS] DOI
```

Add Modal or HPC functions to an existing garden.

**Arguments:**
- `DOI` - DOI of the garden (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-m, --modal-function-ids TEXT` | Comma-separated Modal function IDs to add |
| `-g, --hpc-function-ids TEXT` | Comma-separated HPC function IDs to add |
| `--replace` | Replace existing functions instead of adding |

**Examples:**
```bash
# Add new functions to existing garden
garden-ai garden add-functions "10.26311/my-garden" \
  -m "new-modal-func-id"

# Replace all functions
garden-ai garden add-functions "10.26311/my-garden" \
  -m "func1,func2" \
  --replace
```

### Delete Garden

```bash
garden-ai garden delete [OPTIONS] DOI
```

**Arguments:**
- `DOI` - DOI of the garden to delete (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-f, --force` | Skip confirmation prompt |

**Example:**
```bash
garden-ai garden delete "10.26311/my-garden" --force
```

---

## Function Management

Functions are the deployable units that run on Modal (cloud) or HPC systems. Use the `modal` or `hpc` subcommands to manage each type.

---

## Modal Function Management

### List Modal Functions

```bash
garden-ai function modal list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --limit INTEGER` | Maximum results (default: 50) |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Show Modal Function Details

```bash
garden-ai function modal show [OPTIONS] FUNCTION_ID
```

**Arguments:**
- `FUNCTION_ID` - Modal function ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-c, --code` | Show function code |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Update Modal Function Metadata

```bash
garden-ai function modal update [OPTIONS] FUNCTION_ID
```

**Arguments:**
- `FUNCTION_ID` - Modal function ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-t, --title TEXT` | New title |
| `-d, --description TEXT` | New description |
| `-a, --authors TEXT` | New comma-separated authors |
| `--tags TEXT` | New comma-separated tags |

---

## Modal App Management

Modal apps contain one or more functions and are deployed as a unit.

### Deploy Modal App

```bash
garden-ai function modal app deploy [OPTIONS] FILE
```

Deploy a Modal app from a Python file.

**Arguments:**
- `FILE` - Path to Modal Python file (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --name TEXT` | App name (auto-detected if not provided) |
| `-t, --title TEXT` | Title for functions (defaults to app name) |
| `-a, --authors TEXT` | Comma-separated list of authors |
| `--tags TEXT` | Comma-separated list of tags |
| `--base-image TEXT` | Base Docker image (default: python:3.11-slim) |
| `-r, --requirements TEXT` | Comma-separated pip requirements |
| `--wait / --no-wait` | Wait for deployment to complete (default: wait) |
| `--timeout FLOAT` | Deployment timeout in seconds (default: 300.0) |

**Example:**
```bash
garden-ai function modal app deploy my_app.py \
  -t "Structure Relaxation with MACE" \
  -a "Jane Doe, John Smith" \
  --tags "materials-science,MACE"
```

**What happens:**
1. CLI uploads the Modal app file
2. Garden deploys it to Modal infrastructure
3. Returns function IDs for use in garden creation
4. Functions become callable via Garden SDK

### List Modal Apps

```bash
garden-ai function modal app list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Show Modal App Details

```bash
garden-ai function modal app show [OPTIONS] APP_ID
```

**Arguments:**
- `APP_ID` - Modal app ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-c, --code` | Show file contents |
| `--show-app-text` | Show the app_text field (deployed code) |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Delete Modal App

```bash
garden-ai function modal app delete [OPTIONS] APP_ID
```

**Arguments:**
- `APP_ID` - Modal app ID to delete (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-f, --force` | Skip confirmation |

---

## HPC Function Management

### Deploy HPC Function

```bash
garden-ai function hpc deploy [OPTIONS] FILE
```

Deploy an HPC function from a Python file.

**Arguments:**
- `FILE` - Path to Python file with function (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --name TEXT` | Function name (auto-detected if not provided) |
| `-t, --title TEXT` | Function title |
| `-e, --endpoint-ids TEXT` | Comma-separated endpoint IDs (required) |
| `-a, --authors TEXT` | Comma-separated authors |
| `-d, --description TEXT` | Function description |
| `--tags TEXT` | Comma-separated tags |
| `-r, --requirements TEXT` | Comma-separated pip requirements |

**Example:**
```bash
garden-ai function hpc deploy my_hpc_script.py \
  -t "DFT Structure Relaxation" \
  -e "endpoint-uuid-1,endpoint-uuid-2" \
  -a "Jane Doe, John Smith" \
  --tags "materials-science,DFT,VASP"
```

### List HPC Functions

```bash
garden-ai function hpc list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Show HPC Function Details

```bash
garden-ai function hpc show [OPTIONS] FUNCTION_ID
```

**Arguments:**
- `FUNCTION_ID` - Function ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-c, --code` | Show function code |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Update HPC Function

```bash
garden-ai function hpc update [OPTIONS] FUNCTION_ID
```

**Arguments:**
- `FUNCTION_ID` - Function ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --name TEXT` | New function name |
| `-t, --title TEXT` | New title |
| `-d, --description TEXT` | New description |
| `-a, --authors TEXT` | New comma-separated authors |
| `--tags TEXT` | New comma-separated tags |
| `-e, --endpoint-ids TEXT` | New comma-separated endpoint IDs |

### Delete HPC Function

```bash
garden-ai function hpc delete [OPTIONS] FUNCTION_ID
```

**Arguments:**
- `FUNCTION_ID` - Function ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-f, --force` | Skip confirmation |

---

## HPC Endpoint Management

HPC endpoints are registered Globus Compute endpoints for running HPC functions.

### Create Endpoint

```bash
garden-ai function hpc endpoint create [OPTIONS]
```

Register a new HPC endpoint.

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --name TEXT` | Endpoint name (required) |
| `-g, --gcmu-id TEXT` | Globus Compute endpoint UUID |

**Example:**
```bash
garden-ai function hpc endpoint create \
  -n "polaris-gpu" \
  -g "5aafb4c1-27b2-40d8-a038-a0277611868f"
```

### List Endpoints

```bash
garden-ai function hpc endpoint list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --limit INTEGER` | Maximum results (default: 50) |
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON |

### Show Endpoint Details

```bash
garden-ai function hpc endpoint show [OPTIONS] ENDPOINT_ID
```

**Arguments:**
- `ENDPOINT_ID` - Endpoint ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--pretty` | Pretty-print JSON output |

### Update Endpoint

```bash
garden-ai function hpc endpoint update [OPTIONS] ENDPOINT_ID
```

**Arguments:**
- `ENDPOINT_ID` - Endpoint ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-n, --name TEXT` | New name |
| `-g, --gcmu-id TEXT` | New GCMU ID |

### Delete Endpoint

```bash
garden-ai function hpc endpoint delete [OPTIONS] ENDPOINT_ID
```

**Arguments:**
- `ENDPOINT_ID` - Endpoint ID (required)

**Options:**
| Option | Description |
|--------|-------------|
| `-f, --force` | Skip confirmation |

---

## MCP Server Commands

The Garden MCP server enables AI assistants to interact with Garden-AI.

### Setup MCP Config

```bash
garden-ai mcp setup [OPTIONS]
```

Add config file for an MCP client.

**Options:**
| Option | Description |
|--------|-------------|
| `--client TEXT` | Client type: 'claude', 'claude code', 'gemini', 'cursor', 'windsurf' |
| `--path TEXT` | Path to initialize config file for any other MCP client |

**Examples:**
```bash
# Setup for Claude Code
garden-ai mcp setup --client "claude code"

# Setup for custom client
garden-ai mcp setup --path ~/.config/my-client/mcp.json
```

### Start MCP Server

```bash
garden-ai mcp serve
```

Start the Garden MCP server for AI assistant integration.

---

## Complete Publication Workflow Example

Here's the full CLI workflow for publishing a Modal app to Garden:

```bash
# 1. Ensure logged in
garden-ai whoami

# 2. Test the Modal app locally first
uv run modal run my_app.py

# 3. Deploy the app to Garden
garden-ai function modal app deploy my_app.py \
  -t "Protein Binding Affinity Prediction" \
  -a "Jane Doe, John Smith" \
  --tags "drug-discovery,binding-affinity"

# Output shows function ID, e.g., "modal-func-abc123"

# 4. Create a garden containing the function
garden-ai garden create \
  -t "Drug Discovery ML Models" \
  -a "Jane Doe, John Smith" \
  -d "Machine learning models for drug discovery workflows" \
  --tags "drug-discovery,ML,screening" \
  -m "modal-func-abc123"

# Output shows garden DOI, e.g., "10.26311/garden-xyz789"

# 5. Verify the garden
garden-ai garden show "10.26311/garden-xyz789"

# 6. Users can now access via SDK:
# from garden_ai import GardenClient
# client = GardenClient()
# garden = client.get_garden("10.26311/garden-xyz789")
# result = garden.predict_binding_affinity(molecules)
```

---

## Error Handling

Common CLI errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| "Not logged in" | No active session | Run `garden-ai login` |
| "Garden not found" | Invalid DOI | Verify DOI with `garden-ai garden list` |
| "Function not found" | Invalid function ID | Check `garden-ai function list` |
| "Permission denied" | Not owner of resource | Contact garden owner |
| "Deployment timeout" | Build took too long | Increase `--timeout` value |
| "Invalid endpoint" | Endpoint not registered | Create with `garden-ai function hpc endpoint create` |

---

## Tips for Agent-Driven Workflows

When using the CLI in automated/agent workflows:

1. **Use JSON output** for parsing:
   ```bash
   garden-ai garden list --json --pretty
   ```

2. **Use --force** to skip confirmations:
   ```bash
   garden-ai garden delete DOI --force
   ```

3. **Check deployment status**:
   ```bash
   garden-ai function modal show APP_ID
   ```

4. **Chain commands** for full workflows:
   ```bash
   # Deploy and capture function ID, then create garden
   FUNC_ID=$(garden-ai function modal deploy app.py --json | jq -r '.function_id')
   garden-ai garden create -t "My Garden" -a "Author" -m "$FUNC_ID"
   ```

---

## Testing Deployed Functions

After deployment, verify the function works through the Garden SDK.

### Test Via Python SDK

```python
from garden_ai import GardenClient

# Initialize client (handles auth automatically)
client = GardenClient()

# Get the garden by DOI
garden = client.get_garden("10.26311/garden-xyz789")

# Test Modal function (direct call)
result = garden.predict_binding_affinity(molecules=["CCO", "CCCO"])
print(f"Result: {result}")

# Test HPC function (async via submit/poll pattern)
job_id = garden.relax_structures.submit(structures=[...])
print(f"Submitted job: {job_id}")

# Poll for completion
status = garden.get_job_status(job_id)
print(f"Status: {status.status}")  # "pending", "running", "completed", etc.

# Retrieve results when complete
if status.status == "completed":
    results = garden.get_results(job_id)
    print(f"Results: {results}")
```

### Quick Verification Script

Create a test script to verify deployment:

```python
#!/usr/bin/env python3
"""Quick verification of deployed Garden function."""
from garden_ai import GardenClient

DOI = "10.26311/garden-xyz789"  # Replace with actual DOI
TEST_INPUT = ["CCO"]  # Replace with domain-appropriate test data

def main():
    client = GardenClient()
    garden = client.get_garden(DOI)

    # List available functions
    print(f"Garden: {garden.metadata.title}")
    print(f"Modal functions: {[f.name for f in garden.modal_functions]}")

    # Test first Modal function
    if garden.modal_functions:
        func = garden.modal_functions[0]
        print(f"\nTesting Modal function: {func.name}")
        result = getattr(garden, func.name)(TEST_INPUT)
        print(f"Result: {result}")

    print("\n✅ Deployment verified!")

if __name__ == "__main__":
    main()
```

Run with:
```bash
uv run python verify_deployment.py
```

### Troubleshooting Deployed Functions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Function not found" | Function not in garden | Check `garden-ai garden show DOI` |
| "Authentication error" | Not logged in | Run `garden-ai login` |
| "Endpoint unavailable" | HPC endpoint offline | Check endpoint status, try different endpoint |
| "Timeout" | Function took too long | Increase timeout or use HPC for long jobs |
| "Serialization error" | Non-serializable input/output | Use primitive types (lists, dicts, strings) |
