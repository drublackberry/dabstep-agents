# Catalog Persistence in Multi-Agent System

## Overview

The multi-agent system implements a **hybrid state management approach** that combines instance-level caching with orchestrator-level shared state. This ensures efficient catalog reuse while maintaining clean separation of concerns.

Additionally, the system uses **pre-read directory listings** to eliminate hallucinations and ensure consistent file discovery by providing the agent with hardcoded directory contents rather than requiring it to write exploration code.

## Architecture

### 0. Pre-Read Directory Listing (Anti-Hallucination)

The system includes a `get_directory_listing()` function that hardcodes directory exploration:

```python
def get_directory_listing(ctx_path: str) -> Dict[str, Any]:
    """Pre-read directory contents to provide to the agent."""
    # Returns:
    # {
    #   "directory_path": str,
    #   "total_files": int,
    #   "total_directories": int,
    #   "files": [
    #     {
    #       "name": str,
    #       "path": str,
    #       "size_bytes": int,
    #       "extension": str,
    #       "category": "data" | "documentation" | "code" | "other"
    #     }
    #   ],
    #   "directories": [...]
    # }
```

**Benefits:**
- Eliminates hallucinations about file existence
- Provides accurate file paths and metadata
- Agent doesn't need to write directory exploration code
- Consistent file discovery across all operations
- Reduces token usage by avoiding redundant exploration code

**How it works:**
- Called automatically in `catalog_data_sources()` and `extract_domain_knowledge()`
- Directory listing is injected into the agent's prompt
- Agent is explicitly told NOT to write directory listing code
- Agent uses the provided file paths to read file contents

### 1. LibrarianAgent Instance-Level Cache

The `LibrarianAgent` maintains an internal cache of the catalog:

```python
class LibrarianAgent(BaseCodeAgent):
    def __init__(self, ...):
        self._catalog_cache: Optional[Dict[str, Any]] = None
        self._catalog_ctx_path: Optional[str] = None
```

**Benefits:**
- Fast access without re-computation
- Automatic reuse within the same agent instance
- Works independently of orchestrator

**Methods:**
- `catalog_data_sources(ctx_path, force_refresh=False)` - Creates or retrieves cached catalog
- `get_catalog()` - Returns cached catalog if available
- `has_catalog(ctx_path)` - Checks if catalog exists for given path
- `extract_domain_knowledge(query, ctx_path, catalog=None)` - Uses cached catalog automatically

### 2. MultiAgentOrchestrator Shared State

The orchestrator maintains shared state accessible to all agents:

```python
class MultiAgentOrchestrator:
    def __init__(self, ...):
        self.shared_state: Dict[str, Any] = {
            "catalog": None,
            "domain_knowledge": {},
            "trajectories": [],
            "execution_results": []
        }
```

**Benefits:**
- Central source of truth for all agents
- Enables cross-agent data sharing
- Persists across multiple task executions
- Organized by task for easy retrieval

**Methods:**
- `get_catalog()` - Returns the shared catalog
- `get_domain_knowledge(task=None)` - Returns domain knowledge for specific task or all tasks

## Usage Patterns

### Pattern 1: Automatic Caching (Recommended)

The simplest approach - let the agent manage caching automatically:

```python
librarian = LibrarianAgent(model_id=..., ctx_path=ctx_path)

# First call creates catalog and caches it
catalog = librarian.catalog_data_sources(ctx_path)

# Subsequent calls use cache automatically
domain_knowledge1 = librarian.extract_domain_knowledge("query 1", ctx_path)
domain_knowledge2 = librarian.extract_domain_knowledge("query 2", ctx_path)
# Both queries use the same cached catalog
```

### Pattern 2: Orchestrator-Managed State

Let the orchestrator manage shared state across all agents:

```python
orchestrator = MultiAgentOrchestrator(model_id=..., ctx_path=ctx_path)

# First task creates catalog and stores in shared state
result1 = orchestrator.solve_task("task 1")

# Second task reuses catalog from shared state
result2 = orchestrator.solve_task("task 2")

# Access shared catalog
catalog = orchestrator.get_catalog()

# Access domain knowledge by task
knowledge = orchestrator.get_domain_knowledge("task 1")
```

### Pattern 3: Manual Catalog Passing

Explicitly pass catalog between operations for fine-grained control:

```python
librarian = LibrarianAgent(model_id=..., ctx_path=ctx_path)

# Create catalog once
catalog = librarian.catalog_data_sources(ctx_path)

# Pass catalog explicitly to domain knowledge extraction
knowledge1 = librarian.extract_domain_knowledge(
    "query 1", ctx_path, catalog=catalog
)
knowledge2 = librarian.extract_domain_knowledge(
    "query 2", ctx_path, catalog=catalog
)
```

### Pattern 4: Force Refresh

Force recreation of catalog when data changes:

```python
librarian = LibrarianAgent(model_id=..., ctx_path=ctx_path)

# Initial catalog
catalog1 = librarian.catalog_data_sources(ctx_path)

# ... data files are updated ...

# Force refresh to get updated catalog
catalog2 = librarian.catalog_data_sources(ctx_path, force_refresh=True)
```

## How It Works

### Catalog Creation Flow

1. **First Call to `catalog_data_sources()`:**
   - Agent explores the directory
   - Reads and catalogs all files
   - Stores result in `_catalog_cache`
   - Returns catalog

2. **Subsequent Calls:**
   - Checks if `_catalog_cache` exists and matches `ctx_path`
   - If yes, returns cached catalog immediately
   - If no, creates new catalog

### Domain Knowledge Extraction Flow

1. **Call to `extract_domain_knowledge()`:**
   - Checks if `catalog` parameter is provided
   - If not, checks `_catalog_cache`
   - If cache miss, calls `catalog_data_sources()` automatically
   - Uses catalog to identify critical sources
   - Reads all critical sources (per updated prompt)
   - Reads relevant non-critical sources
   - Returns structured domain knowledge

### Orchestrator Integration Flow

1. **First Task in `solve_task()`:**
   - Checks `shared_state["catalog"]`
   - If None, calls `librarian.catalog_data_sources()`
   - Stores result in `shared_state["catalog"]`
   - Passes catalog to `extract_domain_knowledge()`
   - Stores domain knowledge in `shared_state["domain_knowledge"][task]`

2. **Subsequent Tasks:**
   - Reuses `shared_state["catalog"]`
   - No redundant catalog creation
   - Each task's domain knowledge stored separately

## Benefits of This Approach

### 1. **Efficiency**
- Catalog created only once per context path
- No redundant file I/O operations
- Fast access for multiple queries

### 2. **Flexibility**
- Works at both agent and orchestrator levels
- Supports manual catalog passing when needed
- Force refresh available for dynamic data

### 3. **Smolagents Alignment**
- Uses standard Python instance variables (no special smolagents features needed)
- Natural object-oriented design
- Compatible with smolagents agent lifecycle

### 4. **Separation of Concerns**
- LibrarianAgent manages its own cache
- Orchestrator manages cross-agent state
- Clear ownership and responsibilities

### 5. **Critical Sources Always Read**
- Updated prompt ensures critical sources are always read
- Catalog provides metadata about which sources are critical
- Domain knowledge extraction uses catalog to identify critical files

## State Lifecycle

### Agent Lifetime
- Cache persists for the lifetime of the `LibrarianAgent` instance
- Cache is lost when agent is destroyed
- Suitable for single-session workflows

### Orchestrator Lifetime
- Shared state persists for the lifetime of the `MultiAgentOrchestrator` instance
- Shared across all agents managed by the orchestrator
- Suitable for multi-task workflows

### Persistence Across Sessions
- Current implementation: In-memory only
- For cross-session persistence, consider:
  - Saving catalog to JSON file (requires write permissions)
  - Using external state store (database, Redis, etc.)
  - Implementing custom serialization/deserialization

## Testing

Run the test suite to verify catalog persistence:

```bash
# Test LibrarianAgent catalog persistence
python src/test/test_librarian_agent.py

# Run examples demonstrating different patterns
python src/examples/catalog_persistence_example.py
```

## Best Practices

1. **Use Automatic Caching by Default**
   - Let the agent handle caching automatically
   - Only use manual passing for special cases

2. **Use Orchestrator for Multi-Task Workflows**
   - Leverage shared state when solving multiple related tasks
   - Access catalog and domain knowledge through orchestrator methods

3. **Force Refresh When Data Changes**
   - Use `force_refresh=True` if underlying data files are modified
   - Otherwise, stale catalog may be used

4. **Check Cache Before Creating**
   - Use `has_catalog()` to check if catalog exists
   - Use `get_catalog()` to retrieve cached catalog

5. **Store Domain Knowledge by Task**
   - Orchestrator automatically stores domain knowledge per task
   - Use `get_domain_knowledge(task)` to retrieve specific task knowledge

## Future Enhancements

Potential improvements to consider:

1. **File-Based Persistence**
   - Save catalog to JSON for cross-session reuse
   - Implement cache invalidation based on file timestamps

2. **Catalog Versioning**
   - Track catalog version/timestamp
   - Automatic invalidation when files change

3. **Partial Catalog Updates**
   - Update only changed files instead of full refresh
   - More efficient for large datasets

4. **Distributed State**
   - Share state across multiple orchestrator instances
   - Use external state store (Redis, database)

5. **Catalog Compression**
   - Compress large catalogs for memory efficiency
   - Lazy loading of catalog sections

## Summary

The hybrid approach provides:
- ✅ Efficient catalog reuse within agent instances
- ✅ Shared state across multiple agents via orchestrator
- ✅ Flexibility for different usage patterns
- ✅ Natural integration with smolagents architecture
- ✅ Clear separation of concerns
- ✅ Support for critical source requirements

This design balances simplicity, efficiency, and flexibility while aligning with smolagents best practices.
