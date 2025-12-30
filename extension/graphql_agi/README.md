# GraphQL AGI Extension

A comprehensive extension for Kuzu that combines **GraphQL**, **LanceDB**, **AGI Reasoning**, and **Ladybug Debugging** into a unified AI-native graph database interface.

## Features

### GraphQL API Layer
- Automatic schema generation from Kuzu catalog
- GraphQL to Cypher query translation
- Relay-style pagination support
- Nested query resolution
- Introspection support

### LanceDB Vector Storage
- Seamless vector embedding storage
- Multiple embedding providers (OpenAI, Voyage, Cohere, Ollama)
- HNSW and IVF-PQ indexing
- Hybrid search (vector + keyword)
- Automatic embedding generation

### AGI Reasoning Engine
- Chain-of-Thought (CoT) reasoning
- Tree of Thoughts (ToT) exploration
- ReAct (Reasoning + Acting)
- Self-consistency with voting
- Reflexion with self-correction
- Plan-and-Execute strategy
- Multi-hop graph reasoning

### Ladybug Debugger
- Query complexity analysis
- Performance monitoring
- Anti-pattern detection
- Query plan visualization
- Slow query logging
- Interactive HTML reports

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphQL AGI Gateway                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   GraphQL   │  │   LanceDB   │  │  Ladybug Debugger   │ │
│  │   Schema    │  │   Vector    │  │  Query Analyzer     │ │
│  │   Layer     │  │   Store     │  │  Performance Mon    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐ │
│  │              AGI Reasoning Engine                       │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │ │
│  │  │ Planning │  │Knowledge │  │  Autonomous Agent    │  │ │
│  │  │  System  │  │  Graph   │  │  Tool Executor       │  │ │
│  │  └──────────┘  └──────────┘  └──────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                │
│  ┌────────────────────────┴────────────────────────────┐   │
│  │              Kuzu Graph Database Core                │   │
│  │    Cypher Engine • HNSW Indices • Storage Layer      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Python Package

```bash
cd extension/graphql_agi/python
pip install -e ".[all]"
```

### C++ Extension

```bash
cd extension/graphql_agi
mkdir build && cd build
cmake ..
make
```

## Quick Start

### Python API

```python
from graphql_agi import GraphQLAGI

# Initialize
agi = GraphQLAGI("./my_database")

# GraphQL Query
result = agi.query('''
    query {
        users(first: 10) {
            id
            name
            email
        }
    }
''')

# Semantic Search
similar = agi.semantic_search(
    query="machine learning expert",
    table="users",
    top_k=5
)

# AGI Reasoning
answer = agi.reason(
    question="What projects connect AI researchers?",
    strategy="chain_of_thought"
)

# Query Analysis
analysis = agi.analyze_query("query { users { id } }")
print(f"Complexity: {analysis['complexity']}")
```

### FastAPI Server

```bash
# Start the server
graphql-agi

# Or with uvicorn
uvicorn graphql_agi.server:app --reload
```

Then access:
- API Docs: http://localhost:8000/docs
- GraphQL: http://localhost:8000/graphql
- Schema: http://localhost:8000/graphql/schema

### REST API Examples

```bash
# GraphQL Query
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ users { id name } }"}'

# Semantic Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI researcher", "table": "users", "top_k": 5}'

# AGI Reasoning
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{"question": "What connects these users?"}'

# Query Analysis
curl -X POST http://localhost:8000/debug/analyze \
  -H "Content-Type: application/json" \
  -d '"query { users { id } }"'
```

## Reasoning Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `chain_of_thought` | Sequential step-by-step reasoning | Simple questions |
| `tree_of_thoughts` | Explores multiple paths with backtracking | Complex problems |
| `react` | Interleaves reasoning with tool use | Data gathering |
| `self_consistency` | Multiple chains with voting | High confidence |
| `reflexion` | Self-reflection and correction | Iterative improvement |
| `plan_and_execute` | Plans first, then executes | Multi-step tasks |

## Configuration

```python
from graphql_agi import GraphQLAGI, GraphQLAGIConfig

config = GraphQLAGIConfig(
    # Database
    database_path="./my_db",

    # GraphQL
    max_query_depth=15,
    max_query_complexity=1000,

    # LanceDB
    lancedb_uri="lance://./vectors",
    vector_dimension=1536,
    embedding_model="text-embedding-3-large",

    # AGI
    default_reasoning_strategy="chain_of_thought",
    max_reasoning_steps=10,
    llm_model="claude-3-opus",

    # Ladybug
    enable_debugging=True,
    slow_query_threshold_ms=100
)

agi = GraphQLAGI(config=config)
```

## Testing

```bash
cd extension/graphql_agi/python
pytest tests/ -v
```

## Project Structure

```
extension/graphql_agi/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── graphql_agi_extension.h
│   ├── graphql/
│   │   ├── schema_registry.h
│   │   └── query_translator.h
│   ├── lancedb/
│   │   └── lancedb_connector.h
│   ├── agi/
│   │   ├── reasoning_engine.h
│   │   └── planning_system.h
│   └── ladybug/
│       └── debugger.h
├── src/
│   ├── graphql_agi_extension.cpp
│   ├── graphql/
│   │   ├── schema_registry.cpp
│   │   └── query_translator.cpp
│   └── ...
└── python/
    ├── pyproject.toml
    ├── graphql_agi/
    │   ├── __init__.py
    │   ├── server.py
    │   ├── core/
    │   ├── lancedb/
    │   ├── agi/
    │   └── ladybug/
    └── tests/
```

## License

MIT License - See LICENSE file for details.
