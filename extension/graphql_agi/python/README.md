# GraphQL AGI

AI-native GraphQL interface for Kuzu with LanceDB vector storage and AGI reasoning capabilities.

## Features

- **GraphQL API**: Auto-generated schema from Kuzu database catalog
- **Vector Search**: LanceDB integration with hybrid graph-vector queries
- **AGI Reasoning**: Chain-of-Thought, Tree of Thoughts, ReAct patterns
- **Planning System**: Multi-step query planning and optimization
- **Ladybug Debugger**: Query analysis and performance monitoring

## Installation

```bash
pip install graphql-agi
```

## Quick Start

```python
from graphql_agi import GraphQLAGI

# Initialize
agi = GraphQLAGI(database_path="./my_db")

# Execute GraphQL query
result = agi.query("""
    query {
        Person(limit: 10) {
            name
            age
        }
    }
""")

# Semantic search
results = agi.semantic_search("find experts in machine learning", limit=5)

# AGI reasoning
answer = agi.reason("What patterns exist in the knowledge graph?")
```

## Server Mode

```bash
# Start FastAPI server
python -m graphql_agi.server --port 8000
```

## License

MIT
