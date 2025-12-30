"""
GraphQL Schema Builder

Provides utilities for building and managing GraphQL schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GraphQLTypeKind(str, Enum):
    """GraphQL type kinds."""
    SCALAR = "SCALAR"
    OBJECT = "OBJECT"
    INTERFACE = "INTERFACE"
    UNION = "UNION"
    ENUM = "ENUM"
    INPUT_OBJECT = "INPUT_OBJECT"
    LIST = "LIST"
    NON_NULL = "NON_NULL"


@dataclass
class GraphQLField:
    """Represents a GraphQL field."""
    name: str
    type_name: str
    description: str = ""
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    deprecated: bool = False
    deprecation_reason: str = ""
    is_embeddable: bool = False
    is_semantic_search: bool = False


@dataclass
class GraphQLType:
    """Represents a GraphQL type."""
    name: str
    kind: GraphQLTypeKind
    description: str = ""
    fields: List[GraphQLField] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    possible_types: List[str] = field(default_factory=list)
    kuzu_table_name: str = ""
    is_node_type: bool = False
    is_relation_type: bool = False
    supports_vector_search: bool = False
    embedding_field: str = ""


class SchemaBuilder:
    """
    GraphQL Schema Builder

    Builds GraphQL schemas from Kuzu catalog and custom type definitions.

    Example:
        >>> builder = SchemaBuilder()
        >>> builder.add_type(GraphQLType(
        ...     name="User",
        ...     kind=GraphQLTypeKind.OBJECT,
        ...     fields=[
        ...         GraphQLField(name="id", type_name="ID!"),
        ...         GraphQLField(name="name", type_name="String!"),
        ...     ]
        ... ))
        >>> schema = builder.build()
    """

    def __init__(self):
        self.types: Dict[str, GraphQLType] = {}
        self.directives: List[Dict[str, Any]] = []

        self._add_built_in_types()
        self._add_built_in_directives()

    def _add_built_in_types(self) -> None:
        """Add built-in GraphQL scalar types."""
        scalars = ["ID", "String", "Int", "Float", "Boolean", "DateTime", "JSON", "Vector"]
        for scalar in scalars:
            self.types[scalar] = GraphQLType(
                name=scalar,
                kind=GraphQLTypeKind.SCALAR,
                description=f"Built-in {scalar} scalar type"
            )

        # Node interface
        self.types["Node"] = GraphQLType(
            name="Node",
            kind=GraphQLTypeKind.INTERFACE,
            description="Relay Node interface",
            fields=[GraphQLField(name="id", type_name="ID!")]
        )

        # PageInfo for pagination
        self.types["PageInfo"] = GraphQLType(
            name="PageInfo",
            kind=GraphQLTypeKind.OBJECT,
            description="Relay PageInfo",
            fields=[
                GraphQLField(name="hasNextPage", type_name="Boolean!"),
                GraphQLField(name="hasPreviousPage", type_name="Boolean!"),
                GraphQLField(name="startCursor", type_name="String"),
                GraphQLField(name="endCursor", type_name="String"),
            ]
        )

    def _add_built_in_directives(self) -> None:
        """Add built-in GraphQL directives."""
        self.directives = [
            {
                "name": "deprecated",
                "description": "Marks an element as deprecated",
                "locations": ["FIELD_DEFINITION", "ENUM_VALUE"],
                "arguments": [
                    {"name": "reason", "type": "String", "default": "No longer supported"}
                ]
            },
            {
                "name": "include",
                "description": "Include field if condition is true",
                "locations": ["FIELD", "FRAGMENT_SPREAD", "INLINE_FRAGMENT"],
                "arguments": [
                    {"name": "if", "type": "Boolean!", "required": True}
                ]
            },
            {
                "name": "skip",
                "description": "Skip field if condition is true",
                "locations": ["FIELD", "FRAGMENT_SPREAD", "INLINE_FRAGMENT"],
                "arguments": [
                    {"name": "if", "type": "Boolean!", "required": True}
                ]
            },
            {
                "name": "semantic",
                "description": "Enable semantic search on this field",
                "locations": ["FIELD_DEFINITION"],
                "arguments": [
                    {"name": "model", "type": "String", "default": "text-embedding-3-large"}
                ]
            }
        ]

    def add_type(self, graphql_type: GraphQLType) -> None:
        """Add a type to the schema."""
        self.types[graphql_type.name] = graphql_type

    def add_field_to_type(
        self,
        type_name: str,
        field: GraphQLField
    ) -> None:
        """Add a field to an existing type."""
        if type_name in self.types:
            self.types[type_name].fields.append(field)

    def add_directive(
        self,
        name: str,
        description: str,
        locations: List[str],
        arguments: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add a custom directive."""
        self.directives.append({
            "name": name,
            "description": description,
            "locations": locations,
            "arguments": arguments or []
        })

    def build_from_kuzu(self, connection) -> None:
        """Build schema from Kuzu database catalog."""
        # Get tables
        result = connection.execute("CALL show_tables() RETURN *")
        tables = list(result)

        # Create Query and Mutation root types
        query_type = GraphQLType(
            name="Query",
            kind=GraphQLTypeKind.OBJECT,
            description="Root query type"
        )

        mutation_type = GraphQLType(
            name="Mutation",
            kind=GraphQLTypeKind.OBJECT,
            description="Root mutation type"
        )

        for table in tables:
            table_name = table[0] if isinstance(table, tuple) else table.get('name', '')
            table_type = table[1] if isinstance(table, tuple) else table.get('type', '')

            if table_type == "NODE":
                # Create object type for node table
                node_type = GraphQLType(
                    name=table_name,
                    kind=GraphQLTypeKind.OBJECT,
                    kuzu_table_name=table_name,
                    is_node_type=True,
                    interfaces=["Node"],
                    fields=[GraphQLField(name="id", type_name="ID!")]
                )

                # Get columns
                try:
                    col_result = connection.execute(f"CALL table_info('{table_name}') RETURN *")
                    for col in col_result:
                        col_name = col[0] if isinstance(col, tuple) else col.get('name', '')
                        col_type = col[1] if isinstance(col, tuple) else col.get('type', '')
                        graphql_type = self._kuzu_to_graphql_type(col_type)

                        node_type.fields.append(GraphQLField(
                            name=col_name,
                            type_name=graphql_type
                        ))
                except Exception:
                    pass

                self.add_type(node_type)

                # Add query fields
                query_type.fields.append(GraphQLField(
                    name=self._to_camel_case(table_name),
                    type_name=table_name,
                    arguments=[{"name": "id", "type": "ID!"}]
                ))

                query_type.fields.append(GraphQLField(
                    name=self._to_camel_case(table_name) + "s",
                    type_name=f"[{table_name}!]!",
                    arguments=[
                        {"name": "first", "type": "Int"},
                        {"name": "after", "type": "String"},
                        {"name": "filter", "type": f"{table_name}Filter"}
                    ]
                ))

                # Add mutation fields
                mutation_type.fields.append(GraphQLField(
                    name=f"create{table_name}",
                    type_name=f"{table_name}!",
                    arguments=[{"name": "input", "type": f"Create{table_name}Input!"}]
                ))

        self.add_type(query_type)
        self.add_type(mutation_type)

    def build(self) -> str:
        """Build the GraphQL schema as SDL."""
        lines = []

        # Schema definition
        lines.append("schema {")
        lines.append("  query: Query")
        if "Mutation" in self.types and self.types["Mutation"].fields:
            lines.append("  mutation: Mutation")
        lines.append("}")
        lines.append("")

        # Directives
        for directive in self.directives:
            args = ""
            if directive.get("arguments"):
                arg_strs = []
                for arg in directive["arguments"]:
                    arg_str = f"{arg['name']}: {arg['type']}"
                    if "default" in arg:
                        arg_str += f' = "{arg["default"]}"'
                    arg_strs.append(arg_str)
                args = f"({', '.join(arg_strs)})"

            locations = " | ".join(directive["locations"])
            lines.append(f"directive @{directive['name']}{args} on {locations}")
        lines.append("")

        # Types
        for name, graphql_type in self.types.items():
            if graphql_type.kind == GraphQLTypeKind.SCALAR:
                lines.append(f"scalar {name}")
                lines.append("")
                continue

            if graphql_type.kind == GraphQLTypeKind.ENUM:
                lines.append(f"enum {name} {{")
                for field in graphql_type.fields:
                    lines.append(f"  {field.name}")
                lines.append("}")
                lines.append("")
                continue

            if graphql_type.kind == GraphQLTypeKind.INTERFACE:
                lines.append(f"interface {name} {{")
            elif graphql_type.kind == GraphQLTypeKind.INPUT_OBJECT:
                lines.append(f"input {name} {{")
            else:
                implements = ""
                if graphql_type.interfaces:
                    implements = " implements " + " & ".join(graphql_type.interfaces)
                lines.append(f"type {name}{implements} {{")

            for field in graphql_type.fields:
                args = ""
                if field.arguments:
                    arg_strs = [f"{a['name']}: {a['type']}" for a in field.arguments]
                    args = f"({', '.join(arg_strs)})"

                deprecated = ""
                if field.deprecated:
                    reason = field.deprecation_reason or "No longer supported"
                    deprecated = f' @deprecated(reason: "{reason}")'

                lines.append(f"  {field.name}{args}: {field.type_name}{deprecated}")

            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def validate(self) -> Dict[str, Any]:
        """Validate the schema."""
        errors = []
        warnings = []

        # Check required types
        if "Query" not in self.types:
            errors.append("Missing Query type")

        # Check type references
        for name, graphql_type in self.types.items():
            for field in graphql_type.fields:
                type_name = self._extract_type_name(field.type_name)
                if type_name not in self.types:
                    warnings.append(f"Unknown type '{type_name}' in {name}.{field.name}")

            for iface in graphql_type.interfaces:
                if iface not in self.types:
                    errors.append(f"Type {name} implements unknown interface {iface}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _kuzu_to_graphql_type(self, kuzu_type: str) -> str:
        """Convert Kuzu type to GraphQL type."""
        type_map = {
            "INT64": "Int",
            "INT32": "Int",
            "FLOAT": "Float",
            "DOUBLE": "Float",
            "STRING": "String",
            "BOOL": "Boolean",
            "DATE": "DateTime",
            "TIMESTAMP": "DateTime",
            "UUID": "ID",
        }
        return type_map.get(kuzu_type.upper(), "String")

    def _to_camel_case(self, name: str) -> str:
        """Convert to camelCase."""
        if not name:
            return name
        result = name[0].lower()
        capitalize_next = False
        for char in name[1:]:
            if char == "_":
                capitalize_next = True
            elif capitalize_next:
                result += char.upper()
                capitalize_next = False
            else:
                result += char
        return result

    def _extract_type_name(self, type_ref: str) -> str:
        """Extract base type name from type reference."""
        name = type_ref
        while name and name[0] in "[!":
            name = name[1:]
        while name and name[-1] in "]!":
            name = name[:-1]
        return name
