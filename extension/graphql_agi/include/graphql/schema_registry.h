#pragma once

/**
 * GraphQL Schema Registry
 *
 * Manages the GraphQL schema definition and type system.
 * Automatically maps Kuzu node/relationship types to GraphQL types.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <functional>
#include <variant>

namespace kuzu {
namespace graphql_agi {

// Forward declarations
class GraphQLType;
class GraphQLField;
class GraphQLResolver;

/**
 * GraphQL scalar types
 */
enum class GraphQLScalarType {
    ID,
    String,
    Int,
    Float,
    Boolean,
    DateTime,
    JSON,
    Vector,     // Custom: for embeddings
    Embedding   // Custom: for semantic vectors
};

/**
 * GraphQL type kind
 */
enum class GraphQLTypeKind {
    Scalar,
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    List,
    NonNull,
    Connection,  // Relay-style pagination
    Edge,
    Node
};

/**
 * Represents a GraphQL field argument
 */
struct GraphQLArgument {
    std::string name;
    std::string typeName;
    bool required = false;
    std::optional<std::string> defaultValue;
    std::string description;
};

/**
 * Represents a GraphQL field
 */
struct GraphQLFieldDefinition {
    std::string name;
    std::string typeName;
    std::string description;
    std::vector<GraphQLArgument> arguments;
    bool deprecated = false;
    std::string deprecationReason;

    // AGI-specific metadata
    bool isEmbeddable = false;       // Can generate embeddings
    bool isSemanticSearch = false;   // Supports semantic search
    std::string semanticModel;       // Embedding model to use
};

/**
 * Represents a GraphQL type definition
 */
struct GraphQLTypeDefinition {
    std::string name;
    GraphQLTypeKind kind;
    std::string description;
    std::vector<GraphQLFieldDefinition> fields;
    std::vector<std::string> interfaces;
    std::vector<std::string> possibleTypes; // For Union/Interface

    // Kuzu mapping
    std::string kuzuTableName;
    bool isNodeType = false;
    bool isRelationType = false;

    // AGI metadata
    bool supportsVectorSearch = false;
    std::string embeddingField;
    uint32_t embeddingDimension = 1536;
};

/**
 * Represents a GraphQL directive
 */
struct GraphQLDirective {
    std::string name;
    std::vector<std::string> locations;
    std::vector<GraphQLArgument> arguments;
    std::string description;
};

/**
 * GraphQL Schema Registry
 * Manages all type definitions and provides schema introspection
 */
class SchemaRegistry {
public:
    SchemaRegistry() = default;
    ~SchemaRegistry() = default;

    /**
     * Register a type definition
     */
    void registerType(const GraphQLTypeDefinition& typeDef);

    /**
     * Register a directive
     */
    void registerDirective(const GraphQLDirective& directive);

    /**
     * Get type by name
     */
    const GraphQLTypeDefinition* getType(const std::string& name) const;

    /**
     * Get all registered types
     */
    std::vector<const GraphQLTypeDefinition*> getAllTypes() const;

    /**
     * Get directive by name
     */
    const GraphQLDirective* getDirective(const std::string& name) const;

    /**
     * Build schema from Kuzu catalog
     * Automatically creates GraphQL types for all node and relationship tables
     */
    void buildFromKuzuCatalog(main::ClientContext* context);

    /**
     * Generate SDL (Schema Definition Language) representation
     */
    std::string toSDL() const;

    /**
     * Validate the schema for consistency
     */
    struct ValidationResult {
        bool valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    ValidationResult validate() const;

    /**
     * Add AGI-specific types
     */
    void addAGITypes();

    /**
     * Add LanceDB-specific types
     */
    void addLanceDBTypes();

    /**
     * Add Ladybug debugging types
     */
    void addLadybugTypes();

    // Built-in Query and Mutation root types
    static constexpr const char* QUERY_TYPE = "Query";
    static constexpr const char* MUTATION_TYPE = "Mutation";
    static constexpr const char* SUBSCRIPTION_TYPE = "Subscription";

private:
    std::unordered_map<std::string, GraphQLTypeDefinition> types_;
    std::unordered_map<std::string, GraphQLDirective> directives_;

    void addBuiltInScalars();
    void addBuiltInDirectives();
    void addRelayTypes();
    std::string kuzuTypeToGraphQL(const std::string& kuzuType) const;
};

/**
 * AGI-specific GraphQL types for intelligent queries
 */
namespace AGITypes {

// Semantic search input
struct SemanticSearchInput {
    std::string query;
    std::optional<std::string> model;
    std::optional<uint32_t> topK;
    std::optional<double> threshold;
    std::optional<std::vector<std::string>> filters;
};

// Reasoning request
struct ReasoningRequest {
    std::string question;
    std::vector<std::string> context;
    std::optional<std::string> strategy; // "chain_of_thought", "tree_of_thoughts", etc.
    std::optional<uint32_t> maxSteps;
};

// Planning request
struct PlanningRequest {
    std::string goal;
    std::vector<std::string> constraints;
    std::optional<std::string> planningAlgorithm;
    std::optional<uint32_t> maxDepth;
};

// Knowledge graph query
struct KnowledgeGraphQuery {
    std::string entityId;
    std::optional<uint32_t> hops;
    std::optional<std::vector<std::string>> relationshipTypes;
    std::optional<bool> includeEmbeddings;
};

} // namespace AGITypes

} // namespace graphql_agi
} // namespace kuzu
