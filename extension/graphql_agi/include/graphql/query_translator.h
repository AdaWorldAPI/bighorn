#pragma once

/**
 * GraphQL to Cypher Query Translator
 *
 * Translates GraphQL queries into optimized Cypher queries.
 * Supports:
 * - Field selections → RETURN clauses
 * - Arguments → WHERE predicates
 * - Nested objects → Pattern matching
 * - Connections → Pagination
 * - Semantic search → Vector similarity queries
 */

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>

#include "graphql/schema_registry.h"

namespace kuzu {
namespace graphql_agi {

/**
 * Represents a parsed GraphQL operation
 */
enum class GraphQLOperationType {
    Query,
    Mutation,
    Subscription
};

/**
 * Represents a GraphQL variable
 */
struct GraphQLVariable {
    std::string name;
    std::string typeName;
    std::optional<std::string> defaultValue;
};

/**
 * Represents a GraphQL selection (field or fragment)
 */
struct GraphQLSelection;

struct GraphQLFieldSelection {
    std::string name;
    std::optional<std::string> alias;
    std::unordered_map<std::string, std::string> arguments;
    std::vector<std::shared_ptr<GraphQLSelection>> selections;

    // AGI extensions
    bool isSemanticSearch = false;
    bool isReasoningQuery = false;
    bool isVectorOperation = false;
};

struct GraphQLFragmentSpread {
    std::string fragmentName;
};

struct GraphQLInlineFragment {
    std::string typeCondition;
    std::vector<std::shared_ptr<GraphQLSelection>> selections;
};

struct GraphQLSelection {
    std::variant<GraphQLFieldSelection, GraphQLFragmentSpread, GraphQLInlineFragment> selection;
};

/**
 * Represents a parsed GraphQL document
 */
struct GraphQLDocument {
    GraphQLOperationType operationType;
    std::optional<std::string> operationName;
    std::vector<GraphQLVariable> variables;
    std::vector<std::shared_ptr<GraphQLSelection>> selections;
    std::unordered_map<std::string, std::vector<std::shared_ptr<GraphQLSelection>>> fragments;
};

/**
 * Represents a translated Cypher query
 */
struct CypherQuery {
    std::string query;
    std::unordered_map<std::string, std::string> parameters;

    // For hybrid queries
    std::optional<std::string> vectorSearchQuery;
    std::optional<std::string> postProcessingQuery;

    // Execution hints
    bool requiresVectorDB = false;
    bool requiresAGIProcessing = false;
    std::vector<std::string> requiredExtensions;
};

/**
 * Query translation options
 */
struct TranslationOptions {
    bool enableOptimizations = true;
    bool generateParameterizedQueries = true;
    bool includeDebugInfo = false;
    uint32_t defaultLimit = 100;
    std::optional<uint32_t> maxLimit;

    // AGI options
    bool enableSemanticExpansion = false;
    bool enableQueryUnderstanding = false;
    std::optional<std::string> embeddingModel;
};

/**
 * GraphQL to Cypher Query Translator
 */
class QueryTranslator {
public:
    QueryTranslator(const SchemaRegistry& schema, const TranslationOptions& options = {});
    ~QueryTranslator() = default;

    /**
     * Parse a GraphQL query string
     */
    struct ParseResult {
        bool success;
        std::optional<GraphQLDocument> document;
        std::vector<std::string> errors;
    };
    ParseResult parse(const std::string& query);

    /**
     * Translate a GraphQL document to Cypher
     */
    struct TranslationResult {
        bool success;
        std::vector<CypherQuery> queries;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;

        // Debug info
        std::optional<std::string> queryPlan;
        std::optional<std::string> optimizationNotes;
    };
    TranslationResult translate(const GraphQLDocument& document,
                                const std::unordered_map<std::string, std::string>& variables = {});

    /**
     * Convenience method: parse and translate in one call
     */
    TranslationResult translateQuery(const std::string& graphqlQuery,
                                      const std::unordered_map<std::string, std::string>& variables = {});

    /**
     * Validate a GraphQL query against the schema
     */
    struct ValidationResult {
        bool valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    ValidationResult validate(const GraphQLDocument& document);

    /**
     * Get query complexity score
     */
    uint32_t calculateComplexity(const GraphQLDocument& document);

    /**
     * Check if query depth exceeds limit
     */
    bool checkDepth(const GraphQLDocument& document, uint32_t maxDepth);

private:
    const SchemaRegistry& schema_;
    TranslationOptions options_;

    // Translation helpers
    std::string translateSelection(const GraphQLFieldSelection& field,
                                    const std::string& parentAlias,
                                    uint32_t depth);

    std::string translateArguments(const std::unordered_map<std::string, std::string>& args,
                                    const std::string& alias);

    std::string translateFilter(const std::string& fieldName,
                                 const std::string& op,
                                 const std::string& value);

    std::string translatePagination(const std::unordered_map<std::string, std::string>& args);

    std::string translateOrderBy(const std::string& orderByArg);

    // Semantic query translation
    std::string translateSemanticSearch(const GraphQLFieldSelection& field);

    // AGI query translation
    std::string translateReasoningQuery(const GraphQLFieldSelection& field);
    std::string translatePlanningQuery(const GraphQLFieldSelection& field);
    std::string translateKnowledgeGraphQuery(const GraphQLFieldSelection& field);

    // Optimization
    void optimizeQuery(CypherQuery& query);
    void pushDownFilters(CypherQuery& query);
    void optimizeJoins(CypherQuery& query);
};

/**
 * GraphQL Lexer for tokenization
 */
class GraphQLLexer {
public:
    enum class TokenType {
        Name,
        IntValue,
        FloatValue,
        StringValue,
        BlockStringValue,
        Punctuator,
        EOF_TOKEN
    };

    struct Token {
        TokenType type;
        std::string value;
        uint32_t line;
        uint32_t column;
    };

    GraphQLLexer(const std::string& source);
    Token nextToken();
    Token peekToken();

private:
    std::string source_;
    size_t pos_ = 0;
    uint32_t line_ = 1;
    uint32_t column_ = 1;

    void skipWhitespace();
    void skipComment();
    Token readName();
    Token readNumber();
    Token readString();
    Token readBlockString();
};

/**
 * GraphQL Parser
 */
class GraphQLParser {
public:
    GraphQLParser(const std::string& source);
    QueryTranslator::ParseResult parse();

private:
    GraphQLLexer lexer_;
    GraphQLLexer::Token currentToken_;

    void advance();
    bool expect(const std::string& value);
    bool match(const std::string& value);

    std::optional<GraphQLDocument> parseDocument();
    std::optional<std::vector<GraphQLVariable>> parseVariableDefinitions();
    std::optional<std::vector<std::shared_ptr<GraphQLSelection>>> parseSelectionSet();
    std::optional<GraphQLFieldSelection> parseField();
    std::optional<std::unordered_map<std::string, std::string>> parseArguments();
};

} // namespace graphql_agi
} // namespace kuzu
