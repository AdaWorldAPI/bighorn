/**
 * GraphQL AGI Extension - Main Implementation
 *
 * This is the entry point for the GraphQL AGI extension,
 * combining GraphQL, LanceDB, AGI reasoning, and Ladybug debugging.
 */

#include "graphql_agi_extension.h"
#include "graphql/schema_registry.h"
#include "graphql/query_translator.h"
#include "lancedb/lancedb_connector.h"
#include "agi/reasoning_engine.h"
#include "agi/planning_system.h"
#include "ladybug/debugger.h"

#include "extension/extension.h"
#include "function/scalar_function.h"
#include "function/table_function.h"
#include "main/client_context.h"
#include "main/database.h"

#include <memory>
#include <mutex>

namespace kuzu {
namespace graphql_agi {

// Global instances (managed per database)
static std::mutex g_mutex;
static std::unordered_map<main::Database*, std::shared_ptr<SchemaRegistry>> g_schemaRegistries;
static std::unordered_map<main::Database*, std::shared_ptr<LanceDBConnector>> g_lanceConnectors;
static std::unordered_map<main::Database*, std::shared_ptr<ReasoningEngine>> g_reasoningEngines;
static std::unordered_map<main::Database*, std::shared_ptr<LadybugDebugger>> g_debuggers;

/**
 * GraphQL Query Function
 * Executes a GraphQL query and returns results
 */
struct GraphQLQueryFunction {
    static constexpr const char* name = "GRAPHQL_QUERY";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // graphql_query(query: STRING) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{common::LogicalTypeID::STRING},
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        // graphql_query(query: STRING, variables: STRING) -> STRING
        auto funcWithVars = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{
                common::LogicalTypeID::STRING,
                common::LogicalTypeID::STRING
            },
            common::LogicalTypeID::STRING,
            executeFuncWithVars
        );
        functions.push_back(std::move(funcWithVars));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr) {
        auto context = reinterpret_cast<main::ClientContext*>(dataPtr);
        auto& queryParam = params[0];

        for (auto i = 0u; i < result.state->getSelVector().getSelSize(); ++i) {
            auto pos = result.state->getSelVector()[i];
            auto query = queryParam->getValue<common::ku_string_t>(pos).getAsString();

            auto resultJson = executeGraphQL(context, query, "{}");
            result.setValue(pos, common::ku_string_t{resultJson});
        }
    }

    static void executeFuncWithVars(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                                     common::ValueVector& result,
                                     void* dataPtr) {
        auto context = reinterpret_cast<main::ClientContext*>(dataPtr);
        auto& queryParam = params[0];
        auto& varsParam = params[1];

        for (auto i = 0u; i < result.state->getSelVector().getSelSize(); ++i) {
            auto pos = result.state->getSelVector()[i];
            auto query = queryParam->getValue<common::ku_string_t>(pos).getAsString();
            auto vars = varsParam->getValue<common::ku_string_t>(pos).getAsString();

            auto resultJson = executeGraphQL(context, query, vars);
            result.setValue(pos, common::ku_string_t{resultJson});
        }
    }

    static std::string executeGraphQL(main::ClientContext* context,
                                       const std::string& query,
                                       const std::string& variables);
};

/**
 * Semantic Search Function
 * Performs vector similarity search using LanceDB
 */
struct SemanticSearchFunction {
    static constexpr const char* name = "SEMANTIC_SEARCH";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // semantic_search(query: STRING, table: STRING, top_k: INT64) -> TABLE
        auto func = std::make_unique<function::TableFunction>(
            name,
            executeFunc,
            bindFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static std::unique_ptr<function::TableFuncBindData> bindFunc(
        main::ClientContext* context,
        function::TableFuncBindInput* input);

    static void executeFunc(function::TableFuncInput& input,
                            function::TableFuncOutput& output);
};

/**
 * AGI Reason Function
 * Performs multi-step reasoning with the AGI engine
 */
struct AGIReasonFunction {
    static constexpr const char* name = "AGI_REASON";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // agi_reason(question: STRING) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{common::LogicalTypeID::STRING},
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * AGI Plan Function
 * Creates an execution plan for a goal
 */
struct AGIPlanFunction {
    static constexpr const char* name = "AGI_PLAN";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // agi_plan(goal: STRING) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{common::LogicalTypeID::STRING},
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Create Embedding Function
 * Generates embeddings for text using configured LLM
 */
struct CreateEmbeddingFunction {
    static constexpr const char* name = "CREATE_AGI_EMBEDDING";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // create_agi_embedding(text: STRING) -> LIST[FLOAT]
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{common::LogicalTypeID::STRING},
            common::LogicalTypeID::LIST,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Vector Store Function
 * Stores vectors in LanceDB
 */
struct VectorStoreFunction {
    static constexpr const char* name = "VECTOR_STORE";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // vector_store(entity_id: STRING, entity_type: STRING, vector: LIST[FLOAT]) -> BOOL
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{
                common::LogicalTypeID::STRING,
                common::LogicalTypeID::STRING,
                common::LogicalTypeID::LIST
            },
            common::LogicalTypeID::BOOL,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Ladybug Debug Function
 * Analyzes and debugs queries
 */
struct LadybugAnalyzeFunction {
    static constexpr const char* name = "LADYBUG_ANALYZE";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // ladybug_analyze(query: STRING) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{common::LogicalTypeID::STRING},
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Get Query Plan Function
 */
struct GetQueryPlanFunction {
    static constexpr const char* name = "LADYBUG_QUERY_PLAN";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // ladybug_query_plan(query: STRING) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{common::LogicalTypeID::STRING},
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Multi-hop Reasoning Function
 */
struct MultiHopReasonFunction {
    static constexpr const char* name = "AGI_MULTIHOP";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // agi_multihop(start_entity: STRING, question: STRING, max_hops: INT64) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{
                common::LogicalTypeID::STRING,
                common::LogicalTypeID::STRING,
                common::LogicalTypeID::INT64
            },
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Hybrid Search Function
 * Combines vector similarity with graph traversal
 */
struct HybridSearchFunction {
    static constexpr const char* name = "HYBRID_SEARCH";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // hybrid_search(query: STRING, table: STRING, vector_weight: DOUBLE) -> TABLE
        auto func = std::make_unique<function::TableFunction>(
            name,
            executeFunc,
            bindFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static std::unique_ptr<function::TableFuncBindData> bindFunc(
        main::ClientContext* context,
        function::TableFuncBindInput* input);

    static void executeFunc(function::TableFuncInput& input,
                            function::TableFuncOutput& output);
};

/**
 * GraphQL Schema Introspection Function
 */
struct GraphQLSchemaFunction {
    static constexpr const char* name = "GRAPHQL_SCHEMA";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // graphql_schema() -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{},
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

/**
 * Knowledge Graph Subgraph Function
 */
struct KGSubgraphFunction {
    static constexpr const char* name = "KG_SUBGRAPH";

    static function::function_set getFunctionSet() {
        function::function_set functions;

        // kg_subgraph(entity_id: STRING, radius: INT64) -> STRING
        auto func = std::make_unique<function::ScalarFunction>(
            name,
            std::vector<common::LogicalTypeID>{
                common::LogicalTypeID::STRING,
                common::LogicalTypeID::INT64
            },
            common::LogicalTypeID::STRING,
            executeFunc
        );
        functions.push_back(std::move(func));

        return functions;
    }

    static void executeFunc(const std::vector<std::shared_ptr<common::ValueVector>>& params,
                            common::ValueVector& result,
                            void* dataPtr);
};

// ============================================================================
// Extension Loading
// ============================================================================

void GraphQLAGIExtension::load(main::ClientContext* context) {
    auto& db = *context->getDatabase();

    std::lock_guard<std::mutex> lock(g_mutex);

    // Initialize components if not already done
    if (g_schemaRegistries.find(&db) == g_schemaRegistries.end()) {
        // Create schema registry and build from catalog
        auto schemaRegistry = std::make_shared<SchemaRegistry>();
        schemaRegistry->buildFromKuzuCatalog(context);
        schemaRegistry->addAGITypes();
        schemaRegistry->addLanceDBTypes();
        schemaRegistry->addLadybugTypes();
        g_schemaRegistries[&db] = schemaRegistry;

        // Create LanceDB connector
        LanceDBConfig lanceConfig;
        lanceConfig.uri = "lance://memory";  // Default to in-memory
        auto lanceConnector = std::make_shared<LanceDBConnector>(lanceConfig);
        lanceConnector->connect();
        g_lanceConnectors[&db] = lanceConnector;

        // Create vector store
        auto vectorStore = std::make_shared<VectorStore>(lanceConnector);

        // Create knowledge graph
        auto knowledgeGraph = std::make_shared<KnowledgeGraph>(context);

        // Create reasoning engine
        ReasoningConfig reasoningConfig;
        auto reasoningEngine = std::make_shared<ReasoningEngine>(
            vectorStore, knowledgeGraph, reasoningConfig);
        reasoningEngine->registerGraphQueryTool(context);
        reasoningEngine->registerVectorSearchTool();
        g_reasoningEngines[&db] = reasoningEngine;

        // Create Ladybug debugger
        LadybugConfig ladybugConfig;
        auto debugger = std::make_shared<LadybugDebugger>(ladybugConfig);
        g_debuggers[&db] = debugger;
    }

    // Register all functions
    extension::ExtensionUtils::registerFunction(db, GraphQLQueryFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, SemanticSearchFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, AGIReasonFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, AGIPlanFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, CreateEmbeddingFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, VectorStoreFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, LadybugAnalyzeFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, GetQueryPlanFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, MultiHopReasonFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, HybridSearchFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, GraphQLSchemaFunction::getFunctionSet());
    extension::ExtensionUtils::registerFunction(db, KGSubgraphFunction::getFunctionSet());
}

void GraphQLAGIExtension::initializeGraphQLSchema(main::ClientContext* context) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& db = *context->getDatabase();

    if (g_schemaRegistries.find(&db) != g_schemaRegistries.end()) {
        g_schemaRegistries[&db]->buildFromKuzuCatalog(context);
    }
}

void GraphQLAGIExtension::initializeLanceDB(main::ClientContext* context,
                                             const std::string& uri) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& db = *context->getDatabase();

    LanceDBConfig config;
    config.uri = uri;

    auto connector = std::make_shared<LanceDBConnector>(config);
    connector->connect();
    g_lanceConnectors[&db] = connector;
}

void GraphQLAGIExtension::initializeAGI(main::ClientContext* context) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& db = *context->getDatabase();

    if (g_reasoningEngines.find(&db) != g_reasoningEngines.end()) {
        g_reasoningEngines[&db]->registerGraphQueryTool(context);
    }
}

void GraphQLAGIExtension::initializeLadybug(main::ClientContext* context) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& db = *context->getDatabase();

    LadybugConfig config;
    config.enableTracing = true;
    config.enableMetrics = true;
    config.enableVisualization = true;

    g_debuggers[&db] = std::make_shared<LadybugDebugger>(config);
}

// ============================================================================
// Function Implementations
// ============================================================================

std::string GraphQLQueryFunction::executeGraphQL(main::ClientContext* context,
                                                  const std::string& query,
                                                  const std::string& variables) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& db = *context->getDatabase();

    auto schemaIt = g_schemaRegistries.find(&db);
    if (schemaIt == g_schemaRegistries.end()) {
        return R"({"errors": [{"message": "GraphQL AGI not initialized"}]})";
    }

    // Create translator
    TranslationOptions options;
    options.enableOptimizations = true;
    QueryTranslator translator(*schemaIt->second, options);

    // Parse variables
    std::unordered_map<std::string, std::string> vars;
    // TODO: Parse JSON variables

    // Translate and execute
    auto result = translator.translateQuery(query, vars);

    if (!result.success) {
        std::string errors = R"({"errors": [)";
        for (size_t i = 0; i < result.errors.size(); ++i) {
            if (i > 0) errors += ",";
            errors += R"({"message": ")" + result.errors[i] + R"("})";
        }
        errors += "]}";
        return errors;
    }

    // Execute Cypher queries
    std::string resultJson = R"({"data": {})";

    for (const auto& cypherQuery : result.queries) {
        try {
            auto queryResult = context->query(cypherQuery.query);
            // TODO: Convert result to JSON
        } catch (const std::exception& e) {
            return R"({"errors": [{"message": ")" + std::string(e.what()) + R"("}]})";
        }
    }

    return resultJson;
}

} // namespace graphql_agi
} // namespace kuzu

// ============================================================================
// C API Exports
// ============================================================================

extern "C" {

KUZU_API void graphql_agi_init(kuzu::main::ClientContext* context) {
    kuzu::graphql_agi::GraphQLAGIExtension::load(context);
}

KUZU_API const char* graphql_agi_name() {
    return kuzu::graphql_agi::GraphQLAGIExtension::EXTENSION_NAME;
}

KUZU_API const char* graphql_agi_version() {
    return "1.0.0";
}

}
