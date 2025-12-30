#pragma once

/**
 * GraphQL AGI Extension for Kuzu Database
 *
 * A comprehensive integration combining:
 * - GraphQL API layer for intuitive query interface
 * - LanceDB vector storage for AI-native embeddings
 * - AGI reasoning engine for autonomous intelligence
 * - Ladybug debugging and visualization system
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    GraphQL AGI Gateway                       │
 * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
 * │  │   GraphQL   │  │   LanceDB   │  │  Ladybug Debugger   │ │
 * │  │   Schema    │  │   Vector    │  │  Query Analyzer     │ │
 * │  │   Layer     │  │   Store     │  │  Performance Mon    │ │
 * │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
 * │         │                │                     │            │
 * │  ┌──────┴────────────────┴─────────────────────┴──────────┐ │
 * │  │              AGI Reasoning Engine                       │ │
 * │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │ │
 * │  │  │ Planning │  │Knowledge │  │  Autonomous Agent    │  │ │
 * │  │  │  System  │  │  Graph   │  │  Tool Executor       │  │ │
 * │  │  └──────────┘  └──────────┘  └──────────────────────┘  │ │
 * │  └────────────────────────────────────────────────────────┘ │
 * │                            │                                │
 * │  ┌────────────────────────┴────────────────────────────┐   │
 * │  │              Kuzu Graph Database Core                │   │
 * │  │    Cypher Engine • HNSW Indices • Storage Layer      │   │
 * │  └──────────────────────────────────────────────────────┘   │
 * └─────────────────────────────────────────────────────────────┘
 *
 * @author GraphQL AGI Team
 * @version 1.0.0
 */

#include "extension/extension.h"
#include "main/client_context.h"
#include "main/database.h"

namespace kuzu {
namespace graphql_agi {

// Forward declarations
class GraphQLSchema;
class LanceDBConnector;
class AGIReasoningEngine;
class LadybugDebugger;

/**
 * Main extension class for GraphQL AGI
 */
class GraphQLAGIExtension : public extension::Extension {
public:
    static constexpr const char* EXTENSION_NAME = "graphql_agi";

    /**
     * Load the extension into the database context
     * Registers all GraphQL, LanceDB, AGI, and Ladybug functions
     */
    static void load(main::ClientContext* context);

    /**
     * Get extension name
     */
    static const char* name() { return EXTENSION_NAME; }

    /**
     * Initialize GraphQL schema from Cypher schema
     */
    static void initializeGraphQLSchema(main::ClientContext* context);

    /**
     * Initialize LanceDB vector store connection
     */
    static void initializeLanceDB(main::ClientContext* context,
                                   const std::string& uri);

    /**
     * Initialize AGI reasoning capabilities
     */
    static void initializeAGI(main::ClientContext* context);

    /**
     * Initialize Ladybug debugging system
     */
    static void initializeLadybug(main::ClientContext* context);
};

// Extension configuration options
struct GraphQLAGIConfig {
    // GraphQL settings
    bool enableIntrospection = true;
    bool enableBatchQueries = true;
    uint32_t maxQueryDepth = 15;
    uint32_t maxQueryComplexity = 1000;

    // LanceDB settings
    std::string lanceDBUri = "lance://memory";
    uint32_t vectorDimension = 1536;
    std::string distanceMetric = "cosine";
    uint32_t hnswM = 16;
    uint32_t hnswEfConstruction = 200;

    // AGI settings
    bool enableAutonomousMode = false;
    uint32_t maxReasoningDepth = 10;
    uint32_t maxPlanningSteps = 50;
    double confidenceThreshold = 0.75;
    std::string defaultEmbeddingModel = "text-embedding-3-large";

    // Ladybug settings
    bool enableQueryTracing = true;
    bool enablePerformanceMetrics = true;
    bool enableVisualization = true;
    std::string tracingLevel = "detailed";
};

} // namespace graphql_agi
} // namespace kuzu

// C API exports for extension loading
extern "C" {
    KUZU_API void graphql_agi_init(kuzu::main::ClientContext* context);
    KUZU_API const char* graphql_agi_name();
    KUZU_API const char* graphql_agi_version();
}
