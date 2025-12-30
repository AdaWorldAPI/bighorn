#pragma once

/**
 * AGI Reasoning Engine
 *
 * Implements advanced reasoning capabilities for intelligent query processing:
 * - Chain-of-Thought (CoT) reasoning
 * - Tree of Thoughts (ToT) exploration
 * - Graph-based reasoning over knowledge graphs
 * - Multi-hop inference
 * - Semantic understanding and query expansion
 */

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_map>
#include <queue>

#include "lancedb/lancedb_connector.h"

namespace kuzu {
namespace graphql_agi {

// Forward declarations
class KnowledgeGraph;
class PlanningSystem;
class ToolExecutor;

/**
 * Represents a reasoning step in the chain
 */
struct ReasoningStep {
    uint32_t stepNumber;
    std::string thought;
    std::string action;
    std::optional<std::string> observation;
    double confidence;
    std::chrono::milliseconds duration;

    // Traceability
    std::vector<std::string> evidenceIds;
    std::vector<std::string> sourceQueries;
};

/**
 * Represents a complete reasoning trace
 */
struct ReasoningTrace {
    std::string traceId;
    std::string originalQuery;
    std::vector<ReasoningStep> steps;
    std::string finalAnswer;
    double overallConfidence;
    std::chrono::milliseconds totalDuration;

    // Metrics
    uint32_t graphHops;
    uint32_t vectorSearches;
    uint32_t llmCalls;
};

/**
 * Reasoning strategy options
 */
enum class ReasoningStrategy {
    ChainOfThought,    // Sequential step-by-step reasoning
    TreeOfThoughts,    // Branching exploration with backtracking
    GraphOfThoughts,   // Graph-structured reasoning
    ReAct,             // Reasoning + Acting paradigm
    SelfConsistency,   // Multiple chains with voting
    Reflexion,         // Self-reflection and correction
    PlanAndExecute     // High-level planning then execution
};

/**
 * Configuration for reasoning
 */
struct ReasoningConfig {
    ReasoningStrategy strategy = ReasoningStrategy::ChainOfThought;
    uint32_t maxSteps = 10;
    uint32_t maxBranches = 5;        // For ToT
    double confidenceThreshold = 0.7;
    double temperatureDecay = 0.1;   // Reduce randomness as reasoning progresses

    // Knowledge graph settings
    uint32_t maxGraphHops = 3;
    bool useSemanticExpansion = true;
    uint32_t maxRelatedEntities = 20;

    // LLM settings
    std::string modelName = "claude-3-opus";
    uint32_t maxTokens = 4096;
    double temperature = 0.3;

    // Caching
    bool enableReasoningCache = true;
    uint32_t cacheMaxSize = 1000;
};

/**
 * Represents a thought node in Tree of Thoughts
 */
struct ThoughtNode {
    std::string id;
    std::string thought;
    double score;
    std::vector<std::shared_ptr<ThoughtNode>> children;
    std::weak_ptr<ThoughtNode> parent;

    bool isTerminal = false;
    std::optional<std::string> conclusion;
};

/**
 * AGI Reasoning Engine
 */
class ReasoningEngine {
public:
    ReasoningEngine(std::shared_ptr<VectorStore> vectorStore,
                    std::shared_ptr<KnowledgeGraph> knowledgeGraph,
                    const ReasoningConfig& config = {});
    ~ReasoningEngine() = default;

    /**
     * Execute reasoning for a query
     */
    struct ReasoningResult {
        bool success;
        std::string answer;
        double confidence;
        ReasoningTrace trace;
        std::vector<std::string> relatedEntities;
        std::vector<std::string> usedSources;
        std::optional<std::string> error;
    };

    ReasoningResult reason(const std::string& query,
                            const std::unordered_map<std::string, std::string>& context = {});

    /**
     * Execute Chain-of-Thought reasoning
     */
    ReasoningResult chainOfThought(const std::string& query);

    /**
     * Execute Tree of Thoughts reasoning
     */
    ReasoningResult treeOfThoughts(const std::string& query,
                                     uint32_t numBranches = 3,
                                     uint32_t maxDepth = 5);

    /**
     * Execute ReAct (Reasoning + Acting)
     */
    ReasoningResult react(const std::string& query,
                           const std::vector<std::function<std::string(std::string)>>& actions);

    /**
     * Semantic query expansion
     * Enriches a query with related concepts and entities
     */
    struct ExpandedQuery {
        std::string originalQuery;
        std::string expandedQuery;
        std::vector<std::string> relatedConcepts;
        std::vector<std::string> synonyms;
        std::vector<std::string> relatedEntities;
    };

    ExpandedQuery expandQuery(const std::string& query);

    /**
     * Multi-hop reasoning over knowledge graph
     */
    struct MultiHopResult {
        std::vector<std::string> path;
        std::string answer;
        double confidence;
        std::vector<ReasoningStep> steps;
    };

    MultiHopResult multiHopReason(const std::string& startEntity,
                                    const std::string& question,
                                    uint32_t maxHops = 3);

    /**
     * Answer validation
     * Checks if an answer is consistent and well-grounded
     */
    struct ValidationResult {
        bool valid;
        double groundedness;
        double consistency;
        std::vector<std::string> supportingEvidence;
        std::vector<std::string> contradictions;
    };

    ValidationResult validateAnswer(const std::string& answer,
                                      const std::vector<std::string>& context);

    /**
     * Self-reflection and correction
     */
    std::string reflect(const std::string& originalAnswer,
                          const std::vector<std::string>& feedback);

    /**
     * Get or set configuration
     */
    ReasoningConfig getConfig() const { return config_; }
    void setConfig(const ReasoningConfig& config) { config_ = config; }

    /**
     * Tool registration for ReAct
     */
    void registerTool(const std::string& name,
                       std::function<std::string(const std::string&)> toolFunc);

    void registerGraphQueryTool(main::ClientContext* context);
    void registerVectorSearchTool();
    void registerWebSearchTool();

private:
    std::shared_ptr<VectorStore> vectorStore_;
    std::shared_ptr<KnowledgeGraph> knowledgeGraph_;
    ReasoningConfig config_;

    std::unordered_map<std::string, std::function<std::string(const std::string&)>> tools_;

    // Internal reasoning methods
    ReasoningStep executeStep(const std::string& thought, const std::string& action);
    std::vector<std::string> generateThoughts(const std::string& context, uint32_t count);
    double evaluateThought(const std::string& thought, const std::string& goal);
    std::string selectAction(const std::string& thought);
    std::string executeAction(const std::string& action);

    // Tree search methods
    void expandNode(std::shared_ptr<ThoughtNode> node);
    std::shared_ptr<ThoughtNode> selectBestLeaf(std::shared_ptr<ThoughtNode> root);
    double backpropagate(std::shared_ptr<ThoughtNode> node, double value);

    // Knowledge integration
    std::vector<std::string> retrieveRelevantContext(const std::string& query);
    std::string integrateKnowledge(const std::vector<std::string>& facts);
};

/**
 * Knowledge Graph interface for reasoning
 */
class KnowledgeGraph {
public:
    KnowledgeGraph(main::ClientContext* context);
    ~KnowledgeGraph() = default;

    /**
     * Entity operations
     */
    struct Entity {
        std::string id;
        std::string type;
        std::unordered_map<std::string, std::string> properties;
        std::optional<Vector> embedding;
    };

    std::optional<Entity> getEntity(const std::string& id);
    std::vector<Entity> searchEntities(const std::string& query, uint32_t limit = 10);
    std::vector<Entity> getSimilarEntities(const std::string& id, uint32_t limit = 10);

    /**
     * Relationship operations
     */
    struct Relationship {
        std::string id;
        std::string type;
        std::string sourceId;
        std::string targetId;
        std::unordered_map<std::string, std::string> properties;
    };

    std::vector<Relationship> getRelationships(const std::string& entityId,
                                                  const std::optional<std::string>& relationType = std::nullopt);

    std::vector<Entity> getNeighbors(const std::string& entityId,
                                        uint32_t hops = 1,
                                        const std::optional<std::vector<std::string>>& relTypes = std::nullopt);

    /**
     * Path finding
     */
    struct Path {
        std::vector<Entity> entities;
        std::vector<Relationship> relationships;
        double score;
    };

    std::vector<Path> findPaths(const std::string& startId,
                                  const std::string& endId,
                                  uint32_t maxLength = 4,
                                  uint32_t maxPaths = 10);

    /**
     * Subgraph extraction
     */
    struct Subgraph {
        std::vector<Entity> entities;
        std::vector<Relationship> relationships;
    };

    Subgraph extractSubgraph(const std::string& entityId, uint32_t radius = 2);

    /**
     * Graph queries
     */
    std::string executeCypher(const std::string& query);

    /**
     * Semantic operations
     */
    std::vector<Entity> semanticSearch(const Vector& queryVector, uint32_t topK = 10);
    std::vector<Entity> findConceptuallyRelated(const std::string& concept);

private:
    main::ClientContext* context_;

    std::string entityToCypher(const Entity& entity);
    Entity parseEntityFromResult(const std::string& result);
};

/**
 * Autonomous Agent for complex task execution
 */
class AutonomousAgent {
public:
    AutonomousAgent(std::shared_ptr<ReasoningEngine> reasoningEngine,
                    std::shared_ptr<PlanningSystem> planningSystem);
    ~AutonomousAgent() = default;

    /**
     * Agent state
     */
    enum class AgentState {
        Idle,
        Planning,
        Executing,
        Reasoning,
        Reflecting,
        Completed,
        Failed
    };

    /**
     * Execute a high-level task
     */
    struct TaskResult {
        bool success;
        std::string result;
        std::vector<std::string> steps;
        AgentState finalState;
        std::chrono::milliseconds duration;
    };

    TaskResult executeTask(const std::string& task,
                            const std::unordered_map<std::string, std::string>& context = {});

    /**
     * Get current state
     */
    AgentState getState() const { return state_; }

    /**
     * Cancel current task
     */
    void cancel();

    /**
     * Set callbacks for monitoring
     */
    using StateCallback = std::function<void(AgentState, const std::string&)>;
    void setStateCallback(StateCallback callback);

private:
    std::shared_ptr<ReasoningEngine> reasoningEngine_;
    std::shared_ptr<PlanningSystem> planningSystem_;
    AgentState state_ = AgentState::Idle;
    StateCallback stateCallback_;

    void setState(AgentState state, const std::string& message = "");
};

} // namespace graphql_agi
} // namespace kuzu
