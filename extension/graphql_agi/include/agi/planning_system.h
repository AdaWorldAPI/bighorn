#pragma once

/**
 * AGI Planning System
 *
 * Implements hierarchical task planning and execution:
 * - Goal decomposition
 * - Action planning
 * - Constraint satisfaction
 * - Plan optimization
 * - Dynamic replanning
 */

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_map>
#include <variant>
#include <chrono>

namespace kuzu {
namespace graphql_agi {

/**
 * Represents a condition (precondition or effect)
 */
struct Condition {
    std::string predicate;
    std::vector<std::string> arguments;
    bool negated = false;

    bool matches(const Condition& other) const;
    std::string toString() const;
};

/**
 * Represents an action that can be taken
 */
struct Action {
    std::string name;
    std::vector<std::string> parameters;
    std::vector<Condition> preconditions;
    std::vector<Condition> effects;
    double cost = 1.0;
    std::optional<double> duration;

    // For tool actions
    std::optional<std::function<std::string(std::vector<std::string>)>> executor;
};

/**
 * Represents a state in the planning world
 */
class WorldState {
public:
    WorldState() = default;

    void addFact(const Condition& fact);
    void removeFact(const Condition& fact);
    bool hasFact(const Condition& fact) const;
    bool satisfies(const std::vector<Condition>& conditions) const;

    WorldState apply(const Action& action) const;
    std::vector<Condition> getAllFacts() const;
    std::string toString() const;

private:
    std::vector<Condition> facts_;
};

/**
 * Represents a step in a plan
 */
struct PlanStep {
    uint32_t stepNumber;
    std::string actionName;
    std::vector<std::string> boundArguments;
    std::optional<std::string> description;
    double estimatedCost;

    // Dependencies
    std::vector<uint32_t> dependsOn;

    // Execution state
    enum class Status {
        Pending,
        InProgress,
        Completed,
        Failed,
        Skipped
    };
    Status status = Status::Pending;
    std::optional<std::string> result;
    std::optional<std::string> error;
};

/**
 * Represents a complete plan
 */
struct Plan {
    std::string planId;
    std::string goal;
    std::vector<PlanStep> steps;
    double totalCost;
    std::optional<double> estimatedDuration;

    // Plan quality metrics
    double optimalityScore;
    bool isParallel = false;
    std::vector<std::vector<uint32_t>> parallelGroups;

    // Validation
    bool isValid = true;
    std::vector<std::string> validationErrors;
};

/**
 * Planning algorithm options
 */
enum class PlanningAlgorithm {
    ForwardStateSpace,    // A* search in state space
    BackwardChaining,     // Goal regression
    GraphPlan,            // Planning graph approach
    HierarchicalTaskNetwork, // HTN planning
    MonteCarlo,           // MCTS for planning
    LLMGuided             // LLM-assisted planning
};

/**
 * Planning configuration
 */
struct PlanningConfig {
    PlanningAlgorithm algorithm = PlanningAlgorithm::ForwardStateSpace;
    uint32_t maxPlanLength = 50;
    uint32_t maxSearchNodes = 10000;
    double timeoutSeconds = 30.0;
    bool allowParallelActions = true;
    bool optimizeForCost = true;
    bool optimizeForDuration = false;

    // LLM-guided options
    bool useLLMHeuristics = false;
    bool useLLMActionGeneration = false;
    std::string llmModel = "claude-3-opus";
};

/**
 * Result of a planning operation
 */
struct PlanningResult {
    bool success;
    std::optional<Plan> plan;
    std::chrono::milliseconds planningTime;
    uint32_t nodesExplored;
    std::vector<std::string> failureReasons;

    // Alternative plans (if found)
    std::vector<Plan> alternativePlans;
};

/**
 * AGI Planning System
 */
class PlanningSystem {
public:
    PlanningSystem(const PlanningConfig& config = {});
    ~PlanningSystem() = default;

    /**
     * Create a plan to achieve a goal
     */
    PlanningResult createPlan(const WorldState& initialState,
                               const std::vector<Condition>& goals);

    /**
     * Create plan from natural language goal
     */
    PlanningResult createPlanFromNL(const std::string& goal,
                                      const WorldState& initialState);

    /**
     * Register an action
     */
    void registerAction(const Action& action);

    /**
     * Register actions from a domain
     */
    void loadDomain(const std::string& domainDefinition);

    /**
     * Execute a plan
     */
    struct ExecutionResult {
        bool success;
        std::vector<std::string> stepResults;
        std::optional<std::string> error;
        WorldState finalState;
    };

    ExecutionResult executePlan(Plan& plan, WorldState& state);

    /**
     * Re-plan when execution fails
     */
    PlanningResult replan(const Plan& failedPlan,
                           uint32_t failedStep,
                           const WorldState& currentState,
                           const std::vector<Condition>& goals);

    /**
     * Validate a plan
     */
    struct ValidationResult {
        bool valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        std::vector<uint32_t> problematicSteps;
    };

    ValidationResult validatePlan(const Plan& plan, const WorldState& initialState);

    /**
     * Optimize a plan
     */
    Plan optimizePlan(const Plan& plan);

    /**
     * Parallelize plan steps where possible
     */
    Plan parallelizePlan(const Plan& plan);

    /**
     * Get all registered actions
     */
    std::vector<Action> getAvailableActions() const;

    /**
     * Natural language to action translation
     */
    std::optional<Action> parseNLAction(const std::string& nlAction);

    /**
     * Configuration
     */
    PlanningConfig getConfig() const { return config_; }
    void setConfig(const PlanningConfig& config) { config_ = config; }

    /**
     * Built-in actions for graph/vector operations
     */
    void registerGraphActions(main::ClientContext* context);
    void registerVectorActions(std::shared_ptr<VectorStore> vectorStore);
    void registerReasoningActions(std::shared_ptr<ReasoningEngine> engine);

private:
    PlanningConfig config_;
    std::unordered_map<std::string, Action> actions_;

    // Planning algorithms
    PlanningResult forwardStateSpace(const WorldState& initial,
                                       const std::vector<Condition>& goals);
    PlanningResult backwardChaining(const WorldState& initial,
                                      const std::vector<Condition>& goals);
    PlanningResult graphPlan(const WorldState& initial,
                              const std::vector<Condition>& goals);
    PlanningResult hierarchicalPlan(const WorldState& initial,
                                      const std::vector<Condition>& goals);
    PlanningResult monteCarloTreeSearch(const WorldState& initial,
                                          const std::vector<Condition>& goals);
    PlanningResult llmGuidedPlan(const WorldState& initial,
                                   const std::vector<Condition>& goals);

    // Heuristics
    double estimateCostToGoal(const WorldState& state,
                               const std::vector<Condition>& goals);
    double relaxedPlanHeuristic(const WorldState& state,
                                  const std::vector<Condition>& goals);

    // Action instantiation
    std::vector<Action> getApplicableActions(const WorldState& state);
    std::vector<std::vector<std::string>> findBindings(const Action& action,
                                                          const WorldState& state);
};

/**
 * Hierarchical Task Network (HTN) planning support
 */
class HTNPlanner {
public:
    /**
     * Compound task (can be decomposed)
     */
    struct CompoundTask {
        std::string name;
        std::vector<std::string> parameters;

        // Methods to decompose this task
        struct Method {
            std::string name;
            std::vector<Condition> preconditions;
            std::vector<std::variant<std::string, CompoundTask>> subtasks;
        };
        std::vector<Method> methods;
    };

    HTNPlanner();

    void registerCompoundTask(const CompoundTask& task);
    void registerPrimitiveAction(const Action& action);

    PlanningResult plan(const std::string& taskName,
                         const std::vector<std::string>& arguments,
                         const WorldState& initialState);

private:
    std::unordered_map<std::string, CompoundTask> compoundTasks_;
    std::unordered_map<std::string, Action> primitiveActions_;

    std::optional<std::vector<PlanStep>> decomposeTask(
        const std::string& taskName,
        const std::vector<std::string>& arguments,
        const WorldState& state);
};

/**
 * Tool Executor for executing plan actions
 */
class ToolExecutor {
public:
    ToolExecutor(main::ClientContext* context);

    /**
     * Register a tool
     */
    using ToolFunction = std::function<std::string(const std::vector<std::string>&)>;
    void registerTool(const std::string& name,
                       ToolFunction func,
                       const std::vector<std::string>& parameterNames);

    /**
     * Execute a tool
     */
    struct ToolResult {
        bool success;
        std::string output;
        std::optional<std::string> error;
        std::chrono::milliseconds duration;
    };

    ToolResult executeTool(const std::string& toolName,
                            const std::vector<std::string>& arguments);

    /**
     * Get tool definitions
     */
    std::vector<std::string> getAvailableTools() const;
    std::optional<std::vector<std::string>> getToolParameters(const std::string& toolName) const;

    /**
     * Built-in tools
     */
    void registerBuiltInTools();
    void registerCypherQueryTool();
    void registerVectorSearchTool(std::shared_ptr<VectorStore> vectorStore);
    void registerWebFetchTool();
    void registerFileOperationTools();

private:
    main::ClientContext* context_;

    struct ToolDefinition {
        std::string name;
        ToolFunction function;
        std::vector<std::string> parameterNames;
    };

    std::unordered_map<std::string, ToolDefinition> tools_;
};

} // namespace graphql_agi
} // namespace kuzu
