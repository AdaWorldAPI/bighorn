#pragma once

/**
 * Ladybug Query Analyzer & Debugger
 *
 * Comprehensive debugging and visualization system for GraphQL AGI:
 * - Query performance analysis
 * - Execution tracing
 * - Real-time monitoring
 * - Visual query plan exploration
 * - AGI reasoning visualization
 */

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace kuzu {
namespace graphql_agi {

/**
 * Trace severity levels
 */
enum class TraceSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical
};

/**
 * Represents a single trace event
 */
struct TraceEvent {
    std::string eventId;
    std::chrono::system_clock::time_point timestamp;
    TraceSeverity severity;
    std::string category;
    std::string message;
    std::unordered_map<std::string, std::string> metadata;

    // Timing
    std::chrono::microseconds duration;
    std::optional<std::string> parentEventId;

    // Source location
    std::optional<std::string> sourceFile;
    std::optional<uint32_t> sourceLine;
    std::optional<std::string> function;
};

/**
 * Query execution span for distributed tracing
 */
struct ExecutionSpan {
    std::string spanId;
    std::string traceId;
    std::optional<std::string> parentSpanId;
    std::string operationName;

    std::chrono::system_clock::time_point startTime;
    std::chrono::system_clock::time_point endTime;

    enum class Status { Ok, Error, Timeout };
    Status status = Status::Ok;

    std::vector<TraceEvent> events;
    std::unordered_map<std::string, std::string> tags;
    std::vector<std::pair<std::string, std::string>> logs;
};

/**
 * Performance metrics for a query
 */
struct QueryMetrics {
    std::string queryId;
    std::string queryHash;

    // Timing metrics
    std::chrono::microseconds parseTime;
    std::chrono::microseconds planTime;
    std::chrono::microseconds executionTime;
    std::chrono::microseconds totalTime;

    // Resource metrics
    uint64_t memoryUsedBytes;
    uint64_t peakMemoryBytes;
    uint32_t threadsUsed;

    // I/O metrics
    uint64_t rowsScanned;
    uint64_t rowsReturned;
    uint64_t bytesRead;
    uint64_t bytesWritten;

    // Cache metrics
    uint32_t cacheHits;
    uint32_t cacheMisses;

    // Vector search metrics (for AGI queries)
    uint32_t vectorSearches;
    uint64_t vectorsCompared;
    std::chrono::microseconds vectorSearchTime;

    // LLM metrics
    uint32_t llmCalls;
    uint32_t llmTokensUsed;
    std::chrono::microseconds llmLatency;

    // Reasoning metrics
    uint32_t reasoningSteps;
    uint32_t graphHops;
    double reasoningConfidence;
};

/**
 * Query plan node for visualization
 */
struct QueryPlanNode {
    std::string nodeId;
    std::string operatorType;
    std::string description;

    // Estimated costs
    double estimatedRows;
    double estimatedCost;

    // Actual metrics (after execution)
    std::optional<uint64_t> actualRows;
    std::optional<std::chrono::microseconds> actualTime;

    // Children
    std::vector<std::shared_ptr<QueryPlanNode>> children;

    // Properties
    std::unordered_map<std::string, std::string> properties;

    // Warnings
    std::vector<std::string> warnings;
};

/**
 * Configuration for Ladybug debugger
 */
struct LadybugConfig {
    // Tracing settings
    bool enableTracing = true;
    TraceSeverity minSeverity = TraceSeverity::Info;
    uint32_t maxTraceEvents = 10000;
    bool enableDistributedTracing = false;

    // Performance monitoring
    bool enableMetrics = true;
    bool enableSlowQueryLog = true;
    std::chrono::milliseconds slowQueryThreshold{100};

    // Query plan analysis
    bool captureQueryPlans = true;
    bool explainAnalyze = false;  // Include actual execution stats

    // Visualization
    bool enableVisualization = true;
    std::string visualizationFormat = "json";  // "json", "dot", "svg"

    // Output
    std::optional<std::string> traceOutputFile;
    std::optional<std::string> metricsOutputFile;
    bool enableConsoleOutput = false;

    // Sampling
    double samplingRate = 1.0;  // 0.0 to 1.0
    bool alwaysSampleErrors = true;
};

/**
 * Ladybug Query Analyzer
 */
class QueryAnalyzer {
public:
    QueryAnalyzer(const LadybugConfig& config = {});
    ~QueryAnalyzer() = default;

    /**
     * Analyze a GraphQL query before execution
     */
    struct AnalysisResult {
        bool valid;
        double estimatedComplexity;
        double estimatedCost;
        std::vector<std::string> warnings;
        std::vector<std::string> optimizationSuggestions;
        std::shared_ptr<QueryPlanNode> queryPlan;
    };

    AnalysisResult analyzeQuery(const std::string& graphqlQuery);

    /**
     * Analyze a Cypher query
     */
    AnalysisResult analyzeCypherQuery(const std::string& cypherQuery,
                                       main::ClientContext* context);

    /**
     * Get query plan
     */
    std::shared_ptr<QueryPlanNode> getQueryPlan(const std::string& query,
                                                  main::ClientContext* context);

    /**
     * Explain query (text format)
     */
    std::string explainQuery(const std::string& query,
                              main::ClientContext* context,
                              bool analyze = false);

    /**
     * Detect anti-patterns
     */
    struct AntiPattern {
        std::string name;
        std::string description;
        std::string location;
        std::string suggestion;
        TraceSeverity severity;
    };

    std::vector<AntiPattern> detectAntiPatterns(const std::string& query);

    /**
     * Suggest indices
     */
    struct IndexSuggestion {
        std::string tableName;
        std::string columnName;
        std::string indexType;
        std::string rationale;
        double estimatedImpact;
    };

    std::vector<IndexSuggestion> suggestIndices(const std::string& query,
                                                   main::ClientContext* context);

private:
    LadybugConfig config_;

    double calculateComplexity(const std::string& query);
    std::vector<std::string> findOptimizations(const std::string& query);
};

/**
 * Performance Monitor for real-time metrics
 */
class PerformanceMonitor {
public:
    PerformanceMonitor(const LadybugConfig& config = {});
    ~PerformanceMonitor() = default;

    /**
     * Start monitoring a query
     */
    std::string startQuery(const std::string& queryId, const std::string& query);

    /**
     * Record a metric
     */
    void recordMetric(const std::string& queryId,
                       const std::string& metricName,
                       double value);

    /**
     * End query monitoring
     */
    QueryMetrics endQuery(const std::string& queryId);

    /**
     * Get aggregated statistics
     */
    struct AggregateStats {
        uint64_t totalQueries;
        uint64_t successfulQueries;
        uint64_t failedQueries;

        double avgQueryTime;
        double p50QueryTime;
        double p95QueryTime;
        double p99QueryTime;

        double avgMemoryUsage;
        double peakMemoryUsage;

        double cacheHitRate;

        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point endTime;
    };

    AggregateStats getAggregateStats() const;
    AggregateStats getAggregateStats(std::chrono::system_clock::time_point start,
                                       std::chrono::system_clock::time_point end) const;

    /**
     * Get slow queries
     */
    std::vector<QueryMetrics> getSlowQueries(uint32_t limit = 10) const;

    /**
     * Get queries by pattern
     */
    std::vector<QueryMetrics> getQueriesByPattern(const std::string& pattern) const;

    /**
     * Export metrics
     */
    std::string exportMetrics(const std::string& format = "json") const;
    void exportToPrometheus(const std::string& endpoint);

    /**
     * Reset statistics
     */
    void reset();

private:
    LadybugConfig config_;
    mutable std::mutex mutex_;

    std::unordered_map<std::string, QueryMetrics> activeQueries_;
    std::vector<QueryMetrics> completedQueries_;

    std::chrono::system_clock::time_point startTime_;
};

/**
 * Debug Tracer for detailed execution tracing
 */
class DebugTracer {
public:
    DebugTracer(const LadybugConfig& config = {});
    ~DebugTracer() = default;

    /**
     * Start a new trace
     */
    std::string startTrace(const std::string& operationName);

    /**
     * Start a span within a trace
     */
    std::string startSpan(const std::string& traceId,
                           const std::string& operationName,
                           const std::optional<std::string>& parentSpanId = std::nullopt);

    /**
     * End a span
     */
    void endSpan(const std::string& spanId,
                  ExecutionSpan::Status status = ExecutionSpan::Status::Ok);

    /**
     * Log an event
     */
    void logEvent(const std::string& spanId,
                   const std::string& message,
                   TraceSeverity severity = TraceSeverity::Info,
                   const std::unordered_map<std::string, std::string>& metadata = {});

    /**
     * Add tag to span
     */
    void addTag(const std::string& spanId,
                 const std::string& key,
                 const std::string& value);

    /**
     * Get trace
     */
    struct Trace {
        std::string traceId;
        std::vector<ExecutionSpan> spans;
        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point endTime;
        std::chrono::microseconds totalDuration;
    };

    std::optional<Trace> getTrace(const std::string& traceId) const;

    /**
     * Export trace
     */
    std::string exportTrace(const std::string& traceId,
                             const std::string& format = "json") const;

    /**
     * Export to OpenTelemetry/Jaeger format
     */
    std::string exportToJaeger(const std::string& traceId) const;
    std::string exportToZipkin(const std::string& traceId) const;

private:
    LadybugConfig config_;
    mutable std::mutex mutex_;

    std::unordered_map<std::string, Trace> traces_;
    std::unordered_map<std::string, ExecutionSpan> activeSpans_;

    std::string generateId();
};

/**
 * Visualization Engine for query plans and traces
 */
class VisualizationEngine {
public:
    VisualizationEngine();
    ~VisualizationEngine() = default;

    /**
     * Render query plan as DOT graph
     */
    std::string renderQueryPlanDot(const std::shared_ptr<QueryPlanNode>& plan);

    /**
     * Render query plan as SVG
     */
    std::string renderQueryPlanSvg(const std::shared_ptr<QueryPlanNode>& plan);

    /**
     * Render query plan as JSON (for web viewers)
     */
    std::string renderQueryPlanJson(const std::shared_ptr<QueryPlanNode>& plan);

    /**
     * Render trace as timeline
     */
    std::string renderTraceTimeline(const DebugTracer::Trace& trace);

    /**
     * Render reasoning trace (for AGI debugging)
     */
    std::string renderReasoningTrace(const ReasoningTrace& trace);

    /**
     * Render knowledge graph subgraph
     */
    std::string renderKnowledgeGraph(const KnowledgeGraph::Subgraph& subgraph);

    /**
     * Generate interactive HTML report
     */
    std::string generateHtmlReport(const QueryMetrics& metrics,
                                     const std::shared_ptr<QueryPlanNode>& plan,
                                     const std::optional<DebugTracer::Trace>& trace);

private:
    std::string escapeHtml(const std::string& str);
    std::string escapeDot(const std::string& str);
    std::string colorForSeverity(TraceSeverity severity);
    std::string colorForDuration(std::chrono::microseconds duration);
};

/**
 * Main Ladybug Debugger class
 */
class LadybugDebugger {
public:
    LadybugDebugger(const LadybugConfig& config = {});
    ~LadybugDebugger() = default;

    /**
     * Get component instances
     */
    QueryAnalyzer& getAnalyzer() { return analyzer_; }
    PerformanceMonitor& getMonitor() { return monitor_; }
    DebugTracer& getTracer() { return tracer_; }
    VisualizationEngine& getVisualizer() { return visualizer_; }

    /**
     * Convenience method: Full query analysis and monitoring
     */
    struct DebugSession {
        std::string sessionId;
        std::string queryId;
        std::string traceId;

        QueryAnalyzer::AnalysisResult analysis;
        std::optional<QueryMetrics> metrics;
        std::optional<DebugTracer::Trace> trace;
    };

    DebugSession startDebugSession(const std::string& query);
    void endDebugSession(DebugSession& session);

    /**
     * Interactive debugging
     */
    void setBreakpoint(const std::string& operatorType);
    void clearBreakpoints();
    void stepThrough(const std::string& sessionId);

    /**
     * Configuration
     */
    LadybugConfig getConfig() const { return config_; }
    void setConfig(const LadybugConfig& config);

private:
    LadybugConfig config_;
    QueryAnalyzer analyzer_;
    PerformanceMonitor monitor_;
    DebugTracer tracer_;
    VisualizationEngine visualizer_;

    std::vector<std::string> breakpoints_;
};

} // namespace graphql_agi
} // namespace kuzu
