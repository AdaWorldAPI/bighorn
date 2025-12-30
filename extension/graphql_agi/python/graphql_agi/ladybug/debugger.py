"""
Ladybug Query Analyzer & Debugger

Comprehensive debugging and visualization system for GraphQL AGI.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceSeverity(str, Enum):
    """Trace event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LadybugConfig:
    """Configuration for Ladybug debugger."""
    enable_tracing: bool = True
    min_severity: TraceSeverity = TraceSeverity.INFO
    max_trace_events: int = 10000
    enable_distributed_tracing: bool = False

    enable_metrics: bool = True
    enable_slow_query_log: bool = True
    slow_query_threshold_ms: int = 100

    capture_query_plans: bool = True
    explain_analyze: bool = False

    enable_visualization: bool = True
    visualization_format: str = "json"

    trace_output_file: Optional[str] = None
    metrics_output_file: Optional[str] = None
    enable_console_output: bool = False

    sampling_rate: float = 1.0
    always_sample_errors: bool = True


@dataclass
class TraceEvent:
    """A single trace event."""
    event_id: str
    timestamp: float
    severity: TraceSeverity
    category: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_us: int = 0
    parent_event_id: Optional[str] = None


@dataclass
class QueryMetrics:
    """Performance metrics for a query."""
    query_id: str
    query_hash: str
    query_text: str = ""

    parse_time_ms: float = 0.0
    plan_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0

    memory_used_bytes: int = 0
    peak_memory_bytes: int = 0
    threads_used: int = 1

    rows_scanned: int = 0
    rows_returned: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    cache_hits: int = 0
    cache_misses: int = 0

    vector_searches: int = 0
    vectors_compared: int = 0
    vector_search_time_ms: float = 0.0

    llm_calls: int = 0
    llm_tokens_used: int = 0
    llm_latency_ms: float = 0.0

    reasoning_steps: int = 0
    graph_hops: int = 0
    reasoning_confidence: float = 0.0


@dataclass
class QueryPlanNode:
    """Node in a query plan tree."""
    node_id: str
    operator_type: str
    description: str = ""

    estimated_rows: float = 0.0
    estimated_cost: float = 0.0

    actual_rows: Optional[int] = None
    actual_time_us: Optional[int] = None

    children: List[QueryPlanNode] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    valid: bool
    complexity: float
    estimated_cost: float
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    query_plan: Optional[str] = None
    anti_patterns: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DebugSession:
    """A debugging session."""
    session_id: str
    query_id: str
    trace_id: str
    query_text: str
    start_time: float
    end_time: Optional[float] = None
    analysis: Optional[QueryAnalysis] = None
    metrics: Optional[QueryMetrics] = None
    events: List[TraceEvent] = field(default_factory=list)


class LadybugDebugger:
    """
    Ladybug Query Analyzer & Debugger

    Provides query analysis, performance monitoring, and visualization.

    Example:
        >>> debugger = LadybugDebugger(config)
        >>>
        >>> # Analyze a query
        >>> analysis = debugger.analyze_query("query { users { id } }")
        >>> print(f"Complexity: {analysis.complexity}")
        >>>
        >>> # Start a debug session
        >>> session = debugger.start_session("query { users { id } }")
        >>> # ... execute query ...
        >>> debugger.end_session(session, success=True)
    """

    def __init__(self, config: Optional[LadybugConfig] = None):
        self.config = config or LadybugConfig()
        self._sessions: Dict[str, DebugSession] = {}
        self._completed_queries: List[QueryMetrics] = []
        self._events: List[TraceEvent] = []
        self._start_time = time.time()

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a GraphQL or Cypher query.

        Args:
            query: The query to analyze

        Returns:
            QueryAnalysis with complexity, warnings, and suggestions
        """
        valid = True
        warnings = []
        suggestions = []
        anti_patterns = []

        # Calculate complexity
        complexity = self._calculate_complexity(query)

        # Estimate cost
        estimated_cost = self._estimate_cost(query)

        # Detect anti-patterns
        anti_patterns = self._detect_anti_patterns(query)
        for pattern in anti_patterns:
            warnings.append(f"{pattern['name']}: {pattern['description']}")
            suggestions.append(pattern.get('suggestion', ''))

        # Generate query plan if it's Cypher
        query_plan = None
        if self._is_cypher(query):
            query_plan = self._generate_query_plan(query)

        # Check for common issues
        if complexity > 100:
            warnings.append("High query complexity may impact performance")
            suggestions.append("Consider breaking into smaller queries")

        if "SELECT *" in query.upper() or "RETURN *" in query.upper():
            warnings.append("Selecting all columns may be inefficient")
            suggestions.append("Select only needed columns")

        if complexity > 500:
            valid = False
            warnings.append("Query complexity exceeds maximum allowed")

        return QueryAnalysis(
            valid=valid,
            complexity=complexity,
            estimated_cost=estimated_cost,
            warnings=warnings,
            suggestions=[s for s in suggestions if s],
            query_plan=query_plan,
            anti_patterns=anti_patterns
        )

    def start_session(self, query: str) -> DebugSession:
        """
        Start a debug session for a query.

        Args:
            query: The query being executed

        Returns:
            DebugSession object
        """
        session = DebugSession(
            session_id=str(uuid.uuid4()),
            query_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            query_text=query,
            start_time=time.time()
        )

        # Analyze the query
        session.analysis = self.analyze_query(query)

        self._sessions[session.session_id] = session

        # Log session start
        self._log_event(
            session.trace_id,
            TraceSeverity.INFO,
            "session",
            f"Started debug session for query"
        )

        return session

    def end_session(
        self,
        session: DebugSession,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        End a debug session.

        Args:
            session: The session to end
            success: Whether the query succeeded
            error: Optional error message
        """
        session.end_time = time.time()
        duration_ms = (session.end_time - session.start_time) * 1000

        # Create metrics
        session.metrics = QueryMetrics(
            query_id=session.query_id,
            query_hash=self._hash_query(session.query_text),
            query_text=session.query_text,
            total_time_ms=duration_ms
        )

        # Log session end
        severity = TraceSeverity.INFO if success else TraceSeverity.ERROR
        message = f"Session completed in {duration_ms:.2f}ms"
        if error:
            message += f" - Error: {error}"

        self._log_event(
            session.trace_id,
            severity,
            "session",
            message
        )

        # Store completed query metrics
        self._completed_queries.append(session.metrics)

        # Check for slow query
        if self.config.enable_slow_query_log:
            if duration_ms > self.config.slow_query_threshold_ms:
                logger.warning(
                    f"Slow query detected: {duration_ms:.2f}ms - "
                    f"{session.query_text[:100]}..."
                )

    def log_event(
        self,
        session: DebugSession,
        severity: TraceSeverity,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an event in a debug session."""
        self._log_event(
            session.trace_id,
            severity,
            category,
            message,
            metadata
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated query metrics.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self._completed_queries:
            return {
                "total_queries": 0,
                "avg_query_time_ms": 0,
                "p50_query_time_ms": 0,
                "p95_query_time_ms": 0,
                "p99_query_time_ms": 0,
                "cache_hit_rate": 0,
                "slow_query_count": 0
            }

        times = sorted(q.total_time_ms for q in self._completed_queries)
        total = len(times)

        cache_hits = sum(q.cache_hits for q in self._completed_queries)
        cache_misses = sum(q.cache_misses for q in self._completed_queries)
        cache_total = cache_hits + cache_misses

        slow_count = sum(
            1 for q in self._completed_queries
            if q.total_time_ms > self.config.slow_query_threshold_ms
        )

        return {
            "total_queries": total,
            "avg_query_time_ms": sum(times) / total,
            "p50_query_time_ms": times[int(total * 0.5)] if total else 0,
            "p95_query_time_ms": times[int(total * 0.95)] if total else 0,
            "p99_query_time_ms": times[int(total * 0.99)] if total else 0,
            "cache_hit_rate": cache_hits / cache_total if cache_total else 0,
            "slow_query_count": slow_count,
            "uptime_seconds": time.time() - self._start_time
        }

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the slowest queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of slow query details
        """
        slow = sorted(
            self._completed_queries,
            key=lambda q: q.total_time_ms,
            reverse=True
        )[:limit]

        return [
            {
                "query_id": q.query_id,
                "query_text": q.query_text[:200] + "..." if len(q.query_text) > 200 else q.query_text,
                "total_time_ms": q.total_time_ms,
                "rows_returned": q.rows_returned
            }
            for q in slow
        ]

    def explain_query(self, query: str, analyze: bool = False) -> str:
        """
        Explain a query's execution plan.

        Args:
            query: The query to explain
            analyze: Whether to include actual execution stats

        Returns:
            Query plan as formatted text
        """
        plan = self._generate_query_plan(query)

        if analyze:
            # Would execute and gather actual stats
            plan += "\n\n(ANALYZE not implemented in mock)"

        return plan

    def render_query_plan_json(
        self,
        plan_node: QueryPlanNode
    ) -> Dict[str, Any]:
        """Render query plan as JSON for visualization."""
        return {
            "id": plan_node.node_id,
            "type": plan_node.operator_type,
            "description": plan_node.description,
            "estimatedRows": plan_node.estimated_rows,
            "estimatedCost": plan_node.estimated_cost,
            "actualRows": plan_node.actual_rows,
            "actualTimeUs": plan_node.actual_time_us,
            "properties": plan_node.properties,
            "warnings": plan_node.warnings,
            "children": [
                self.render_query_plan_json(c) for c in plan_node.children
            ]
        }

    def generate_html_report(
        self,
        session: DebugSession
    ) -> str:
        """Generate an interactive HTML debug report."""
        metrics_json = "null"
        if session.metrics:
            import json
            metrics_json = json.dumps({
                "total_time_ms": session.metrics.total_time_ms,
                "rows_returned": session.metrics.rows_returned,
                "cache_hits": session.metrics.cache_hits
            })

        analysis_json = "null"
        if session.analysis:
            import json
            analysis_json = json.dumps({
                "complexity": session.analysis.complexity,
                "warnings": session.analysis.warnings,
                "suggestions": session.analysis.suggestions
            })

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ladybug Debug Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin: 10px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; font-size: 12px; }}
        .warning {{ background: #FFF3CD; border-left: 4px solid #FFC107; padding: 10px; margin: 5px 0; }}
        .suggestion {{ background: #D1ECF1; border-left: 4px solid #17A2B8; padding: 10px; margin: 5px 0; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Ladybug Debug Report</h1>

    <div class="card">
        <h2>Session Info</h2>
        <p><strong>Session ID:</strong> {session.session_id}</p>
        <p><strong>Query ID:</strong> {session.query_id}</p>
        <p><strong>Duration:</strong> {(session.end_time or time.time()) - session.start_time:.3f}s</p>
    </div>

    <div class="card">
        <h2>Query</h2>
        <pre>{session.query_text}</pre>
    </div>

    <div class="card">
        <h2>Metrics</h2>
        <div class="metric">
            <div class="metric-value">{session.metrics.total_time_ms if session.metrics else 0:.2f}ms</div>
            <div class="metric-label">Total Time</div>
        </div>
        <div class="metric">
            <div class="metric-value">{session.metrics.rows_returned if session.metrics else 0}</div>
            <div class="metric-label">Rows Returned</div>
        </div>
    </div>

    <div class="card">
        <h2>Analysis</h2>
        <p><strong>Complexity:</strong> {session.analysis.complexity if session.analysis else 0:.1f}</p>

        <h3>Warnings</h3>
        {''.join(f'<div class="warning">{w}</div>' for w in (session.analysis.warnings if session.analysis else []))}

        <h3>Suggestions</h3>
        {''.join(f'<div class="suggestion">{s}</div>' for s in (session.analysis.suggestions if session.analysis else []))}
    </div>

    {'<div class="card"><h2>Query Plan</h2><pre>' + session.analysis.query_plan + '</pre></div>' if session.analysis and session.analysis.query_plan else ''}

    <script>
        const metrics = {metrics_json};
        const analysis = {analysis_json};
        console.log('Ladybug Debug Data:', {{ metrics, analysis }});
    </script>
</body>
</html>
"""

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        complexity = 1.0

        # Count nesting depth (approximation)
        max_depth = 0
        current_depth = 0
        for char in query:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1

        complexity += max_depth * 5

        # Count field selections
        field_count = query.count(':') + query.count('{')
        complexity += field_count * 2

        # Check for expensive operations
        query_upper = query.upper()
        if "MATCH" in query_upper:
            complexity += 10
        if "OPTIONAL" in query_upper:
            complexity += 5
        if "*" in query:
            complexity += 20

        return complexity

    def _estimate_cost(self, query: str) -> float:
        """Estimate query execution cost."""
        return self._calculate_complexity(query) * 1.5

    def _detect_anti_patterns(self, query: str) -> List[Dict[str, str]]:
        """Detect anti-patterns in query."""
        patterns = []

        query_upper = query.upper()

        if "SELECT *" in query_upper or "RETURN *" in query_upper:
            patterns.append({
                "name": "Select All",
                "description": "Selecting all columns can be inefficient",
                "suggestion": "Select only the columns you need",
                "severity": "warning"
            })

        if query.count('{') > 5:
            patterns.append({
                "name": "Deep Nesting",
                "description": "Query has deep nesting which may cause N+1 problems",
                "suggestion": "Consider flattening the query or using batching",
                "severity": "warning"
            })

        if "CARTESIAN" in query_upper or ("MATCH" in query_upper and query.count("MATCH") > 2):
            patterns.append({
                "name": "Potential Cartesian Product",
                "description": "Multiple unconnected patterns may create cartesian products",
                "suggestion": "Ensure patterns are connected or use explicit joins",
                "severity": "error"
            })

        return patterns

    def _is_cypher(self, query: str) -> bool:
        """Check if query is Cypher (vs GraphQL)."""
        cypher_keywords = ["MATCH", "CREATE", "MERGE", "DELETE", "SET", "RETURN"]
        query_upper = query.upper()
        return any(kw in query_upper for kw in cypher_keywords)

    def _generate_query_plan(self, query: str) -> str:
        """Generate a query execution plan."""
        # Mock query plan
        return f"""
Query Plan for: {query[:50]}...

┌─────────────────────────────────────────────────────┐
│                    Result                           │
│                  (estimated: 100 rows)              │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                 Projection                          │
│               (select fields)                       │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Filter                             │
│              (apply predicates)                     │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                 Table Scan                          │
│               (scan all nodes)                      │
└─────────────────────────────────────────────────────┘
"""

    def _hash_query(self, query: str) -> str:
        """Create a hash of the query for deduplication."""
        import hashlib
        normalized = " ".join(query.split()).lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _log_event(
        self,
        trace_id: str,
        severity: TraceSeverity,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a trace event."""
        if severity.value < self.config.min_severity.value:
            return

        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=message,
            metadata=metadata or {}
        )

        self._events.append(event)

        # Trim old events if over limit
        if len(self._events) > self.config.max_trace_events:
            self._events = self._events[-self.config.max_trace_events:]

        if self.config.enable_console_output:
            logger.log(
                logging.DEBUG if severity == TraceSeverity.DEBUG else
                logging.INFO if severity == TraceSeverity.INFO else
                logging.WARNING if severity == TraceSeverity.WARNING else
                logging.ERROR,
                f"[{category}] {message}"
            )
