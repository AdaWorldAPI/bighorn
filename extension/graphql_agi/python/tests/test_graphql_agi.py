"""
Comprehensive Test Suite for GraphQL AGI

Tests all components:
- GraphQL schema and query translation
- LanceDB vector store
- AGI reasoning engine
- Planning system
- Ladybug debugger
"""

import pytest
from typing import Any, Dict, List

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def vector_config():
    """Vector store configuration."""
    from graphql_agi.lancedb.vector_store import VectorStoreConfig
    return VectorStoreConfig(
        uri="lance://memory",
        dimension=1536,
        distance_metric="cosine"
    )


@pytest.fixture
def vector_store(vector_config):
    """Vector store instance."""
    from graphql_agi.lancedb.vector_store import VectorStore
    return VectorStore(vector_config)


@pytest.fixture
def reasoning_config():
    """Reasoning engine configuration."""
    from graphql_agi.agi.reasoning import ReasoningConfig, ReasoningStrategy
    return ReasoningConfig(
        strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_steps=5,
        confidence_threshold=0.7
    )


@pytest.fixture
def ladybug_config():
    """Ladybug debugger configuration."""
    from graphql_agi.ladybug.debugger import LadybugConfig
    return LadybugConfig(
        enable_tracing=True,
        enable_metrics=True,
        slow_query_threshold_ms=100
    )


@pytest.fixture
def ladybug_debugger(ladybug_config):
    """Ladybug debugger instance."""
    from graphql_agi.ladybug.debugger import LadybugDebugger
    return LadybugDebugger(ladybug_config)


# ============================================================================
# Vector Store Tests
# ============================================================================

class TestVectorStore:
    """Tests for LanceDB vector store."""

    def test_create_vector_store(self, vector_config):
        """Test vector store initialization."""
        from graphql_agi.lancedb.vector_store import VectorStore
        store = VectorStore(vector_config)
        assert store is not None

    def test_embed_text(self, vector_store):
        """Test text embedding generation."""
        embedding = vector_store.embed("Hello, world!")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # Default dimension

    def test_store_vector(self, vector_store):
        """Test storing a vector."""
        embedding = vector_store.embed("Test entity")
        success = vector_store.store(
            entity_id="test_1",
            entity_type="TestType",
            vector=embedding,
            content="Test entity content"
        )
        assert success is True

    def test_search_vectors(self, vector_store):
        """Test vector search."""
        # Store some vectors
        for i in range(5):
            embedding = vector_store.embed(f"Entity {i}")
            vector_store.store(
                entity_id=f"entity_{i}",
                entity_type="TestType",
                vector=embedding,
                content=f"Content for entity {i}"
            )

        # Search
        query_embedding = vector_store.embed("Entity 2")
        results = vector_store.search(
            query_vector=query_embedding,
            table="TestType",
            top_k=3
        )

        assert len(results) <= 3
        for result in results:
            assert hasattr(result, 'entity_id')
            assert hasattr(result, 'score')
            assert 0 <= result.score <= 1

    def test_delete_vector(self, vector_store):
        """Test vector deletion."""
        embedding = vector_store.embed("Delete me")
        vector_store.store(
            entity_id="delete_me",
            entity_type="TestType",
            vector=embedding
        )

        success = vector_store.delete("delete_me", "TestType")
        assert success is True

    def test_cosine_similarity(self, vector_store):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        sim_same = vector_store._cosine_similarity(vec1, vec2)
        sim_orthogonal = vector_store._cosine_similarity(vec1, vec3)

        assert sim_same == pytest.approx(1.0)
        assert sim_orthogonal == pytest.approx(0.0)


# ============================================================================
# Reasoning Engine Tests
# ============================================================================

class TestReasoningEngine:
    """Tests for AGI reasoning engine."""

    def test_chain_of_thought(self, vector_store, reasoning_config):
        """Test chain-of-thought reasoning."""
        from graphql_agi.agi.reasoning import ReasoningEngine

        # Mock graph connection
        class MockConn:
            def execute(self, query):
                return []

        engine = ReasoningEngine(
            config=reasoning_config,
            vector_store=vector_store,
            graph_connection=MockConn()
        )

        result = engine.reason(
            question="What is the capital of France?",
            strategy=reasoning_config.strategy
        )

        assert hasattr(result, 'success')
        assert hasattr(result, 'answer')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'steps')
        assert len(result.steps) > 0

    def test_reasoning_step_structure(self, vector_store, reasoning_config):
        """Test reasoning step structure."""
        from graphql_agi.agi.reasoning import ReasoningEngine

        class MockConn:
            def execute(self, query):
                return []

        engine = ReasoningEngine(
            config=reasoning_config,
            vector_store=vector_store,
            graph_connection=MockConn()
        )

        result = engine.reason("Test question")

        for step in result.steps:
            assert hasattr(step, 'step_number')
            assert hasattr(step, 'thought')
            assert hasattr(step, 'action')
            assert hasattr(step, 'confidence')
            assert step.step_number > 0

    def test_tool_registration(self, vector_store, reasoning_config):
        """Test custom tool registration."""
        from graphql_agi.agi.reasoning import ReasoningEngine

        class MockConn:
            def execute(self, query):
                return []

        engine = ReasoningEngine(
            config=reasoning_config,
            vector_store=vector_store,
            graph_connection=MockConn()
        )

        def custom_tool(input_str: str) -> str:
            return f"Custom result for: {input_str}"

        engine.register_tool("custom_tool", custom_tool)

        assert "custom_tool" in engine.tools

    def test_reasoning_strategies(self, vector_store):
        """Test different reasoning strategies."""
        from graphql_agi.agi.reasoning import (
            ReasoningEngine,
            ReasoningConfig,
            ReasoningStrategy
        )

        class MockConn:
            def execute(self, query):
                return []

        strategies = [
            ReasoningStrategy.CHAIN_OF_THOUGHT,
            ReasoningStrategy.TREE_OF_THOUGHTS,
            ReasoningStrategy.REACT,
            ReasoningStrategy.SELF_CONSISTENCY,
        ]

        for strategy in strategies:
            config = ReasoningConfig(
                strategy=strategy,
                max_steps=3
            )
            engine = ReasoningEngine(
                config=config,
                vector_store=vector_store,
                graph_connection=MockConn()
            )

            result = engine.reason("Test question", strategy=strategy)
            assert result is not None
            assert hasattr(result, 'answer')


# ============================================================================
# Planning System Tests
# ============================================================================

class TestPlanningSystem:
    """Tests for AGI planning system."""

    def test_create_plan(self, vector_store, reasoning_config):
        """Test plan creation."""
        from graphql_agi.agi.reasoning import ReasoningEngine
        from graphql_agi.agi.planning import PlanningSystem, PlanningConfig

        class MockConn:
            def execute(self, query):
                return []

        reasoning_engine = ReasoningEngine(
            config=reasoning_config,
            vector_store=vector_store,
            graph_connection=MockConn()
        )

        planning_config = PlanningConfig(max_plan_length=10)
        planner = PlanningSystem(
            config=planning_config,
            reasoning_engine=reasoning_engine,
            graph_connection=MockConn()
        )

        result = planner.create_plan(
            goal="Find all users connected to AI projects",
            constraints=[]
        )

        assert hasattr(result, 'success')
        assert hasattr(result, 'plan')

    def test_world_state(self):
        """Test world state management."""
        from graphql_agi.agi.planning import WorldState, Condition

        state = WorldState()

        # Add facts
        state.add_fact(Condition("connected", ["db"]))
        state.add_fact(Condition("available", ["api"]))

        assert state.has_fact(Condition("connected", ["db"]))
        assert not state.has_fact(Condition("connected", ["other"]))

        # Remove fact
        state.remove_fact(Condition("connected", ["db"]))
        assert not state.has_fact(Condition("connected", ["db"]))

    def test_action_registration(self, vector_store, reasoning_config):
        """Test action registration."""
        from graphql_agi.agi.reasoning import ReasoningEngine
        from graphql_agi.agi.planning import PlanningSystem, PlanningConfig, Action, Condition

        class MockConn:
            def execute(self, query):
                return []

        reasoning_engine = ReasoningEngine(
            config=reasoning_config,
            vector_store=vector_store,
            graph_connection=MockConn()
        )

        planner = PlanningSystem(
            config=PlanningConfig(),
            reasoning_engine=reasoning_engine,
            graph_connection=MockConn()
        )

        custom_action = Action(
            name="custom_action",
            parameters=["param1"],
            preconditions=[Condition("ready", [])],
            effects=[Condition("done", [])],
            cost=2.0
        )

        planner.register_action(custom_action)
        assert "custom_action" in planner.actions


# ============================================================================
# Ladybug Debugger Tests
# ============================================================================

class TestLadybugDebugger:
    """Tests for Ladybug debugger."""

    def test_query_analysis(self, ladybug_debugger):
        """Test query analysis."""
        query = """
            query {
                users(first: 10) {
                    id
                    name
                    posts {
                        title
                    }
                }
            }
        """

        analysis = ladybug_debugger.analyze_query(query)

        assert hasattr(analysis, 'valid')
        assert hasattr(analysis, 'complexity')
        assert hasattr(analysis, 'warnings')
        assert hasattr(analysis, 'suggestions')
        assert analysis.complexity > 0

    def test_debug_session(self, ladybug_debugger):
        """Test debug session lifecycle."""
        query = "MATCH (n) RETURN n LIMIT 10"

        session = ladybug_debugger.start_session(query)

        assert hasattr(session, 'session_id')
        assert hasattr(session, 'query_id')
        assert session.query_text == query

        ladybug_debugger.end_session(session, success=True)

        assert session.end_time is not None
        assert session.metrics is not None

    def test_metrics_collection(self, ladybug_debugger):
        """Test metrics collection."""
        # Execute some queries
        for i in range(5):
            session = ladybug_debugger.start_session(f"Query {i}")
            ladybug_debugger.end_session(session, success=True)

        metrics = ladybug_debugger.get_metrics()

        assert metrics['total_queries'] == 5
        assert 'avg_query_time_ms' in metrics
        assert 'p50_query_time_ms' in metrics
        assert 'p95_query_time_ms' in metrics

    def test_slow_query_detection(self, ladybug_config):
        """Test slow query detection."""
        import time
        from graphql_agi.ladybug.debugger import LadybugDebugger

        config = ladybug_config
        config.slow_query_threshold_ms = 1  # Very low threshold

        debugger = LadybugDebugger(config)

        session = debugger.start_session("Slow query")
        time.sleep(0.01)  # Sleep 10ms
        debugger.end_session(session, success=True)

        slow_queries = debugger.get_slow_queries()
        assert len(slow_queries) >= 1

    def test_anti_pattern_detection(self, ladybug_debugger):
        """Test anti-pattern detection."""
        query = "SELECT * FROM users"

        analysis = ladybug_debugger.analyze_query(query)

        assert len(analysis.anti_patterns) > 0
        assert any("Select All" in p['name'] for p in analysis.anti_patterns)

    def test_html_report_generation(self, ladybug_debugger):
        """Test HTML report generation."""
        session = ladybug_debugger.start_session("Test query")
        ladybug_debugger.end_session(session, success=True)

        html = ladybug_debugger.generate_html_report(session)

        assert "<!DOCTYPE html>" in html
        assert "Ladybug" in html
        assert session.session_id in html


# ============================================================================
# Schema Builder Tests
# ============================================================================

class TestSchemaBuilder:
    """Tests for GraphQL schema builder."""

    def test_schema_initialization(self):
        """Test schema builder initialization."""
        from graphql_agi.core.schema import SchemaBuilder

        builder = SchemaBuilder()

        assert "ID" in builder.types
        assert "String" in builder.types
        assert "Node" in builder.types
        assert len(builder.directives) > 0

    def test_add_type(self):
        """Test adding custom types."""
        from graphql_agi.core.schema import (
            SchemaBuilder,
            GraphQLType,
            GraphQLTypeKind,
            GraphQLField
        )

        builder = SchemaBuilder()

        user_type = GraphQLType(
            name="User",
            kind=GraphQLTypeKind.OBJECT,
            fields=[
                GraphQLField(name="id", type_name="ID!"),
                GraphQLField(name="name", type_name="String!"),
                GraphQLField(name="email", type_name="String"),
            ]
        )

        builder.add_type(user_type)

        assert "User" in builder.types
        assert len(builder.types["User"].fields) == 3

    def test_schema_validation(self):
        """Test schema validation."""
        from graphql_agi.core.schema import (
            SchemaBuilder,
            GraphQLType,
            GraphQLTypeKind,
            GraphQLField
        )

        builder = SchemaBuilder()

        # Add Query type (required)
        query_type = GraphQLType(
            name="Query",
            kind=GraphQLTypeKind.OBJECT,
            fields=[
                GraphQLField(name="users", type_name="[User!]!")
            ]
        )
        builder.add_type(query_type)

        validation = builder.validate()

        assert validation['valid'] is True
        # Should have warning about undefined User type
        assert len(validation['warnings']) > 0

    def test_build_sdl(self):
        """Test SDL generation."""
        from graphql_agi.core.schema import (
            SchemaBuilder,
            GraphQLType,
            GraphQLTypeKind,
            GraphQLField
        )

        builder = SchemaBuilder()

        builder.add_type(GraphQLType(
            name="Query",
            kind=GraphQLTypeKind.OBJECT,
            fields=[GraphQLField(name="hello", type_name="String!")]
        ))

        sdl = builder.build()

        assert "schema {" in sdl
        assert "query: Query" in sdl
        assert "type Query" in sdl
        assert "hello: String!" in sdl


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self, vector_store, reasoning_config, ladybug_debugger):
        """Test full GraphQL AGI pipeline."""
        from graphql_agi.agi.reasoning import ReasoningEngine
        from graphql_agi.agi.planning import PlanningSystem, PlanningConfig

        class MockConn:
            def execute(self, query):
                return [{"id": "1", "name": "Test"}]

        # 1. Store some vectors
        for i in range(3):
            embedding = vector_store.embed(f"Entity {i} is about AI research")
            vector_store.store(
                entity_id=f"entity_{i}",
                entity_type="Research",
                vector=embedding,
                content=f"Entity {i} content"
            )

        # 2. Create reasoning engine
        reasoning_engine = ReasoningEngine(
            config=reasoning_config,
            vector_store=vector_store,
            graph_connection=MockConn()
        )

        # 3. Create planner
        planner = PlanningSystem(
            config=PlanningConfig(),
            reasoning_engine=reasoning_engine,
            graph_connection=MockConn()
        )

        # 4. Debug a query
        session = ladybug_debugger.start_session("Test query")

        # 5. Perform search
        query_vec = vector_store.embed("AI research")
        results = vector_store.search(query_vec, "Research", top_k=2)

        # 6. Reason about results
        reasoning_result = reasoning_engine.reason(
            "What AI research is available?"
        )

        # 7. Create a plan
        plan_result = planner.create_plan("Find AI researchers")

        ladybug_debugger.end_session(session, success=True)

        # Verify all components worked
        assert len(results) > 0
        assert reasoning_result.answer is not None
        assert plan_result.success


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
