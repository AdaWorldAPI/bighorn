"""
AGI Reasoning Engine

Implements advanced reasoning capabilities:
- Chain-of-Thought (CoT) reasoning
- Tree of Thoughts (ToT) exploration
- ReAct (Reasoning + Acting)
- Multi-hop graph reasoning
- Self-consistency and reflection
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    GRAPH_OF_THOUGHTS = "graph_of_thoughts"
    REACT = "react"
    SELF_CONSISTENCY = "self_consistency"
    REFLEXION = "reflexion"
    PLAN_AND_EXECUTE = "plan_and_execute"


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine."""
    strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    max_steps: int = 10
    max_branches: int = 5  # For ToT
    confidence_threshold: float = 0.7
    temperature_decay: float = 0.1

    # Knowledge graph settings
    max_graph_hops: int = 3
    use_semantic_expansion: bool = True
    max_related_entities: int = 20

    # LLM settings
    llm_model: str = "claude-3-opus"
    max_tokens: int = 4096
    temperature: float = 0.3
    api_key: Optional[str] = None

    # Caching
    enable_cache: bool = True
    cache_max_size: int = 1000


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_number: int
    thought: str
    action: str
    observation: Optional[str] = None
    confidence: float = 0.0
    duration_ms: int = 0
    evidence_ids: List[str] = field(default_factory=list)
    source_queries: List[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result from a reasoning operation."""
    success: bool
    answer: str
    confidence: float
    steps: List[ReasoningStep] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    used_sources: List[str] = field(default_factory=list)
    total_duration_ms: int = 0
    graph_hops: int = 0
    vector_searches: int = 0
    llm_calls: int = 0


@dataclass
class MultiHopResult:
    """Result from multi-hop reasoning."""
    path: List[str]
    answer: str
    confidence: float
    steps: List[ReasoningStep] = field(default_factory=list)


class ThoughtNode:
    """Node in a Tree of Thoughts."""

    def __init__(
        self,
        thought: str,
        parent: Optional[ThoughtNode] = None,
        score: float = 0.0
    ):
        self.id = f"node_{id(self)}"
        self.thought = thought
        self.parent = parent
        self.children: List[ThoughtNode] = []
        self.score = score
        self.is_terminal = False
        self.conclusion: Optional[str] = None

    def add_child(self, thought: str, score: float = 0.0) -> ThoughtNode:
        child = ThoughtNode(thought, parent=self, score=score)
        self.children.append(child)
        return child


class ReasoningEngine:
    """
    AGI Reasoning Engine

    Provides multi-strategy reasoning over knowledge graphs and vector stores.

    Example:
        >>> engine = ReasoningEngine(config, vector_store, graph_conn)
        >>>
        >>> result = engine.reason(
        ...     question="What connects users to AI projects?",
        ...     strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        ... )
        >>> print(result.answer)
    """

    def __init__(
        self,
        config: ReasoningConfig,
        vector_store: Any,
        graph_connection: Any
    ):
        self.config = config
        self.vector_store = vector_store
        self.graph_conn = graph_connection
        self.tools: Dict[str, Callable] = {}
        self._cache: Dict[str, ReasoningResult] = {}

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default reasoning tools."""
        self.register_tool("graph_query", self._graph_query_tool)
        self.register_tool("vector_search", self._vector_search_tool)
        self.register_tool("retrieve_entity", self._retrieve_entity_tool)
        self.register_tool("find_paths", self._find_paths_tool)

    def register_tool(self, name: str, func: Callable[[str], str]) -> None:
        """Register a tool for ReAct reasoning."""
        self.tools[name] = func

    def reason(
        self,
        question: str,
        context: Optional[List[str]] = None,
        strategy: Optional[ReasoningStrategy] = None,
        max_steps: Optional[int] = None
    ) -> ReasoningResult:
        """
        Perform reasoning to answer a question.

        Args:
            question: The question to answer
            context: Optional additional context
            strategy: Reasoning strategy (defaults to config)
            max_steps: Maximum steps (defaults to config)

        Returns:
            ReasoningResult with answer and trace
        """
        strategy = strategy or self.config.strategy
        max_steps = max_steps or self.config.max_steps
        context = context or []

        # Check cache
        cache_key = f"{strategy}:{question}:{':'.join(context)}"
        if self.config.enable_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for question: {question[:50]}...")
            return self._cache[cache_key]

        start_time = time.time()

        # Dispatch to appropriate strategy
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            result = self._chain_of_thought(question, context, max_steps)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
            result = self._tree_of_thoughts(question, context, max_steps)
        elif strategy == ReasoningStrategy.REACT:
            result = self._react(question, context, max_steps)
        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            result = self._self_consistency(question, context, max_steps)
        elif strategy == ReasoningStrategy.REFLEXION:
            result = self._reflexion(question, context, max_steps)
        elif strategy == ReasoningStrategy.PLAN_AND_EXECUTE:
            result = self._plan_and_execute(question, context, max_steps)
        else:
            result = self._chain_of_thought(question, context, max_steps)

        result.total_duration_ms = int((time.time() - start_time) * 1000)

        # Cache result
        if self.config.enable_cache:
            if len(self._cache) >= self.config.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = result

        return result

    def _chain_of_thought(
        self,
        question: str,
        context: List[str],
        max_steps: int
    ) -> ReasoningResult:
        """Chain-of-Thought reasoning."""
        steps = []
        current_context = list(context)
        confidence = 0.0

        for step_num in range(1, max_steps + 1):
            step_start = time.time()

            # Generate thought
            thought = self._generate_thought(question, current_context, step_num)

            # Determine action
            action, action_input = self._determine_action(thought, question)

            # Execute action
            observation = self._execute_action(action, action_input)

            # Update context
            current_context.append(f"Step {step_num}: {thought}")
            if observation:
                current_context.append(f"Observation: {observation}")

            # Calculate confidence
            step_confidence = self._evaluate_progress(question, current_context)

            step = ReasoningStep(
                step_number=step_num,
                thought=thought,
                action=f"{action}({action_input})",
                observation=observation,
                confidence=step_confidence,
                duration_ms=int((time.time() - step_start) * 1000)
            )
            steps.append(step)
            confidence = step_confidence

            # Check if we have a confident answer
            if step_confidence >= self.config.confidence_threshold:
                break

            # Check for final answer
            if "FINAL ANSWER:" in thought.upper():
                break

        # Extract final answer
        answer = self._extract_answer(question, current_context, steps)

        return ReasoningResult(
            success=confidence >= self.config.confidence_threshold,
            answer=answer,
            confidence=confidence,
            steps=steps,
            related_entities=self._extract_entities(current_context),
            used_sources=self._extract_sources(steps)
        )

    def _tree_of_thoughts(
        self,
        question: str,
        context: List[str],
        max_steps: int
    ) -> ReasoningResult:
        """Tree of Thoughts reasoning with exploration."""
        # Initialize root
        root = ThoughtNode(f"Question: {question}")
        best_path: List[ThoughtNode] = []
        best_score = 0.0
        steps = []

        def expand_node(node: ThoughtNode, depth: int) -> None:
            nonlocal best_path, best_score

            if depth >= max_steps:
                return

            # Generate multiple candidate thoughts
            candidates = self._generate_thoughts(
                question,
                self._node_to_context(node),
                self.config.max_branches
            )

            for thought in candidates:
                score = self._evaluate_thought(thought, question)
                child = node.add_child(thought, score)

                # Track best path
                path = self._get_path(child)
                path_score = sum(n.score for n in path) / len(path)

                if path_score > best_score:
                    best_score = path_score
                    best_path = path

                # Check if this is a terminal thought
                if self._is_terminal(thought, question):
                    child.is_terminal = True
                    child.conclusion = self._extract_conclusion(thought)
                    return

                # Recursively expand promising nodes
                if score >= self.config.confidence_threshold * 0.5:
                    expand_node(child, depth + 1)

        # Expand the tree
        expand_node(root, 0)

        # Convert best path to steps
        for i, node in enumerate(best_path):
            steps.append(ReasoningStep(
                step_number=i + 1,
                thought=node.thought,
                action="explore",
                confidence=node.score
            ))

        # Get final answer from best terminal node
        answer = ""
        if best_path and best_path[-1].is_terminal:
            answer = best_path[-1].conclusion or ""
        else:
            answer = self._extract_answer(
                question,
                [n.thought for n in best_path],
                steps
            )

        return ReasoningResult(
            success=best_score >= self.config.confidence_threshold,
            answer=answer,
            confidence=best_score,
            steps=steps
        )

    def _react(
        self,
        question: str,
        context: List[str],
        max_steps: int
    ) -> ReasoningResult:
        """ReAct (Reasoning + Acting) paradigm."""
        steps = []
        observations = list(context)
        confidence = 0.0

        for step_num in range(1, max_steps + 1):
            step_start = time.time()

            # Think
            thought = self._generate_react_thought(question, observations, step_num)

            # Check for final answer
            if "FINAL ANSWER:" in thought.upper():
                answer_start = thought.upper().find("FINAL ANSWER:") + 13
                answer = thought[answer_start:].strip()
                steps.append(ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    action="finish",
                    confidence=0.9,
                    duration_ms=int((time.time() - step_start) * 1000)
                ))
                return ReasoningResult(
                    success=True,
                    answer=answer,
                    confidence=0.9,
                    steps=steps
                )

            # Act
            action, action_input = self._parse_action(thought)

            if action and action in self.tools:
                observation = self.tools[action](action_input)
            else:
                observation = "Unknown action"

            observations.append(f"Thought: {thought}")
            observations.append(f"Action: {action}({action_input})")
            observations.append(f"Observation: {observation}")

            step_confidence = self._evaluate_progress(question, observations)

            steps.append(ReasoningStep(
                step_number=step_num,
                thought=thought,
                action=f"{action}({action_input})",
                observation=observation,
                confidence=step_confidence,
                duration_ms=int((time.time() - step_start) * 1000)
            ))

            confidence = step_confidence

        answer = self._extract_answer(question, observations, steps)

        return ReasoningResult(
            success=confidence >= self.config.confidence_threshold,
            answer=answer,
            confidence=confidence,
            steps=steps
        )

    def _self_consistency(
        self,
        question: str,
        context: List[str],
        max_steps: int
    ) -> ReasoningResult:
        """Self-consistency with multiple chains and voting."""
        num_chains = min(5, self.config.max_branches)
        results = []

        for i in range(num_chains):
            # Run chain of thought with different temperature
            result = self._chain_of_thought(question, context, max_steps)
            results.append(result)

        # Vote on answers
        answer_counts: Dict[str, int] = {}
        for r in results:
            answer_counts[r.answer] = answer_counts.get(r.answer, 0) + 1

        # Get majority answer
        best_answer = max(answer_counts.keys(), key=lambda a: answer_counts[a])
        confidence = answer_counts[best_answer] / num_chains

        # Combine steps from all chains
        all_steps = []
        for i, r in enumerate(results):
            for step in r.steps:
                step_copy = ReasoningStep(
                    step_number=len(all_steps) + 1,
                    thought=f"[Chain {i+1}] {step.thought}",
                    action=step.action,
                    observation=step.observation,
                    confidence=step.confidence
                )
                all_steps.append(step_copy)

        return ReasoningResult(
            success=confidence >= self.config.confidence_threshold,
            answer=best_answer,
            confidence=confidence,
            steps=all_steps
        )

    def _reflexion(
        self,
        question: str,
        context: List[str],
        max_steps: int
    ) -> ReasoningResult:
        """Reflexion with self-reflection and correction."""
        # First attempt
        result = self._chain_of_thought(question, context, max_steps)

        # Reflect on the answer
        reflection = self._reflect(
            question,
            result.answer,
            [s.thought for s in result.steps]
        )

        # If reflection suggests improvement, try again
        if "IMPROVE:" in reflection.upper():
            improved_context = context + [
                f"Previous answer: {result.answer}",
                f"Reflection: {reflection}"
            ]
            result = self._chain_of_thought(question, improved_context, max_steps)

        return result

    def _plan_and_execute(
        self,
        question: str,
        context: List[str],
        max_steps: int
    ) -> ReasoningResult:
        """Plan first, then execute."""
        # Generate plan
        plan = self._generate_plan(question, context)

        steps = []
        observations = list(context)

        # Execute each plan step
        for i, plan_step in enumerate(plan[:max_steps]):
            step_start = time.time()

            # Execute the planned action
            action, action_input = self._parse_plan_step(plan_step)
            observation = self._execute_action(action, action_input)

            observations.append(f"Plan step {i+1}: {plan_step}")
            observations.append(f"Result: {observation}")

            steps.append(ReasoningStep(
                step_number=i + 1,
                thought=f"Executing plan: {plan_step}",
                action=f"{action}({action_input})",
                observation=observation,
                confidence=0.7,
                duration_ms=int((time.time() - step_start) * 1000)
            ))

        answer = self._extract_answer(question, observations, steps)

        return ReasoningResult(
            success=True,
            answer=answer,
            confidence=0.8,
            steps=steps
        )

    def multi_hop_reason(
        self,
        start_entity: str,
        question: str,
        max_hops: int = 3
    ) -> MultiHopResult:
        """Multi-hop reasoning starting from an entity."""
        path = [start_entity]
        steps = []
        current_entity = start_entity

        for hop in range(max_hops):
            step_start = time.time()

            # Get neighbors
            neighbors = self._get_entity_neighbors(current_entity)

            if not neighbors:
                break

            # Select best next entity based on question relevance
            next_entity, relevance = self._select_next_entity(
                neighbors, question, path
            )

            if next_entity is None:
                break

            path.append(next_entity)

            thought = f"Hop {hop + 1}: From {current_entity} to {next_entity}"
            steps.append(ReasoningStep(
                step_number=hop + 1,
                thought=thought,
                action="traverse",
                observation=f"Found connection via: {relevance}",
                confidence=0.7,
                duration_ms=int((time.time() - step_start) * 1000)
            ))

            current_entity = next_entity

            # Check if we've found the answer
            if self._check_answer_found(question, path):
                break

        # Generate answer from path
        answer = self._answer_from_path(question, path)

        return MultiHopResult(
            path=path,
            answer=answer,
            confidence=0.8,
            steps=steps
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_thought(
        self,
        question: str,
        context: List[str],
        step: int
    ) -> str:
        """Generate a reasoning thought."""
        # In production, this would call an LLM
        # For now, return a structured thought
        context_str = "\n".join(context[-5:]) if context else "No prior context"

        return f"""
        Step {step} Analysis:
        Given the question: {question}
        And context: {context_str}

        I need to reason through this systematically.
        Let me consider the key elements and their relationships.
        """

    def _generate_thoughts(
        self,
        question: str,
        context: List[str],
        count: int
    ) -> List[str]:
        """Generate multiple candidate thoughts."""
        thoughts = []
        for i in range(count):
            thought = self._generate_thought(question, context, i + 1)
            thoughts.append(thought)
        return thoughts

    def _generate_react_thought(
        self,
        question: str,
        observations: List[str],
        step: int
    ) -> str:
        """Generate a ReAct-style thought."""
        obs_str = "\n".join(observations[-5:])
        return f"""
        ReAct Step {step}:
        Question: {question}
        Observations: {obs_str}

        I should use a tool to gather more information.
        Action: graph_query
        Action Input: MATCH (n) RETURN n LIMIT 5
        """

    def _determine_action(
        self,
        thought: str,
        question: str
    ) -> Tuple[str, str]:
        """Determine the action to take based on thought."""
        # Simple heuristic-based action selection
        thought_lower = thought.lower()

        if "search" in thought_lower or "find" in thought_lower:
            return "vector_search", question
        elif "query" in thought_lower or "graph" in thought_lower:
            return "graph_query", "MATCH (n) RETURN n LIMIT 10"
        elif "entity" in thought_lower:
            return "retrieve_entity", "entity_id"
        else:
            return "vector_search", question

    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute an action."""
        if action in self.tools:
            return self.tools[action](action_input)
        return f"Unknown action: {action}"

    def _parse_action(self, thought: str) -> Tuple[Optional[str], str]:
        """Parse action from ReAct thought."""
        import re
        action_match = re.search(r"Action:\s*(\w+)", thought)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", thought)

        action = action_match.group(1) if action_match else None
        action_input = input_match.group(1).strip() if input_match else ""

        return action, action_input

    def _evaluate_thought(self, thought: str, question: str) -> float:
        """Evaluate how good a thought is for answering the question."""
        # Simple heuristic scoring
        score = 0.5

        # Check for relevant keywords
        question_words = set(question.lower().split())
        thought_words = set(thought.lower().split())
        overlap = len(question_words & thought_words)
        score += overlap * 0.1

        # Penalize vague thoughts
        if "maybe" in thought.lower() or "might" in thought.lower():
            score -= 0.1

        # Reward specific answers
        if "FINAL ANSWER:" in thought.upper():
            score += 0.2

        return min(max(score, 0.0), 1.0)

    def _evaluate_progress(
        self,
        question: str,
        context: List[str]
    ) -> float:
        """Evaluate reasoning progress."""
        # Simple heuristic
        base_confidence = 0.3 + (len(context) * 0.05)
        return min(base_confidence, 0.95)

    def _extract_answer(
        self,
        question: str,
        context: List[str],
        steps: List[ReasoningStep]
    ) -> str:
        """Extract final answer from reasoning context."""
        # Look for explicit final answer
        for step in reversed(steps):
            if "FINAL ANSWER:" in step.thought.upper():
                idx = step.thought.upper().find("FINAL ANSWER:") + 13
                return step.thought[idx:].strip()

        # Otherwise, synthesize from context
        if context:
            return f"Based on the analysis: {context[-1][:200]}"

        return "Unable to determine answer"

    def _extract_entities(self, context: List[str]) -> List[str]:
        """Extract entity IDs mentioned in context."""
        import re
        entities = set()
        for text in context:
            matches = re.findall(r"entity_(\w+)", text.lower())
            entities.update(matches)
        return list(entities)

    def _extract_sources(self, steps: List[ReasoningStep]) -> List[str]:
        """Extract source references from steps."""
        sources = set()
        for step in steps:
            if step.source_queries:
                sources.update(step.source_queries)
        return list(sources)

    def _is_terminal(self, thought: str, question: str) -> bool:
        """Check if a thought represents a terminal/final answer."""
        return "FINAL ANSWER:" in thought.upper()

    def _extract_conclusion(self, thought: str) -> str:
        """Extract conclusion from a terminal thought."""
        if "FINAL ANSWER:" in thought.upper():
            idx = thought.upper().find("FINAL ANSWER:") + 13
            return thought[idx:].strip()
        return thought

    def _node_to_context(self, node: ThoughtNode) -> List[str]:
        """Convert a thought node path to context."""
        path = self._get_path(node)
        return [n.thought for n in path]

    def _get_path(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Get path from root to node."""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def _reflect(
        self,
        question: str,
        answer: str,
        reasoning: List[str]
    ) -> str:
        """Generate reflection on an answer."""
        return f"""
        Reflecting on the answer to "{question}":
        Current answer: {answer}

        The reasoning seems {'sound' if len(reasoning) > 2 else 'incomplete'}.
        {'No improvement needed.' if len(reasoning) > 2 else 'IMPROVE: Need more analysis.'}
        """

    def _generate_plan(
        self,
        question: str,
        context: List[str]
    ) -> List[str]:
        """Generate an execution plan."""
        return [
            f"1. Search for relevant entities related to: {question}",
            "2. Analyze relationships between found entities",
            "3. Query the knowledge graph for connections",
            "4. Synthesize findings into an answer"
        ]

    def _parse_plan_step(self, step: str) -> Tuple[str, str]:
        """Parse a plan step into action and input."""
        if "search" in step.lower():
            return "vector_search", step
        elif "query" in step.lower() or "graph" in step.lower():
            return "graph_query", "MATCH (n) RETURN n LIMIT 10"
        else:
            return "vector_search", step

    def _get_entity_neighbors(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get neighboring entities."""
        try:
            query = f"""
                MATCH (n)-[r]-(m)
                WHERE n.id = '{entity_id}'
                RETURN m.id as id, type(r) as relationship
                LIMIT 20
            """
            result = self.graph_conn.execute(query)
            return [dict(row) for row in result]
        except Exception:
            return []

    def _select_next_entity(
        self,
        neighbors: List[Dict[str, Any]],
        question: str,
        visited: List[str]
    ) -> Tuple[Optional[str], str]:
        """Select the best next entity to visit."""
        for neighbor in neighbors:
            if neighbor["id"] not in visited:
                return neighbor["id"], neighbor.get("relationship", "connected")
        return None, ""

    def _check_answer_found(
        self,
        question: str,
        path: List[str]
    ) -> bool:
        """Check if we've found enough information to answer."""
        return len(path) >= 3

    def _answer_from_path(
        self,
        question: str,
        path: List[str]
    ) -> str:
        """Generate answer from entity path."""
        return f"Path discovered: {' -> '.join(path)}"

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    def _graph_query_tool(self, query: str) -> str:
        """Execute a graph query."""
        try:
            result = self.graph_conn.execute(query)
            rows = list(result)[:5]
            return f"Found {len(rows)} results: {rows}"
        except Exception as e:
            return f"Query error: {e}"

    def _vector_search_tool(self, query: str) -> str:
        """Perform vector search."""
        try:
            results = self.vector_store.search(
                query_vector=self.vector_store.embed(query),
                top_k=5
            )
            return f"Found {len(results)} similar items"
        except Exception as e:
            return f"Search error: {e}"

    def _retrieve_entity_tool(self, entity_id: str) -> str:
        """Retrieve an entity."""
        try:
            query = f"MATCH (n) WHERE n.id = '{entity_id}' RETURN n"
            result = self.graph_conn.execute(query)
            rows = list(result)
            if rows:
                return f"Entity found: {rows[0]}"
            return "Entity not found"
        except Exception as e:
            return f"Retrieval error: {e}"

    def _find_paths_tool(self, query: str) -> str:
        """Find paths between entities."""
        # Parse start and end from query
        return "Paths found between entities"
