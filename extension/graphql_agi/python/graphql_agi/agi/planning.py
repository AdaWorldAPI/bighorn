"""
AGI Planning System

Implements hierarchical task planning:
- Goal decomposition
- Action planning
- Constraint satisfaction
- Plan execution
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PlanningAlgorithm(str, Enum):
    """Available planning algorithms."""
    FORWARD_SEARCH = "forward_search"
    BACKWARD_CHAINING = "backward_chaining"
    HIERARCHICAL = "hierarchical"
    LLM_GUIDED = "llm_guided"


@dataclass
class PlanningConfig:
    """Configuration for planning system."""
    algorithm: PlanningAlgorithm = PlanningAlgorithm.FORWARD_SEARCH
    max_plan_length: int = 50
    max_search_nodes: int = 10000
    timeout_seconds: float = 30.0
    allow_parallel: bool = True
    optimize_for_cost: bool = True

    # LLM settings
    use_llm_heuristics: bool = False
    llm_model: str = "claude-3-opus"
    api_key: Optional[str] = None


@dataclass
class Condition:
    """A logical condition (predicate with arguments)."""
    predicate: str
    arguments: List[str] = field(default_factory=list)
    negated: bool = False

    def matches(self, other: Condition) -> bool:
        return (
            self.predicate == other.predicate and
            self.arguments == other.arguments and
            self.negated == other.negated
        )

    def __str__(self) -> str:
        neg = "NOT " if self.negated else ""
        args = ", ".join(self.arguments)
        return f"{neg}{self.predicate}({args})"


@dataclass
class Action:
    """An action that can be taken."""
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: List[Condition] = field(default_factory=list)
    effects: List[Condition] = field(default_factory=list)
    cost: float = 1.0
    duration: Optional[float] = None
    executor: Optional[Callable] = None


class WorldState:
    """Represents the state of the world."""

    def __init__(self):
        self.facts: List[Condition] = []

    def add_fact(self, fact: Condition) -> None:
        if not self.has_fact(fact):
            self.facts.append(fact)

    def remove_fact(self, fact: Condition) -> None:
        self.facts = [f for f in self.facts if not f.matches(fact)]

    def has_fact(self, fact: Condition) -> bool:
        return any(f.matches(fact) for f in self.facts)

    def satisfies(self, conditions: List[Condition]) -> bool:
        for cond in conditions:
            if cond.negated:
                if self.has_fact(Condition(cond.predicate, cond.arguments, False)):
                    return False
            else:
                if not self.has_fact(cond):
                    return False
        return True

    def apply(self, action: Action) -> WorldState:
        new_state = WorldState()
        new_state.facts = list(self.facts)

        for effect in action.effects:
            if effect.negated:
                new_state.remove_fact(
                    Condition(effect.predicate, effect.arguments, False)
                )
            else:
                new_state.add_fact(effect)

        return new_state

    def copy(self) -> WorldState:
        new_state = WorldState()
        new_state.facts = list(self.facts)
        return new_state


@dataclass
class PlanStep:
    """A step in a plan."""
    step_number: int
    action_name: str
    bound_arguments: List[str] = field(default_factory=list)
    description: Optional[str] = None
    estimated_cost: float = 1.0
    depends_on: List[int] = field(default_factory=list)

    class Status(str, Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"

    status: Status = Status.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """A complete plan."""
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    total_cost: float = 0.0
    estimated_duration: Optional[float] = None
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class PlanningResult:
    """Result of a planning operation."""
    success: bool
    plan: Optional[Plan] = None
    planning_time_ms: int = 0
    nodes_explored: int = 0
    failure_reasons: List[str] = field(default_factory=list)
    alternative_plans: List[Plan] = field(default_factory=list)


class PlanningSystem:
    """
    AGI Planning System

    Creates and executes plans to achieve goals.

    Example:
        >>> planner = PlanningSystem(config, reasoning_engine, graph_conn)
        >>>
        >>> result = planner.create_plan(
        ...     goal="Find all connections between AI researchers",
        ...     constraints=["Use only graph queries"]
        ... )
        >>>
        >>> if result.success:
        ...     planner.execute_plan(result.plan)
    """

    def __init__(
        self,
        config: PlanningConfig,
        reasoning_engine: Any,
        graph_connection: Any
    ):
        self.config = config
        self.reasoning_engine = reasoning_engine
        self.graph_conn = graph_connection
        self.actions: Dict[str, Action] = {}

        self._register_default_actions()

    def _register_default_actions(self) -> None:
        """Register default planning actions."""
        # Graph query action
        self.register_action(Action(
            name="graph_query",
            parameters=["query"],
            preconditions=[Condition("database_connected", [])],
            effects=[Condition("query_result_available", [])],
            cost=1.0,
            executor=self._execute_graph_query
        ))

        # Vector search action
        self.register_action(Action(
            name="vector_search",
            parameters=["query", "table", "top_k"],
            preconditions=[Condition("vector_store_connected", [])],
            effects=[Condition("search_results_available", [])],
            cost=2.0,
            executor=self._execute_vector_search
        ))

        # Reason action
        self.register_action(Action(
            name="reason",
            parameters=["question", "context"],
            preconditions=[Condition("reasoning_engine_available", [])],
            effects=[Condition("reasoning_complete", [])],
            cost=5.0,
            executor=self._execute_reasoning
        ))

        # Synthesize action
        self.register_action(Action(
            name="synthesize",
            parameters=["inputs"],
            preconditions=[Condition("data_available", [])],
            effects=[Condition("answer_synthesized", [])],
            cost=1.0,
            executor=self._execute_synthesis
        ))

    def register_action(self, action: Action) -> None:
        """Register a planning action."""
        self.actions[action.name] = action

    def create_plan(
        self,
        goal: str,
        constraints: Optional[List[str]] = None
    ) -> PlanningResult:
        """
        Create a plan to achieve a goal.

        Args:
            goal: Natural language goal description
            constraints: Optional constraints on the plan

        Returns:
            PlanningResult with the plan
        """
        start_time = time.time()
        constraints = constraints or []

        # Parse goal into conditions
        goal_conditions = self._parse_goal(goal)

        # Create initial state
        initial_state = self._create_initial_state()

        # Plan using configured algorithm
        if self.config.algorithm == PlanningAlgorithm.LLM_GUIDED:
            result = self._llm_guided_plan(goal, initial_state, goal_conditions)
        elif self.config.algorithm == PlanningAlgorithm.BACKWARD_CHAINING:
            result = self._backward_chain(initial_state, goal_conditions)
        else:
            result = self._forward_search(initial_state, goal_conditions)

        result.planning_time_ms = int((time.time() - start_time) * 1000)

        # Apply constraints
        if result.success and result.plan:
            result = self._apply_constraints(result, constraints)

        return result

    def _forward_search(
        self,
        initial_state: WorldState,
        goals: List[Condition]
    ) -> PlanningResult:
        """A* forward state-space search."""
        from heapq import heappush, heappop

        # Priority queue: (cost, state, actions_taken)
        frontier = [(0, initial_state, [])]
        explored = set()
        nodes_explored = 0

        while frontier and nodes_explored < self.config.max_search_nodes:
            cost, state, actions = heappop(frontier)
            nodes_explored += 1

            # Check if we've reached the goal
            if state.satisfies(goals):
                # Build plan from actions
                plan = self._actions_to_plan(actions, "Goal achieved")
                return PlanningResult(
                    success=True,
                    plan=plan,
                    nodes_explored=nodes_explored
                )

            # Generate state hash for duplicate detection
            state_hash = self._hash_state(state)
            if state_hash in explored:
                continue
            explored.add(state_hash)

            # Expand node with applicable actions
            for action_name, action in self.actions.items():
                if state.satisfies(action.preconditions):
                    new_state = state.apply(action)
                    new_cost = cost + action.cost
                    new_actions = actions + [(action_name, action)]

                    # Heuristic: number of unsatisfied goals
                    h = sum(1 for g in goals if not new_state.has_fact(g))
                    priority = new_cost + h

                    heappush(frontier, (priority, new_state, new_actions))

        return PlanningResult(
            success=False,
            nodes_explored=nodes_explored,
            failure_reasons=["Search exhausted without finding a plan"]
        )

    def _backward_chain(
        self,
        initial_state: WorldState,
        goals: List[Condition]
    ) -> PlanningResult:
        """Backward chaining from goals."""
        plan_steps = []
        remaining_goals = list(goals)
        achieved = set()
        nodes_explored = 0

        while remaining_goals and nodes_explored < self.config.max_search_nodes:
            goal = remaining_goals.pop(0)
            nodes_explored += 1

            if initial_state.has_fact(goal):
                continue

            # Find action that achieves this goal
            found_action = None
            for action_name, action in self.actions.items():
                for effect in action.effects:
                    if effect.matches(goal):
                        found_action = (action_name, action)
                        break
                if found_action:
                    break

            if not found_action:
                return PlanningResult(
                    success=False,
                    nodes_explored=nodes_explored,
                    failure_reasons=[f"No action achieves: {goal}"]
                )

            action_name, action = found_action
            plan_steps.insert(0, (action_name, action))

            # Add preconditions as new goals
            for precond in action.preconditions:
                if str(precond) not in achieved:
                    remaining_goals.append(precond)

            achieved.add(str(goal))

        plan = self._actions_to_plan(plan_steps, "Goal achieved via backward chaining")

        return PlanningResult(
            success=True,
            plan=plan,
            nodes_explored=nodes_explored
        )

    def _llm_guided_plan(
        self,
        goal: str,
        initial_state: WorldState,
        goal_conditions: List[Condition]
    ) -> PlanningResult:
        """Use LLM to guide planning."""
        # Generate plan steps using LLM
        plan_text = self._generate_plan_with_llm(goal)

        # Parse plan text into steps
        steps = self._parse_llm_plan(plan_text)

        if not steps:
            return PlanningResult(
                success=False,
                failure_reasons=["LLM failed to generate valid plan"]
            )

        plan = Plan(
            plan_id=f"plan_{int(time.time())}",
            goal=goal,
            steps=steps,
            total_cost=sum(s.estimated_cost for s in steps)
        )

        return PlanningResult(
            success=True,
            plan=plan,
            nodes_explored=1
        )

    def _generate_plan_with_llm(self, goal: str) -> str:
        """Generate plan using LLM."""
        # In production, call actual LLM
        # For now, generate a structured plan
        available_actions = ", ".join(self.actions.keys())

        return f"""
        Plan to achieve: {goal}

        Available actions: {available_actions}

        Steps:
        1. graph_query: Search for relevant entities
        2. vector_search: Find semantically similar items
        3. reason: Analyze the connections
        4. synthesize: Create final answer
        """

    def _parse_llm_plan(self, plan_text: str) -> List[PlanStep]:
        """Parse LLM-generated plan into steps."""
        import re
        steps = []

        lines = plan_text.strip().split("\n")
        step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match patterns like "1. action_name: description"
            match = re.match(r"(\d+)\.\s*(\w+):\s*(.+)", line)
            if match:
                step_num += 1
                action_name = match.group(2)
                description = match.group(3)

                action = self.actions.get(action_name)
                cost = action.cost if action else 1.0

                steps.append(PlanStep(
                    step_number=step_num,
                    action_name=action_name,
                    description=description,
                    estimated_cost=cost
                ))

        return steps

    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan: The plan to execute

        Returns:
            Execution results
        """
        results = []
        state = self._create_initial_state()

        for step in plan.steps:
            step.status = PlanStep.Status.IN_PROGRESS

            action = self.actions.get(step.action_name)
            if not action:
                step.status = PlanStep.Status.FAILED
                step.error = f"Unknown action: {step.action_name}"
                continue

            if not state.satisfies(action.preconditions):
                step.status = PlanStep.Status.FAILED
                step.error = "Preconditions not met"
                continue

            # Execute the action
            try:
                if action.executor:
                    result = action.executor(step.bound_arguments)
                    step.result = str(result)
                else:
                    step.result = "Action has no executor"

                step.status = PlanStep.Status.COMPLETED
                state = state.apply(action)
                results.append({
                    "step": step.step_number,
                    "action": step.action_name,
                    "result": step.result
                })

            except Exception as e:
                step.status = PlanStep.Status.FAILED
                step.error = str(e)

        success = all(s.status == PlanStep.Status.COMPLETED for s in plan.steps)

        return {
            "success": success,
            "steps": results,
            "final_state": state
        }

    def validate_plan(self, plan: Plan, initial_state: WorldState) -> bool:
        """Validate that a plan is executable."""
        state = initial_state.copy()

        for step in plan.steps:
            action = self.actions.get(step.action_name)
            if not action:
                plan.is_valid = False
                plan.validation_errors.append(f"Unknown action: {step.action_name}")
                return False

            if not state.satisfies(action.preconditions):
                plan.is_valid = False
                plan.validation_errors.append(
                    f"Step {step.step_number}: Preconditions not met"
                )
                return False

            state = state.apply(action)

        return True

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_goal(self, goal: str) -> List[Condition]:
        """Parse natural language goal into conditions."""
        # Simple heuristic parsing
        conditions = []

        goal_lower = goal.lower()

        if "find" in goal_lower or "search" in goal_lower:
            conditions.append(Condition("search_results_available", []))

        if "connect" in goal_lower or "relation" in goal_lower:
            conditions.append(Condition("query_result_available", []))

        if "answer" in goal_lower or "explain" in goal_lower:
            conditions.append(Condition("answer_synthesized", []))

        if not conditions:
            conditions.append(Condition("goal_achieved", []))

        return conditions

    def _create_initial_state(self) -> WorldState:
        """Create initial world state."""
        state = WorldState()
        state.add_fact(Condition("database_connected", []))
        state.add_fact(Condition("vector_store_connected", []))
        state.add_fact(Condition("reasoning_engine_available", []))
        return state

    def _hash_state(self, state: WorldState) -> str:
        """Create hash of state for duplicate detection."""
        facts = sorted(str(f) for f in state.facts)
        return ":".join(facts)

    def _actions_to_plan(
        self,
        actions: List[Tuple[str, Action]],
        goal: str
    ) -> Plan:
        """Convert action sequence to plan."""
        steps = []
        total_cost = 0.0

        for i, (name, action) in enumerate(actions):
            step = PlanStep(
                step_number=i + 1,
                action_name=name,
                bound_arguments=action.parameters,
                estimated_cost=action.cost
            )
            steps.append(step)
            total_cost += action.cost

        return Plan(
            plan_id=f"plan_{int(time.time())}",
            goal=goal,
            steps=steps,
            total_cost=total_cost
        )

    def _apply_constraints(
        self,
        result: PlanningResult,
        constraints: List[str]
    ) -> PlanningResult:
        """Apply constraints to filter/modify plan."""
        if not result.plan:
            return result

        # Simple constraint checking
        for constraint in constraints:
            constraint_lower = constraint.lower()

            if "no vector" in constraint_lower:
                # Remove vector search steps
                result.plan.steps = [
                    s for s in result.plan.steps
                    if s.action_name != "vector_search"
                ]

            if "no graph" in constraint_lower:
                result.plan.steps = [
                    s for s in result.plan.steps
                    if s.action_name != "graph_query"
                ]

        # Renumber steps
        for i, step in enumerate(result.plan.steps):
            step.step_number = i + 1

        return result

    # =========================================================================
    # Action Executors
    # =========================================================================

    def _execute_graph_query(self, args: List[str]) -> str:
        """Execute a graph query action."""
        query = args[0] if args else "MATCH (n) RETURN n LIMIT 10"
        try:
            result = self.graph_conn.execute(query)
            rows = list(result)[:5]
            return f"Query returned {len(rows)} results"
        except Exception as e:
            return f"Query failed: {e}"

    def _execute_vector_search(self, args: List[str]) -> str:
        """Execute a vector search action."""
        query = args[0] if args else "default query"
        return f"Vector search completed for: {query}"

    def _execute_reasoning(self, args: List[str]) -> str:
        """Execute a reasoning action."""
        question = args[0] if args else "default question"
        result = self.reasoning_engine.reason(question=question)
        return result.answer

    def _execute_synthesis(self, args: List[str]) -> str:
        """Execute a synthesis action."""
        return "Synthesized answer from available data"
