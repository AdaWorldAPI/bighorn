"""
NARS (Non-Axiomatic Reasoning System) - Resource-bounded inference.

Implements basic NARS-style inference with truth values that track
both frequency and confidence. Designed for real-time AGI reasoning
under computational constraints.

Based on Pei Wang's NARS theory:
- Truth values: (frequency, confidence)
- Evidence accumulation
- Resource-bounded inference
- Non-monotonic reasoning

Reference: https://cis.temple.edu/~pwang/NARS-Intro.html
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


# =============================================================================
# TRUTH VALUES
# =============================================================================

@dataclass
class TruthValue:
    """
    NARS truth value with frequency and confidence.

    - frequency (f): Proportion of positive evidence [0, 1]
    - confidence (c): Amount of evidence relative to max [0, 1)

    The confidence approaches 1 as evidence accumulates but never reaches it,
    reflecting that we can never be 100% certain.
    """

    frequency: float = 0.5  # f ∈ [0, 1]
    confidence: float = 0.5  # c ∈ [0, 1)

    # Evidential horizon (k): determines how quickly confidence grows
    K: float = 1.0

    def __post_init__(self):
        """Clamp values to valid ranges."""
        self.frequency = max(0.0, min(1.0, self.frequency))
        self.confidence = max(0.0, min(0.99, self.confidence))

    @classmethod
    def from_evidence(cls, positive: int, total: int, k: float = 1.0) -> "TruthValue":
        """
        Create truth value from evidence counts.

        Args:
            positive: Number of positive observations
            total: Total number of observations
            k: Evidential horizon parameter

        Returns:
            TruthValue with computed f and c
        """
        if total == 0:
            return cls(0.5, 0.0)

        f = positive / total
        c = total / (total + k)
        return cls(f, c, k)

    @property
    def expectation(self) -> float:
        """Expected value: E = c * (f - 0.5) + 0.5"""
        return self.confidence * (self.frequency - 0.5) + 0.5

    @property
    def evidence(self) -> Tuple[float, float]:
        """
        Convert back to evidence counts.

        Returns:
            (positive_evidence, total_evidence)
        """
        if self.confidence >= 1.0:
            return (self.frequency * 1000, 1000)

        w = self.confidence / (1 - self.confidence)  # total evidence
        return (self.frequency * w, w)

    def revision(self, other: "TruthValue") -> "TruthValue":
        """
        Revise this truth value with new evidence.

        Combines evidence from two sources about the same statement.

        Args:
            other: New evidence

        Returns:
            Revised truth value
        """
        # Get evidence counts
        w1_pos, w1 = self.evidence
        w2_pos, w2 = other.evidence

        # Combine evidence
        total = w1 + w2
        if total == 0:
            return TruthValue(0.5, 0.0)

        f = (w1_pos + w2_pos) / total
        c = total / (total + self.K)

        return TruthValue(f, c, self.K)

    def __repr__(self) -> str:
        return f"<{self.frequency:.2f}, {self.confidence:.2f}>"


# =============================================================================
# STATEMENTS
# =============================================================================

class Copula(str, Enum):
    """NARS copulas (statement relations)."""
    INHERITANCE = "-->"      # S --> P: S is a type of P
    SIMILARITY = "<->"       # S <-> P: S and P are similar
    IMPLICATION = "==>"      # S ==> P: If S then P
    EQUIVALENCE = "<=>"      # S <=> P: S iff P
    INSTANCE = "{--"         # {S} --> P: S is an instance of P
    PROPERTY = "--]"         # S --> [P]: S has property P
    INST_PROP = "{-]"        # {S} --> [P]: Instance S has property P


@dataclass
class Term:
    """A term (concept) in NARS."""
    name: str
    is_variable: bool = False
    is_compound: bool = False
    components: List["Term"] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.is_variable:
            return f"${self.name}"
        return self.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Statement:
    """A NARS statement with subject, copula, and predicate."""
    subject: Term
    copula: Copula
    predicate: Term
    truth: TruthValue = field(default_factory=TruthValue)

    def __repr__(self) -> str:
        return f"{self.subject} {self.copula.value} {self.predicate} {self.truth}"

    @classmethod
    def parse(cls, text: str, truth: TruthValue = None) -> "Statement":
        """
        Parse statement from string.

        Examples:
            "bird --> animal"
            "robin --> bird"
            "robin --> animal"
        """
        # Find copula
        for copula in Copula:
            if copula.value in text:
                parts = text.split(copula.value)
                if len(parts) == 2:
                    subj = Term(parts[0].strip())
                    pred = Term(parts[1].strip())
                    return cls(subj, copula, pred, truth or TruthValue())

        raise ValueError(f"Could not parse statement: {text}")


# =============================================================================
# INFERENCE RULES
# =============================================================================

class InferenceRule(str, Enum):
    """NARS inference rules."""
    # First-order syllogistic rules
    DEDUCTION = "deduction"       # (M --> P, S --> M) |- S --> P
    INDUCTION = "induction"       # (M --> P, M --> S) |- S --> P
    ABDUCTION = "abduction"       # (P --> M, S --> M) |- S --> P
    EXEMPLIFICATION = "exemplification"  # (P --> M, S --> M) |- P --> S

    # Composition rules
    INTERSECTION = "intersection"
    UNION = "union"
    DIFFERENCE = "difference"

    # Higher-order rules
    CONDITIONAL_DEDUCTION = "conditional_deduction"
    CONDITIONAL_ABDUCTION = "conditional_abduction"

    # Revision
    REVISION = "revision"


def deduction_truth(t1: TruthValue, t2: TruthValue) -> TruthValue:
    """
    Deduction truth function.

    (M --> P) <t1>
    (S --> M) <t2>
    |-
    (S --> P) <f, c>

    f = f1 * f2
    c = f1 * f2 * c1 * c2
    """
    f = t1.frequency * t2.frequency
    c = t1.frequency * t2.frequency * t1.confidence * t2.confidence
    return TruthValue(f, c)


def induction_truth(t1: TruthValue, t2: TruthValue) -> TruthValue:
    """
    Induction truth function.

    (M --> P) <t1>
    (M --> S) <t2>
    |-
    (S --> P) <f, c>

    Weaker than deduction due to uncertainty about M's coverage.
    """
    f = t1.frequency
    c = t2.frequency * t1.confidence * t2.confidence / (t2.frequency * t1.confidence * t2.confidence + 1)
    return TruthValue(f, c)


def abduction_truth(t1: TruthValue, t2: TruthValue) -> TruthValue:
    """
    Abduction truth function.

    (P --> M) <t1>
    (S --> M) <t2>
    |-
    (S --> P) <f, c>

    Inference to best explanation - weakest but creative.
    """
    f = t2.frequency
    c = t1.frequency * t1.confidence * t2.confidence / (t1.frequency * t1.confidence * t2.confidence + 1)
    return TruthValue(f, c)


def revision_truth(t1: TruthValue, t2: TruthValue) -> TruthValue:
    """Combine evidence for same statement."""
    return t1.revision(t2)


# =============================================================================
# NARS REASONER
# =============================================================================

class NARSReasoner:
    """
    Basic NARS inference engine.

    Performs resource-bounded reasoning with truth value tracking.
    """

    def __init__(self):
        """Initialize reasoner."""
        self.beliefs: Dict[str, Statement] = {}
        self.goals: List[Statement] = []
        self.inference_trace: List[str] = []

    def add_belief(self, statement: Statement) -> None:
        """Add or revise a belief."""
        key = f"{statement.subject}_{statement.copula.value}_{statement.predicate}"

        if key in self.beliefs:
            # Revise existing belief
            existing = self.beliefs[key]
            revised_truth = revision_truth(existing.truth, statement.truth)
            statement.truth = revised_truth
            self.inference_trace.append(
                f"Revised: {statement} (was {existing.truth})"
            )

        self.beliefs[key] = statement

    def infer(
        self,
        premises: List[str],
        rule: str,
    ) -> Tuple[str, TruthValue]:
        """
        Perform single inference step.

        Args:
            premises: List of premise strings
            rule: Inference rule to apply

        Returns:
            (conclusion_string, truth_value)
        """
        self.inference_trace = []

        # Parse premises
        parsed = [Statement.parse(p) for p in premises]

        if len(parsed) < 2:
            return ("", TruthValue(0.5, 0.0))

        p1, p2 = parsed[0], parsed[1]

        # Apply inference rule
        if rule == InferenceRule.DEDUCTION.value:
            return self._deduction(p1, p2)
        elif rule == InferenceRule.INDUCTION.value:
            return self._induction(p1, p2)
        elif rule == InferenceRule.ABDUCTION.value:
            return self._abduction(p1, p2)
        elif rule == InferenceRule.REVISION.value:
            return self._revision(p1, p2)
        else:
            return ("", TruthValue(0.5, 0.0))

    def _deduction(
        self,
        p1: Statement,
        p2: Statement,
    ) -> Tuple[str, TruthValue]:
        """
        Deduction: (M --> P, S --> M) |- S --> P

        "If birds fly and robins are birds, then robins fly."
        """
        # Check if middle term matches
        if p1.subject == p2.predicate:
            # M --> P, S --> M  =>  S --> P
            conclusion = f"{p2.subject} {Copula.INHERITANCE.value} {p1.predicate}"
            truth = deduction_truth(p1.truth, p2.truth)
            self.inference_trace.append(
                f"Deduction: {p1} + {p2} => {conclusion} {truth}"
            )
            return (conclusion, truth)

        return ("", TruthValue(0.5, 0.0))

    def _induction(
        self,
        p1: Statement,
        p2: Statement,
    ) -> Tuple[str, TruthValue]:
        """
        Induction: (M --> P, M --> S) |- S --> P

        "If birds fly and birds have wings, then things with wings fly."
        """
        if p1.subject == p2.subject:
            # M --> P, M --> S  =>  S --> P
            conclusion = f"{p2.predicate} {Copula.INHERITANCE.value} {p1.predicate}"
            truth = induction_truth(p1.truth, p2.truth)
            self.inference_trace.append(
                f"Induction: {p1} + {p2} => {conclusion} {truth}"
            )
            return (conclusion, truth)

        return ("", TruthValue(0.5, 0.0))

    def _abduction(
        self,
        p1: Statement,
        p2: Statement,
    ) -> Tuple[str, TruthValue]:
        """
        Abduction: (P --> M, S --> M) |- S --> P

        "If birds fly and penguins fly (supposedly), then penguins are birds."
        """
        if p1.predicate == p2.predicate:
            # P --> M, S --> M  =>  S --> P
            conclusion = f"{p2.subject} {Copula.INHERITANCE.value} {p1.subject}"
            truth = abduction_truth(p1.truth, p2.truth)
            self.inference_trace.append(
                f"Abduction: {p1} + {p2} => {conclusion} {truth}"
            )
            return (conclusion, truth)

        return ("", TruthValue(0.5, 0.0))

    def _revision(
        self,
        p1: Statement,
        p2: Statement,
    ) -> Tuple[str, TruthValue]:
        """
        Revision: Combine evidence for same statement.
        """
        if (p1.subject == p2.subject and
            p1.predicate == p2.predicate and
            p1.copula == p2.copula):

            truth = revision_truth(p1.truth, p2.truth)
            conclusion = f"{p1.subject} {p1.copula.value} {p1.predicate}"
            self.inference_trace.append(
                f"Revision: {p1.truth} + {p2.truth} => {truth}"
            )
            return (conclusion, truth)

        return ("", TruthValue(0.5, 0.0))

    def chain_inference(
        self,
        premises: List[str],
        max_steps: int = 10,
    ) -> List[Tuple[str, TruthValue]]:
        """
        Perform multi-step inference chain.

        Tries to derive new conclusions from premises and derived facts.

        Args:
            premises: Initial premise strings
            max_steps: Maximum inference steps

        Returns:
            List of (conclusion, truth) pairs derived
        """
        # Initialize with premises
        for p in premises:
            stmt = Statement.parse(p, TruthValue(0.9, 0.9))
            self.add_belief(stmt)

        conclusions = []
        statements = list(self.beliefs.values())

        for step in range(max_steps):
            new_found = False

            # Try all pairs
            for i, s1 in enumerate(statements):
                for j, s2 in enumerate(statements):
                    if i >= j:
                        continue

                    # Try each rule
                    for rule in [InferenceRule.DEDUCTION,
                                 InferenceRule.INDUCTION,
                                 InferenceRule.ABDUCTION]:

                        concl, truth = self.infer(
                            [str(s1), str(s2)],
                            rule.value
                        )

                        if concl and truth.confidence > 0.1:
                            # Check if this is new
                            key = concl.replace(" ", "_")
                            if key not in self.beliefs:
                                new_stmt = Statement.parse(concl, truth)
                                self.add_belief(new_stmt)
                                conclusions.append((concl, truth))
                                statements.append(new_stmt)
                                new_found = True

            if not new_found:
                break

        return conclusions

    def get_trace(self) -> List[str]:
        """Get inference trace."""
        return self.inference_trace
