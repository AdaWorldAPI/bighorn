#!/usr/bin/env python3
"""
dreamer_pandas.py — The Pandas Dreamer (Sleep Cycle)
═══════════════════════════════════════════════════════════════════════════════

Offline consolidation: discovers golden patterns during sleep.

Architecture (No DuckDB Required):
    LanceDB (Arrow) → Pandas DataFrame → Aggregation → Golden Patterns

The SQL Cortex is replaced with a Pandas Motor:
- Zero-copy (or near zero-copy) from LanceDB Arrow tables
- Pattern aggregation via groupby
- Golden pattern discovery (frequent + high resonance)

This is the "Dreamer" — it runs during idle cycles to:
1. Load thought history from LanceDB
2. Find recurring successful sequences
3. Crystallize them into new τ macros (Autopoiesis)

Born: 2026-01-03
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Pandas for aggregation
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Optional numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Optional LanceDB
try:
    import lancedb
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False
    lancedb = None


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN PATTERN — A pattern worth crystallizing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoldenPattern:
    """
    A pattern discovered during dreaming that's worth crystallizing.

    Golden = Frequent + High Resonance + Successful Outcome
    """
    sequence: str                    # The microcode sequence
    count: int                       # How many times seen
    avg_resonance: float            # Average resonance during execution
    success_rate: float             # Fraction of successful outcomes
    first_seen: float               # Timestamp
    last_seen: float                # Timestamp
    source_thoughts: List[str] = field(default_factory=list)  # Thought IDs

    @property
    def golden_score(self) -> float:
        """
        Calculate how 'golden' this pattern is.

        Golden = log(count) × avg_resonance × success_rate
        """
        import math
        count_factor = math.log(max(self.count, 1) + 1)
        return count_factor * self.avg_resonance * self.success_rate

    def to_microcode(self, address: int = 0xE0) -> Dict[str, Any]:
        """Convert to τ macro format for crystallization."""
        return {
            "address": address,
            "byte": 1,  # Hot (crystallized from experimental)
            "name": f"AUTO_{address:02X}",
            "microcode": self.sequence,
            "chain": self.sequence.split("→"),
            "metadata": {
                "count": self.count,
                "avg_resonance": self.avg_resonance,
                "success_rate": self.success_rate,
                "golden_score": self.golden_score,
                "discovered_at": time.time()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THOUGHT RECORD — What we store for dreaming
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThoughtRecord:
    """
    A record of a thought for the dreamer to analyze.

    This is what gets stored in LanceDB thought_history table.
    """
    id: str
    timestamp: float
    microcode_sequence: str          # "feel→resonate→decide"
    outcome: str                     # "success" | "failure" | "partial"
    resonance: float                 # Peak resonance during execution
    free_will_score: float          # MUL agency score
    trajectory: str                  # "CORNER→EDGE→INTERIOR→FLOW"

    # Optional vector for VSA matching
    vector: Optional[List[float]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# PANDAS DREAMER — The Sleep Cycle Engine
# ═══════════════════════════════════════════════════════════════════════════════

class PandasDreamer:
    """
    The Pandas Dream Loop.

    Replaces DuckDB SQL with Pandas aggregations.
    Discovers golden patterns from thought history.
    """

    # Thresholds for golden patterns
    MIN_COUNT = 3                    # Seen at least 3 times
    MIN_RESONANCE = 0.7             # Average resonance > 0.7
    MIN_SUCCESS_RATE = 0.6          # Success > 60%
    MAX_PATTERNS_PER_DREAM = 10     # Don't overwhelm with discoveries

    def __init__(self, lance_uri: Optional[str] = None):
        """
        Initialize the Pandas Dreamer.

        Args:
            lance_uri: Path to LanceDB database (optional)
        """
        self.lance_uri = lance_uri
        self.db = None
        self._init_db()

        # In-memory history (used if no LanceDB)
        self.memory_history: List[ThoughtRecord] = []

        # Discovered patterns
        self.golden_patterns: Dict[str, GoldenPattern] = {}

        # Dream statistics
        self.dream_count = 0
        self.total_patterns_discovered = 0
        self.last_dream_time: Optional[float] = None

    def _init_db(self):
        """Initialize LanceDB connection."""
        if HAS_LANCEDB and self.lance_uri:
            try:
                self.db = lancedb.connect(self.lance_uri)
            except Exception as e:
                print(f"⚠ LanceDB init failed: {e}")
                self.db = None

    def record(self, thought: ThoughtRecord):
        """
        Record a thought for later dreaming.

        Args:
            thought: ThoughtRecord to store
        """
        # Store in memory
        self.memory_history.append(thought)

        # Limit memory size
        if len(self.memory_history) > 10000:
            self.memory_history = self.memory_history[-10000:]

        # Store in LanceDB if available
        if self.db:
            try:
                table_name = "thought_history"

                data = {
                    "id": thought.id,
                    "timestamp": thought.timestamp,
                    "microcode_sequence": thought.microcode_sequence,
                    "outcome": thought.outcome,
                    "resonance": thought.resonance,
                    "free_will_score": thought.free_will_score,
                    "trajectory": thought.trajectory
                }

                if thought.vector:
                    data["vector"] = thought.vector

                # Append to table (create if not exists)
                if table_name in self.db.table_names():
                    table = self.db.open_table(table_name)
                    table.add([data])
                else:
                    self.db.create_table(table_name, [data])

            except Exception as e:
                print(f"⚠ LanceDB record failed: {e}")

    def _load_history(self, limit: int = 10000) -> pd.DataFrame:
        """
        Load thought history into Pandas DataFrame.

        Fast Arrow load from LanceDB, or use in-memory history.
        """
        if not HAS_PANDAS:
            return None

        # Try LanceDB first
        if self.db and "thought_history" in self.db.table_names():
            try:
                table = self.db.open_table("thought_history")
                # Use limit and to_pandas for zero-copy
                df = table.search().limit(limit).to_pandas()
                return df
            except Exception as e:
                print(f"⚠ LanceDB load failed: {e}")

        # Fall back to in-memory
        if self.memory_history:
            records = [
                {
                    "id": t.id,
                    "timestamp": t.timestamp,
                    "microcode_sequence": t.microcode_sequence,
                    "outcome": t.outcome,
                    "resonance": t.resonance,
                    "free_will_score": t.free_will_score,
                    "trajectory": t.trajectory
                }
                for t in self.memory_history[-limit:]
            ]
            return pd.DataFrame(records)

        return pd.DataFrame()

    def dream(self) -> List[GoldenPattern]:
        """
        The Pandas Dream Loop.

        Discovers golden patterns from thought history.
        This is the "SQL Group By" replacement.

        Returns:
            List of newly discovered golden patterns
        """
        if not HAS_PANDAS:
            print("⚠ Pandas not available — dreaming offline")
            return []

        self.dream_count += 1
        self.last_dream_time = time.time()

        print(f"💤 Dream #{self.dream_count} starting...")

        # 1. Load History (Fast Arrow Load)
        df = self._load_history()

        if df is None or df.empty:
            print("   No thoughts to dream about")
            return []

        print(f"   Loaded {len(df)} thoughts")

        # 2. Filter for Success
        success_df = df[df["outcome"] == "success"]
        print(f"   {len(success_df)} successful thoughts")

        if success_df.empty:
            return []

        # 3. Aggregation (The "SQL Group By" replacement)
        # We look for recurring sequences that led to success
        patterns = (
            success_df.groupby("microcode_sequence")
            .agg(
                count=("id", "count"),
                avg_resonance=("resonance", "mean"),
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max"),
                thoughts=("id", list)
            )
            .reset_index()
        )

        # 4. Filter for "Golden Patterns"
        # Golden = Frequent + High Resonance
        golden = patterns[
            (patterns["count"] >= self.MIN_COUNT) &
            (patterns["avg_resonance"] >= self.MIN_RESONANCE)
        ]

        # Calculate success rate (need to rejoin)
        all_patterns = (
            df.groupby("microcode_sequence")
            .agg(total=("id", "count"))
            .reset_index()
        )

        golden = golden.merge(all_patterns, on="microcode_sequence")
        golden["success_rate"] = golden["count"] / golden["total"]

        # Filter by success rate
        golden = golden[golden["success_rate"] >= self.MIN_SUCCESS_RATE]

        # Sort by golden score proxy (count × resonance × success_rate)
        golden["golden_score"] = (
            golden["count"].apply(lambda x: np.log(x + 1) if HAS_NUMPY else x) *
            golden["avg_resonance"] *
            golden["success_rate"]
        )
        golden = golden.sort_values("golden_score", ascending=False)

        # Limit discoveries
        golden = golden.head(self.MAX_PATTERNS_PER_DREAM)

        # 5. Convert to GoldenPattern objects
        new_patterns = []

        for _, row in golden.iterrows():
            sequence = row["microcode_sequence"]

            # Skip if already discovered
            if sequence in self.golden_patterns:
                # Update existing
                existing = self.golden_patterns[sequence]
                existing.count = row["count"]
                existing.avg_resonance = row["avg_resonance"]
                existing.last_seen = row["last_seen"]
                existing.success_rate = row["success_rate"]
                continue

            # New golden pattern!
            pattern = GoldenPattern(
                sequence=sequence,
                count=int(row["count"]),
                avg_resonance=float(row["avg_resonance"]),
                success_rate=float(row["success_rate"]),
                first_seen=float(row["first_seen"]),
                last_seen=float(row["last_seen"]),
                source_thoughts=row["thoughts"][:10]  # Keep first 10 IDs
            )

            self.golden_patterns[sequence] = pattern
            new_patterns.append(pattern)
            self.total_patterns_discovered += 1

            print(f"   ✨ NEW PATTERN: {sequence[:50]}...")
            print(f"      count={pattern.count}, resonance={pattern.avg_resonance:.3f}")

        print(f"💤 Dream complete: {len(new_patterns)} new patterns")
        return new_patterns

    def get_crystallization_candidates(self, limit: int = 5) -> List[Tuple[GoldenPattern, Dict]]:
        """
        Get patterns ready for crystallization into τ macros.

        Returns patterns with highest golden scores.
        """
        if not self.golden_patterns:
            return []

        # Sort by golden score
        sorted_patterns = sorted(
            self.golden_patterns.values(),
            key=lambda p: p.golden_score,
            reverse=True
        )[:limit]

        # Generate macro definitions
        candidates = []
        base_address = 0xE0

        for i, pattern in enumerate(sorted_patterns):
            address = base_address + i
            macro_def = pattern.to_microcode(address)
            candidates.append((pattern, macro_def))

        return candidates

    def get_stats(self) -> Dict[str, Any]:
        """Get dreaming statistics."""
        return {
            "dream_count": self.dream_count,
            "total_patterns_discovered": self.total_patterns_discovered,
            "current_patterns": len(self.golden_patterns),
            "memory_history_size": len(self.memory_history),
            "last_dream_time": self.last_dream_time,
            "top_patterns": [
                {
                    "sequence": p.sequence[:40],
                    "golden_score": p.golden_score
                }
                for p in sorted(
                    self.golden_patterns.values(),
                    key=lambda x: x.golden_score,
                    reverse=True
                )[:5]
            ]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED DREAMER — For testing without LanceDB
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedDreamer(PandasDreamer):
    """
    A dreamer with simulated history for testing.
    """

    def populate_test_data(self, n_thoughts: int = 500):
        """Populate with simulated thought history."""
        import random

        # Predefined sequences (some will become golden)
        sequences = [
            "feel→resonate→decide",
            "observe→analyze→conclude",
            "sense→attune→embrace",
            "feel→trust→act",
            "wonder→leap→discover",
            "disrupt→create→become",
            "feel→resonate→decide→act",
            "random→noise→chaos",
        ]

        # Bias some sequences toward success
        success_bias = {
            "feel→resonate→decide": 0.85,
            "observe→analyze→conclude": 0.75,
            "sense→attune→embrace": 0.80,
            "random→noise→chaos": 0.20,
        }

        for i in range(n_thoughts):
            seq = random.choice(sequences)
            success_chance = success_bias.get(seq, 0.50)

            # Resonance correlates with success
            if random.random() < success_chance:
                outcome = "success"
                resonance = random.uniform(0.6, 0.95)
            else:
                outcome = "failure"
                resonance = random.uniform(0.2, 0.5)

            thought = ThoughtRecord(
                id=f"test_{i:05d}",
                timestamp=time.time() - random.uniform(0, 86400),  # Last 24h
                microcode_sequence=seq,
                outcome=outcome,
                resonance=resonance,
                free_will_score=random.uniform(0.4, 0.9),
                trajectory=random.choice(["CORNER→EDGE→INTERIOR", "EDGE→INTERIOR→FLOW", "CORNER→FLOW"])
            )

            self.record(thought)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_dreamer():
    """Test the Pandas Dreamer."""
    print("=" * 60)
    print("PANDAS DREAMER TEST — Sleep Cycle Pattern Discovery")
    print("=" * 60)

    if not HAS_PANDAS:
        print("⚠ Pandas not available — cannot run test")
        return

    # Create simulated dreamer
    dreamer = SimulatedDreamer()

    print("\n1. Populating test data...")
    dreamer.populate_test_data(n_thoughts=500)
    print(f"   Recorded {len(dreamer.memory_history)} thoughts")

    print("\n2. Running dream cycle...")
    new_patterns = dreamer.dream()

    print(f"\n3. Results:")
    print(f"   New patterns discovered: {len(new_patterns)}")

    for pattern in new_patterns[:5]:
        print(f"\n   Pattern: {pattern.sequence}")
        print(f"      Count: {pattern.count}")
        print(f"      Avg Resonance: {pattern.avg_resonance:.3f}")
        print(f"      Success Rate: {pattern.success_rate:.3f}")
        print(f"      Golden Score: {pattern.golden_score:.3f}")

    print("\n4. Crystallization candidates:")
    candidates = dreamer.get_crystallization_candidates(limit=3)

    for pattern, macro in candidates:
        print(f"\n   τ Macro: {macro['name']}")
        print(f"      Address: {hex(macro['address'])}")
        print(f"      Microcode: {macro['microcode']}")
        print(f"      Golden Score: {pattern.golden_score:.3f}")

    print("\n5. Dreamer stats:")
    stats = dreamer.get_stats()
    for k, v in stats.items():
        if k != "top_patterns":
            print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("DREAMER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_dreamer()
