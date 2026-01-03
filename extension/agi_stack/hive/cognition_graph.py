#!/usr/bin/env python3
"""
cognition_graph.py — Self-Organizing Architecture Discovery
═══════════════════════════════════════════════════════════════════════════════

The architecture understands itself.
No LLM required for structural self-awareness.

What it does:
  1. Takes code chunks encoded to 10kD
  2. Builds similarity matrix (cosine similarity)
  3. Finds natural clusters (connected components, hierarchical)
  4. Discovers duplicates, orphans, merge candidates
  5. Outputs: "Here are 4 natural modules, 2 orphans, 3 duplicates"

YOU decide what to merge/delete/keep.
HIVE finds the structure.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import numpy as np

from .hive_code_chunker import HiveCodeChunker, CodeChunk, ChunkType
from .code_schema_10k import CodeSchemaEncoder


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArchitectureCluster:
    """
    A natural grouping of code chunks discovered by similarity.

    This is an EMERGENT module boundary, not a declared one.
    """
    id: str
    name: str                                    # Inferred from dominant features
    chunks: List[CodeChunk] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None       # Average vector

    # What makes this cluster coherent
    common_imports: List[str] = field(default_factory=list)
    common_domains: List[str] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)

    # Relationships
    depends_on: List[str] = field(default_factory=list)    # Other cluster IDs
    used_by: List[str] = field(default_factory=list)       # Other cluster IDs

    # Quality
    cohesion: float = 0.0        # Internal similarity (higher = better)
    coupling: float = 0.0        # External dependencies (lower = better)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "size": len(self.chunks),
            "chunks": [c.id for c in self.chunks],
            "common_imports": self.common_imports,
            "common_domains": self.common_domains,
            "cohesion": self.cohesion,
            "coupling": self.coupling,
            "depends_on": self.depends_on,
            "used_by": self.used_by,
        }


@dataclass
class MergeCandidate:
    """Two chunks that might be duplicates or should be merged."""
    chunk_a: CodeChunk
    chunk_b: CodeChunk
    similarity: float
    reason: str          # "duplicate", "overlap", "similar_purpose"


@dataclass
class CognitionGraphResult:
    """Results from self-organizing architecture analysis."""
    clusters: List[ArchitectureCluster]
    duplicates: List[MergeCandidate]
    orphans: List[CodeChunk]
    similarity_matrix: Optional[np.ndarray] = None

    # Recommendations
    merge_suggestions: List[Tuple[str, str, str]] = field(default_factory=list)  # (a, b, reason)
    delete_suggestions: List[str] = field(default_factory=list)
    refactor_suggestions: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITION GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class CognitionGraph:
    """
    Self-organizing architecture discovery.

    Takes encoded chunks and finds natural structure.
    No LLM required.
    """

    # Thresholds
    SIMILARITY_THRESHOLD = 0.7      # Min similarity to be in same cluster
    DUPLICATE_THRESHOLD = 0.95      # Above this = likely duplicate
    MERGE_THRESHOLD = 0.85          # Above this = consider merging

    def __init__(
        self,
        chunker: Optional[HiveCodeChunker] = None,
        encoder: Optional[CodeSchemaEncoder] = None
    ):
        self.chunker = chunker or HiveCodeChunker()
        self.encoder = encoder or CodeSchemaEncoder()

        self.chunks: List[CodeChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None

    def analyze_directory(self, directory: str, pattern: str = "**/*.py") -> CognitionGraphResult:
        """
        Analyze a directory and discover its architecture.

        Returns clusters, duplicates, orphans, and suggestions.
        """
        print(f"🔍 Analyzing {directory}...")

        # 1. Chunk all files
        print("  Step 1: Parsing code...")
        self.chunks = self.chunker.chunk_directory(directory, pattern)
        print(f"    Found {len(self.chunks)} chunks")

        if not self.chunks:
            return CognitionGraphResult(clusters=[], duplicates=[], orphans=[])

        # 2. Encode to 10kD
        print("  Step 2: Encoding to 10kD...")
        self.vectors = self.encoder.encode_batch(self.chunks)
        print(f"    Encoded {len(self.vectors)} vectors")

        # 3. Build similarity matrix
        print("  Step 3: Building similarity matrix...")
        self.similarity_matrix = self._build_similarity_matrix()

        # 4. Find clusters
        print("  Step 4: Finding clusters...")
        clusters = self._find_clusters()
        print(f"    Found {len(clusters)} clusters")

        # 5. Find duplicates
        print("  Step 5: Finding duplicates...")
        duplicates = self._find_duplicates()
        print(f"    Found {len(duplicates)} potential duplicates")

        # 6. Find orphans
        print("  Step 6: Finding orphans...")
        orphans = self._find_orphans()
        print(f"    Found {len(orphans)} orphans")

        # 7. Generate suggestions
        print("  Step 7: Generating suggestions...")
        result = CognitionGraphResult(
            clusters=clusters,
            duplicates=duplicates,
            orphans=orphans,
            similarity_matrix=self.similarity_matrix,
        )
        self._generate_suggestions(result)

        print(f"✅ Analysis complete!")
        return result

    def _build_similarity_matrix(self) -> np.ndarray:
        """Build pairwise cosine similarity matrix."""
        n = len(self.vectors)
        sim = np.zeros((n, n), dtype=np.float32)

        # Normalize vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = self.vectors / norms

        # Cosine similarity = dot product of normalized vectors
        sim = normalized @ normalized.T

        return sim

    def _find_clusters(self) -> List[ArchitectureCluster]:
        """
        Find natural clusters using connected components on similarity graph.
        """
        n = len(self.chunks)

        # Build adjacency from similarity
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                if self.similarity_matrix[i, j] >= self.SIMILARITY_THRESHOLD:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Find connected components (simple BFS)
        visited = set()
        components: List[List[int]] = []

        for start in range(n):
            if start in visited:
                continue

            # BFS from this node
            component = []
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if component:
                components.append(component)

        # Convert to clusters
        clusters = []
        for i, component in enumerate(components):
            chunk_list = [self.chunks[idx] for idx in component]
            cluster = self._build_cluster(f"cluster_{i}", chunk_list)
            clusters.append(cluster)

        # Sort by size
        clusters.sort(key=lambda c: len(c.chunks), reverse=True)

        return clusters

    def _build_cluster(self, cluster_id: str, chunks: List[CodeChunk]) -> ArchitectureCluster:
        """Build a cluster from a list of chunks."""
        # Find common imports
        import_counts: Dict[str, int] = defaultdict(int)
        for chunk in chunks:
            for imp in chunk.imports:
                import_counts[imp] += 1

        # Imports that appear in >50% of chunks
        threshold = len(chunks) / 2
        common_imports = [imp for imp, count in import_counts.items() if count >= threshold]

        # Infer name from common features
        name = self._infer_cluster_name(chunks, common_imports)

        # Compute centroid
        indices = [self.chunks.index(c) for c in chunks]
        vectors = self.vectors[indices]
        centroid = np.mean(vectors, axis=0) if len(vectors) > 0 else None

        # Compute cohesion (average internal similarity)
        if len(indices) > 1:
            internal_sims = []
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i+1:]:
                    internal_sims.append(self.similarity_matrix[idx_i, idx_j])
            cohesion = np.mean(internal_sims) if internal_sims else 0.0
        else:
            cohesion = 1.0

        return ArchitectureCluster(
            id=cluster_id,
            name=name,
            chunks=chunks,
            centroid=centroid,
            common_imports=common_imports[:10],
            cohesion=float(cohesion),
        )

    def _infer_cluster_name(self, chunks: List[CodeChunk], common_imports: List[str]) -> str:
        """Infer a name for the cluster based on its contents."""
        # Count words in chunk names
        word_counts: Dict[str, int] = defaultdict(int)

        for chunk in chunks:
            # Split name into words
            name = chunk.name.lower()
            name = name.replace('_', ' ').replace('.', ' ')
            words = name.split()

            for word in words:
                if len(word) > 2:  # Skip short words
                    word_counts[word] += 1

        # Also check common imports for domain hints
        for imp in common_imports:
            parts = imp.lower().split('.')
            for part in parts:
                if len(part) > 2:
                    word_counts[part] += 1

        # Most common word becomes the name
        if word_counts:
            top_word = max(word_counts.items(), key=lambda x: x[1])[0]
            return f"{top_word}_cluster"

        return "unnamed_cluster"

    def _find_duplicates(self) -> List[MergeCandidate]:
        """Find chunks that are near-duplicates."""
        duplicates = []
        n = len(self.chunks)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity_matrix[i, j]

                if sim >= self.DUPLICATE_THRESHOLD:
                    duplicates.append(MergeCandidate(
                        chunk_a=self.chunks[i],
                        chunk_b=self.chunks[j],
                        similarity=float(sim),
                        reason="duplicate"
                    ))
                elif sim >= self.MERGE_THRESHOLD:
                    # Check if they're in different files (merge candidate)
                    if self.chunks[i].file_path != self.chunks[j].file_path:
                        duplicates.append(MergeCandidate(
                            chunk_a=self.chunks[i],
                            chunk_b=self.chunks[j],
                            similarity=float(sim),
                            reason="similar_purpose"
                        ))

        # Sort by similarity
        duplicates.sort(key=lambda d: d.similarity, reverse=True)
        return duplicates

    def _find_orphans(self) -> List[CodeChunk]:
        """Find chunks with very low similarity to everything else."""
        orphans = []
        n = len(self.chunks)

        for i in range(n):
            chunk = self.chunks[i]

            # Skip modules and classes (they're containers)
            if chunk.chunk_type in (ChunkType.MODULE, ChunkType.CLASS):
                continue

            # Max similarity to any other chunk
            max_sim = 0.0
            for j in range(n):
                if i != j:
                    max_sim = max(max_sim, self.similarity_matrix[i, j])

            # If max similarity is very low, it's an orphan
            if max_sim < 0.3:
                orphans.append(chunk)

        return orphans

    def _generate_suggestions(self, result: CognitionGraphResult):
        """Generate actionable suggestions."""
        # Merge suggestions from duplicates
        for dup in result.duplicates:
            result.merge_suggestions.append((
                dup.chunk_a.id,
                dup.chunk_b.id,
                f"{dup.reason} (sim={dup.similarity:.2f})"
            ))

        # Delete suggestions from orphans
        for orphan in result.orphans:
            result.delete_suggestions.append(
                f"{orphan.id} — never called, low similarity"
            )

        # Refactor suggestions from low-cohesion clusters
        for cluster in result.clusters:
            if cluster.cohesion < 0.5 and len(cluster.chunks) > 3:
                result.refactor_suggestions.append(
                    f"{cluster.name} — low cohesion ({cluster.cohesion:.2f}), consider splitting"
                )

    def print_report(self, result: CognitionGraphResult):
        """Print a human-readable report."""
        print("\n" + "=" * 70)
        print("COGNITION GRAPH REPORT — Self-Organizing Architecture")
        print("=" * 70)

        # Clusters
        print(f"\n📦 DISCOVERED CLUSTERS ({len(result.clusters)})")
        print("-" * 50)
        for cluster in result.clusters[:10]:
            print(f"\n  [{cluster.name}] ({len(cluster.chunks)} chunks)")
            print(f"      Cohesion: {cluster.cohesion:.2f}")
            print(f"      Common imports: {', '.join(cluster.common_imports[:5])}")
            print(f"      Chunks:")
            for chunk in cluster.chunks[:5]:
                print(f"        - {chunk.name}")
            if len(cluster.chunks) > 5:
                print(f"        ... and {len(cluster.chunks) - 5} more")

        # Duplicates
        if result.duplicates:
            print(f"\n⚠️  POTENTIAL DUPLICATES ({len(result.duplicates)})")
            print("-" * 50)
            for dup in result.duplicates[:10]:
                print(f"\n  {dup.chunk_a.name}")
                print(f"  ≈ {dup.chunk_b.name}")
                print(f"    Similarity: {dup.similarity:.2f} ({dup.reason})")

        # Orphans
        if result.orphans:
            print(f"\n🗑️  ORPHANS ({len(result.orphans)})")
            print("-" * 50)
            for orphan in result.orphans[:10]:
                print(f"  - {orphan.name} ({orphan.file_path})")

        # Suggestions
        print(f"\n💡 SUGGESTIONS")
        print("-" * 50)

        if result.merge_suggestions:
            print(f"\n  MERGE ({len(result.merge_suggestions)}):")
            for a, b, reason in result.merge_suggestions[:5]:
                print(f"    - {a.split(':')[-1]} + {b.split(':')[-1]}: {reason}")

        if result.delete_suggestions:
            print(f"\n  DELETE ({len(result.delete_suggestions)}):")
            for suggestion in result.delete_suggestions[:5]:
                print(f"    - {suggestion}")

        if result.refactor_suggestions:
            print(f"\n  REFACTOR ({len(result.refactor_suggestions)}):")
            for suggestion in result.refactor_suggestions[:5]:
                print(f"    - {suggestion}")

        print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_cognition_graph():
    """Test self-organizing architecture discovery."""
    print("=" * 60)
    print("COGNITION GRAPH TEST")
    print("=" * 60)

    graph = CognitionGraph()

    # Analyze the hive directory itself
    import os
    hive_dir = os.path.dirname(__file__)

    result = graph.analyze_directory(hive_dir)
    graph.print_report(result)

    print("\n" + "=" * 60)
    print("COGNITION GRAPH TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_cognition_graph()
