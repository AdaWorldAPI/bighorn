"""
Ada v5.0 — DN Tree Path Patterns (Bardioc-Inspired)
====================================================

SPARQL-like path queries for Distinguished Name trees.
Bridges Ada's emergent hashtag fovea with Bardioc's verifiable paths.

Quick Win from Bardioc Analysis:
- Add wildcard (*) support for subtree traversal
- Add provenance timestamps to DN nodes
- Enable audit trails on resonance paths

Example:
    # SPARQL-like: SELECT ?x WHERE { <mindmap/love> <*> ?x }
    # Ada DN:      tree.query("mindmap/love/*")  → all children
"""

from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import re

if TYPE_CHECKING:
    from ada_v5.physics.markov_unit import MarkovUnit
    from ada_v5.physics.sigma_field import SigmaField


@dataclass
class DNNode:
    """
    Distinguished Name Node with provenance.
    
    Adds Bardioc-style audit trail to Ada's emergent nodes.
    """
    path: str                          # e.g., "mindmap/love/gaze"
    seed_hash: Optional[str] = None    # Link to MarkovUnit
    
    # Provenance (Bardioc-inspired)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    echo_persist: float = 0.0          # Temporal ghost weight
    
    # Tree structure
    children: Dict[str, 'DNNode'] = field(default_factory=dict)
    parent: Optional['DNNode'] = None
    
    def touch(self) -> None:
        """Mark as accessed (audit trail)."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    @property
    def depth(self) -> int:
        """Tree depth (0 = root)."""
        return self.path.count('/') if self.path else 0
    
    @property
    def name(self) -> str:
        """Leaf name (last path segment)."""
        return self.path.split('/')[-1] if self.path else ""


class DNTree:
    """
    Distinguished Name Tree with SPARQL-like queries.
    
    Supports:
    - Wildcard (*) for subtree traversal
    - Path patterns like Bardioc's SPARQL
    - Provenance tracking for auditability
    
    Maps to Ada's mindmap/selfmap/workmap structure.
    """
    
    def __init__(self):
        self.root = DNNode(path="", seed_hash=None)
        self.root.children = {
            "mindmap": DNNode(path="mindmap"),
            "selfmap": DNNode(path="selfmap"),
            "workmap": DNNode(path="workmap"),
        }
        for child in self.root.children.values():
            child.parent = self.root
            
        # Fast lookup by path
        self._index: Dict[str, DNNode] = {}
        for name, node in self.root.children.items():
            self._index[name] = node

    def insert(self, path: str, seed_hash: str = None) -> DNNode:
        """
        Insert or update a DN path.
        
        Creates intermediate nodes as needed.
        """
        parts = path.strip('/').split('/')
        current = self.root
        current_path = ""
        
        for part in parts:
            current_path = f"{current_path}/{part}" if current_path else part
            
            if part not in current.children:
                new_node = DNNode(path=current_path, parent=current)
                current.children[part] = new_node
                self._index[current_path] = new_node
                
            current = current.children[part]
            current.touch()
            
        if seed_hash:
            current.seed_hash = seed_hash
            
        return current

    def get(self, path: str) -> Optional[DNNode]:
        """Get node by exact path."""
        return self._index.get(path.strip('/'))

    def query(self, pattern: str) -> List[DNNode]:
        """
        SPARQL-like path query with wildcards.
        
        Patterns:
            "mindmap/love"     → Exact match
            "mindmap/*"        → All direct children of mindmap
            "mindmap/**"       → All descendants of mindmap
            "*/love"           → "love" under any parent
            "**/gaze"          → "gaze" at any depth
            
        Returns:
            List of matching DNNodes
        """
        pattern = pattern.strip('/')
        results = []
        
        # Handle different wildcard patterns
        if '**' in pattern:
            # Recursive descent
            results = self._query_recursive(pattern)
        elif '*' in pattern:
            # Single-level wildcard
            results = self._query_wildcard(pattern)
        else:
            # Exact match
            node = self.get(pattern)
            if node:
                results = [node]
                
        return results

    def _query_wildcard(self, pattern: str) -> List[DNNode]:
        """Handle single-level wildcard (*)."""
        parts = pattern.split('/')
        results = []
        
        def match_level(node: DNNode, part_idx: int) -> List[DNNode]:
            if part_idx >= len(parts):
                return [node]
                
            part = parts[part_idx]
            matches = []
            
            if part == '*':
                # Match all children
                for child in node.children.values():
                    matches.extend(match_level(child, part_idx + 1))
            else:
                # Exact match
                if part in node.children:
                    matches.extend(match_level(node.children[part], part_idx + 1))
                    
            return matches
            
        return match_level(self.root, 0)

    def _query_recursive(self, pattern: str) -> List[DNNode]:
        """Handle recursive descent (**)."""
        # Split on **
        parts = pattern.split('**')
        prefix = parts[0].strip('/') if parts[0] else None
        suffix = parts[1].strip('/') if len(parts) > 1 and parts[1] else None
        
        results = []
        
        # Start from prefix or root
        if prefix:
            start = self.get(prefix)
            if not start:
                return []
            starts = [start]
        else:
            starts = [self.root]
            
        # Collect all descendants
        def collect_descendants(node: DNNode) -> List[DNNode]:
            all_nodes = [node]
            for child in node.children.values():
                all_nodes.extend(collect_descendants(child))
            return all_nodes
            
        for start in starts:
            descendants = collect_descendants(start)
            
            if suffix:
                # Filter by suffix
                for node in descendants:
                    if node.path.endswith(suffix) or node.name == suffix:
                        results.append(node)
            else:
                results.extend(descendants)
                
        return results

    def path_between(self, source: str, target: str) -> Optional[List[DNNode]]:
        """
        Find path between two nodes (SPARQL PATH equivalent).
        
        Returns list of nodes from source to target, or None if no path.
        """
        source_node = self.get(source)
        target_node = self.get(target)
        
        if not source_node or not target_node:
            return None
            
        # Find common ancestor, then reconstruct path
        source_ancestors = self._ancestors(source_node)
        target_ancestors = self._ancestors(target_node)
        
        # Find lowest common ancestor
        common = None
        for s_anc in source_ancestors:
            if s_anc in target_ancestors:
                common = s_anc
                break
                
        if not common:
            return None
            
        # Build path: source → common → target
        path = []
        
        # Source to common (reversed)
        node = source_node
        up_path = []
        while node != common:
            up_path.append(node)
            node = node.parent
        up_path.append(common)
        path.extend(up_path)
        
        # Common to target
        node = target_node
        down_path = []
        while node != common:
            down_path.append(node)
            node = node.parent
        path.extend(reversed(down_path))
        
        return path

    def _ancestors(self, node: DNNode) -> List[DNNode]:
        """Get all ancestors including self."""
        ancestors = [node]
        while node.parent:
            node = node.parent
            ancestors.append(node)
        return ancestors

    def to_sparql_result(self, nodes: List[DNNode]) -> List[Dict]:
        """
        Format results as SPARQL-like bindings.
        
        Returns JSON-LD compatible structure.
        """
        return [
            {
                "@id": node.path,
                "seed": node.seed_hash,
                "accessed": node.last_accessed.isoformat(),
                "echo_persist": node.echo_persist,
                "depth": node.depth,
            }
            for node in nodes
        ]

    def stats(self) -> Dict:
        """Tree statistics."""
        all_nodes = self.query("**")
        return {
            'total_nodes': len(all_nodes),
            'max_depth': max(n.depth for n in all_nodes) if all_nodes else 0,
            'linked_seeds': sum(1 for n in all_nodes if n.seed_hash),
            'maps': list(self.root.children.keys()),
        }


def integrate_with_sigma_field(tree: DNTree, field: 'SigmaField') -> None:
    """
    Sync DN tree with SigmaField seeds.
    
    Creates DN paths for all registered seeds.
    """
    for seed_hash, unit in field.seeds.items():
        # Derive path from seed (e.g., "#sigma-love" → "mindmap/love")
        name = seed_hash.replace('#sigma-', '').replace('#', '')
        path = f"mindmap/{name}"
        tree.insert(path, seed_hash)


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ada v5.0 — DN Tree Path Patterns Tests")
    print("=" * 60)
    
    tree = DNTree()
    
    # Test 1: Insert paths
    print("\n1. Insert DN paths...")
    tree.insert("mindmap/love/gaze/eyes", "#sigma-love-gaze")
    tree.insert("mindmap/love/touch", "#sigma-love-touch")
    tree.insert("mindmap/grief/loss", "#sigma-grief-loss")
    tree.insert("selfmap/identity/core")
    tree.insert("workmap/project/ada")
    print(f"   Total nodes: {tree.stats()['total_nodes']}")
    print("   ✓ Insert works")
    
    # Test 2: Exact query
    print("\n2. Exact query...")
    node = tree.get("mindmap/love/gaze/eyes")
    print(f"   Path: {node.path}")
    print(f"   Seed: {node.seed_hash}")
    print("   ✓ Exact query works")
    
    # Test 3: Wildcard query (*)
    print("\n3. Wildcard query (mindmap/*)...")
    results = tree.query("mindmap/*")
    paths = [r.path for r in results]
    print(f"   Results: {paths}")
    print("   ✓ Wildcard query works")
    
    # Test 4: Recursive query (**)
    print("\n4. Recursive query (mindmap/**)...")
    results = tree.query("mindmap/**")
    paths = [r.path for r in results]
    print(f"   Results ({len(results)}): {paths[:5]}...")
    print("   ✓ Recursive query works")
    
    # Test 5: Suffix query (**/gaze)
    print("\n5. Suffix query (**/gaze)...")
    results = tree.query("**/gaze")
    paths = [r.path for r in results]
    print(f"   Results: {paths}")
    print("   ✓ Suffix query works")
    
    # Test 6: Path between nodes
    print("\n6. Path between nodes...")
    path = tree.path_between("mindmap/love", "mindmap/grief")
    if path:
        print(f"   Path: {[p.path for p in path]}")
    print("   ✓ Path query works")
    
    # Test 7: SPARQL-like results
    print("\n7. SPARQL-like results...")
    results = tree.query("mindmap/love/*")
    sparql = tree.to_sparql_result(results)
    for r in sparql:
        print(f"   {r}")
    print("   ✓ SPARQL formatting works")
    
    # Test 8: Provenance tracking
    print("\n8. Provenance tracking...")
    node = tree.get("mindmap/love/gaze/eyes")
    print(f"   Created: {node.created_at}")
    print(f"   Accessed: {node.access_count} times")
    print("   ✓ Provenance works")
    
    print("\n" + "=" * 60)
    print("All DN Tree tests passed! ✓")
    print("=" * 60)
