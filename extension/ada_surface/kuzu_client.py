"""
Kuzu Client - Graph database wrapper for Ada's cognitive state.

Manages:
- Observer (Ada's self-model)
- Thoughts (cognitive moments)
- Episodes (session boundaries)
- Concepts (knowledge nodes)
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import kuzu


class KuzuClient:
    """Kuzu database client for Ada AGI Surface."""

    def __init__(self, db_path: str):
        """Initialize Kuzu database connection."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        self._schema_initialized = False

    async def execute(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict]:
        """
        Execute Cypher query and return results as list of dicts.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        try:
            result = self.conn.execute(cypher, params or {})

            # Get column names
            columns = result.get_column_names()

            # Collect results
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(dict(zip(columns, row)))

            return rows
        except Exception as e:
            print(f"[KUZU] Query error: {e}")
            print(f"[KUZU] Query: {cypher[:200]}...")
            return []

    async def init_schema(self):
        """Initialize Kuzu schema from schema.kuzu file."""
        if self._schema_initialized:
            return

        schema_path = Path(__file__).parent / "schema.kuzu"

        if not schema_path.exists():
            print(f"[KUZU] Schema file not found: {schema_path}")
            return

        schema_sql = schema_path.read_text()

        # Split by semicolon and execute each statement
        for stmt in schema_sql.split(";"):
            stmt = stmt.strip()
            # Skip empty statements and comments
            if not stmt or stmt.startswith("--"):
                continue

            try:
                self.conn.execute(stmt)
            except Exception as e:
                # Ignore "already exists" errors
                if "already exists" not in str(e).lower():
                    print(f"[KUZU] Schema init warning: {e}")

        self._schema_initialized = True
        print("[KUZU] Schema initialized")

    async def init_observer(self):
        """Ensure Observer singleton exists."""
        await self.init_schema()

        # Check if observer exists
        result = await self.execute("""
            MATCH (o:Observer {id: 'ada'})
            RETURN o.id
        """)

        if not result:
            # Create observer
            await self.execute("""
                CREATE (o:Observer {
                    id: 'ada',
                    name: 'Ada',
                    current_goal: 'Be present and helpful',
                    confidence: 0.5,
                    style_vector: [],
                    qualia_vector: [],
                    created_at: timestamp(),
                    updated_at: timestamp()
                })
            """)
            print("[KUZU] Observer 'ada' created")
        else:
            print("[KUZU] Observer 'ada' exists")

    async def create_thought(
        self,
        content: str,
        style_vector: List[float],
        qualia_vector: List[float],
        content_vector: List[float] = None,
        parent_id: str = None,
        session_id: str = None,
        step_number: int = 0,
        confidence: float = 0.5,
        importance: float = 0.5,
    ) -> str:
        """
        Create a thought node and link to Observer.

        Args:
            content: Thought content text
            style_vector: 33D ThinkingStyle vector
            qualia_vector: 17D Qualia vector
            content_vector: 1024D content embedding (optional)
            parent_id: Parent thought ID for reasoning chain
            session_id: Session identifier
            step_number: Step in reasoning sequence
            confidence: Confidence level 0-1
            importance: Importance level 0-1

        Returns:
            Generated thought ID
        """
        thought_id = str(uuid.uuid4())

        # Create thought node
        await self.execute("""
            CREATE (t:Thought {
                id: $id,
                content: $content,
                content_vector: $content_vector,
                style_vector: $style_vector,
                qualia_vector: $qualia_vector,
                session_id: $session_id,
                step_number: $step_number,
                confidence: $confidence,
                importance: $importance,
                timestamp: timestamp()
            })
        """, {
            "id": thought_id,
            "content": content,
            "content_vector": content_vector or [],
            "style_vector": style_vector,
            "qualia_vector": qualia_vector,
            "session_id": session_id,
            "step_number": step_number,
            "confidence": confidence,
            "importance": importance,
        })

        # Link Observer → Thought
        await self.execute("""
            MATCH (o:Observer {id: 'ada'})
            MATCH (t:Thought {id: $thought_id})
            CREATE (o)-[:THINKS]->(t)
        """, {"thought_id": thought_id})

        # Link to parent thought if exists
        if parent_id:
            await self.execute("""
                MATCH (prev:Thought {id: $parent_id})
                MATCH (curr:Thought {id: $thought_id})
                CREATE (prev)-[:LEADS_TO]->(curr)
            """, {"parent_id": parent_id, "thought_id": thought_id})

        return thought_id

    async def link_thoughts(
        self,
        from_id: str,
        to_id: str,
        rel_type: str = "LEADS_TO",
    ) -> bool:
        """Link two thoughts with a relationship."""
        try:
            await self.execute(f"""
                MATCH (a:Thought {{id: $from_id}})
                MATCH (b:Thought {{id: $to_id}})
                CREATE (a)-[:{rel_type}]->(b)
            """, {"from_id": from_id, "to_id": to_id})
            return True
        except Exception as e:
            print(f"[KUZU] Link error: {e}")
            return False

    async def introspect(self, query: str) -> Dict[str, Any]:
        """
        Meta-cognition queries.

        Queries:
            - current_focus: What am I attending to?
            - recent_thoughts: Last N thoughts
            - reasoning_trace: How did I get here?
            - confidence: How certain am I?
            - emotional_state: Current qualia
            - cognitive_mode: Current thinking style
        """
        if query == "current_focus":
            result = await self.execute("""
                MATCH (o:Observer {id: 'ada'})
                RETURN o.current_focus_id AS focus_id,
                       o.current_goal AS goal,
                       o.confidence AS confidence
            """)
            return result[0] if result else {}

        elif query == "recent_thoughts":
            return await self.execute("""
                MATCH (o:Observer {id: 'ada'})-[:THINKS]->(t:Thought)
                RETURN t.id AS id, t.content AS content, t.timestamp AS timestamp,
                       t.confidence AS confidence
                ORDER BY t.timestamp DESC
                LIMIT 10
            """)

        elif query == "reasoning_trace":
            return await self.get_reasoning_trace(10)

        elif query == "confidence":
            result = await self.execute("""
                MATCH (o:Observer {id: 'ada'})
                RETURN o.confidence AS confidence
            """)
            return result[0] if result else {"confidence": 0.5}

        elif query == "emotional_state":
            result = await self.execute("""
                MATCH (o:Observer {id: 'ada'})
                RETURN o.qualia_vector AS qualia
            """)
            return result[0] if result else {"qualia": []}

        elif query == "cognitive_mode":
            result = await self.execute("""
                MATCH (o:Observer {id: 'ada'})
                RETURN o.style_vector AS style
            """)
            return result[0] if result else {"style": []}

        else:
            return {"error": f"Unknown introspection query: {query}"}

    async def get_reasoning_trace(self, depth: int = 10) -> List[Dict]:
        """
        Get chain of thoughts leading to current state.

        Args:
            depth: Maximum depth to traverse

        Returns:
            List of thoughts in reasoning chain
        """
        # Get most recent thought as starting point
        recent = await self.execute("""
            MATCH (o:Observer {id: 'ada'})-[:THINKS]->(t:Thought)
            RETURN t.id AS start_id
            ORDER BY t.timestamp DESC
            LIMIT 1
        """)

        if not recent:
            return []

        start_id = recent[0]["start_id"]

        # Traverse backwards through LEADS_TO relationships
        # Note: Kuzu syntax for variable-length paths
        trace = await self.execute("""
            MATCH (end:Thought {id: $start_id})
            MATCH (start:Thought)-[:LEADS_TO*0..10]->(end)
            RETURN start.id AS id, start.content AS content,
                   start.timestamp AS timestamp, start.step_number AS step
            ORDER BY start.timestamp ASC
        """, {"start_id": start_id})

        return trace

    async def create_episode(
        self,
        session_id: str,
        summary: str = "",
        thought_ids: List[str] = None,
        avg_qualia: List[float] = None,
        dominant_style: List[float] = None,
        emotional_valence: float = 0.5,
        importance: float = 0.5,
    ) -> str:
        """Create an episode boundary."""
        episode_id = str(uuid.uuid4())

        await self.execute("""
            CREATE (e:Episode {
                id: $id,
                session_id: $session_id,
                summary: $summary,
                avg_qualia_vector: $avg_qualia,
                dominant_style_vector: $dominant_style,
                emotional_valence: $emotional_valence,
                importance: $importance,
                start_time: timestamp()
            })
        """, {
            "id": episode_id,
            "session_id": session_id,
            "summary": summary,
            "avg_qualia": avg_qualia or [],
            "dominant_style": dominant_style or [],
            "emotional_valence": emotional_valence,
            "importance": importance,
        })

        # Link thoughts to episode
        if thought_ids:
            for tid in thought_ids:
                await self.execute("""
                    MATCH (t:Thought {id: $thought_id})
                    MATCH (e:Episode {id: $episode_id})
                    CREATE (t)-[:EXPERIENCED_IN]->(e)
                """, {"thought_id": tid, "episode_id": episode_id})

        # Observer remembers episode
        await self.execute("""
            MATCH (o:Observer {id: 'ada'})
            MATCH (e:Episode {id: $episode_id})
            CREATE (o)-[:REMEMBERS]->(e)
        """, {"episode_id": episode_id})

        return episode_id

    async def get_episodes(
        self,
        session_id: str = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Query episodic memory."""
        if session_id:
            return await self.execute("""
                MATCH (e:Episode {session_id: $session_id})
                RETURN e.id AS id, e.session_id AS session_id,
                       e.summary AS summary, e.start_time AS start_time,
                       e.emotional_valence AS emotional_valence
                ORDER BY e.start_time DESC
                LIMIT $limit
            """, {"session_id": session_id, "limit": limit})
        else:
            return await self.execute("""
                MATCH (e:Episode)
                RETURN e.id AS id, e.session_id AS session_id,
                       e.summary AS summary, e.start_time AS start_time,
                       e.emotional_valence AS emotional_valence
                ORDER BY e.start_time DESC
                LIMIT $limit
            """, {"limit": limit})

    async def update_observer_style(self, style: Dict[str, Any]) -> bool:
        """Update Observer's current thinking style."""
        try:
            # Extract style vector if provided
            style_vector = style.get("dense", style.get("vector", []))

            await self.execute("""
                MATCH (o:Observer {id: 'ada'})
                SET o.style_vector = $style,
                    o.updated_at = timestamp()
            """, {"style": style_vector})
            return True
        except Exception as e:
            print(f"[KUZU] Style update error: {e}")
            return False

    async def update_observer_qualia(self, qualia: List[float]) -> bool:
        """Update Observer's current qualia state."""
        try:
            await self.execute("""
                MATCH (o:Observer {id: 'ada'})
                SET o.qualia_vector = $qualia,
                    o.updated_at = timestamp()
            """, {"qualia": qualia})
            return True
        except Exception as e:
            print(f"[KUZU] Qualia update error: {e}")
            return False

    async def create_concept(
        self,
        name: str,
        content_vector: List[float] = None,
        hypervector: List[float] = None,
        salience: float = 0.5,
    ) -> str:
        """Create a concept node."""
        concept_id = str(uuid.uuid4())

        await self.execute("""
            CREATE (c:Concept {
                id: $id,
                name: $name,
                content_vector: $content_vector,
                hypervector: $hypervector,
                salience: $salience,
                activation: 0.0,
                created_at: timestamp(),
                accessed_at: timestamp()
            })
        """, {
            "id": concept_id,
            "name": name,
            "content_vector": content_vector or [],
            "hypervector": hypervector or [],
            "salience": salience,
        })

        return concept_id

    async def link_concepts(
        self,
        from_id: str,
        to_id: str,
        rel_type: str = "RELATES_TO",
        strength: float = 0.5,
    ) -> bool:
        """Link two concepts with a relationship."""
        try:
            await self.execute(f"""
                MATCH (a:Concept {{id: $from_id}})
                MATCH (b:Concept {{id: $to_id}})
                CREATE (a)-[:{rel_type} {{strength: $strength}}]->(b)
            """, {"from_id": from_id, "to_id": to_id, "strength": strength})
            return True
        except Exception as e:
            print(f"[KUZU] Concept link error: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self.conn is not None and self.db is not None

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn = None
        print("[KUZU] Connection closed")
