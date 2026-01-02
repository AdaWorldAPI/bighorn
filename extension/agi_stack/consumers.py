"""
Redis Stream Consumers - Process async events from hive.

Streams:
    - ada:stream:thoughts    -> Persist to Kuzu + LanceDB
    - ada:stream:episodes    -> Mark episode boundaries
    - ada:stream:adaptations -> Style/qualia changes
"""

import asyncio
import json
from typing import Dict, Optional
from datetime import datetime

from .kuzu_client import KuzuClient
from .lance_client import LanceClient

# Stream names
STREAM_THOUGHTS = "ada:stream:thoughts"
STREAM_EPISODES = "ada:stream:episodes"
STREAM_ADAPTATIONS = "ada:stream:adaptations"

# Consumer group name
CONSUMER_GROUP = "ada-agi-surface"
CONSUMER_NAME = "worker-1"


async def start_consumers(
    kuzu: KuzuClient,
    lance: LanceClient,
    redis_url: str,
    redis_token: str,
):
    """
    Start all Redis stream consumers.

    Args:
        kuzu: Kuzu database client
        lance: LanceDB client
        redis_url: Upstash Redis REST URL
        redis_token: Upstash Redis REST token
    """
    if not redis_url or not redis_token:
        print("[CONSUMER] Redis not configured, skipping consumers")
        return

    try:
        from upstash_redis.asyncio import Redis
        redis = Redis(url=redis_url, token=redis_token)

        # Initialize consumer groups
        await init_consumer_groups(redis)

        print("[CONSUMER] Starting consumers...")

        # Run consumers concurrently
        await asyncio.gather(
            thought_consumer(redis, kuzu, lance),
            episode_consumer(redis, kuzu),
            adaptation_consumer(redis, kuzu),
            return_exceptions=True,
        )

    except ImportError:
        print("[CONSUMER] upstash-redis not installed, skipping consumers")
    except Exception as e:
        print(f"[CONSUMER] Failed to start consumers: {e}")


async def init_consumer_groups(redis):
    """Initialize consumer groups for streams."""
    try:
        # Create streams and consumer groups if they don't exist
        for stream in [STREAM_THOUGHTS, STREAM_EPISODES, STREAM_ADAPTATIONS]:
            try:
                await redis.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
                print(f"[CONSUMER] Created group for {stream}")
            except Exception as e:
                # Group may already exist
                if "BUSYGROUP" not in str(e):
                    print(f"[CONSUMER] Group create warning for {stream}: {e}")
    except Exception as e:
        print(f"[CONSUMER] Failed to init groups: {e}")


async def thought_consumer(
    redis,
    kuzu: KuzuClient,
    lance: LanceClient,
):
    """
    Consume thoughts from Redis stream.
    Persist to Kuzu (graph) + LanceDB (vector).
    """
    last_id = "0"

    while True:
        try:
            # Read from stream
            events = await redis.xread(
                {STREAM_THOUGHTS: last_id},
                block=5000,
                count=100,
            )

            if not events:
                continue

            for stream_name, messages in events:
                for msg_id, data in messages:
                    try:
                        await process_thought(kuzu, lance, msg_id, data)
                        last_id = msg_id
                    except Exception as e:
                        print(f"[CONSUMER] Thought processing error: {e}")

        except asyncio.CancelledError:
            print("[CONSUMER] Thought consumer cancelled")
            break
        except Exception as e:
            print(f"[CONSUMER] Thought stream error: {e}")
            await asyncio.sleep(1)


async def process_thought(
    kuzu: KuzuClient,
    lance: LanceClient,
    msg_id: str,
    data: Dict[str, str],
):
    """Process a single thought event."""
    thought_id = data.get("id", msg_id)
    content = data.get("content", "")

    # Parse vectors from JSON
    content_vector = []
    style_33d = []
    qualia_17d = []

    try:
        content_vector = json.loads(data.get("content_vector", "[]"))
    except json.JSONDecodeError:
        pass

    try:
        style_33d = json.loads(data.get("style_33d", "[]"))
    except json.JSONDecodeError:
        style_33d = [0.33, 0.33, 0.34] + [0.11] * 9 + [0.2] * 5 + [0.0] * 16

    try:
        qualia_17d = json.loads(data.get("qualia_17d", "[]"))
    except json.JSONDecodeError:
        qualia_17d = [0.5] * 17

    parent_id = data.get("parent_id") or None
    session_id = data.get("session_id") or None
    step_number = int(data.get("step_number", "0"))

    # 1. Create thought in Kuzu
    await kuzu.create_thought(
        content=content,
        style_vector=style_33d,
        qualia_vector=qualia_17d,
        content_vector=content_vector,
        parent_id=parent_id,
        session_id=session_id,
        step_number=step_number,
    )

    # 2. Index in LanceDB if we have a content vector
    if content_vector and len(content_vector) > 0:
        await lance.upsert(
            id=thought_id,
            vector=content_vector,
            table="thoughts",
            metadata={
                "content": content[:500],
                "session_id": session_id or "",
            },
        )

    print(f"[CONSUMER] Thought processed: {thought_id[:8]}...")


async def episode_consumer(redis, kuzu: KuzuClient):
    """
    Consume episode boundaries from Redis stream.
    Create episodes and link thoughts.
    """
    last_id = "0"

    while True:
        try:
            events = await redis.xread(
                {STREAM_EPISODES: last_id},
                block=5000,
                count=10,
            )

            if not events:
                continue

            for stream_name, messages in events:
                for msg_id, data in messages:
                    try:
                        await process_episode(kuzu, msg_id, data)
                        last_id = msg_id
                    except Exception as e:
                        print(f"[CONSUMER] Episode processing error: {e}")

        except asyncio.CancelledError:
            print("[CONSUMER] Episode consumer cancelled")
            break
        except Exception as e:
            print(f"[CONSUMER] Episode stream error: {e}")
            await asyncio.sleep(1)


async def process_episode(kuzu: KuzuClient, msg_id: str, data: Dict[str, str]):
    """Process a single episode event."""
    episode_id = data.get("id", msg_id)
    session_id = data.get("session_id", "")
    summary = data.get("summary", "")

    thought_ids = []
    avg_qualia = []

    try:
        thought_ids = json.loads(data.get("thought_ids", "[]"))
    except json.JSONDecodeError:
        pass

    try:
        avg_qualia = json.loads(data.get("avg_qualia", "[]"))
    except json.JSONDecodeError:
        avg_qualia = [0.5] * 17

    # Create episode
    await kuzu.create_episode(
        session_id=session_id,
        summary=summary,
        thought_ids=thought_ids,
        avg_qualia=avg_qualia,
    )

    print(f"[CONSUMER] Episode created: {episode_id[:8]}...")


async def adaptation_consumer(redis, kuzu: KuzuClient):
    """
    Consume style/qualia adaptations from Redis stream.
    Update Observer's cognitive state.
    """
    last_id = "0"

    while True:
        try:
            events = await redis.xread(
                {STREAM_ADAPTATIONS: last_id},
                block=5000,
                count=10,
            )

            if not events:
                continue

            for stream_name, messages in events:
                for msg_id, data in messages:
                    try:
                        await process_adaptation(kuzu, msg_id, data)
                        last_id = msg_id
                    except Exception as e:
                        print(f"[CONSUMER] Adaptation processing error: {e}")

        except asyncio.CancelledError:
            print("[CONSUMER] Adaptation consumer cancelled")
            break
        except Exception as e:
            print(f"[CONSUMER] Adaptation stream error: {e}")
            await asyncio.sleep(1)


async def process_adaptation(kuzu: KuzuClient, msg_id: str, data: Dict[str, str]):
    """Process a style/qualia adaptation event."""
    style_33d = []
    qualia_17d = []

    try:
        style_33d = json.loads(data.get("style_33d", "[]"))
    except json.JSONDecodeError:
        pass

    try:
        qualia_17d = json.loads(data.get("qualia_17d", "[]"))
    except json.JSONDecodeError:
        pass

    reason = data.get("reason", "")

    # Update Observer's state
    if style_33d:
        await kuzu.update_observer_style({"dense": style_33d})

    if qualia_17d:
        await kuzu.update_observer_qualia(qualia_17d)

    print(f"[CONSUMER] Adaptation applied: {reason or 'no reason'}")
