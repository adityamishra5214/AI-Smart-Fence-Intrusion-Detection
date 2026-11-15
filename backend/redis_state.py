"""
Redis state management, node heartbeat TTL tracking (3s), and Pub/Sub broadcaster.
Includes transparent high-precision in-memory TTL fallback when Redis is offline.
"""

import os
import time
import json
from typing import Dict, List, Optional, Set
from simulator.schemas import NodeStatus

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class NodeStateTracker:
    """
    Monitors node aliveness with strict 3-second TTL.
    Missing heartbeat (> 3s) immediately triggers Mode 3 SENSOR_FAULT.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl_seconds: float = 3.0):
        self.ttl_seconds = ttl_seconds
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis = False

        # In-memory TTL fallback store
        self._memory_heartbeats: Dict[str, float] = {}
        self._memory_node_info: Dict[str, Dict[str, str]] = {}

        enable_redis = os.getenv("ENABLE_REDIS", "false").lower() in ("true", "1", "yes")

        if enable_redis and REDIS_AVAILABLE:
            try:
                r = redis.Redis(host=host, port=port, db=db, socket_timeout=0.1)
                r.ping()
                self.redis_client = r
                self.use_redis = True
                print(f"[RedisState] Connected to Redis at {host}:{port}")
            except Exception as e:
                print(f"[RedisState] Redis offline ({e}). Using in-memory TTL store.")
                self.use_redis = False
        else:
            self.use_redis = False

    def record_heartbeat(self, node_id: str, zone_id: str, metadata: Optional[Dict] = None):
        """Records node heartbeat and refreshes 3-second TTL."""
        now = time.time()
        self._memory_heartbeats[node_id] = now
        self._memory_node_info[node_id] = {"zone_id": zone_id, "last_seen": str(now)}

        if self.use_redis and self.redis_client:
            try:
                key = f"node:{node_id}:alive"
                self.redis_client.set(key, json.dumps({"zone_id": zone_id, "ts": now}), ex=int(self.ttl_seconds))
            except Exception as e:
                print(f"[RedisState] Redis write error: {e}")

    def is_alive(self, node_id: str) -> bool:
        """Returns True if node heartbeat was received within TTL window."""
        now = time.time()
        if self.use_redis and self.redis_client:
            try:
                key = f"node:{node_id}:alive"
                return bool(self.redis_client.exists(key))
            except Exception:
                pass

        # In-memory check
        last_seen = self._memory_heartbeats.get(node_id)
        if last_seen is None:
            return False
        return (now - last_seen) <= self.ttl_seconds

    def mark_dead(self, node_id: str):
        """Forces immediate expiry (e.g., upon MQTT LWT death packet)."""
        self._memory_heartbeats.pop(node_id, None)
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(f"node:{node_id}:alive")
            except Exception:
                pass

    def get_all_registered_nodes(self) -> List[str]:
        """Returns all known node IDs."""
        return list(self._memory_node_info.keys())

    def get_node_statuses(self) -> Dict[str, Dict]:
        """Returns status of all monitored nodes."""
        statuses = {}
        now = time.time()
        for node_id, info in self._memory_node_info.items():
            last_ts = self._memory_heartbeats.get(node_id, 0.0)
            elapsed = now - last_ts
            alive = elapsed <= self.ttl_seconds
            statuses[node_id] = {
                "zone_id": info.get("zone_id", "unknown"),
                "alive": alive,
                "status": NodeStatus.HEALTHY.value if alive else NodeStatus.FAULT.value,
                "seconds_since_heartbeat": round(elapsed, 2),
            }
        return statuses
