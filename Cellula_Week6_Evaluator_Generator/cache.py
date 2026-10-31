import hashlib
import json
from typing import Any, Dict, List, Optional

import redis

from config import CACHE_TTL, REDIS_URL


class RedisCache:
    """JSON cache stored in Redis under explicit key prefixes."""

    def __init__(self, url: str = REDIS_URL, ttl: int = CACHE_TTL) -> None:
        self.url = url
        self.ttl = ttl
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def ping(self) -> bool:
        return bool(self.client.ping())

    @staticmethod
    def fingerprint(*parts: Any) -> str:
        raw = "|".join(str(part) for part in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        raw = self.client.get(key)
        if raw is None:
            self.client.incr("metrics:cache_miss")
            return None
        self.client.incr("metrics:cache_hit")
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.client.set(key, json.dumps(value), ex=ttl or self.ttl)

    def version(self, name: str = "kb") -> int:
        value = self.client.get(f"version:{name}")
        return int(value) if value else 1

    def bump_version(self, name: str = "kb") -> int:
        return int(self.client.incr(f"version:{name}"))

    def delete_prefix(self, prefix: str) -> int:
        keys: List[str] = list(self.client.scan_iter(match=f"{prefix}*", count=500))
        return self.client.delete(*keys) if keys else 0

    def bump_metric(self, name: str) -> None:
        self.client.incr(f"metrics:{name}")

    def metrics(self) -> Dict[str, int]:
        hits = int(self.client.get("metrics:cache_hit") or 0)
        misses = int(self.client.get("metrics:cache_miss") or 0)
        total = hits + misses
        return {
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": round(100 * hits / total, 1) if total else 0.0,
            "generator_runs": int(self.client.get("metrics:generator_runs") or 0),
            "evaluator_runs": int(self.client.get("metrics:evaluator_runs") or 0),
        }

    def reset_metrics(self) -> None:
        self.client.delete(
            "metrics:cache_hit",
            "metrics:cache_miss",
            "metrics:generator_runs",
            "metrics:evaluator_runs",
        )
