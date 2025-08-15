from __future__ import annotations

from typing import Optional


class Cache:
    def __init__(self, url: str, enabled: bool = False) -> None:
        self.enabled = enabled
        self.client = None
        if enabled:
            try:
                import redis  # type: ignore
                self.client = redis.from_url(url)
            except Exception:
                self.client = None
                self.enabled = False

    def get(self, key: str) -> Optional[bytes]:
        if not self.enabled or self.client is None:
            return None
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.setex(key, ttl_seconds, value)
        except Exception:
            return