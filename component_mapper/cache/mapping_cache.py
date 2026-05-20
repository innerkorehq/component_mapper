import asyncio
import json
import logging
from pathlib import Path
from component_mapper.models import MappingCacheRecord

logger = logging.getLogger(__name__)


class MappingCache:
    def __init__(self, cache_path: str, auto_persist_every: int = 50):
        self._path = Path(cache_path)
        self._store: dict[str, MappingCacheRecord] = {}
        self._lock = asyncio.Lock()
        self._write_count = 0
        self._auto_persist_every = auto_persist_every

    async def load(self) -> None:
        """Load cache from disk. Silent if file missing."""
        if not self._path.exists():
            logger.debug("No mapping cache at %s — starting fresh", self._path)
            return
        try:
            async with asyncio.Lock():
                data = self._path.read_text()
            records = json.loads(data)
            async with self._lock:
                for key, raw in records.items():
                    try:
                        self._store[key] = MappingCacheRecord.model_validate(raw)
                    except Exception:
                        pass
            logger.info(
                "Loaded %d mapping cache records from %s", len(self._store), self._path
            )
        except Exception as exc:
            logger.warning("Failed to load mapping cache: %s", exc)

    async def get(self, fingerprint_hash: str) -> MappingCacheRecord | None:
        async with self._lock:
            return self._store.get(fingerprint_hash)

    async def set(self, fingerprint_hash: str, record: MappingCacheRecord) -> None:
        async with self._lock:
            self._store[fingerprint_hash] = record
            self._write_count += 1
            should_persist = self._write_count % self._auto_persist_every == 0
        if should_persist:
            await self.persist()

    async def persist(self) -> None:
        """Write cache to disk atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            async with self._lock:
                data = {k: v.model_dump(mode="json") for k, v in self._store.items()}
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.replace(self._path)
            logger.debug("Persisted %d mapping cache records", len(data))
        except Exception as exc:
            logger.warning("Failed to persist mapping cache: %s", exc)

    async def increment_hit(self, fingerprint_hash: str) -> None:
        async with self._lock:
            record = self._store.get(fingerprint_hash)
            if record:
                record.hit_count += 1

    @property
    def size(self) -> int:
        return len(self._store)
