from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from diskcache import Cache
from tqdm import tqdm


@dataclass
class Status:
    cache_dir: str
    data: Dict[str, Any] = field(default_factory=dict)
    progress_bars: Dict[str, tqdm] = field(default_factory=dict)

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache = Cache(self.cache_dir)

    def reset(self):
        self.data.clear()
        self.progress_bars.clear()
        self.cache.clear()

    def track(
        self,
        key: str,
        n: int = 0,
        total: int = 100,
        unit: str = "it",
        unit_scale: bool = False,
        disable: bool = False,
    ):
        """Create a new tracking entry."""
        self.data[key] = {
            "n": n,
            "total": total,
            "unit": unit,
            "unit_scale": unit_scale,
            "disable": disable,
        }
        self.cache.set(f"{key}_n", n)
        self.cache.set(f"{key}_total", total)
        self.cache.set(f"{key}_unit", unit)
        self.cache.set(f"{key}_unit_scale", unit_scale)
        self.cache.set(f"{key}_disable", disable)

    def incr(self, key: str, delta: int = 1):
        """Increment a tracked value."""
        self.cache.incr(f"{key}_n", delta)

    def set(self, key: str, value: int, type: str = "n"):
        """Set a tracked value directly."""
        assert type in {"n", "store"}
        self.cache.set(f"{key}_{type}", value)

    def get(self, key: str, type: str = "n"):
        """Get a tracked value from the cache."""
        assert type in {"n", "total", "store"}
        return self.cache.get(f"{key}_{type}", 0)

    def read(self, key: str):
        """Get a tracked value from the cache."""
        return self.data.get(key, {}).get("n", 0)

    def read_all(self):
        """Read all values from the cache."""
        for key in self.data.keys():
            self.data[key]["n"] = self.cache.get(f"{key}_n", 0)
            self.data[key]["total"] = self.cache.get(f"{key}_total", 100)

    def show_progress(self):
        """Display progress bars for all tracked items."""
        self.read_all()
        for key, data in self.data.items():
            if data["disable"]:
                continue
            if key not in self.progress_bars:
                self.progress_bars[key] = tqdm(
                    total=data["total"],
                    unit=data["unit"],
                    unit_scale=data["unit_scale"],
                    desc=key,
                    disable=data["disable"],
                )
            if self.progress_bars[key].n < data["total"]:
                self.progress_bars[key].n = min(data["n"], data["total"])
                self.progress_bars[key].refresh()

    def close(self):
        """Close all progress bars and the cache."""
        for bar in self.progress_bars.values():
            bar.close()
        self.cache.close()
