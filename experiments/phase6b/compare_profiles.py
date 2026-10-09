"""Print decode-step comparisons of profile JSON files (Phase 6B working tool): mean, median, regions, copies, submit."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def row(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d["summary"]["decode"]
    r = s["regions_ms"]
    engine = s.get("engine_submit_ms", {}).get("", None)
    tree = d["environment"]["native_tree_sha256"][:8]
    return (
        f"{str(path.parent.name) + '/' + path.stem.removeprefix('profile-'):58s} {s['wall_ms']:7.1f} ({s['wall_ms_median']:6.1f}) "
        f"plan {r.get('plan', 0):5.1f} io {r.get('io', 0):5.1f} asm {r.get('assemble', 0):5.1f} h2d {r.get('h2d', 0):4.1f} "
        f"other {r.get('other', 0):5.1f} | copy {s['h2d_device_ms']:5.1f} hit {s.get('host_cache_hit_rate') or 0:.3f} "
        f"| prefill {d['summary']['prefill']['wall_ms']:6.0f} | RAM {s.get('peak_resident_bytes', 0) / 1e9:4.1f} "
        f"| engine submit {'-' if engine is None else f'{engine:5.2f}'} | native {tree}"
    )


def main() -> int:
    for argument in sys.argv[1:]:
        for path in sorted(Path().glob(argument)):
            print(row(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
