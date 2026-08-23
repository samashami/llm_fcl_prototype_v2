"""Loading and validation for frozen norm-matched shrinkage schedules."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple


ScheduleKey = Tuple[int, int, str]


def load_shrinkage_schedule(
    path: str | Path, expected_keys: Iterable[ScheduleKey]
) -> Dict[ScheduleKey, float]:
    """Load a complete, unique round/client/layer shrinkage schedule."""
    path = Path(path)
    expected = set(expected_keys)
    schedule: Dict[ScheduleKey, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"round", "client", "layer", "shrinkage_factor"}
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "shrinkage schedule is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (int(row["round"]), int(row["client"]), row["layer"])
                factor = float(row["shrinkage_factor"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid shrinkage schedule value on line {line_number}"
                ) from error
            if key in schedule:
                raise ValueError(f"duplicate shrinkage schedule entry: {key}")
            if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
                raise ValueError(f"invalid shrinkage factor for {key}: {factor}")
            if key[0] == 0 and factor != 1.0:
                raise ValueError(f"round-0 shrinkage factor must be 1 for {key}")
            schedule[key] = factor

    missing = expected.difference(schedule)
    extra = set(schedule).difference(expected)
    if missing:
        preview = ", ".join(map(str, sorted(missing)[:3]))
        raise ValueError(f"shrinkage schedule is missing entries: {preview}")
    if extra:
        preview = ", ".join(map(str, sorted(extra)[:3]))
        raise ValueError(f"shrinkage schedule has unexpected entries: {preview}")
    return schedule

