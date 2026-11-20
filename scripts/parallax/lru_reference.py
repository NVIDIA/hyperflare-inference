"""Produce synthetic policy-decision vectors from DerpSeek's actual _pairs method."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Trusted DerpSeek engine/derpseek_engine/residency.py"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tree = ast.parse(args.source.read_text())
    method = next(
        method
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "_pairs"
    )
    namespace = {"np": np, "TUNE": {}}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(args.source), "exec"),
        namespace,
    )
    rng = np.random.default_rng(20261117)
    cases = []
    for case in range(24):
        tick = 5
        window = [1, 2, 64][case % 3]
        minimum = [0, 1, 4][case % 3]
        age = case % 5
        per_layer = 1 + case % 2
        total = 1 + case % 4
        previous = rng.integers(0, 12, (2, 5)).astype(np.float32)
        misses = rng.integers(0, 5, (2, 5))
        busy = rng.random((2, 2)) < .3
        misses[:, :2] *= busy
        held = np.zeros((2, 5), dtype=bool)
        held[:, :2] = True
        born = rng.integers(0, 5, (2, 2))
        used = rng.integers(0, 5, (2, 2))
        recent = previous.copy()
        recent *= np.float32(1.0 - 1.0 / window)
        recent += misses
        obj = SimpleNamespace(
            recent=recent, held=held, busy=busy, slot_born=born, slot_used=used
        )
        namespace["TUNE"].update(
            victim=0, min_misses=minimum, min_age=age,
            promote=per_layer, promote_tick=total,
        )
        result = namespace["_pairs"](obj, misses, tick)
        pairs = [[layer, int(row) + 1, int(slot)]
                 for layer, entries in result for row, slot in entries]
        cases.append(dict(
            window=window, minimum=minimum, age=age, per_layer=per_layer, total=total,
            previous=previous.tolist(), misses=misses.tolist(), busy=busy.tolist(),
            born=born.tolist(), used=used.tolist(), pairs=pairs,
        ))
    scope = (
        "Synthetic decisions from chronarion/derpseek "
        "engine/derpseek_engine/residency.py _pairs; canonical host and slot "
        "ordering, victim=0. Not serving performance."
    )
    args.output.write_text(json.dumps(
        {"scope": scope, "cases": cases}, separators=(",", ":")
    ) + "\n")
    exchanges = sum(len(case["pairs"]) for case in cases)
    print(f"{len(cases)} reference decision cases; {exchanges} exchanges; "
          f"{args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
