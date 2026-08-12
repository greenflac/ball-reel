"""python3 -m ball_reel  -> run the offline replay and print the ranked run."""

from __future__ import annotations

import sys
from pathlib import Path

from .pipeline import render, run

ROOT = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    live = "--live" in argv
    try:
        result = run(ROOT, live=live)
    except FileNotFoundError as exc:
        print(f"no fixtures: {exc}\nrun: python3 -m ball_reel.tools.make_fixtures")
        return 2
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
