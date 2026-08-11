#!/usr/bin/env python3
"""Parse + check every line of mll_positional.txt."""

from __future__ import annotations

from pathlib import Path

from .check import check_all, check_proof_of_line, check_syntax

DATASET = Path(__file__).resolve().parent.parent / "mll_positional.txt"


def main() -> None:
    lines = [
        line.strip()
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad = 0
    for i, line in enumerate(lines, start=1):
        syntax_ok = check_syntax(line)
        proof_ok = check_proof_of_line(line)
        all_syntax, all_proof = check_all(line)
        if not syntax_ok:
            bad += 1
            print(f"line {i}: check_syntax failed")
        elif not proof_ok:
            bad += 1
            print(f"line {i}: check_proof_of_line failed")
        elif (all_syntax, all_proof) != (syntax_ok, proof_ok):
            bad += 1
            print(
                f"line {i}: check_all mismatch "
                f"(check_syntax={syntax_ok}, check_proof_of_line={proof_ok}, "
                f"check_all={all_syntax, all_proof})"
            )
    print(f"{len(lines)} lines ({bad} failures)")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
