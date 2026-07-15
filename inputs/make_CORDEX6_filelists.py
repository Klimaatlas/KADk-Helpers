#!/usr/bin/env python3

"""
Find relevant files from the local CORDEX6 archive and generate file lists.

Given the output structure from HCLIM, this provides an interface
for KAPy to the CORDEX6 archive.
"""

from pathlib import Path

# Configuration
ROOT = Path("/dmidata/projects/klimaatlas/data/CORDEX6/DAY_EUR11_FULL/")

variables = [
    "hurs",
    "pr",
    "rsds",
    "sfcWind",
    "sfcWindmax",
    "tas",
    "tasmax",
    "tasmin",
]


def main():
    for var in variables:
        outfile = Path(f"CORDEX6-{var}")

        pattern = f"{var}_*.nc"

        SRCDIR = ROOT / var

        files = sorted(
            p
            for p in SRCDIR.rglob(pattern)
        )

        with outfile.open("w") as f:
            for file in files:
                f.write(f"{file}\n")

        print(f"{var}: {len(files)} files written to {outfile}")


if __name__ == "__main__":
    main()