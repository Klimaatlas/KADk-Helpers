#!/usr/bin/env python3

"""
Find relevant files from the local CORDEX6 archive and generate file lists.

Given the output structure from HCLIM, this provides an interface
for KAPy to the CORDEX6 archive.
"""

from pathlib import Path
from collections import defaultdict
import fnmatch

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

removal= ["*_ERA5_evaluation_*"]


#Build an index of the entire collection
index = defaultdict(list)

for p in ROOT.rglob("*.nc"):
    parts = p.stem.split("_")

    var = parts[0]

    index[(var)].append(p)

for lst in index.values():
    lst.sort()


for var in variables:
    outfile = Path(f"CORDEX6-{var}")

    files = index[(var)]

    # Apply removal patterns
    files = [
        str(f) for f in files
        if not any(fnmatch.fnmatch(str(f), p) for p in removal)
    ]

    with outfile.open("w") as f:
        for file in files:
            f.write(f"{file}\n")

    print(f"{var}: {len(files)} files written to {outfile}")

