#!/usr/bin/env python3

"""
Find relevant files from the local CMIP5 archive and generate file lists.

"""

from pathlib import Path
import pandas as pd

# Configuration
ROOT = Path("/dmidata/projects/klimaatlas/data/CMIP5/DAY_FULL/")

variables = [
    "tas",
]


def main():
    for var in variables:
        outfile = Path(f"CMIP5-{var}")

        pattern = f"{var}_*.nc"

        SRCDIR = ROOT 

        files = sorted(
            p
            for p in SRCDIR.rglob(pattern)
        )

        #Check for duplicate file names. 
        df=pd.DataFrame(files, columns=['path'])
        df["filename"]=df.path.apply(lambda x: x.name)
        df.sort_values(by="path", inplace=True)
        #Mark duplicates
        df["duplicated"]=df.duplicated(subset="filename", keep=False)
        #Drop duplicates
        dropped=df.drop_duplicates(subset="filename", keep="last", inplace=False)
        #Convert paths to strings
        output_list=dropped.path.apply(lambda x: str(x))

        with outfile.open("w") as f:
            for file in output_list:
                f.write(f"{file}\n")

        print(f"{var}: {len(files)} files written to {outfile}")


if __name__ == "__main__":
    main()
