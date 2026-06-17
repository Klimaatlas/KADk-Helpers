#!/usr/bin/env python
from pathlib import Path
import fnmatch

SRCDIR = Path("/dmidata/projects/klimaatlas/data/CORDEX5/DAY_EUR11_FULL")
OUTDIR = Path(".")
VARS = ["tas", "tasmin", "tasmax","pr","sfcWind","sfcWindmax","rsds","hurs"]

# --------------------------------------------------
# Collect files
# --------------------------------------------------
files = {
    f"CORDEX5-{v}": list((SRCDIR / v).glob(f"{v}_EUR-11_*"))
    for v in VARS
}

# RekLiEs naming format (tas only)
files["ReKliEs-tas"] = list((SRCDIR / "tas").glob("tas_own_ReKliEs-DWD_ICHEC-EC-EARTH_*"))

# --------------------------------------------------
# Removal patterns
# --------------------------------------------------
patterns = [
    # * RCMs using CNRM CM5 v1 as a boundary condition, as there are recognised problems
    "*CNRM-CERFACS-CNRM*v1*",
    # * WRF331 (because its old and there is a newer version available in the ensemble)
    "*WRF331*",
    # * Some HadREM3 historical runs that spill over beyond end of 2005
    "*_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_day_20060101-20101231.nc",
    # * RegCM4 experiments that start in Dec 2005 rather than Jan 2006
    "*ICTP-RegCM4-6_v1*20051201-2005123?*",
    "*ICTP-RegCM4-6_v1*20051201-20060101*",
    # * NOAA model only has rcp26 run, but not historical. 
    "*EUR-11_NOAA-GFDL-GFDL-ESM2G_rcp26_r1i1p1_GERICS-REMO2015_v1_day*",
    # * Duplicated rsds from MOHC that end on 30th Dec
    "*rsds_EUR-11_MOHC-HadGEM2-ES_rcp85_r1i1p1_ICTP-RegCM4-6_v1_day_20*0101-2*1230.nc"
]

# Remove files for IPSL WRF381P. The archive has files that cover both 5 year and 10 year time windows, that appear to be complete duplicates of each other. We remove all 10 year versions. But it requires a lot of manual work!
years = [
    "19510101-1960","19610101-1970","19710101-1980","19810101-1990","19910101-2000",
    "20110101-2020","20210101-2030","20310101-2040","20410101-2050","20510101-2060",
    "20610101-2070","20710101-2080","20810101-2090","20910101-2100"
]

patterns += [
    f"*_EUR-11_IPSL-IPSL-CM5A-MR_*_r1i1p1_IPSL-WRF381P_v1_day_{y}*"
    for y in years
]

# Apply removal patterns
def keep_file(path):
    s = str(path)
    return not any(fnmatch.fnmatch(s, p) for p in patterns)

for v in files:
    files[v] = [f for f in files[v] if keep_file(f)]

# --------------------------------------------------
# Add back two v1 models (tas only)
# --------------------------------------------------
files["CORDEX5-tas"] += list((SRCDIR / "tas").glob(
    "tas_EUR-11_CNRM-CERFACS-CNRM-CM5_*CLMcom-ETH-COSMO*v1*"
))
files["CORDEX5-tas"] += list((SRCDIR / "tas").glob(
    "tas_EUR-11_CNRM-CERFACS-CNRM-CM5_*GERICS-REMO2015*v1*"
))

# --------------------------------------------------
# Finalize lists (unique + sorted)
# --------------------------------------------------
for v in files:
    files[v] = sorted({str(f) for f in files[v]})

# --------------------------------------------------
# Write file lists
# --------------------------------------------------
for v, flist in files.items():
    outfile = Path(OUTDIR) / v
    outfile.write_text("\n".join(flist))

print("File counts:")
for v in files:
    print(v, len(files[v]))
