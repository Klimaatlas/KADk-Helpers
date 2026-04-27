#!/bin/bash
#
# This script finds relevant files from the local DMI CORDEX6 results and 
# generates filelists. Given the output structure from HCLIM, this 
# is the easiet way to interface the data into KAPy
#

# Config
SRCDIR="/net/isilon/ifs/arch/home/fbo/HCLIM/HCLIM2CMOR/PP/work/output_CMORlight/CORDEX/CMIP6/DD/EUR-12/HCLIMcom-DMI/IPSL-CM6A-LR/"
vars=("tas" "tasmin" "tasmax")

# Loop over variables and find files
for var in "${vars[@]}"; do
    # Find and link files
    find $SRCDIR -path "*/merged_files/*" -name "${var}_*_day_*.nc" > "../inputs/CORDEX6-${var}"
done
