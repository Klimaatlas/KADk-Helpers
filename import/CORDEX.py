# Import CORDEX data
#
# Generally CORDEX data can be imported using the defaults, but as ever there
# are some files that simply fail because they are not 100% standards compliant
# and therefore need special handling.

import KAPy
import os
import numpy as np
import re

def import_CORDEX(inFiles,varID,internalVarName, cutoutArgs, **kwargs):
    # Import --------------------------------------
    #Import using the default import functionality
    da=KAPy.defaultImport(inFiles=inFiles,
                            varID=varID,
                            internalVarName=internalVarName)

    #Post import modifications ------------
    inFnames=[os.path.basename(f) for f in inFiles]
    
    #Problems with x and y coordinates being indentical that needs to be fixed        
    if  any(re.search(".*_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_IPSL-WRF381P_v1_day_20060101-21001231.nc", f) for f in inFnames):
        da.coords['x']=np.arange(0,da.x.size)
        da.coords['y']=np.arange(0,da.y.size)

    #Problems with mixed calendars in two output files
    if  any(re.search(".*_EUR-11_ICHEC-EC-EARTH_rcp26_r12i1p1_CLMcom-CCLM4-8-17_v1_day*", f) for f in inFnames):
        da.coords['time']=[f.change_calendar("proleptic_gregorian",has_year_zero=True) for f in da.coords['time'].values]

    #Problem with auxillary coordinate units that have time as a dimension
    #The problem seems to be primarily with REMO2015
    if  any(re.search(".*GERICS-REMO2015*", f) for f in inFnames):
        if len(da.lon.shape)==3:
            useThis=da.lon.isel(time=0,drop=True)
            da = da.drop_vars('lon')
            da = da.assign_coords(lon=useThis)

    #Apply cutouts-----------------
    if cutoutArgs["method"] == "lonlatbox":
        da=KAPy.cutout_lonlat(da,**cutoutArgs,varID=varID)

    return(da)

    