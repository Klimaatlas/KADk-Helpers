# Import CORDEX data
#
# Generally CORDEX data can be imported using the defaults, but as ever there
# are some files that simply fail because they are not 100% standards compliant
# and therefore need special handling.

import KAPy
import os
import numpy as np
import re
import xarray as xr

def import_CORDEX(inFiles,varCode,internalVarName, checks,cutoutArgs, **kwargs):
    # Import --------------------------------------
    #Get list of filenames
    inFnames=[os.path.basename(f) for f in inFiles]
   
    # The sfcWind HadGEM2 / RegCM4 historical run has the supplementary coordinates labelled as "xlon" and "xlat",
    # which is a mistake - they should be "lon" and "lat". The problem is only in the historical runs though
    # and a nice workaround based on the fact that xarray interprets the first file as the template for the
    # rest. Reversing the file order, so that the rcps are first, achieves this.
    if  any(re.search(".*sfcWind_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1_ICTP-RegCM4-6_v1_day.*", f) for f in inFnames):
        time_coder=xr.coders.CFDatetimeCoder(use_cftime=True)
        try:
            dsIn = xr.open_mfdataset(
                inFiles,
                chunks={"time": 256},
                combine="nested",
                concat_dim="time",
                decode_times=time_coder,
                decode_cf=False,
                join="override",
                compat="override",
                coords="minimal",
                data_vars=["sfcWind"])
            
        except Exception as e:
            raise RuntimeError(f"Opening following NetCDF files:\n '{inFiles}'\n failed with error:\n{e}")	
        
        #Correct supplementary coordinates
        dsIn = dsIn.set_coords(["lat", "lon"])
        
        # Select the desired variable to give a and rename to the variable code
        da = dsIn[internalVarName]
        da.name= varCode

        # Drop the m10 degenerate dimension. 
        # We drop time_bnds explicitly here - this mirrors what xarray would do anyway when working with
        # a dataarray. Might need to fix in the future.
        da = da.squeeze(drop=True)
        da["time"].attrs.pop("bounds")
        da.attrs.pop("coordinates")
    
    else:
        #Import using the default import functionality
        da=KAPy.defaultImport(inFiles=inFiles,
                                varCode=varCode,
                                internalVarName=internalVarName,
                                checks=checks)

    #Post import modifications ------------
    
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

    #NorESM - crCLIM model seems to have strange temperature values in the summer of 2100. We remove
    #this year for tas, tasmin and tasmax.
    if  any(re.search("tas.*_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_.*", f) for f in inFnames):
        da=da.sel(time=slice(None,"2100-01-01"))

    #Apply cutouts-----------------
    if cutoutArgs["method"] == "lonlatbox":
        da=KAPy.cutout_lonlat(da,**cutoutArgs,varCode=varCode)

    return(da)

    