import KAPy
import xarray as xr

def import_CLIM4CITIES(inFiles,varCode,internalVarName, checks, **kwargs):

    """
    Import CLIM4CITIES data

    The CLIM4CITIES dataset works well with the default import, but requires the following modifications
     * the datetime dimension needs to be renamed to time

    Args:
        config (_type_): _description_
        inFiles (_type_): _description_
        inpID (_type_): _description_
    """
    """
    #Debugging setup
    config=KAPy.getConfig("./config/config.yaml")  
    inFiles=["./inputs/predictions_Kobenhavn_2023.nc"]
    inpID="C4C-Kbh"
    """

    #Import using the default import functionality
    # da=KAPy.defaultImport(inFiles=inFiles,
    #                         varCode=varCode,
    #                         internalVarName=internalVarName,
    #                         checks=checks)
    
    #-------------Modified defaultImport--------------
    time_coder=xr.coders.CFDatetimeCoder(use_cftime=True)
    inFiles = sorted(inFiles)  #Helps ensure monotonic time

    try:
        dsIn =xr.open_mfdataset(inFiles,
                                combine='by_coords' if checks=="all" else "nested",
                                concat_dim=None if checks=="all" else "time",
                                decode_times=time_coder, 
                                join="exact" if checks=="all" else "override" , 
                                compat="no_conflicts" if checks=="all" else "override",
                                coords="minimal",
                                data_vars="minimal",
                                chunks={"time": 256},
                                parallel=True,
                                preprocess=lambda ds: ds[[internalVarName]])

    except Exception as e:
        raise RuntimeError(f"Opening following NetCDF files:\n '{inFiles}'\n failed with error:\n{e}")    
    

    # Apply some checkes on the results (if requested)
    # if checks=="all":
    #     if not dsIn.indexes["time"].is_monotonic_increasing:
    #         raise ValueError(f"Time coordinate is not monotonic in file set: '{inFiles}'.")
    #     if dsIn.indexes["time"].has_duplicates:
    #         raise ValueError(f"Duplicate timestamps detected file set: '{inFiles}'.")

    # Select the desired variable to give a and rename to the variable code
    da = dsIn[internalVarName]
    da.name= varCode

    # Drop degenerate dimensions. If any remain, throw an error
    da = da.squeeze(drop=True)
    if len(da.dims) != 3:
        raise ImportError(
            f"Extra dimensions found during import - there should be only "
            + f"three dimensions after degenerate dimensions are dropped but "
            + f"found {len(da.dims)} i.e. {da.dims}."
        )

    # Drop coordinates that are not associated with a dimension. Often you seen
    # height or level coming in as a coordinate, when it is perhaps more appropriate as
    # an attribute. However, different models handle this differently, and some have
    # already dropped it. The different between the two can cause problems when we
    # come to the point of merging ensemble members.
    for thisCoord in da.coords.keys():
        if len(da[thisCoord].dims)==0:
            da.attrs[thisCoord]=da[thisCoord].values
            da= da.drop_vars(thisCoord)


    #-------------End modified defaultImport--------------

    #Modify the datetime variable
    if "datetime" in da.dims:
        da = da.rename({"datetime": "time"})
    elif "date" in da.dims:
        da = da.rename({"date": "time"})
    elif "time" not in da.dims:
        raise ValueError("Cannot figure out how to handle naming of time dimension.")

    #Set lon,lat coordinates
    da.lon.attrs['standard_name']="longitude"
    da.lon.attrs['units']="degrees_east"
    da.lat.attrs['standard_name']="latitude"
    da.lat.attrs['units']="degrees_north"

    #Set temperature units
    da.attrs["units"]="degC"

    #Calculate daily averages
    if varCode=="tas":
        da=da.resample(time='D').mean()
    elif varCode=="tasmax":
        da=da.resample(time='D').max()
    else:
        raise ValueError(f"Unknown variable ID {config["inputs"][inpID]["varID"]}")

    return(da)

