import KAPy
import xarray as xr

def import_CERRA(inFiles,varCode,internalVarName, checks, cutoutArgs,**kwargs):

    """
    Import CERRA data

    The CERRA dataset uses defaultImport(), but requires the following modifications
     * the valid_time dimension needs to be renamed to time
     * the dataset is on a three-hourly time steps - need to calculate daily values from there.
     * use the native chunking of the dataset to start with

    Args:
        config (_type_): _description_
        inFiles (_type_): _description_
        inpID (_type_): _description_
    """

    #Import using the default import functionality
    da=KAPy.defaultImport(inFiles=inFiles,
                            varCode=varCode,
                            internalVarName=internalVarName,
                            checks=checks,
                            chunks={})  #Use native chunking
    
    #Modify the datetime variable
    da = da.rename({"valid_time": "time"})

    #Apply cutouts-----------------
    if cutoutArgs["method"] == "lonlatbox":
        da=KAPy.cutout_lonlat(da,**cutoutArgs,varCode=varCode)

    #Calculate daily averages
    if varCode=="tas":
        da=da.resample(time='D').mean()
    elif varCode=="tasmax":
        da=da.resample(time='D').max()
    elif varCode=="tasmin":
        da=da.resample(time='D').min()
    else:
        raise ValueError(f"Unknown variable ID {config["inputs"][inpID]["varID"]}")

    return(da)

