import KAPy
import xarray as xr

def import_CERRA(input_files,variable_code,internal_variable_name, checks, cutout_arguments,**kwargs):

    """
    Import CERRA data

    The CERRA dataset uses defaultImport(), but requires the following modifications
     * the valid_time dimension needs to be renamed to time
     * the dataset is on a three-hourly time steps - need to calculate daily values from there.
     * use the native chunking of the dataset to start with

    Args:
        config (_type_): _description_
        input_files (_type_): _description_
        inpID (_type_): _description_
    """

    #Import using the default import functionality
    da=KAPy.default_import(input_files=input_files,
                            variable_code=variable_code,
                            internal_variable_name=internal_variable_name,
                            checks=checks,
                            chunks={})  #Use native chunking
    
    #Modify the datetime variable
    da = da.rename({"valid_time": "time"})

    #Correct the coordinates attribute, which has been invalidated
    #when we drop expver and rename valid_time to time:
    da.encoding['coordinates'] = "latitude longitude"

    #Apply cutouts-----------------
    if cutout_arguments["method"] == "lonlatbox":
        da=KAPy.cutout_lonlat(da,**cutout_arguments,variable_code=variable_code)

    #Calculate daily averages
    if variable_code=="tas":
        da=da.resample(time='D').mean()
    elif variable_code=="tasmax":
        da=da.resample(time='D').max()
    elif variable_code=="tasmin":
        da=da.resample(time='D').min()
    else:
        raise ValueError(f"Unknown variable ID {config["inputs"][inpID]["varID"]}")

    return(da)

