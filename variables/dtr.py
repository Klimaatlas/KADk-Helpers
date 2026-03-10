import xarray as xr

#Diurnal temperature range and skewness

def dtr_skew(tasmax,tasmin,tas):
    dtr=(tasmax-tasmin) 
    dtr.attrs["units"] = "degC"

    skew=tas-(tasmax+tasmin)/2 
    skew.attrs["units"] = "degC"

    rtn=xr.Dataset({"dtr": dtr,
                   "skew": skew})

    return(rtn)
    

#Inversion function
def invert_dtr_skew(tas,skew,dtr):
    tasmax=tas-skew+dtr/2
    tasmax.attrs['units']="degC"

    tasmin=tas-skew-dtr/2
    tasmin.attrs['units']="degC"

    rtn=xr.Dataset({"tasmin": tasmin,
                   "tasmax": tasmax})

    return(rtn)
