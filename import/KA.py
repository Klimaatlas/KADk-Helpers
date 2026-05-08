# Import KA data
#
# Import Klimaatlas daily-biasadjusted data.
# Generally this goes straight in, but are a few problems with the temperature units that
# need to be tweaked first.

#from KAPy.workflow import KAPy
import KAPy
import xarray as xr

def import_KA(inFiles,
              varCode,
              internalVarName,
              checks,
              cutoutArgs,
              **kwargs):
    # Import --------------------------------------
    #Import using the default import functionality
    da=KAPy.defaultImport(inFiles=inFiles,
                            varCode=varCode,
                            internalVarName=internalVarName,
                            checks=checks)

    #Post import modifications ------------
    # Correct temperature units
    if da.attrs['units']=="degrees C":
        da.attrs['units']="degC"


    #Apply cutouts-----------------
    if cutoutArgs["method"] == "lonlatbox":
        da=KAPy.cutout_lonlat(da,**cutoutArgs,varCode=varCode)

    return(da)


# Example --------------------------

if __name__ == "__main__":
    #Setup
    inFiles=["/dmidata/projects/klimaatlas/archive/v2025a/daily_bias_corrected/1KM/tas_KGDK-1_NCC-NorESM1-M_historical_r1i1p1_SMHI-RCA4_v1_day_19810101-20101231.nc"]

    #Import
    da=import_KA(inFiles,
                 varCode="tas",
                 internalVarName="tas",
                 checks="all",
                 cutoutArgs={"method":"none"})
    