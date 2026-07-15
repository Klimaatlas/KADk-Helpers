# Import KA data
#
# Import Klimaatlas daily-biasadjusted data.
# Generally this goes straight in, but are a few problems with the temperature units that
# need to be tweaked first.

import KAPy

def import_KA(input_files,
              variable_code,
              internal_variable_name,
              checks,
              cutout_arguments,
              **kwargs):
    # Import --------------------------------------
    #Import using the default import functionality
    da=KAPy.default_import(input_files=input_files,
                            variable_code=variable_code,
                            internal_variable_name=internal_variable_name,
                            checks=checks)

    #Post import modifications ------------
    # Correct temperature units
    if da.attrs['units']=="degrees C":
        da.attrs['units']="degC"


    #Apply cutouts-----------------
    if cutout_arguments["method"] == "lonlatbox":
        da=KAPy.cutout_lonlat(da,**cutout_arguments,variable_code=variable_code)

    return(da)


# Example --------------------------

if __name__ == "__main__":
    #Setup
    input_files=["/dmidata/projects/klimaatlas/archive/v2025a/daily_bias_corrected/1KM/tas_KGDK-1_NCC-NorESM1-M_historical_r1i1p1_SMHI-RCA4_v1_day_19810101-20101231.nc"]

    #Import
    da=import_KA(input_files,
                 variable_code="tas",
                 internal_variable_name="tas",
                 checks="all",
                 cutout_arguments={"method":"none"})
    