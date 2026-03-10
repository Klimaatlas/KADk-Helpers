# Collection of functions to handle importing of differing data formats
import KAPy
from KAPy import helpers
import xarray as xr
from cdo import Cdo
import xesmf as xe

def import_KGDK(inFiles,varCode,internalVarName, checks,**kwargs):
    """
    Import KGDK data

    Imports data from Klimagrid Danmark (KGDK) as an object for use in Klimaatlas. 
    We need to do much of the import ourselves here, as there are a couple of issues
    with the default dataset that this code aims to correct
    * time steps in input file are duplicated, so we only keep the first 365
    * need to set grid correctly

    Parameters
    ----------
    config : _type_
        Configuration object
    inFiles : _type_
        List of files to import
    inpID : _type_
        ID of the configuration

    Returns
    ----------
    xarray dataarray
        The KGDK dataset represented as a dataarray in xarray

    """
    
    """
    #Debugging setup
    inFiles=["./inputs/tas_klimagrid_2023.nc"]
    inpID="KGDK"
    """
    #Read input files into RAM
    time_coder=xr.coders.CFDatetimeCoder(use_cftime=True)
    dsIn =xr.open_mfdataset(inFiles,
                            combine='nested',
                            decode_times=time_coder,
                            join="override", 
                            concat_dim='time')
    dsIn=dsIn.compute()

    #Regrid 20km tas data onto 10km tasmax grid to allow calculation of e.g. skewness
    if varCode=="tas":
        refGrd=helpers.readFile("/dmidata/projects/klimaatlas/dev/inputs/KGDK/tasmaxobs_2004-2019.nc")
        regrdr=xe.Regridder(dsIn,refGrd,
                            method="nearest_s2d",
                            unmapped_to_nan=False)  
        dsIn=regrdr(dsIn,keep_attrs=True)

    # Select the desired variable and rename it
    ds = dsIn.rename({internalVarName: varCode})

    #Specify the coordinate system to be UTM. First add the crs information
    #Thanks to ChatGPT for help setting this up, particularly the coordinate definition
    # crs_attrs = {
    #     "grid_mapping_name": "transverse_mercator",
    #     "scale_factor_at_central_meridian": 0.9996,
    #     "longitude_of_central_meridian": 9.0,  # Central meridian for UTM Zone 32
    #     "latitude_of_projection_origin": 0.0,
    #     "false_easting": 500000.0,  # Standard for UTM
    #     "false_northing": 0.0,  # Northern Hemisphere; use 10000000.0 for Southern Hemisphere
    #     "semi_major_axis": 6378137.0,  # WGS84 ellipsoid
    #     "inverse_flattening": 298.257223563,
    # }

    # Add the CRS variable with UTM Zone 32 projection attributes
    #ds["crs"] = xr.DataArray(0, attrs=crs_attrs)
    #ds[thisInp["varCode"]].attrs['grid_mapping']="crs"   

    #Set the CF dimension names correctly
    ds.y.attrs['axis']="Y"
    ds.y.attrs['standard_name']="projection_y_coordinate"
    ds.x.attrs['axis']="X"
    ds.x.attrs['standard_name']="projection_x_coordinate"
    ds.lon.attrs['standard_name']="longitude"
    ds.lon.attrs['units']="degrees_east"
    ds.lat.attrs['standard_name']="latitude"
    ds.lat.attrs['units']="degrees_north"
    ds[varCode].attrs['coordinates'] = "lon lat"

    #Ensure consistent ordering
    ds=ds.transpose("time","y","x")
    
    ## Convert to dataarray
    da = ds[varCode]  

    return(da)
