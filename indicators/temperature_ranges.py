import xarray as xr

#Annual temperature range 
#
#Note that this is defined in Klimaatlas as the range of max(tas)-min(tas) within
#a year.
 
def annual_temperature_range(tas):
    annual_max=tas.resample(time="YS").max()
    annual_min=tas.resample(time="YS").min()
    annual_range=annual_max-annual_min
    out=annual_range.mean(dim="time")
    out.attrs['units']="degC"
  
    return out


#Daily temperature range 
#
#Defined as a the mean difference between tasmax and tasmin
#Note that dtr is also calculated as a variable in some cases e.g.
#as part of bias-adjusting tasmax and tasmin, and therefore the
#same result can be obtained by averaging dtr, rather than
#using tasmax and tasmin.
 
def daily_temperature_range(tasmax,tasmin):
    dtr=(tasmax-tasmin)
    out=dtr.mean(dim="time")
    out.attrs['units']="degC"
  
    return out


# Example --------------------------

if __name__ == "__main__":
    #Setup for debugging with VS code 
    #Assumes a working directory corresponding to the project root
    import sys
    from pathlib import Path
    sys.path.append("./KAPy/workflow")
    import KAPy

    configfile="./config/config.yaml"
    #configfile="./workflow/testing/config.yaml"

    config=KAPy.getConfig(configfile)  
    wf=KAPy.getWorkflow(config)

    # Annual temperature range -------------------------
    indID='i006'
    leaf=next(iter(wf['indicators'][indID]['input_dict']))
    inFiles=wf['indicators'][indID]['input_dict'][leaf]

    tas=KAPy.readFile(inFiles['tas'])

    out=annual_temperature_range(tas=tas)
    
    #Daily temperature range------------------------------
    indID='i007'
    leaf=next(iter(wf['indicators'][indID]['input_dict']))
    inFiles=wf['indicators'][indID]['input_dict'][leaf]

    tasmax=KAPy.readFile(inFiles['tasmax'])
    tasmin=KAPy.readFile(inFiles['tasmin'])

    out=daily_temperature_range(tasmax,tasmin)

    print("Done.")
    

    

