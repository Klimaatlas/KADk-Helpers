# Calculate number of heatwave and warmwave days
# A ’heatwave day’ is indicated when the average of the maximum temperature,
#  over at least three consecutive days, is above 28°C, not counting the
# first two days of each heatwave. A "warmwave" is the same definition for 25C

"""
#Setup for debugging with VS code
import os
print(os.getcwd())
os.chdir("KAPy/workflow")
import KAPy
os.chdir("../..")
config=KAPy.getConfig("./config/config.yaml")  
wf=KAPy.getWorkflow(config)
indID='i008'
outFile=list(wf['indicators'][indID])[0]
outFile='outputs/05.indicators/i008/i008_BC-KGDK_KGDK_historical+rcp85_NCC-NorESM1-M_r1i1p1_SMHI-RCA4_v1.nc'
inFile=wf['indicators'][indID][outFile]
tasmax = KAPy.helpers.readFile(inFile[0])
tasmax=tasmax.compute()
%matplotlib inline
import matplotlib.pyplot as plt


"""

import xarray as xr
import xclim

def genericHeatwave(tasmax,threshold=28):
    #Apply three day rolling mean - right aligned by default
    rollMean=tasmax.rolling({"time":3}).mean()

    #Check for values higher than the threshold, taking
    #care that the units are correct
    aboveThresh=xclim.core.units.convert_units_to(rollMean ,"degC") > threshold

    #A heatwave is defined as three days in a row but only the first day
    #counts towards the number of HW days. 
    #durationCriteria=aboveThresh.rolling({"time":3}).sum()==3

    #Get number of days
    #out=durationCriteria.groupby("time.year").sum().mean(dim="year")
    out=aboveThresh.groupby("time.year").sum().mean(dim="year")

    return out


def heatwaveDays(tasmax):
    return genericHeatwave(tasmax,threshold=28)

def warmwaveDays(tasmax):
    return genericHeatwave(tasmax,threshold=25)

""" 
#Plotting function to check definitions
this = KAPy.helpers.readFile(inFile[0])
selThis={'time':slice(None,30),'x':10,'y':10}
tasmax=this.isel(selThis).compute()

import xarray as xr
import numpy as np
threshold=25

# Load fredriks values into a NumPy array
tasmax = np.loadtxt("/dmidata/projects/klimaatlas/fbo/warmdaytest/CanESM_CLMcom_year2071_rcp85_lat10_lon15.txt")
da = xr.DataArray(values, dims="time", name="tasmax")

#And his indicators
ind=np.loadtxt("/dmidata/projects/klimaatlas/fbo/warmdaytest/CanESM_CLMcom_year2071_rcp85_lat10_lon15_warmdays.txt")
ind = xr.DataArray(ind, dims="time", name="indicator")+1

# Slice
thisSlice={"time": slice(160,190)}
da = da.isel(thisSlice)
ind=ind.isel(thisSlice)

#Apply three day rolling mean - right aligned
rollMean=da.rolling({"time":3}).mean()

#Check for values higher than the threshold, taking
#care that the units are correct
aboveThresh=rollMean > threshold

#A heatwave is defined as three days in a row but only the first day
#counts towards the number of HW days. 
durationCriteria=aboveThresh.rolling({"time":3}).sum()==3

#Get number of days
out=aboveThresh.sum().mean()
print(out)

da.plot.line("o-",label="tasmax")
rollMean.plot.line("x-",label="rollMean")
aboveThresh.plot.line("-x",label="aboveThresh")
#durationCriteria.plot.line("-x",label="durationCriteria")
ind.plot.line("-+",label="FBO warm day")
plt.axhline(y=threshold,color="red")
plt.legend()
plt.savefig("Warm_wave.png", dpi=300)


"""


