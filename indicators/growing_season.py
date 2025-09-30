# Calculate the length of the growing season
# The description in Klimaatlas is as follows:
# Vækstsæsonens længde er antal døgn fra årets første 6 sammenhængende dage med daglige middeltemperaturer over 5 °C,
# til årets sidste 6 sammenhængende dage med daglige middeltemperaturer over 5 °C.

"""
#Setup for debugging with VS code
import os
print(os.getcwd())
os.chdir("../../KAPy/workflow")
import KAPy
os.chdir("../..")
config=KAPy.getConfig("./config/config.yaml")  
wf=KAPy.getWorkflow(config)
indID='i011'
outFile=list(wf['indicators'][indID])[0]
inFile=wf['indicators'][indID][outFile]
tas = KAPy.helpers.readFile(inFile[0])
tas=tas.compute()
%matplotlib inline
import matplotlib.pyplot as plt
"""

import xarray as xr
import numpy as np
import xclim

def growingSeason(tas):
    #The calculation strategy here is that we find the timing of the first and last 
    #six-day periods and subtract the timestamps to get the time difference between
    #them. This is an ok approach, but doesn't account exactly for differences in
    #length of the calendar in different climate models. Should incorporate later

    #Load into memory so that dask behaves itself (for now)
    tas=tas.compute()

    #Look for days over 5 degrees and apply a rolling sum (rightalign by default). 
    growingDay=xclim.core.units.convert_units_to(tas,"degC") >5
    rollSumRight=growingDay.rolling({'time': 6},center=False).sum()

    #We start with the last period over the threshold, as this is right-aligned 
    # and therefore easier easier
    lastSatisfied=xr.where(rollSumRight==6,
                           rollSumRight.time,
                           None)
    lastDay=lastSatisfied.groupby("time.year").max()

    #The first period is easier, but we want the rolling sum to be left aligned here
    #which requires a shift
    rollSumLeft=rollSumRight.shift(time=-5)
    firstSatisfied=xr.where(rollSumLeft==6,
                               rollSumLeft.time,
                               None)
    firstDay=firstSatisfied.groupby("time.year").min()

    #Now get the growing season from the time difference between the two
    #This step doesn't seem to be implemented in dask, so need to do it by hand
    seasonLength=(lastDay-firstDay)/np.timedelta64(1, 'D')

    #Convert to an numeric format, calculate the average, and output
    out=seasonLength.mean(dim="year")
    return out




