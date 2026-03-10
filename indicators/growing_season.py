# Calculate the length of the growing season
# The description in Klimaatlas is as follows:
# Vækstsæsonens længde er antal døgn fra årets første 6 sammenhængende dage med daglige middeltemperaturer over 5 °C,
# til årets sidste 6 sammenhængende dage med daglige middeltemperaturer over 5 °C.

import xarray as xr
import numpy as np
import xclim

def growingSeason(tas):
    tas=tas.unify_chunks()
    #We use xclim as a basis for the calculation
    #This more or less hits the spot straight away
    season=xclim.atmos.growing_season_length(tas,thresh='5.0 degC',window=6)

    #Then we average across the years
    out=season.mean(dim="time")
    return out

def growingSeasonStart(tas):
    tas=tas.unify_chunks()
    # Take function directly from xclim
    gsStart=xclim.atmos.growing_season_start(tas,thresh='5.0 degC', window=6)

    #The Klimaatlas definition is that we use a right-aligned window, whereas
    #xclim uses left-aligned. We therefore need to add 6 extra days
    gsStart=gsStart+6

    #Average over the period
    out=gsStart.mean(dim="time")

    return out


def growingSeasonEnd(tas):
    tas=tas.unify_chunks()
    # Take function directly from xclim
    gsEnd=xclim.atmos.growing_season_end(tas,thresh='5.0 degC', window=6)

    #The Klimaatlas definition is that we use a right-aligned window, whereas
    #xclim uses left-aligned. We therefore need to add 6 extra days
    gsEnd=gsEnd+6

    #Average over the period
    out=gsEnd.mean(dim="time")

    return out

""" 
#Setup for debugging with VS code
import os
print(os.getcwd())
from KAPy.workflow import KAPy
config=KAPy.getConfig("./config/config.yaml")  
wf=KAPy.getWorkflow(config)
%matplotlib inline
import matplotlib.pyplot as plt
import cftime
import datetime

#Plotting function to check definitions
inFile=['outputs/03.calibration/BC-KGDK-tas/tas_BC-KGDK_KGDK_historical+rcp26_CNRM-CERFACS-CNRM-CM5_r1i1p1_GERICS-REMO2015_v2.nc']
this = KAPy.helpers.readFile(inFile[0])

thisYear = 2006
selThis={'x':10,'y':10}
tas=this.isel(selThis).sel(time=str(thisYear)).compute()
threshold=5

#Check for values higher than the threshold, taking
#care that the units are correct
aboveThresh=tas > threshold

#Use growingSeasonStart code
growingDay=xclim.core.units.convert_units_to(tas,"degC") >threshold
rollSumRight=growingDay.rolling({'time': 6},center=False).sum()

#The first period is easier, but we want the rolling sum to be left aligned here
#which requires a shift
rollSumLeft=rollSumRight.shift(time=-5)
firstSatisfied=xr.where(rollSumLeft==6,
                            rollSumLeft.time,
                            None)
firstDay=firstSatisfied.groupby("time.year").min()

#Xclim growing season functions.
#We use indicators here, as they give prettier output
xcStart=xclim.atmos.growing_season_start(tas,thresh='5.0 degC', window=5)
xcEnd=xclim.atmos.growing_season_end(tas,thresh='5.0 degC', window=5)
season=xclim.atmos.growing_season_length(tas,thresh='5.0 degC',window=5)

idx=70
tas.isel({'time': slice(None, idx)}).plot.line("o-",label="tasmax")
aboveThresh.isel({'time': slice(None, idx)}).plot.line("-x",label="aboveThresh")
plt.axhline(y=threshold,color="red",label="Threshold")
plt.axvline(x=cftime.DatetimeGregorian(thisYear, 1, 1,12) + datetime.timedelta(days=xcStart.data[0]-1),
                color="green",label="Xclim start")
plt.legend()
plt.savefig("Season_start.png", dpi=300)

idx=300
tas.isel({'time': slice(idx,None)}).plot.line("o-",label="tasmax")
aboveThresh.isel({'time': slice(idx,None)}).plot.line("-x",label="aboveThresh")
plt.axhline(y=threshold,color="red",label="Threshold")
plt.axvline(x=cftime.DatetimeGregorian(thisYear, 1, 1) + datetime.timedelta(days=xcEnd.data[0]-1),
            color="green",label="Xclim end")
plt.legend()
plt.savefig("Season_end.png", dpi=300)


"""





