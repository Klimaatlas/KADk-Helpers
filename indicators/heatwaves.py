# Heatwaves
#
# Calculate number of heatwave and warmwave days based on the DMI definition.


# Imports -------------------
import xarray as xr
import xclim

# Helper functions ---------------------------------
def identifyHeatwaves(tasmax: xr.DataArray,
                      threshold: float):
    #Apply three day rolling mean - right aligned by default
    rollMean=tasmax.rolling({"time":3}).mean()

    #Check for values higher than the threshold, taking
    #care that the units are correct
    aboveThresh=xclim.core.units.convert_units_to(rollMean ,"degC") > threshold

    return aboveThresh, rollMean


def genericHeatwaveDays(tasmax: xr.DataArray,
                threshold: float):
    """
    Generic implementation of DMI's approach to calculating heatwaves. A ’heatwave day’ is 
    indicated when the average of the maximum temperature,  over at least three consecutive 
    days, is above a threshold - the first two days of each heatwave don't contribute to the count.

    Parameters
    ----------
    tasmax : xr.DataArray
        Daily maximum temperature. The time dimension should be called 'time' and have daily
        resolution. Heatwave statistics are calculated over other (spatial) dimensions but are not
        referenced directly, and can therefore have arbitrary names.
    threshold : float
        Temperature threshold in degrees C, above heatwave counting is triggered.

    Returns
    -------
    xr.DataArray
        Mean annual number of heatwave days. 
    """

    #Identify the heatwaves
    aboveThresh,_=identifyHeatwaves(tasmax=tasmax,threshold=threshold)

    #Get number of days above the threshold
    daysAboveThreshhold=aboveThresh.groupby("time.year").sum().mean(dim="year")

    #Check for missing data. Any missing values in the timeseries
    #should give an NaN. Create a mask and apply.
    nan_mask = tasmax.isnull().any(dim="time")
    res=daysAboveThreshhold.where(~nan_mask)

    return res



# Main functions (to be called externally) ----------------
def heatwaveDays(tasmax):
    """Calculate heatwave days using a 28°C threshold."""    
    return genericHeatwaveDays(tasmax,threshold=28)

def warmwaveDays(tasmax):
    """Calculate warmwave days using a 25°C threshold."""
    return genericHeatwaveDays(tasmax,threshold=25)



# Validation --------------------------------------
if __name__ == "__main__":
    #Imports for use here
    import matplotlib.pyplot as plt    
    import numpy as np
    import pandas as pd

    #Create a synthetic dataset for testing
    #We create a sinusoidal temperature time series with the following components
    # * a low frequency element simulating start of a season or heatwave
    # * a high frequency element that simulates (deterministic) noise
    t=np.arange(0,21)
    y=22+10*np.sin(t/180*2*np.pi)+0.5*np.sin(t*np.exp(1))
    
    #We consider two versions of this data - one with the full dataset
    #and one with NANs in the time series corresponding to a masked field over water
    for addNaNs in [False,True]:
        if not addNaNs:
            print("Full dataset---------------------")
        else:
            print("And now with NaNs----------------")
            y[-8:-6]=np.nan

        #Wrap into a dataarray
        time = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")
        da = xr.DataArray(
            y,
            dims=["time"],
            coords={"time": time},
            name="tasmax",
            attrs={"units": "degC"})
        
        #Identify thresholds
        thisThreshold=25
        aboveThresh,rollMean=identifyHeatwaves(tasmax=da,threshold=thisThreshold)

        #Plot 
        da.plot.line("o",label="tasmax")
        rollMean.plot.line("+-",label="rollMean")
        (thisThreshold-0.5+aboveThresh).plot.line("-x",label="aboveThresh")
        plt.axhline(y=thisThreshold,color="red")
        plt.legend()
        plt.show()
        
        #Get number of days
        out=genericHeatwaveDays(da,threshold=thisThreshold)
        print(f"Number of heatwave days detected: {out.data}")


