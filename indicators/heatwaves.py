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


def DMIHeatwave(tasmax: xr.DataArray,
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

    #A heatwave is defined as three days in a row but only the first day
    #counts towards the number of HW days. 
    #durationCriteria=aboveThresh.rolling({"time":3}).sum()==3

    #Get number of days
    #out=durationCriteria.groupby("time.year").sum().mean(dim="year")
    out=aboveThresh.groupby("time.year").sum().mean(dim="year")

    #TODO: Discuss how we handle null values
    #    # RETTELSE: Vi tjekker om der mangler data langs 'time' aksen, 
    # # men bevarer de rumlige dimensioner ('rlat', 'rlon')
    # nan_mask = tasmax.isnull().any(dim="time")
    
    # # Anvend masken direkte på out
    # return out.where(~nan_mask)

    return out



# Main functions (to be called externally) ----------------
def heatwaveDays(tasmax):
    """Calculate heatwave days using a 28°C threshold."""    
    return DMIHeatwave(tasmax,threshold=28)

def warmwaveDays(tasmax):
    """Calculate warmwave days using a 25°C threshold."""
    return DMIHeatwave(tasmax,threshold=25)



# Validation --------------------------------------
if __name__ == "__main__":
    #Imports for use here
    import matplotlib.pyplot as plt    
    import numpy as np
    import pandas as pd

    #Create a synthetic dataset for testing
    #We create a sinusoidal temperature time series with two components:
    # * a low frequency element ranging from 20 to 30 degrees and back over 90 days. 
    # * a high frequency element that simulates (deterministic) noise
    t=np.arange(0,31)
    y=20+10*np.sin(t/180*2*np.pi)+0.5*np.sin(t*np.exp(1))
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
     
    #Get number of days
    out=DMIHeatwave(da,threshold=thisThreshold)
    print(out)


