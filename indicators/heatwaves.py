"""
heatwaves.py

Generic implementation of DMI's approach to calculating heatwaves. A heatwave day is identified
when the 3-day rolling mean (right-aligned) of daily maximum temperature exceeds the threshold.
In this way, the first two days of each heatwave don't contribute to the count, even though they can
be above the threshold. Note also that we count the number of days where we are in a heatwave state,
rather than the number of heatwave events.

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

# Public API-----------------
__all__ = ["heatwave_days", "warmwave_days", "count_heatwave_days"]

# Imports -------------------
import xarray as xr
import xclim


# Private helper functions ---------------------------------
def _identify_heatwave_days(
    tasmax: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Identify heatwave days for a given threshold and return a true/false mask
    for those points in time/space that are in a heatwave state.
    """
    # Assert what we have
    if "time" not in tasmax.dims:
        raise ValueError("tasmax must contain a 'time' dimension")
    if "units" not in tasmax.attrs:
        raise ValueError("tasmax must define units")

    # Assert daily time resolution
    time_resolution = xr.infer_freq(tasmax.time)
    if time_resolution not in ["D", "1D"]:
        raise ValueError(
            f"tasmax must have daily time resolution, 'D', but received '{time_resolution}'."
        )

    # Apply three day rolling mean - right aligned by default
    rolling_mean = tasmax.rolling(time=3, min_periods=3).mean()

    # Check for values higher than the threshold, taking
    # care that the units are correct
    above_threshold = (
        xclim.core.units.convert_units_to(rolling_mean, "degC") > threshold
    )

    return above_threshold, rolling_mean


# Generic public function -----------------
def count_heatwave_days(tasmax: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Calculate the mean annual number of heatwave days.
    """
    # Identify the heatwaves
    above_threshold, _ = _identify_heatwave_days(tasmax=tasmax, threshold=threshold)

    # Get number of days above the threshold
    days_above_threshold = above_threshold.groupby("time.year").sum().mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = tasmax.isnull().any(dim="time")
    result = days_above_threshold.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def heatwave_days(tasmax: xr.DataArray) -> xr.DataArray:
    """
    Calculate heatwave days using a 28°C threshold.
    """
    return count_heatwave_days(tasmax, threshold=28)


def warmwave_days(tasmax: xr.DataArray) -> xr.DataArray:
    """
    Calculate warmwave days using a 25°C threshold.
    """
    return count_heatwave_days(tasmax, threshold=25)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Create a synthetic dataset for testing
    # We create a sinusoidal temperature time series with the following components
    # * a low frequency element simulating start of a season or heatwave
    # * a high frequency element that simulates (deterministic) noise
    t = np.arange(0, 21)
    src_y = 22 + 10 * np.sin(t / 180 * 2 * np.pi) + 0.5 * np.sin(t * np.exp(1))

    # We consider two versions of this data - one with the full dataset
    # and one with NANs in the time series corresponding to a masked field over water
    for add_nans in [False, True]:
        if not add_nans:
            figure_title = "Full dataset"
            y = src_y.copy()
        else:
            figure_title = "Dataset with NaNs added"
            y = src_y.copy()
            y[-8:-6] = np.nan
        print(f"{figure_title}----------------------------------")

        # Wrap into a dataarray
        time = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")
        da = xr.DataArray(
            y,
            dims=["time"],
            coords={"time": time},
            name="tasmax",
            attrs={"units": "degC"},
        )

        # Identify thresholds
        this_threshold = 25
        above_threshold, rolling_mean = _identify_heatwave_days(
            tasmax=da, threshold=this_threshold
        )

        # Plot
        da.plot.line("o", label="tasmax")
        rolling_mean.plot.line("+-", label="Three-day rollingmean")
        (this_threshold - 0.5 + above_threshold).plot.line(
            "-x", label="Heatwave detection"
        )
        plt.axhline(y=this_threshold, color="red")
        plt.legend()
        plt.title(figure_title)
        plt.show()

        # Get number of days
        out = count_heatwave_days(da, threshold=this_threshold)
        print(f"Number of heatwave days detected: {out.data}")
