"""
wind_days.py

Generic implementation of DMI's approach to calculating extreme wind days. A wind day is identified
when the daily maximum near-surface wind speed (sfcWindmax) strictly exceeds a given threshold.
Note that we count the number of days where we are above the threshold, rather than the number of 
storm events, and then calculate the annual mean across all years.

Parameters
----------
sfcWindmax : xr.DataArray
    Daily maximum near-surface wind speed. The time dimension should be called 'time' and have daily
    resolution. Wind statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
threshold : float
    Wind speed threshold in m/s, above which counting is triggered.

Returns
-------
xr.DataArray
    Mean annual number of extreme wind days.
"""

# Public API-----------------
__all__ = ["daysAbove25ms", "count_wind_days"]

# Imports -------------------
import xarray as xr
import xclim
import numpy as np


# Private helper functions ---------------------------------
def _assert_daily_resolution(time_coord, name: str = "sfcWindmax"):
    """Internal helper to validate time resolution with fallback handling."""
    # Try xarray's native inference first
    time_resolution = xr.infer_freq(time_coord)
    
    # Fallback if infer_freq returns None (common with chunked or climate model calendars)
    if time_resolution is None:
        if len(time_coord) > 1:
            # Calculate the median delta between time steps in days
            time_deltas = time_coord.diff(dim="time")
            # Convert to days depending on whether it is a standard datetime or cftime
            if np.issubdtype(time_deltas.dtype, np.timedelta64):
                median_days = np.median(time_deltas.dt.days)
            else:  # Handle cftime Objects
                median_days = np.median([t.days for t in time_deltas.values])
                
            if median_days == 1:
                time_resolution = "D"

    if time_resolution not in ["D", "1D"]:
        raise ValueError(
            f"{name} must have daily time resolution, 'D', but received '{time_resolution}'."
        )


def _identify_wind_days(
    sfcWindmax: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Identify wind days for a given threshold and return a true/false mask
    for those points in time/space that exceed the wind speed threshold.
    """
    # Assert what we have
    if "time" not in sfcWindmax.dims:
        raise ValueError("sfcWindmax must contain a 'time' dimension")
    if "units" not in sfcWindmax.attrs:
        raise ValueError("sfcWindmax must define units")

    # Assert daily time resolution using the robust validator
    _assert_daily_resolution(sfcWindmax.time, name="sfcWindmax")

    # Check for values higher than the threshold, taking
    # care that the units are correct (m/s)
    wind_ms = xclim.core.units.convert_units_to(sfcWindmax, "m/s")
    above_threshold = wind_ms > threshold

    return above_threshold, wind_ms


# Generic public function -----------------
def count_wind_days(sfcWindmax: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Calculate the mean annual number of wind days exceeding a specific threshold.
    """
    # Identify the wind days
    above_threshold, _ = _identify_wind_days(sfcWindmax=sfcWindmax, threshold=threshold)

    # Get number of days above the threshold grouped by year and then average
    days_above_threshold = above_threshold.groupby("time.year").sum(dim="time").mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give a NaN. Create a mask and apply.
    nan_mask = sfcWindmax.isnull().any(dim="time")
    result = days_above_threshold.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def daysAbove25ms(sfcWindmax: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate wind days strictly exceeding a 25 m/s threshold.
    """
    return count_wind_days(sfcWindmax, threshold=25.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing
    # We create a wind time series simulating a brief storm event
    t = np.arange(0, 21)
    src_y = 12 + 10 * np.sin(t / 180 * 2 * np.pi) + 8 * np.exp(-((t - 10) ** 2) / 4)

    # We consider two versions of this data - one with the full dataset
    # and one with NaNs in the time series corresponding to a masked field over water
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
            name="sfcWindmax",
            attrs={"units": "m/s"},
        )

        # Identify thresholds
        this_threshold = 25.0
        above_threshold, wind_speed = _identify_wind_days(
            sfcWindmax=da, threshold=this_threshold
        )

        # Plot
        da.plot.line("o", label="sfcWindmax")
        (this_threshold - 0.5 + above_threshold).plot.line(
            "-x", label="Storm detection"
        )
        plt.axhline(y=this_threshold, color="red", linestyle="--")
        plt.legend()
        plt.title(figure_title)
        plt.ylabel("Wind Speed (m/s)")
        plt.show()

        # Get number of days (Rettet fra det fejlagtige count_25ms_wind_days)
        out = daysAbove25ms(da)
        print(f"Number of wind days detected (>25 m/s): {out.data}")
