"""
freeze_thaw_days.py

Generic implementation of DMI's approach to calculating freeze-thaw days. A freeze-thaw day is
identified when the daily minimum temperature (tasmin) is strictly below freezing (0°C) AND the
daily maximum temperature (tasmax) is strictly above freezing (0°C) within the same 24-hour period.
The script counts the annual number of freeze-thaw days and returns the long-term mean across all years.

Parameters
----------
tasmax : xr.DataArray
    Daily maximum temperature. The time dimension must be called 'time' and have daily resolution.
tasmin : xr.DataArray
    Daily minimum temperature. The time dimension must be called 'time' and have daily resolution.
threshold : float
    Temperature threshold in degrees C defining the freezing point (default 0°C).

Returns
-------
xr.DataArray
    Mean annual number of freeze-thaw days.
"""

# Public API-----------------
__all__ = ["freezeThawDays", "count_freeze_thaw_days"]

# Imports -------------------
import xarray as xr
import xclim
import numpy as np


# Private helper functions ---------------------------------
def _assert_daily_resolution(time_coord, name: str = "DataArray"):
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


def _identify_freeze_thaw_days(
    tasmax: xr.DataArray, tasmin: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Identify freeze-thaw days for a given threshold and return a true/false mask
    along with the converted tasmin and tasmax arrays.
    """
    # Assert what we have
    if "time" not in tasmax.dims or "time" not in tasmin.dims:
        raise ValueError("Inputs must contain a 'time' dimension")
    if "units" not in tasmax.attrs or "units" not in tasmin.attrs:
        raise ValueError("Both 'tasmin' and 'tasmax' must define units")

    # Assert daily time resolution using the robust validator
    _assert_daily_resolution(tasmax.time, name="tasmax")
    _assert_daily_resolution(tasmin.time, name="tasmin")

    # Ensure temperature units are converted to degC
    tasmin_degC = xclim.core.units.convert_units_to(tasmin, "degC")
    tasmax_degC = xclim.core.units.convert_units_to(tasmax, "degC")

    # Condition: frost at night (tasmin < 0) AND thaw during the day (tasmax > 0)
    condition = (tasmin_degC < threshold) & (tasmax_degC > threshold)

    return condition, tasmin_degC, tasmax_degC


# Generic public function -----------------
def count_freeze_thaw_days(
    tasmax: xr.DataArray, tasmin: xr.DataArray, threshold: float = 0.0
) -> xr.DataArray:
    """
    Calculate the mean annual number of freeze-thaw days.
    """
    # Identify the freeze-thaw days
    condition, _, _ = _identify_freeze_thaw_days(tasmax=tasmax, tasmin=tasmin, threshold=threshold)

    # Convert boolean mask to integer, sum per year, and then average over all years
    days_above_threshold = condition.astype(int).groupby("time.year").sum(dim="time").mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = tasmin.isnull().any(dim="time") | tasmax.isnull().any(dim="time")
    result = days_above_threshold.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def freezeThawDays(tasmax: xr.DataArray, tasmin: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate freeze-thaw days using a 0°C threshold.
    """
    return count_freeze_thaw_days(tasmax=tasmax, tasmin=tasmin, threshold=0.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing (21 days)
    t = np.arange(0, 21)

    # Simulate a mean temperature line dropping, with a constant daily range of 6 degrees (+/- 3C)
    mean_temp = 2.0 - (t * 0.3)  # Drops from 2C down to -4C
    tasmax_values = mean_temp + 3.0
    tasmin_values = mean_temp - 3.0

    # We consider two versions of this data - one with the full dataset
    # and one with NaNs in the time series corresponding to a masked field over water
    for add_nans in [False, True]:
        if not add_nans:
            figure_title = "Full dataset"
            tasmax_test = tasmax_values.copy()
            tasmin_test = tasmin_values.copy()
        else:
            figure_title = "Dataset with NaNs added"
            tasmax_test = tasmax_values.copy()
            tasmin_test = tasmin_values.copy()
            tasmin_test[-8:-6] = np.nan
        print(f"{figure_title}----------------------------------")

        # Wrap into dataarrays
        time = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")
        da_max = xr.DataArray(tasmax_test, dims=["time"], coords={"time": time}, attrs={"units": "degC"})
        da_min = xr.DataArray(tasmin_test, dims=["time"], coords={"time": time}, attrs={"units": "degC"})

        # Identify freeze-thaw mask for plotting
        this_threshold = 0.0
        condition, tasmin_c, tasmax_c = _identify_freeze_thaw_days(
            tasmax=da_max, tasmin=da_min, threshold=this_threshold
        )

        # Plot diagnostics (only for the full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            plt.plot(t, tasmax_test, 'r-o', label="tasmax")
            plt.plot(t, tasmin_test, 'b-o', label="tasmin")
            (this_threshold - 0.5 + condition).plot.line(
                "g-x", label="Freeze-Thaw Detected (min<0 AND max>0)"
            )
            plt.axhline(y=this_threshold, color="black", linestyle="--")
            plt.legend()
            plt.title(figure_title)
            plt.ylabel("Temperature (°C)")
            plt.xlabel("Time (Days)")
            plt.show()

        # Get number of days using the updated positional/keyword arguments signature
        out = freezeThawDays(tasmax=da_max, tasmin=da_min)
        print(f"Mean annual number of freeze-thaw days: {out.data}")
