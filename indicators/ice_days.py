"""
ice_days.py

Generic implementation of DMI's approach to calculating ice days. An ice day is identified
when the daily maximum temperature (tasmax) is strictly below a given threshold (default 0°C).
Note that we count the number of days where the maximum temperature does not exceed the freezing
point, rather than event spans, and then calculate the annual mean across all years.

Parameters
----------
tasmax : xr.DataArray
    Daily maximum temperature. The time dimension should be called 'time' and have daily
    resolution. Ice day statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
threshold : float
    Temperature threshold in degrees C, below which ice day counting is triggered.

Returns
-------
xr.DataArray
    Mean annual number of ice days.
"""

# Public API-----------------
__all__ = ["iceDays", "count_ice_days"]

# Imports -------------------
import xarray as xr
import xclim
import numpy as np


# Private helper functions ---------------------------------
def _identify_ice_days(
    tasmax: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Identify ice days for a given threshold and return a true/false mask
    for those points in time/space where the maximum temperature is below the threshold.
    """
    # Assert what we have
    if "time" not in tasmax.dims:
        raise ValueError("tasmax must contain a 'time' dimension")
    if "units" not in tasmax.attrs:
        raise ValueError("tasmax must define units")

    # Try xarray's native inference first
    time_resolution = xr.infer_freq(tasmax.time)

    # Fallback if infer_freq returns None (common with short datasets or climate calendars)
    if time_resolution is None:
        if len(tasmax.time) > 1:
            # Calculate the median delta between time steps in days
            time_deltas = tasmax.time.diff(dim="time")
            # Convert to days depending on whether it is a standard datetime or cftime
            if np.issubdtype(time_deltas.dtype, np.timedelta64):
                median_days = np.median(time_deltas.dt.days)
            else:  # Handle cftime Objects
                median_days = np.median([t.days for t in time_deltas.values])

            if median_days == 1:
                time_resolution = "D"

    if time_resolution not in ["D", "1D"]:
        raise ValueError(
            f"tasmax must have daily time resolution, 'D', but received '{time_resolution}'."
        )

    # Ensure temperature units are converted to degC
    tasmax_degC = xclim.core.units.convert_units_to(tasmax, "degC")
    above_threshold = tasmax_degC < threshold

    return above_threshold, tasmax_degC


# Generic public function -----------------
def count_ice_days(tasmax: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Calculate the mean annual number of ice days.
    """
    # Identify the ice days
    below_threshold, _ = _identify_ice_days(tasmax=tasmax, threshold=threshold)

    # Convert boolean mask to integer, sum per year, and then average over all years
    days_below_threshold = below_threshold.astype(int).groupby("time.year").sum(dim="time").mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = tasmax.isnull().any(dim="time")
    result = days_below_threshold.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def iceDays(tasmax: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate ice days using a 0°C threshold.
    """
    return count_ice_days(tasmax, threshold=0.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing (21 days)
    t = np.arange(0, 21)
    # Simulate a temperature cycle going below and above 0°C
    src_y = -3.0 + 5.0 * np.sin(t / 20 * 2 * np.pi)

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
            name="tasmax",
            attrs={"units": "degC"},
        )

        # Identify thresholds
        this_threshold = 0.0
        below_threshold, tasmax_degC = _identify_ice_days(
            tasmax=da, threshold=this_threshold
        )

        # Plot diagnostics (only for the full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da.plot.line("o", label="tasmax")
            (this_threshold - 0.5 + below_threshold).plot.line(
                "-x", label="Ice Day Detection (<0°C)"
            )
            plt.axhline(y=this_threshold, color="red", linestyle="--")
            plt.legend()
            plt.title(figure_title)
            plt.ylabel("Temperature (°C)")
            plt.show()

        # Get number of days
        out = iceDays(da)
        print(f"Mean annual number of ice days: {out.data}")
