"""
snowfall_days.py

Generic implementation of DMI's approach to calculating snowfall days. A snowfall day is identified
when the estimated daily snowfall amount strictly exceeds a given threshold (default 1.0 mm).
The fraction of precipitation falling as snow (alpha) is determined linearly based on daily mean
temperature (tas): alpha is 1 if tas <= 0°C, 0 if tas >= 2°C, and 1 - tas/2 in between.
The script counts the annual number of snowfall days and returns the long-term mean.

Parameters
----------
pr : xr.DataArray
    Daily precipitation amount. The time dimension must be called 'time' and have daily resolution.
tas : xr.DataArray
    Daily mean temperature. The time dimension must be called 'time' and have daily resolution.
threshold : float
    Snowfall threshold in mm, above which a snowfall day is triggered.

Returns
-------
xr.DataArray
    Mean annual number of snowfall days.
"""

# Public API-----------------
__all__ = ["snowfallDays", "count_snowfall_days"]

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


def _identify_snowfall_days(
    pr: xr.DataArray, tas: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Identify snowfall days for a given threshold and return a true/false mask
    along with the calculated daily snowfall amount array.
    """
    # Assert what we have
    if "time" not in pr.dims or "time" not in tas.dims:
        raise ValueError("Inputs must contain a 'time' dimension")
    if "units" not in pr.attrs or "units" not in tas.attrs:
        raise ValueError("Both 'pr' and 'tas' must define units")

    # Assert daily time resolution using the robust validator
    _assert_daily_resolution(pr.time, name="pr")
    _assert_daily_resolution(tas.time, name="tas")

    # Ensure precipitation units are converted to mm
    pr_mm = xclim.core.units.convert_units_to(pr, "mm")

    # Handle temperature units (convert Kelvin to Celsius if necessary)
    tas_c = tas.copy()
    if tas_c.attrs.get("units") in ["K", "Kelvin"]:
        tas_c = tas_c - 273.15

    # Calculate alpha (fraction of precipitation falling as snow)
    # alpha is 1 if T <= 0, 0 if T >= 2, and linear transition in between
    alpha = xr.where(
        tas_c <= 0,
        1.0,
        xr.where(tas_c >= 2, 0.0, 1.0 - (tas_c / 2.0))
    )

    # Calculate daily snowfall amount
    snowfall = pr_mm * alpha

    # Check for values strictly higher than the threshold
    above_threshold = snowfall > threshold

    return above_threshold, snowfall


# Generic public function -----------------
def count_snowfall_days(pr: xr.DataArray, tas: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Calculate the mean annual number of snowfall days exceeding a specific threshold.
    """
    # Identify the snowfall days
    above_threshold, _ = _identify_snowfall_days(pr=pr, tas=tas, threshold=threshold)

    # Convert boolean mask to integer and calculate annual sum, then mean across all years
    snow_days = above_threshold.astype(int)
    annual_sum = snow_days.groupby("time.year").sum(dim="time").mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = pr.isnull().any(dim="time") | tas.isnull().any(dim="time")
    result = annual_sum.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def snowfallDays(pr: xr.DataArray, tas: xr.DataArray) -> xr.DataArray:
    """
    Calculate snowfall days strictly exceeding a 1.0 mm threshold.
    """
    return count_snowfall_days(pr=pr, tas=tas, threshold=1.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing
    t = np.arange(0, 30)

    # Simulate a dropping temperature crossing the 0-2 degC boundary
    tas_values = 4.0 - (t * 0.25)  # Starts at 4.0C, drops past 0C
    # Simulate steady precipitation of 4mm every day
    pr_values = np.repeat(4.0, len(t))

    # We consider two versions of this data - one with the full dataset
    # and one with NaNs in the time series corresponding to a masked field over water
    for add_nans in [False, True]:
        if not add_nans:
            figure_title = "Full dataset"
            tas_test = tas_values.copy()
            pr_test = pr_values.copy()
        else:
            figure_title = "Dataset with NaNs added"
            tas_test = tas_values.copy()
            pr_test = pr_values.copy()
            pr_test[-5:-3] = np.nan
        print(f"{figure_title}----------------------------------")

        # Wrap into DataArrays directly to match the new signature
        time = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")

        da_pr = xr.DataArray(pr_test, dims=["time"], coords={"time": time}, attrs={"units": "mm"})
        da_tas = xr.DataArray(tas_test, dims=["time"], coords={"time": time}, attrs={"units": "degC"})

        # Identify thresholds and calculate snowfall array
        this_threshold = 1.0
        above_threshold, snowfall_amt = _identify_snowfall_days(
            pr=da_pr, tas=da_tas, threshold=this_threshold
        )

        # Plot diagnostics (only for the full dataset setup visualization)
        if not add_nans:
            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            ax1.plot(t, tas_test, 'g-', label="Temperature (°C)")
            ax2.plot(t, snowfall_amt, 'b-^', label="Calculated Snowfall (mm)")
            ax2.plot(t, above_threshold * 3, 'r-x', label="Snow Day Detected (>1mm)")

            ax1.set_xlabel('Time (Days)')
            ax1.set_ylabel('Temperature', color='g')
            ax2.set_ylabel('Snowfall / Detection', color='b')
            plt.title(figure_title)
            plt.axhline(y=this_threshold, color="gray", linestyle="--")

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
            plt.show()

        # Get number of days using direct parameters
        out = snowfallDays(pr=da_pr, tas=da_tas)
        print(f"Number of snowfall days detected (>1.0 mm): {out.data}")
