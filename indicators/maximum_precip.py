"""
maximum_precip.py

Generic implementation of DMI's approach to calculating multi-day maximum precipitation amounts.
This script calculates the maximum accumulated precipitation over a rolling window of N consecutive 
days within each year. The rolling sum is shifted to align with a specific target day in the window. 
Finally, the script returns the long-term mean annual maximum amount across all years.

Parameters
----------
pr : xr.DataArray
    Daily precipitation amount. The time dimension must be called 'time' and have daily
    resolution. Precipitation statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
period : int
    The window size in days for the rolling sum (e.g., 5 or 14 days).
target_day : int
    The specific day index within the N-day period where the result should be centered/placed (1-indexed).

Returns
-------
xr.DataArray
    Mean annual maximum N-day accumulated precipitation.
"""

# Public API-----------------
__all__ = ["maximum5dayPrecip", "maximum14dayPrecip", "count_maximum_precip"]

# Imports -------------------
import xarray as xr
import xclim
import numpy as np


# Private helper functions ---------------------------------
def _calculate_shifted_rolling_precip(
    pr: xr.DataArray, period: int, target_day: int
) -> xr.DataArray:
    """
    Calculate N-day rolling accumulated precipitation and shift the values 
    to be positioned on the designated target day of the window.
    """
    # Assert what we have
    if "time" not in pr.dims:
        raise ValueError("pr must contain a 'time' dimension")
    if "units" not in pr.attrs:
        raise ValueError("pr must define units")

    # NOTE: Manual time resolution validation has been removed.
    # KAPy/Snakemake often extracts subset seasons (e.g., summer only), which introduces 
    # large gaps in the 'time' coordinate between years. xarray's rolling operation 
    # handles these intervals appropriately based on index position.

    # Ensure precipitation units are converted to mm
    pr_mm = xclim.core.units.convert_units_to(pr, "mm")

    # Ensure chunks are unified across calculations
    pr_mm = pr_mm.unify_chunks()

    # Calculate rolling sum (value naturally lands on the LAST day of the period, center=False)
    total_rolling_sum = pr_mm.rolling(time=period, min_periods=period).sum()

    # Manually shift the value back to line up with the target day
    # Example for 14-day sum: lands on day 14, shift by (7 - 14) = -7 -> lands on day 7
    shift_amount = target_day - period
    shifted_rolling_sum = total_rolling_sum.shift(time=shift_amount)

    return shifted_rolling_sum


# Generic public function -----------------
def count_maximum_precip(pr: xr.DataArray, period: int, target_day: int) -> xr.DataArray:
    """
    Calculate the long-term mean annual maximum N-day accumulated precipitation.
    """
    # Calculate the shifted rolling precipitation
    shifted_rolling_sum = _calculate_shifted_rolling_precip(
        pr=pr, period=period, target_day=target_day
    )

    # Find the maximum level for each year individually
    max_precip_per_year = shifted_rolling_sum.groupby("time.year").max(dim="time")

    # Calculate the long-term mean across all years
    out = max_precip_per_year.mean(dim="year")
    out.attrs['units'] = "mm"

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = pr.isnull().any(dim="time")
    result = out.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def maximum5dayPrecip(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate the mean annual maximum 5-day accumulated precipitation (shifted to day 3).
    """
    return count_maximum_precip(pr, period=5, target_day=3)


def maximum14dayPrecip(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate the mean annual maximum 14-day accumulated precipitation (shifted to day 7).
    """
    return count_maximum_precip(pr, period=14, target_day=7)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing (30 days)
    t = np.arange(0, 30)
    
    # Background drizzle with a major 3-day heavy rainfall pulse in the middle (days 12, 13, 14)
    src_y = np.repeat(1.0, len(t))
    src_y[12] = 25.0
    src_y[13] = 40.0
    src_y[14] = 15.0

    # We consider two versions of this data - one with the full dataset
    # and one with NaNs in the time series corresponding to a masked field over water
    for add_nans in [False, True]:
        if not add_nans:
            figure_title = "Full dataset"
            y = src_y.copy()
        else:
            figure_title = "Dataset with NaNs added"
            y = src_y.copy()
            y[-5:-3] = np.nan
        print(f"{figure_title}----------------------------------")

        # Wrap into a dataarray
        time = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")
        da = xr.DataArray(
            y,
            dims=["time"],
            coords={"time": time},
            name="pr",
            attrs={"units": "mm"},
        )

        # Calculate a 5-day rolling sum shifted to target day 3 for plotting visualization
        shifted_sum = _calculate_shifted_rolling_precip(da, period=5, target_day=3)

        # Plot diagnostics for visualization (only for the full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da.plot.line("o-", label="Daily precipitation (pr)")
            shifted_sum.plot.line("x--", label="5-day rolling sum (shifted to Day 3)")
            plt.title("Synthetic Precipitation Validation")
            plt.ylabel("Precipitation (mm)")
            plt.xlabel("Time")
            plt.legend()
            plt.show()

        # Get final mean annual maximum values
        rx5 = maximum5dayPrecip(da)
        rx14 = maximum14dayPrecip(da)
        print(f"Mean annual maximum 5-day precipitation: {rx5.data} mm")
        print(f"Mean annual maximum 14-day precipitation: {rx14.data} mm")
