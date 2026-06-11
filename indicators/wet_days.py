"""
wet_days.py

Generic implementation of DMI's approach to calculating heavy precipitation days. A heavy 
precipitation day is identified when the daily precipitation amount (pr) is greater than or 
equal to a given threshold (e.g., 10 mm or 20 mm). The script counts the annual number of 
days exceeding the threshold and returns the long-term mean across all years.

Parameters
----------
pr : xr.DataArray
    Daily precipitation amount. The time dimension should be called 'time' and have daily
    resolution. Precipitation statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
threshold : float
    Precipitation threshold in mm, at or above which heavy precipitation day counting is triggered.

Returns
-------
xr.DataArray
    Mean annual number of heavy precipitation days.
"""

# Public API-----------------
__all__ = ["daysAbove10mm", "daysAbove20mm", "count_precip_days"]

# Imports -------------------
import xarray as xr
import xclim
import numpy as np


# Private helper functions ---------------------------------
def _assert_daily_resolution(time_coord, name: str = "pr"):
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


def _identify_precip_days(
    pr: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Identify heavy precipitation days for a given threshold and return a true/false mask
    for those points in time/space that are equal to or greater than the threshold.
    """
    # Assert what we have
    if "time" not in pr.dims:
        raise ValueError("pr must contain a 'time' dimension")
    if "units" not in pr.attrs:
        raise ValueError("pr must define units")

    # Assert daily time resolution using the robust validator
    _assert_daily_resolution(pr.time)

    # Ensure precipitation units are converted to mm
    pr_mm = xclim.core.units.convert_units_to(pr, "mm")
    above_threshold = pr_mm >= threshold

    return above_threshold, pr_mm


# Generic public function -----------------
def count_precip_days(pr: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Calculate the mean annual number of heavy precipitation days.
    """
    # Identify the heavy precipitation days
    above_threshold, _ = _identify_precip_days(pr=pr, threshold=threshold)

    # Convert boolean mask to integer, sum per year, and then average over all years
    days_above_threshold = above_threshold.astype(int).groupby("time.year").sum(dim="time").mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = pr.isnull().any(dim="time")
    result = days_above_threshold.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def daysAbove10mm(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate heavy precipitation days using a 10 mm threshold.
    """
    return count_precip_days(pr, threshold=10.0)


def daysAbove20mm(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate heavy precipitation days using a 20 mm threshold.
    """
    return count_precip_days(pr, threshold=20.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing (30 days)
    t = np.arange(0, 30)
    # Simulate a rainfall event pattern peaking above 20 mm
    src_y = 5.0 + 18.0 * np.sin(t / 29 * np.pi)

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

        # Identify thresholds
        this_threshold = 10.0
        above_threshold, pr_mm = _identify_precip_days(
            pr=da, threshold=this_threshold
        )

        # Plot diagnostics (only for the full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da.plot.line("o", label="pr")
            (this_threshold - 0.5 + above_threshold).plot.line(
                "-x", label="Heavy Precip Detection (>=10mm)"
            )
            plt.axhline(y=this_threshold, color="red", linestyle="--")
            plt.axhline(y=20.0, color="orange", linestyle="--")
            plt.legend()
            plt.title(figure_title)
            plt.ylabel("Precipitation (mm)")
            plt.show()

        # Get number of days (Rettede funktionsnavne)
        out10 = daysAbove10mm(da)
        out20 = daysAbove20mm(da)
        print(f"Mean annual number of days >= 10mm: {out10.data}")
        print(f"Mean annual number of days >= 20mm: {out20.data}")
