"""
dry_days.py

Generic implementation of DMI's approach to calculating dry days. A dry day is identified
when the daily precipitation amount (pr) is strictly below a given threshold (default 1.0 mm).
The script counts the annual number of dry days and returns the long-term mean across all years.

Parameters
----------
pr : xr.DataArray
    Daily precipitation amount. The time dimension should be called 'time' and have daily
    resolution. Precipitation statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
threshold : float
    Precipitation threshold in mm, below which a day is considered dry.

Returns
-------
xr.DataArray
    Mean annual number of dry days.
"""

# Public API-----------------
__all__ = ["dryDays", "count_dry_days"]

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


def _identify_dry_days(
    pr: xr.DataArray, threshold: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Identify dry days for a given threshold and return a true/false mask
    for those points in time/space that are strictly below the threshold.
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
    below_threshold = pr_mm < threshold

    return below_threshold, pr_mm


# Generic public function -----------------
def count_dry_days(pr: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    """
    Calculate the mean annual number of dry days.
    """
    # Identify the dry days
    below_threshold, _ = _identify_dry_days(pr=pr, threshold=threshold)

    # Convert boolean mask to integer, sum per year, and then average over all years
    days_below_threshold = below_threshold.astype(int).groupby("time.year").sum(dim="time").mean(dim="year")

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = pr.isnull().any(dim="time")
    result = days_below_threshold.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def dryDays(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate dry days strictly using a 1.0 mm threshold.
    """
    return count_dry_days(pr, threshold=1.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing (21 days)
    t = np.arange(0, 21)
    
    # Simulate alternating dry and wet days (some days under 1.0mm, some above)
    src_y = 0.5 + 2.0 * np.sin(t / 20 * 2 * np.pi) ** 2

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
            name="pr",
            attrs={"units": "mm"},
        )

        # Identify thresholds
        this_threshold = 1.0
        below_threshold, pr_mm = _identify_dry_days(
            pr=da, threshold=this_threshold
        )

        # Plot diagnostics (only for the full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da.plot.line("o", label="pr")
            (this_threshold - 0.5 + below_threshold).plot.line(
                "-x", label="Dry Day Detection (<1mm)"
            )
            plt.axhline(y=this_threshold, color="red", linestyle="--")
            plt.legend()
            plt.title(figure_title)
            plt.ylabel("Precipitation (mm)")
            plt.xlabel("Time (Days)")
            plt.show()

        # Get number of days
        out = dryDays(da)
        print(f"Mean annual number of dry days: {out.data}")
