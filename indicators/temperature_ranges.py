"""
temperature_ranges.py
Implementation of DMI's approach to calculating annual and daily temperature ranges.
Designed to integrate seamlessly with the KAPy workflow framework.
"""

# Public API-----------------
__all__ = ["annualTemperatureRange", "dailyTemperatureRange"]

# Imports -------------------
import xarray as xr
import xclim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Private helper functions ---------------------------------
def _assert_temperature_input(da: xr.DataArray, name: str):
    """Internal helper to validate temperature input structure."""
    if "time" not in da.dims:
        raise ValueError(f"{name} must contain a time dimension")
    if "units" not in da.attrs:
        raise ValueError(f"{name} must define units")

    # Try xarray's native inference first
    time_resolution = xr.infer_freq(da.time)

    # Fallback if infer_freq returns None (common with chunked or climate model calendars)
    if time_resolution is None:
        if len(da.time) > 1:
            # Calculate the median delta between time steps in days
            time_deltas = da.time.diff(dim="time")
            # Convert to days depending on whether it is a standard datetime or cftime
            if np.issubdtype(time_deltas.dtype, np.timedelta64):
                median_days = np.median(time_deltas.dt.days)
            else:  # Handle cftime Objects
                median_days = np.median([t.days for t in time_deltas.values])

            if median_days == 1:
                time_resolution = "D"

    if time_resolution not in ["D", "1D"]:
        raise ValueError(
            f"{name} must have daily time resolution, 'D', but received {time_resolution}."
        )

# Generic public functions -----------------
def annualTemperatureRange(tas: xr.DataArray) -> xr.DataArray:
    """
    Calculate the annual temperature range, defined as the mean across all years
    of the difference between the absolute maximum and minimum temperature within each year.
    """
    _assert_temperature_input(tas, "tas")

    # Ensure correct units (degC)
    tas_c = xclim.core.units.convert_units_to(tas, "degC")

    # Group by year using xarray's groupby instead of resample to match climate conventions
    annual_max = tas_c.groupby("time.year").max(dim="time")
    annual_min = tas_c.groupby("time.year").min(dim="time")
    annual_range = annual_max - annual_min
    out = annual_range.mean(dim="year")
    out.attrs["units"] = "degC"

    # Mask out missing data grid cells
    nan_mask = tas.isnull().any(dim="time")
    return out.where(~nan_mask)

def dailyTemperatureRange(tasmax: xr.DataArray, tasmin: xr.DataArray) -> xr.DataArray:
    """
    Calculate the daily temperature range, defined as the long-term mean
    difference between daily maximum and daily minimum temperature.
    Accepts individual 'tasmax' and 'tasmin' DataArrays as required by KAPy.
    """
    _assert_temperature_input(tasmax, "tasmax")
    _assert_temperature_input(tasmin, "tasmin")

    # Ensure correct units (degC)
    tasmax_c = xclim.core.units.convert_units_to(tasmax, "degC")
    tasmin_c = xclim.core.units.convert_units_to(tasmin, "degC")

    dtr = tasmax_c - tasmin_c
    out = dtr.mean(dim="time")
    out.attrs["units"] = "degC"

    # Mask out missing data grid cells
    nan_mask = tasmax.isnull().any(dim="time") | tasmin.isnull().any(dim="time")
    return out.where(~nan_mask)

# Validation --------------------------------------
if __name__ == "__main__":
    # Setup synthetic dataset covering 2 years to test annual/daily ranges
    t = np.arange(0, 730)
    time_coord = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")

    # Simulate seasonal cycle and daily noise
    seasonal_cycle = 10 + 15 * np.sin(t / 365 * 2 * np.pi)
    daily_noise = 5 * np.sin(t * np.exp(1))

    tas_base = seasonal_cycle + daily_noise
    tasmax_base = tas_base + 4.0  # DTR of exactly 8 degrees
    tasmin_base = tas_base - 4.0

    for add_nans in [False, True]:
        figure_title = "Dataset with NaNs added" if add_nans else "Full dataset"
        print(f"\n{figure_title} ----------------------------------")

        # Copy base arrays to prevent mutation across loop iterations
        tas_mock = tas_base.copy()
        tasmax_mock = tasmax_base.copy()
        tasmin_mock = tasmin_base.copy()

        if add_nans:
            tas_mock[-50:-40] = np.nan
            tasmax_mock[-50:-40] = np.nan
            tasmin_mock[-50:-40] = np.nan

        da_tas = xr.DataArray(tas_mock, dims=["time"], coords={"time": time_coord}, attrs={"units": "degC"})
        da_tasmax = xr.DataArray(tasmax_mock, dims=["time"], coords={"time": time_coord}, attrs={"units": "degC"})
        da_tasmin = xr.DataArray(tasmin_mock, dims=["time"], coords={"time": time_coord}, attrs={"units": "degC"})

        # Calculate indices - passing variables directly to match new signature
        atr_out = annualTemperatureRange(da_tas)
        dtr_out = dailyTemperatureRange(da_tasmax, da_tasmin)

        print(f"Calculated Annual Temperature Range: {atr_out.data} degC")
        print(f"Calculated Daily Temperature Range: {dtr_out.data} degC")

        # Quick diagnostic plot for the full dataset
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da_tasmax.plot(label="tasmax", alpha=0.5)
            da_tasmin.plot(label="tasmin", alpha=0.5)
            da_tas.plot(label="tas mean", color="black")
            plt.title("Synthetic Temperature Range Input Data")
            plt.legend()
            plt.show()
