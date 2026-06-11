"""
frost_during_growing_season.py

Generic implementation of DMI's approach to calculating post-start frost days.
The script first identifies the start of the growing season (GSS), defined as the first
6-day period of the year with daily mean temperature (tas) strictly above 5°C for all 6 days
(positioned on the 6th day). It then counts the number of days from this GSS date up until
June 23rd where the daily minimum temperature (tasmin) drops strictly below 0°C (frost).
If GSS occurs after June 23rd for a given year, it returns 0. Finally, it returns the
long-term mean annual number of post-start frost days.

Parameters
----------
tas : xr.DataArray
    Daily mean temperature. The time dimension must be called 'time' and have daily resolution.
tasmin : xr.DataArray
    Daily minimum temperature. The time dimension must be called 'time' and have daily resolution.
threshold_tas : float
    Temperature threshold in degrees C for identifying the growing season start (default 5.0°C).
window_tas : int
    Consecutive days required above threshold_tas to trigger growing season start (default 6 days).
threshold_frost : float
    Temperature threshold in degrees C below which a frost day is counted (default 0.0°C).

Returns
-------
xr.DataArray
    Mean annual number of frost days occurring during the early growing season.
"""

# Public API-----------------
__all__ = ["frostGrowingSeason", "count_growing_season_frost_days"]

# Imports -------------------
import xarray as xr
import xclim


# Private helper functions ---------------------------------
def _calculate_post_start_frost_annual(
    tas: xr.DataArray, tasmin: xr.DataArray, threshold_tas: float, window_tas: int, threshold_frost: float
) -> xr.DataArray:
    """
    Identify growing season start and count subsequent frost days up until June 23rd
    for each calendar year using a calendar-agnostic block-by-block array function.
    """
    # Assert what we have
    if "time" not in tas.dims or "time" not in tasmin.dims:
        raise ValueError("Inputs must contain a 'time' dimension")
    if "units" not in tas.attrs or "units" not in tasmin.attrs:
        raise ValueError("Both 'tas' and 'tasmin' must define units")

    # NOTE: Manual time resolution validation (xr.infer_freq) has been removed
    # for full compatibility with KAPy/Snakemake pipeline adjustments.

    # Unit conversion and temperature scale safe checks
    tas_c = xclim.core.units.convert_units_to(tas, "degC")
    tasmin_c = xclim.core.units.convert_units_to(tasmin, "degC")

    # Force entire time dimension into one chunk to scan full years seamlessly
    if tas_c.chunks is not None:
        tas_c = tas_c.chunk({"time": -1})
    if tasmin_c.chunks is not None:
        tasmin_c = tasmin_c.chunk({"time": -1})

    # Prepare inputs to map_blocks by aligning them into a single structure
    combined = xr.Dataset({"tas": tas_c, "tasmin": tasmin_c})

    def _find_frost_block(block):
        import numpy as np

        tas_np = block.tas.values
        tasmin_np = block.tasmin.values

        # Output template matching full temporal array shape
        out_np = np.zeros_like(tas_np, dtype=float)
        time_axis = block.tas.get_axis_num("time")

        years = block.time.dt.year.values
        months = block.time.dt.month.values
        days = block.time.dt.day.values
        unique_years = np.unique(years)

        # Move time axis to axis 0 for universal dimension indexing (1D, 2D, 3D grids)
        if time_axis != 0:
            tas_np = np.moveaxis(tas_np, time_axis, 0)
            tasmin_np = np.moveaxis(tasmin_np, time_axis, 0)
            out_np = np.moveaxis(out_np, time_axis, 0)

        spatial_shape = tas_np.shape[1:]

        for spatial_idx in np.ndindex(spatial_shape):
            tas_ts = tas_np[(slice(None),) + spatial_idx]
            tasmin_ts = tasmin_np[(slice(None),) + spatial_idx]

            for year in unique_years:
                year_mask = years == year
                year_indices = np.where(year_mask)[0]

                if len(year_indices) < window_tas:
                    continue

                # Isolate the current year's timeseries and calendar components properly
                year_tas = tas_ts[year_indices]
                year_tasmin = tasmin_ts[year_indices]
                year_months = months[year_indices]
                year_days = days[year_indices]

                # 1. Calendar-agnostic detection of June 23rd index for this specific year
                june23_loc = np.where((year_months == 6) & (year_days == 23))[0]
                if len(june23_loc) == 0:
                    continue
                june23_idx = june23_loc[0]

                # 2. Find the first 6-day growing season start (GSS)
                is_warm = (year_tas > threshold_tas).astype(int)
                gss_idx = None

                for i in range(len(is_warm) - window_tas + 1):
                    if np.sum(is_warm[i : i + window_tas]) == window_tas:
                        gss_idx = i + window_tas - 1  # 6th day index inside the year
                        break

                # 3. If GSS is not found, or happens after June 23rd, count remains 0
                if gss_idx is None or gss_idx > june23_idx:
                    continue

                # 4. Count frost days from GSS (inclusive) until June 23rd (inclusive)
                early_season_tasmin = year_tasmin[gss_idx : june23_idx + 1]
                frost_count = np.sum(early_season_tasmin < threshold_frost)

                # FIX: Map back to the absolute global timeline index using year_indices
                target_time_idx = year_indices[gss_idx]
                out_np[(target_time_idx,) + spatial_idx] = float(frost_count)

        if time_axis != 0:
            out_np = np.moveaxis(out_np, 0, time_axis)

        return block.tas.copy(data=out_np)

    # Run block operations
    centered_counts = combined.map_blocks(_find_frost_block, template=combined.tas)

    # Return annual maximums to isolate the single annual count value per year
    return centered_counts.groupby("time.year").max(dim="time")


# Generic public function -----------------
def count_growing_season_frost_days(
    tas: xr.DataArray, tasmin: xr.DataArray, threshold_tas: float = 5.0, window_tas: int = 6, threshold_frost: float = 0.0
) -> xr.DataArray:
    """
    Calculate the long-term mean annual number of early growing season frost days.
    """
    # Calculate the annual counts
    annual_counts = _calculate_post_start_frost_annual(
        tas=tas, tasmin=tasmin, threshold_tas=threshold_tas, window_tas=window_tas, threshold_frost=threshold_frost
    )

    # Average the counts across all years
    out = annual_counts.mean(dim="year")
    out.attrs['units'] = "days"

    # Spatial NaN masking to preserve grid boundaries
    nan_mask = tas.isnull().any(dim="time") | tasmin.isnull().any(dim="time")
    result = out.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def frostGrowingSeason(tas: xr.DataArray, tasmin: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate the number of frost days (<0°C) from the growing season start until June 23rd.
    """
    return count_growing_season_frost_days(tas=tas, tasmin=tasmin, threshold_tas=5.0, window_tas=6, threshold_frost=0.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Create a synthetic dataset (180 days covering Jan 1st to June 30th)
    t = np.arange(0, 182)  # Adjust to 182 to ensure June 23rd (day 174) is fully enveloped
    time = pd.Timestamp("2000-01-01") + pd.to_timedelta(t, unit="D")

    # Simulate a rising spring temperature trend
    # June 23rd will be index 174 (since Jan 1st is day 0)
    tas_values = -2.0 + (t * 0.08)  # Crosses 5°C around day 88 (early April)

    # Simulate overnight minimums with occasional frost events dipping below 0°C
    tasmin_values = tas_values - 4.0
    # Artificially inject exactly 3 frost days AFTER the growing season has started, but before June 23rd
    tasmin_values[95] = -1.5
    tasmin_values[102] = -0.8
    tasmin_values[110] = -2.2

    for add_nans in [False, True]:
        figure_title = "Dataset with NaNs added" if add_nans else "Full dataset"
        print(f"\n{figure_title} ----------------------------------")

        tas_test = tas_values.copy()
        tasmin_test = tasmin_values.copy()

        if add_nans:
            tas_test[-10:] = np.nan
            tasmin_test[-10:] = np.nan

        # Wrap into dataarrays directly to match new signature
        da_tas = xr.DataArray(tas_test, dims=["time"], coords={"time": time}, attrs={"units": "degC"})
        da_tasmin = xr.DataArray(tasmin_test, dims=["time"], coords={"time": time}, attrs={"units": "degC"})

        # Get number of days
        out = frostGrowingSeason(tas=da_tas, tasmin=da_min if 'da_min' in locals() else da_tasmin)
        print(f"Mean annual number of frost days: {out.data}")
