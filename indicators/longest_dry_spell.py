"""
longest_dry_spell.py

Generic implementation of DMI's approach to calculating the longest dry spell. A dry day is 
identified when daily precipitation (pr) is strictly below a given threshold (default 1.0 mm).
This script identifies all consecutive dry periods across the full timeseries, maps the duration 
of each spell to its center day, finds the annual maximum dry spell length based on these centered 
days, and finally returns the long-term mean annual maximum dry spell length across all years.

Parameters
----------
pr : xr.DataArray
    Daily precipitation amount. The time dimension must be called 'time' and have daily
    resolution. Precipitation statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
threshold : float
    Precipitation threshold in mm, below which a day is considered dry.

Returns
-------
xr.DataArray
    Mean annual longest dry spell length (days).
"""

# Public API-----------------
__all__ = ["longestDrySpell", "count_longest_dry_spell"]

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


def _identify_centered_dry_spells(
    pr: xr.DataArray, threshold: float
) -> xr.DataArray:
    """
    Identify all consecutive dry spells in the full timeseries and return an array 
    where the total duration of each spell is assigned strictly to its center day.
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

    # Force the entire time dimension into a single chunk to prevent chunk boundary errors
    if pr.chunks is not None:
        pr_mm = pr_mm.chunk({"time": -1})

    # Create a dry day mask (1 for dry, 0 for wet)
    is_dry = (pr_mm < threshold).astype(int)
    is_wet = is_dry == 0

    # Calculate the duration of all dry spells (accumulated on the last day of each spell)
    cum_sum = is_dry.cumsum(dim="time")
    reset = cum_sum.where(is_wet).ffill(dim="time").fillna(0)
    spell_durations_at_end = cum_sum - reset

    # Identify where the spells end
    is_spell_end = (spell_durations_at_end > 0) & (
        is_dry.shift(time=-1, fill_value=0) == 0
    )

    # Isolate the total duration values onto the final day of each dry spell
    total_spell_lengths = spell_durations_at_end.where(is_spell_end, 0)

    # Re-verify that the time dimension remains unified before map_blocks
    if total_spell_lengths.chunks is not None:
        total_spell_lengths = total_spell_lengths.chunk({"time": -1})

    # Inner block helper to shift values to the middle of each period
    def _shift_block_to_center(block):
        lengths_np = block.values
        out_np = np.zeros_like(lengths_np)
        time_axis = block.get_axis_num("time")
        end_indices = np.where(lengths_np > 0)

        # Robust check for empty coordinate arrays
        if len(end_indices) == 0 or len(end_indices[0]) == 0:
            return block

        coords_list = np.transpose(end_indices)

        for coord in coords_list:
            idx_tuple = tuple(coord)
            duration = int(lengths_np[idx_tuple])
            time_idx = idx_tuple[time_axis]

            # Calculate the midpoint index along the time axis
            steps_back = duration // 2
            center_time_idx = time_idx - steps_back

            # Defensive check: if the spell started before the data series, clamp to index 0
            if center_time_idx < 0:
                center_time_idx = 0

            center_idx_list = list(idx_tuple)
            center_idx_list[time_axis] = center_time_idx
            center_idx_tuple = tuple(center_idx_list)

            out_np[center_idx_tuple] = duration

        return block.copy(data=out_np)

    # Map the block shifting function over dask chunks while preserving coordinates
    centered_spells = total_spell_lengths.map_blocks(
        _shift_block_to_center, template=total_spell_lengths
    )

    return centered_spells


# Generic public function -----------------
def count_longest_dry_spell(pr: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Calculate the long-term mean annual longest dry spell length.
    """
    # Identify and center the dry spells
    centered_spells = _identify_centered_dry_spells(pr=pr, threshold=threshold)

    # Find the maximum dry spell for each year based on the centered days
    annual_max = centered_spells.groupby("time.year").max(dim="time")

    # Calculate the long-term mean across all years
    out = annual_max.mean(dim="year")
    out.attrs['units'] = "days"

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = pr.isnull().any(dim="time")
    result = out.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def longestDrySpell(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate the mean annual longest dry spell strictly using a 1.0 mm threshold.
    """
    return count_longest_dry_spell(pr, threshold=1.0)


# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create a synthetic dataset for testing (30 days)
    t = np.arange(0, 30)
    
    # Simulate an 11-day severe dry spell block in the middle (days 10 to 20)
    # Background has heavy rain (10mm), dry spell days have 0mm.
    src_y = np.repeat(10.0, len(t))
    src_y[10:21] = 0.0  # 11 days of dry weather

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

        # Extract centered spell array for validation visualization
        this_threshold = 1.0
        centered_spells_da = _identify_centered_dry_spells(da, threshold=this_threshold)

        # Plot diagnostic comparison (only for full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da.plot.line("o-", label="Daily precipitation (pr)")
            centered_spells_da.plot.line("x--", label="Identified Dry Spell (duration at center day)")
            plt.axhline(y=this_threshold, color="red", linestyle="--", label="Dry threshold (1.0 mm)")
            plt.title("Synthetic Dry Spell Validation (Duration maps to center day)")
            plt.ylabel("Precipitation / Duration")
            plt.xlabel("Time")
            plt.legend()
            plt.show()

        # Get final statistic
        out = longestDrySpell(da)
        print(f"Mean annual maximum dry spell length: {out.data} days")
