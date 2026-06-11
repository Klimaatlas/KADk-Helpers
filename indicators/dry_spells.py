"""
dry_spell_counts.py

Generic implementation of DMI's approach to counting dry spells of specific lengths. A dry day 
is identified when daily precipitation (pr) is strictly below a given threshold (default 1.0 mm).
The script identifies consecutive dry spells over the full timeseries, filters them based on a 
minimum duration (e.g., 5 or 10 days), maps each valid event as a count of '1' to its center day, 
sums the events per calendar year, and returns the long-term mean annual number of spells.

Parameters
----------
pr : xr.DataArray
    Daily precipitation amount. The time dimension must be called 'time' and have daily
    resolution. Precipitation statistics are calculated over other (spatial) dimensions but are not
    referenced directly, and can therefore have arbitrary names.
threshold : float
    Precipitation threshold in mm, below which a day is considered dry.
min_duration : int
    The minimum required length (in days) for a consecutive dry period to be counted.

Returns
-------
xr.DataArray
    Mean annual number of dry spells matching or exceeding the minimum duration.
"""

# Public API-----------------
__all__ = ["fiveDayDrySpells", "tenDayDrySpells", "count_dry_spells"]

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


def _identify_centered_dry_spell_counts(
    pr: xr.DataArray, threshold: float, min_duration: int
) -> xr.DataArray:
    """
    Identify consecutive dry spells, filter them by minimum duration, and return 
    an array where each valid spell is marked with a '1.0' on its center day.
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

    # Inner block helper to filter by duration and shift counts to the center day
    def _count_and_shift_to_center(block):
        lengths_np = block.values
        out_np = np.zeros_like(lengths_np)
        time_axis = block.get_axis_num("time")
        end_indices = np.where(lengths_np > 0)

        # Check for empty coordinate arrays
        if len(end_indices) == 0 or len(end_indices[0]) == 0:
            return block

        coords_list = np.transpose(end_indices)

        for coord in coords_list:
            idx_tuple = tuple(coord)
            duration = int(lengths_np[idx_tuple])

            # Filter dry spells based on minimum duration
            if duration >= min_duration:
                time_idx = idx_tuple[time_axis]

                # Calculate the midpoint index along the time axis
                steps_back = duration // 2
                center_time_idx = time_idx - steps_back

                if center_time_idx < 0:
                    center_time_idx = 0

                center_idx_list = list(idx_tuple)
                center_idx_list[time_axis] = center_time_idx
                center_idx_tuple = tuple(center_idx_list)

                # Set the value to 1.0 to mark a single valid dry spell event
                out_np[center_idx_tuple] = 1.0

        return block.copy(data=out_np)

    # Map the block shifting function over dask chunks while preserving coordinates
    centered_events = total_spell_lengths.map_blocks(
        _count_and_shift_to_center, template=total_spell_lengths
    )

    return centered_events


# Generic public function -----------------
def count_dry_spells(pr: xr.DataArray, threshold: float, min_duration: int) -> xr.DataArray:
    """
    Calculate the long-term mean annual number of dry spells matching or exceeding a minimum duration.
    """
    # Identify and position valid dry spell events
    centered_events = _identify_centered_dry_spell_counts(
        pr=pr, threshold=threshold, min_duration=min_duration
    )

    # Sum the events per calendar year
    annual_count = centered_events.groupby("time.year").sum(dim="time")

    # Calculate the long-term mean across all years
    out = annual_count.mean(dim="year")
    out.attrs['units'] = "spells"

    # Check for missing data. Any missing values in the timeseries
    # should give an NaN. Create a mask and apply.
    nan_mask = pr.isnull().any(dim="time")
    result = out.where(~nan_mask)

    return result


# Convenience wrappers ---------------------
def fiveDayDrySpells(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate the mean annual number of dry spells lasting at least 5 days (< 1.0 mm).
    """
    return count_dry_spells(pr, threshold=1.0, min_duration=5)


def tenDayDrySpells(pr: xr.DataArray,**kwargs) -> xr.DataArray:
    """
    Calculate the mean annual number of dry spells lasting at least 10 days (< 1.0 mm).
    """
    return count_dry_spells(pr, threshold=1.0, min_duration=10)

# Validation --------------------------------------
if __name__ == "__main__":
    # Imports for use here
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Create a synthetic dataset for testing (40 days)
    t = np.arange(0, 40)
    
    # Simulate two distinct dry spells:
    # Spell 1: A short 6-day dry spell (days 5 to 10) -> Should count for 5-day, but not 10-day
    # Spell 2: A long 12-day dry spell (days 20 to 31) -> Should count for both 5-day and 10-day
    src_y = np.repeat(10.0, len(t))
    src_y[5:11] = 0.0
    src_y[20:32] = 0.0

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


        # Extract centered 5-day event markings for visualization
        this_threshold = 1.0
        events_5day = _identify_centered_dry_spell_counts(da, threshold=this_threshold, min_duration=5)

        # Plot diagnostic comparison (only for full dataset)
        if not add_nans:
            plt.figure(figsize=(10, 4))
            da.plot.line("o-", label="Daily precipitation (pr)")
            events_5day.plot.line("x--", label="5-day event markers (1.0 at center day)")
            plt.axhline(y=this_threshold, color="red", linestyle="--", label="Dry threshold (1.0 mm)")
            plt.title("Synthetic Dry Spell Counting Validation")
            plt.ylabel("Precipitation / Count Marker")
            plt.xlabel("Time")
            plt.legend()
            plt.show()

        # Get final statistics
        out5 = count5dayDrySpells(da)
        out10 = count10dayDrySpells(da)
        print(f"Mean annual number of 5-day dry spells: {out5.data}")
        print(f"Mean annual number of 10-day dry spells: {out10.data}")
