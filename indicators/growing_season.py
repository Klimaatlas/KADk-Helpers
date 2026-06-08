import xarray as xr
import xclim


def generic_growing_season_start_annual(tas, threshold=5.0, window=6):
    """Intern hjælpefunktion der returnerer de årlige værdier før midling."""
    tas_c = tas
    if tas_c.attrs.get("units") in ["K", "Kelvin"]:
        tas_c = tas_c - 273.15
    if tas.chunks is not None:
        tas_c = tas_c.chunk({"time": -1})

    is_warm = (tas_c > threshold).astype(int)

    def _find_gss_block(block):
        import numpy as np
        warm_np = block.values
        out_np = np.zeros_like(warm_np, dtype=float)
        time_axis = block.get_axis_num("time")
        years = block.time.dt.year.values
        unique_years = np.unique(years)

        if time_axis != 0:
            warm_np = np.moveaxis(warm_np, time_axis, 0)
            out_np = np.moveaxis(out_np, time_axis, 0)

        spatial_shape = warm_np.shape[1:]
        for spatial_idx in np.ndindex(spatial_shape):
            full_ts = warm_np[(slice(None),) + spatial_idx]
            for year in unique_years:
                year_mask = years == year
                year_indices = np.where(year_mask)[0]  # RETTELSE: Hent arrayet direkte ud af tuplen
                
                if len(year_indices) < window:
                    continue

                year_warm = full_ts[year_indices]
                for i in range(len(year_warm) - window + 1):
                    if np.sum(year_warm[i : i + window]) == window:
                        gss_day_value = float(i + window)
                        target_time_idx = year_indices[i + window - 1]
                        out_np[(target_time_idx,) + spatial_idx] = gss_day_value
                        break

        if time_axis != 0:
            out_np = np.moveaxis(out_np, 0, time_axis)
        return block.copy(data=out_np)

    centered_gss = is_warm.map_blocks(_find_gss_block, template=is_warm)
    return centered_gss.groupby("time.year").max(dim="time")


def generic_growing_season_end_annual(tas, threshold=5.0, window=6):
    """Intern hjælpefunktion der returnerer de årlige værdier før midling."""
    tas_c = tas
    if tas_c.attrs.get("units") in ["K", "Kelvin"]:
        tas_c = tas_c - 273.15
    if tas.chunks is not None:
        tas_c = tas_c.chunk({"time": -1})

    is_warm = (tas_c > threshold).astype(int)

    def _find_gse_block(block):
        import numpy as np
        warm_np = block.values
        out_np = np.zeros_like(warm_np, dtype=float)
        time_axis = block.get_axis_num("time")
        years = block.time.dt.year.values
        unique_years = np.unique(years)

        if time_axis != 0:
            warm_np = np.moveaxis(warm_np, time_axis, 0)
            out_np = np.moveaxis(out_np, time_axis, 0)

        spatial_shape = warm_np.shape[1:]
        for spatial_idx in np.ndindex(spatial_shape):
            full_ts = warm_np[(slice(None),) + spatial_idx]
            for year in unique_years:
                year_mask = years == year
                year_indices = np.where(year_mask)[0]  # RETTELSE: Hent arrayet direkte ud af tuplen
                
                if len(year_indices) < window:  # Nu virker dette tjek korrekt, da year_indices er et array
                    continue
                    
                year_warm = full_ts[year_indices]
                # Loop baglæns fra slutningen af året
                for i in range(len(year_warm) - window, -1, -1):
                    if np.sum(year_warm[i : i + window]) == window:
                        gse_day_value = float(i + window)
                        target_time_idx = year_indices[i + window - 1]  # RETTELSE: year_indices er nu et array
                        out_np[(target_time_idx,) + spatial_idx] = gse_day_value
                        break

        if time_axis != 0:
            out_np = np.moveaxis(out_np, 0, time_axis)
        return block.copy(data=out_np)

    centered_gse = is_warm.map_blocks(_find_gse_block, template=is_warm)
    return centered_gse.groupby("time.year").max(dim="time")


# Workflow-funktioner med camelCase navngivning:

def growingSeasonStart(tas):
    annual_gss = generic_growing_season_start_annual(tas, threshold=5.0, window=6)
    out = annual_gss.where(annual_gss > 0).mean(dim="year")
    nan_mask = tas.isnull().any(dim="time")
    return out.where(~nan_mask)


def growingSeasonEnd(tas):
    annual_gse = generic_growing_season_end_annual(tas, threshold=5.0, window=6)
    out = annual_gse.where(annual_gse > 0).mean(dim="year")
    nan_mask = tas.isnull().any(dim="time")
    return out.where(~nan_mask)


def growingSeasonLength(tas):
    # 1. Hent de årlige start- og slutdage
    annual_gss = generic_growing_season_start_annual(tas, threshold=5.0, window=6)
    annual_gse = generic_growing_season_end_annual(tas, threshold=5.0, window=6)
    
    # 2. Beregn længden for hvert år (End - Start + 1)
    annual_length = (annual_gse - annual_gss + 1.0)
    valid_years = (annual_gss > 0) & (annual_gse > 0)
    
    # 3. Beregn gennemsnittet over alle gyldige år
    out = annual_length.where(valid_years).mean(dim="year")

    # 4. Spatial maskering
    nan_mask = tas.isnull().any(dim="time")
    return out.where(~nan_mask)
