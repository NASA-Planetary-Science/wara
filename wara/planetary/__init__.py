"""Planetary gamma-ray / neutron spectroscopy tools for WARA.

This subpackage hosts the data-visualization and analysis tools for NASA
planetary missions that carried gamma-ray and/or neutron spectrometers. The
first mission being wired in is **Lunar Prospector** (LP-GRS).

Step 1: a self-contained 3D Moon rendered with the correct latitude/longitude
coordinate system (:mod:`wara.planetary.moon`).
Step 2: search/download/read LP-GRS products from the NASA PDS and select/sum
spectra by date, (lat, lon), altitude, ... (:mod:`wara.planetary.lp`).
"""
from .abundance import (
    ABUNDANCE_ELEMENTS,
    ABUNDANCE_RESOLUTIONS,
    LP_ABUNDANCE_BASE_URL,
    abundance_grid,
    download_abundance,
    read_abundance,
)
from .lola import (
    LOLA_BASE_URL,
    LOLA_RESOLUTIONS,
    download_lola_dem,
    elevation_grid,
    read_lola_dem,
)
from .lp import (
    LP_DATA_DIR,
    LP_DOCUMENTS,
    LP_GRS_BASE_URL,
    LP_METADATA_CSV,
    LP_MISSION_PAGE_URL,
    LOW_ALTITUDE_START,
    fetch_document,
    LPGrsDay,
    LPGrsProduct,
    build_orbit_metadata,
    doy_to_date,
    download_products,
    filter_products,
    list_grs_products,
    load_orbit_metadata,
    read_grs_day,
)
from .moon import (
    LUNAR_LANDMARKS,
    R_MOON_KM,
    lonlat_to_xyz,
    xyz_to_lonlat,
    sphere_mesh,
    graticule_traces,
    default_texture_path,
    load_texture_intensity,
    moon_figure,
)

__all__ = [
    "ABUNDANCE_ELEMENTS",
    "ABUNDANCE_RESOLUTIONS",
    "LP_ABUNDANCE_BASE_URL",
    "abundance_grid",
    "download_abundance",
    "read_abundance",
    "LOLA_BASE_URL",
    "LOLA_RESOLUTIONS",
    "download_lola_dem",
    "elevation_grid",
    "read_lola_dem",
    "LP_DATA_DIR",
    "LP_DOCUMENTS",
    "LP_GRS_BASE_URL",
    "LP_MISSION_PAGE_URL",
    "fetch_document",
    "LP_METADATA_CSV",
    "LOW_ALTITUDE_START",
    "build_orbit_metadata",
    "load_orbit_metadata",
    "LPGrsDay",
    "LPGrsProduct",
    "doy_to_date",
    "download_products",
    "filter_products",
    "list_grs_products",
    "read_grs_day",
    "LUNAR_LANDMARKS",
    "R_MOON_KM",
    "lonlat_to_xyz",
    "xyz_to_lonlat",
    "sphere_mesh",
    "graticule_traces",
    "default_texture_path",
    "load_texture_intensity",
    "moon_figure",
]
