# LUCAS Land Use Change Visualization Module (`LUC_viz`)

This Python module (`LUC_viz.py`) provides functions designed for visualizing Land Use and Land Cover (LUC) change datasets from the LUCAS LUC project.

## Overview

*   **Interactive Dominant Land Use Mapping:** Visualize the most prevalent land use type for a specific time point within a defined region using an interactive map (`explore_dominant_LUC`).
*   **Composition Comparison (Pie Charts):** Compare the overall land use composition between two time points using side-by-side pie charts (`compare_LUC_pie`).
*   **Change Analysis (Bar Charts):** Analyze both the relative percentage change and the change in land use proportions between two time points using bar charts (`compare_LUC_bar`).
*   **Specific Land Cover Type Change Mapping:** Map the spatial distribution of the change in fraction for a single land cover type between two time points (`compare_LCTYPE`).

## Dependencies

The module relies on several core scientific Python libraries:

*   `xarray`: For handling multi-dimensional labeled data (the input datasets).
*   `geopandas`: For spatial data manipulation and creating interactive maps.
*   `matplotlib`: For generating static plots (pie and bar charts).
*   `cartopy`: For adding geographic context (coastlines, borders) to maps.
*   `rasterio`: Used internally for polygonizing raster data.
*   `pandas`: For data manipulation, particularly within plotting functions.
*   `numpy`: For numerical operations.
*   `affine`: For handling affine transformations during polygonization.
*   `IPython`: For displaying interactive map outputs in Jupyter environments.
*   `shapely`: For geometric operations.

## Data Requirements

The functions in this module expect input in the format of LUCAS LUC datasets, namely data as `xarray.Dataset` objects with the following structure:

*   A data variable named `landCoverFrac` containing the fraction of each land cover type.
*   Coordinates:
    *   `time`: Representing the time points of the data.
    *   `lat`: Latitude values.
    *   `lon`: Longitude values (expected in a 0-360 degree range).
    *   `lctype`: Land cover type identifier (integers 1-16, corresponding to the predefined `lctype_to_name` mapping within the module).

## Module Functions

*   `explore_dominant_LUC(ds, time, bbox=None, title=None)`: Creates an interactive map showing the dominant land use type for a given time and optional region.
*   `compare_LUC_pie(ds1, time1, ds2, time2, bbox=None)`: Displays two pie charts comparing the land use composition between two datasets/times.
*   `compare_LUC_bar(ds1, time1, ds2, time2, bbox=quebec_bbox_360)`: Generates two bar charts showing relative percentage change and absolute proportion change between two datasets/times.
*   `compare_LCTYPE(ds1, time1, ds2, time2, lctype, bbox)`: Plots a map illustrating the change in fraction for a specific land cover type between two datasets/times.

## Notes

*   The module includes predefined bounding boxes (`quebec_bbox_360`, `project_bbox`) for convenience.
*   Input longitude data should be in the 0-360 degree range. The interactive map function (`explore_dominant_LUC`) handles conversion to -180 to 180 for display.
*   Color mappings for land use types are defined in `land_use_colormap`. 

