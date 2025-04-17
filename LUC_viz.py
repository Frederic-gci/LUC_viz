from affine import Affine
import geopandas as gpd
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import shape
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Constants
lctype_to_name = {
    1: "Tropical broadleaf evergreen trees",
    2: "Tropical deciduous trees",
    3: "Temperate broadleaf evergreen trees",
    4: "Temperate deciduous trees",
    5: "Evergreen coniferous trees",
    6: "Deciduous coniferous trees",
    7: "Coniferous shrubs",
    8: "Deciduous shrubs",
    9: "C3 grass",
    10: "C4 grass",
    11: "Tundra",
    12: "Swamp",
    13: "Non-irrigated crops",
    14: "Irrigated crops",
    15: "Urban",
    16: "Bare",
}

# Reverse dictionary to find lctype from name
name_to_lctype = {v: k for k, v in lctype_to_name.items()}

land_use_colormap = {
    1: "#006400",  # Tropical broadleaf evergreen trees (Deep Green)
    2: "#228B22",  # Tropical deciduous trees (Medium Green)
    3: "#556B2F",  # Temperate broadleaf evergreen trees (Dark Olive Green)
    4: "#008080",  # Temperate deciduous trees (Dark Teal/Pine Green)
    5: "#8FBC8F",  # Evergreen coniferous trees (Lighter Green - DarkSeaGreen)
    6: "#20B2AA",  # Deciduous coniferous trees (Lighter Teal - LightSeaGreen)
    7: "#808000",  # Coniferous shrubs (Olive)
    8: "#BDB76B",  # Deciduous shrubs (Dark Khaki)
    9: "#7CFC00",  # C3 grass (LawnGreen - brighter, more distinct green)
    10: "#32CD32",  # C4 grass (LimeGreen - slightly darker than C3)
    11: "#BC8F8F",  # Tundra (RosyBrown - mix of sparse veg/rock)
    12: "#8B4513",  # Swamp (SaddleBrown)
    13: "#FFD700",  # Non-irrigated crops (Gold)
    14: "#DAA520",  # Irrigated crops (GoldenRod)
    15: "#404040",  # Urban (Darker Gray)
    16: "#D3D3D3",  # Bare (LightGray)
}

# Quebec bounding box (lon_min, lon_max, lat_min, lat_max) (as 0-360 range)
quebec_bbox_360 = [280, 303, 44.8, 62.3]
# Project bbox includes most of Ontario and the Great Lakes
project_bbox = [266, 307.3, 41.7, 62.5]


# %% Load select_bbox
def _select_bbox(ds, bbox):
    """Select data within a bounding box.

    Args:
        ds: xarray Dataset or DataArray
        bbox: list of [lon_min, lon_max, lat_min, lat_max] using 0-360 range

    Returns:
        Subset of the dataset within the bounding box
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    return ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def _get_dominant_land_use(ds, time, region_bbox=None):
    """Get the dominant land use type for a given time and region."""
    land_use = ds.landCoverFrac.sel(time=time)
    if region_bbox:
        land_use = _select_bbox(land_use, region_bbox)

    # First get the dominant type
    dominant_type = land_use.fillna(0).argmax(dim="lctype") + 1

    # Then mask the NA values with -1
    all_nan_mask = land_use.isnull().all(dim="lctype")
    dominant_type = dominant_type.where(~all_nan_mask, -1)  # Use -1 for NA values

    return dominant_type


# %%
def _polygonize_dominant_land_use(dominant_type: xr.DataArray):
    """Polygonize the dominant land use xarray.DataArray.

    Args:
        dominant_type: xarray.DataArray with the dominant land use types

    Returns:
        GeoDataFrame with polygons for each contiguous area of same land use type
    """
    # First convert the xarray to the format expected by rasterio.features
    values = dominant_type.values

    # Create the affine transform
    lon = dominant_type.lon.values
    lat = dominant_type.lat.values
    transform = Affine.translation(lon[0], lat[0]) * Affine.scale(
        (lon[-1] - lon[0]) / (len(lon) - 1), (lat[-1] - lat[0]) / (len(lat) - 1)
    )

    # Use rasterio.features to get polygons
    shapes = features.shapes(
        values.astype("int32"),
        transform=transform,
        connectivity=8,  # Use 8-connectivity for diagonal connections
    )

    # Convert to list of dictionaries with geometry and properties
    geometries = []
    properties = []
    for geom, value in shapes:
        if value > 0:  # Skip nodata (-1) values and 0 values
            geometries.append(shape(geom))
            properties.append(
                {
                    "land_use_type": int(value),
                    "land_use_name": lctype_to_name[int(value)],
                }
            )

    if not geometries:  # Check if we have any valid polygons
        return gpd.GeoDataFrame(
            [],  # Empty data list
            columns=["land_use_type", "land_use_name", "geometry"],
            crs="EPSG:4326",
        )

    # Create GeoDataFrame with explicit geometry column
    gdf = gpd.GeoDataFrame(
        data=properties,  # Use data keyword argument
        geometry=geometries,
        crs="EPSG:4326",
    )

    # Convert longitudes from 0-360 to -180-180 if needed
    if gdf.geometry.bounds.minx.min() > 180:
        gdf.geometry = gdf.geometry.translate(xoff=-360)

    return gdf


# %%
def explore_dominant_LUC(ds, time, region_bbox=None, title=None):
    """Create an interactive map of dominant land use types using geopandas.explore.

    Args:
        ds: xarray Dataset
        time: time to plot
        region_bbox: optional bounding box
        title: optional title

    Returns:
        folium.Map: Interactive map with colored polygons and legend
    """
    # Get dominant land use and convert to polygons
    dominant_type = _get_dominant_land_use(ds, time, region_bbox)
    gdf = _polygonize_dominant_land_use(dominant_type)

    # Create a colorlist from the land_use_colormap that matches the land_use_type of the dataset
    # sorted alphabetically by their land_use_name value
    color_list = [
        land_use_colormap[type]
        for type in (
            name_to_lctype[name] for name in sorted(gdf["land_use_name"].unique())  # type: ignore
        )
    ]

    # Create the map
    m = gdf.explore(
        column="land_use_name",
        tiles="CartoDB positron",
        categorical=True,
        cmap=color_list,
        tooltip=["land_use_name"],
        tooltip_kwds={"labels": False},
        legend=True,
        legend_kwds={"caption": "Land Use Type"},
        style_kwds={
            "fillOpacity": 0.7,
        },
    )

    # Get HTML representation and wrap in a styled div
    map_html = m._repr_html_()
    styled_html = f'<div style="max-width: 1000px;">{map_html}</div>'
    return HTML(styled_html)


def _get_df_of_total_land_use(ds1, time1, ds2, time2, bbox=None):
    """Get a dataframe of the total land use for two times"""
    t1_data = ds1.landCoverFrac.sel(time=time1)
    t2_data = ds2.landCoverFrac.sel(time=time2)
    if bbox:
        t1_data = _select_bbox(t1_data, bbox)
        t2_data = _select_bbox(t2_data, bbox)
    t1_sum = t1_data.sum(dim=["lat", "lon", "time"], skipna=True)
    t2_sum = t2_data.sum(dim=["lat", "lon", "time"], skipna=True)

    # Get both datasets in a pandas dataframe with columns lctype, time, and sum
    t1_df = t1_sum.to_dataframe(name="t1")
    t2_df = t2_sum.to_dataframe(name="t2")
    df = pd.merge(t1_df, t2_df, on="lctype", how="outer")
    df = df.loc[df.any(axis=1)]  # remove rows with all 0
    df = df.reset_index()  # lctype is now a column instead of the index
    df["lctype"] = df["lctype"].astype(int)
    df["name"] = df["lctype"].map(lctype_to_name)
    df["color"] = df["lctype"].map(land_use_colormap)

    return df


def _test_inputs(ds1, time1, ds2, time2, bbox=None):
    """Test the inputs to the compare_land_use function"""
    if not isinstance(ds1, xr.Dataset):
        print("ds1 must be an xarray Dataset")
        return False
    if not isinstance(ds2, xr.Dataset):
        print("ds2 must be an xarray Dataset")
        return False
    # test if t1 is part of the time coordinate of ds1
    if len(ds1.time.sel(time=time1)) == 0:
        print(f"{time1} must be part of the time coordinate of ds1")
        return False
    # test if t2 is part of the time coordinate of ds2
    if len(ds2.time.sel(time=time2)) == 0:
        print(f"{time2} must be part of the time coordinate of ds2")
        return False
    if ds1.landCoverFrac.sel(time=time1).sum() == 0:
        print(f"ds1 has no data for {time1}")
        return False
    if ds2.landCoverFrac.sel(time=time2).sum() == 0:
        print(f"ds2 has no data for {time2}")
        return False
    if bbox:
        ds1 = _select_bbox(ds1, bbox)
        ds2 = _select_bbox(ds2, bbox)
        # See if length of lat or lon is 0
        if ds1.lat.size == 0 or ds1.lon.size == 0:
            print("bbox is not valid for ds1")
            return False
        if ds2.lat.size == 0 or ds2.lon.size == 0:
            print("bbox is not valid for ds2")
            return False
    return True


def compare_LUC_pie(ds1, time1, ds2, time2, region_bbox=None):
    """Create two pie charts of the total land use percent for two years"""
    if not _test_inputs(ds1, time1, ds2, time2, region_bbox):
        return
    df = _get_df_of_total_land_use(ds1, time1, ds2, time2, region_bbox)

    # Plot the pie charts
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(aspect="equal"))

    wedges1, texts1, autotexts1 = ax[0].pie(
        df["t1"], autopct="%1.1f%%", colors=df["color"], startangle=90
    )
    ax[0].set_title(f"Land Use Composition - {time1}")

    # Plot second pie
    wedges2, texts2, autotexts2 = ax[1].pie(
        df["t2"], autopct="%1.1f%%", colors=df["color"], startangle=90
    )
    ax[1].set_title(f"Land Use Composition - {time2}")

    # Create a single legend for both plots (optional, can get crowded)
    fig.legend(
        wedges2,
        df["name"],
        title="Land Use Types",
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
    )
    plt.tight_layout(rect=(0, 0, 0.85, 1))  # Adjust layout only if legend exists

    plt.show()


# %%
def compare_LUC_bar(ds1, time1, ds2, time2, bbox=quebec_bbox_360):
    """
    Plot both the relative percentage change and the change in percentage points
    of land use types between two time points, side-by-side.

    Relative Change: (Area_t2 - Area_t1) / Area_t1 * 100%. Excludes types with zero area at time1.
    Share Change: %Share_t2 - %Share_t1 (change in percentage points of total area).

    Args:
        ds1: xarray Dataset for the first time point.
        time1: Time identifier for the first dataset.
        ds2: xarray Dataset for the second time point.
        time2: Time identifier for the second dataset.
        bbox: Optional bounding box [lon_min, lon_max, lat_min, lat_max] (0-360).
    """
    if not _test_inputs(ds1, time1, ds2, time2, bbox):
        return
    df = _get_df_of_total_land_use(ds1, time1, ds2, time2, bbox)

    t1_total = df["t1"].sum()
    t2_total = df["t2"].sum()

    df["change_percent"] = ((df["t2"] - df["t1"]) / df["t1"]) * 100
    df["t1_share"] = df["t1"] / t1_total
    df["t2_share"] = df["t2"] / t2_total

    # --- Prepare data for plotting ---
    red_green = ["#d62728" if v < 0 else "#2ca02c" for v in df["change_percent"]]
    labels = df["name"]
    # --- Plotting ---
    plt.style.use("seaborn-v0_8-talk")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, max(6, len(labels) * 0.4)), sharey=True
    )  # Adjust height based on labels
    fig.suptitle(f"Land Use Comparison: {time1} vs {time2}", fontsize=16)

    # Plot 1: Relative Percentage Change
    bars1 = ax1.barh(labels, df["change_percent"], color=red_green)
    ax1.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    max_abs_change1 = (
        np.max(np.abs(df["change_percent"])) if len(df["change_percent"]) > 0 else 1
    )
    ax1.set_xlim(
        -max_abs_change1 * 1.15, max_abs_change1 * 1.15
    )  # Slightly more padding
    ax1.axvline(0, color="grey", linewidth=0.8)
    ax1.set_xlabel(f"Change relative to {time1} Area (%)")
    ax1.set_ylabel("Land Use Type")  # Shared y-label
    ax1.set_title("Relative Percentage Change")
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    # Plot 2: Absolute Share Comparison (Grouped Bar Chart)
    bar_height = 0.35
    y_pos = np.arange(len(labels))

    bars_t1 = ax2.barh(
        y_pos + bar_height / 2,
        df["t1_share"],
        bar_height,
        label=f"{time1}",
        color="#1f77b4",
    )  # Blue
    bars_t2 = ax2.barh(
        y_pos - bar_height / 2,
        df["t2_share"],
        bar_height,
        label=f"{time2}",
        color="#ff7f0e",
    )  # Orange

    # Add value labels on bars
    ax2.bar_label(bars_t1, fmt="%.3f", padding=3, fontsize=9)
    ax2.bar_label(bars_t2, fmt="%.3f", padding=3, fontsize=9)

    # Customize plot 2
    # Determine appropriate xlim. Max share value + padding.
    max_share = max(
        np.max(df["t1_share"]) if len(df["t1_share"]) > 0 else 0,
        np.max(df["t2_share"]) if len(df["t2_share"]) > 0 else 0,
    )
    ax2.set_xlim(0, max_share * 1.1)

    ax2.set_yticks(y_pos)  # Ensure ticks are centered between the grouped bars
    ax2.set_yticklabels(
        labels
    )  # Set labels (already done by sharey=True, but good practice)
    ax2.set_xlabel("Proportion of Total Area")
    ax2.set_title("Land Use Proportions")
    ax2.legend()
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    # Add horizontal lines between groups for better separation
    for y in y_pos[:-1]:
        ax2.axhline(y + 0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout for suptitle
    plt.show()


# %%
def compare_LCTYPE(ds1, time1, ds2, time2, lctype, bbox):
    """For a given LCTYPE, plot the variation in land occupation between time1 and time2 as a raster map."""
    if not _test_inputs(ds1, time1, ds2, time2, bbox):
        return

    # Check if lctype is valid
    if lctype not in lctype_to_name:
        print(
            f"Error: Invalid lctype '{lctype}'. Must be one of {list(lctype_to_name.keys())}"
        )
        return

    # 1. Extract data for the specific lctype and times within the bbox
    lc_frac1 = ds1.landCoverFrac.sel(lctype=lctype, time=time1)
    lc_frac2 = ds2.landCoverFrac.sel(lctype=lctype, time=time2)

    if bbox:
        lc_frac1 = _select_bbox(lc_frac1, bbox)
        lc_frac2 = _select_bbox(lc_frac2, bbox)

    # Remove time dimension since we've already selected a specific time
    lc_frac1 = lc_frac1.squeeze(dim="time")
    lc_frac2 = lc_frac2.squeeze(dim="time")

    change = lc_frac2 - lc_frac1

    # Handle cases where change might be all NaN after subtraction
    if change.isnull().all():
        print(
            f"Warning: Change calculation resulted in all NaN values for lctype "
            f"{lctype} ({lctype_to_name[lctype]}) between {time1} and {time2}."
        )
        return

    # 3. Plotting using xarray and cartopy for background map
    lctype_name = lctype_to_name[lctype]

    # Determine min/max for colormap scaling, centering around 0
    vmin = change.min().item()
    vmax = change.max().item()
    # Center the colormap around 0 if the range spans zero
    max_abs = max(abs(vmin), abs(vmax))
    if vmin < 0 < vmax:
        vmin = -max_abs
        vmax = max_abs
    elif vmin >= 0:
        vmin = 0
    elif vmax <= 0:
        vmax = 0

    # Ensure vmin and vmax are different for the colormap
    if abs(vmin - vmax) < 1e-9:
        vmin -= 0.01
        vmax += 0.01

    # Create figure and axes with Cartopy projection
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={
            "projection": ccrs.PlateCarree()
        },  # Use PlateCarree for lat/lon data
    )
    ax.add_feature(cfeature.OCEAN, zorder=1)
    ax.add_feature(cfeature.COASTLINE, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle="-", zorder=1)
    ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=1)
    ax.add_feature(cfeature.RIVERS, zorder=1)
    # Add provincial and states borders
    provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
        edgecolor="gray",
        linestyle=":",
    )
    ax.add_feature(provinces, zorder=1)

    # Plot the data using xarray's plot, specifying the axes and data transform
    # Assuming 'change' coordinates are standard lat/lon (EPSG:4326)
    img = change.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),  # Specify data CRS
        cmap="PiYG",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={
            "label": f"Fraction Change ({lctype_name})",
            "shrink": 0.61,  # Shrink colorbar to match the map size
        },
        zorder=0,  # Plot data below borders on other map elements (zorder=0)
    )

    plt.tight_layout()  # minimize whitespace

    plt.title(f"Change in '{lctype_name}' Fraction ({time1} to {time2})", pad=20)

    # Add gridlines and labels
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0,  # Hide gridlines by setting width to 0
    )
    gl.top_labels = False
    gl.right_labels = False

    plt.show()
