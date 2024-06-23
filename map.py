import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# load the world map from the natural earth dataset
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# filter out Antarctica for a better view of other regions
world = world[(world.pop_est > 0) & (world.name != "Antarctica")]

# define cluster locations
locations = {
    "Cape Town": (-33.9249, 18.4241),
    "Hong Kong": (22.3193, 114.1694),
    "Canada": (43.65107, -79.347015),
    "London": (51.5074, -0.1278),
    "Northern California": (37.7749, -122.4194)
}

# create a GeoDataFrame for the locations
geometry = [Point(xy[1], xy[0]) for xy in locations.values()]
gdf_locations = gpd.GeoDataFrame(geometry, columns=['geometry'])
gdf_locations['Location'] = locations.keys()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(ax=ax, color='lightgray', edgecolor='white')
gdf_locations.plot(ax=ax, color='blue', markersize=100)

ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.set_axis_off()

plt.savefig("results/figures/clusters_map.pdf", bbox_inches='tight')