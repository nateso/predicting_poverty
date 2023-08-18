import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point
from datetime import datetime, timedelta

from tqdm.auto import tqdm
import numpy as np


def recenter_data(cluster_gpd, client, fltr, extreme=False):
    print('Download data from OSM')
    new_lats, new_lons = [], []
    old_ids = []
    old_count = 0
    # load all populated areas from OSM within the country
    roi_multip = gpd.GeoDataFrame(geometry=cluster_gpd['roi'].copy(), crs='EPSG:4326')
    roi_multip = roi_multip.geometry.unary_union

    # run the query
    bpolys = gpd.GeoDataFrame(geometry=[roi_multip], crs='EPSG:4326')
    populated_areas = download_OSM_data(bpolys, fltr, client)

    populated_areas = preprocess_response(populated_areas)

    print('Recenter locations...')
    # recenter the locations
    for j, cluster_info in tqdm(cluster_gpd.iterrows(), total=len(cluster_gpd)):
        cluster_id = cluster_info['cluster_id']

        nearest_geom, dist = get_nearest_geom(cluster_info, populated_areas, extreme)
        new_lat, new_lon = recenter_location(cluster_info, nearest_geom, dist)

        if np.isnan(new_lat):
            print(f'{cluster_id} could not find new location, use old location instead')
            new_lat, new_lon = cluster_info.lat, cluster_info.lon
            old_count += 1
            old_ids.append(cluster_id)
        new_lats.append(new_lat)
        new_lons.append(new_lon)
    return new_lats, new_lons, old_count, old_ids


def recenter_location(cluster_info, nearest_geom, dist):
    if nearest_geom is None:
        new_lon, new_lat = np.nan, np.nan
    elif dist < 1:
        # if the min distance is smaller than 1 meter, just stay
        new_lon, new_lat = float(cluster_info.lon), float(cluster_info.lat)
    else:
        intersection = nearest_geom.intersection(cluster_info.roi)
        closest_point = nearest_points(intersection, cluster_info.geometry)[
            0]  # select the first, as this is the closes point on the polygon
        new_lon, new_lat = float(closest_point.x), float(closest_point.y)
    return new_lat, new_lon


def get_nearest_geom(cluster_info, response_df, extreme=False):
    lon, lat = float(cluster_info.lon), float(cluster_info.lat)
    point = Point(lon, lat)
    proj_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs('EPSG:3857')
    proj_geom = response_df.copy().to_crs('EPSG:3857')

    # set the radius
    if cluster_info.rural == 1:
        if extreme:
            radius = 10000
        else:
            radius = 5000
    else:
        if extreme:
            radius = 5000
        else:
            radius = 2000

    # Use a spatial index to find the nearest point
    sindex = proj_geom.sindex
    nearest_idx, dist = sindex.nearest(proj_point.values[0][0], max_distance=radius, return_all=False,
                                       return_distance=True)
    if len(dist) == 0:
        nearest_geom = None
        dist = np.nan
    else:
        nearest_geom = proj_geom.iloc[nearest_idx.flatten()[1]].values[0]
        nearest_geom_df = gpd.GeoDataFrame(geometry=[nearest_geom], crs='EPSG:3857').to_crs('EPSG:4326')
        nearest_geom = nearest_geom_df.geometry[0]
        # Transform the polygon to the target CRS (EPSG 4326)
        # nearest_geom = nearest_geom.transform(transformer.transform)

    return nearest_geom, float(dist)


def preprocess_response(response_df):
    # split all multipolygons into single polygons
    response_df = response_df.explode(index_parts=False).reset_index(drop=True)

    # ensure that all geometries are valid
    invalid_mask = ~response_df.geometry.is_valid
    print(f'fixing {sum(invalid_mask)} invalid geometries')
    response_df.geometry[invalid_mask] = response_df.geometry[invalid_mask].buffer(0)

    return response_df


def download_OSM_data(bpolys, fltr, client, tags=False):
    pre = datetime.now()
    tm = "2023-07-01/2023-07-31/P1M"
    if tags:
        response = client.elements.geometry.post(bpolys=bpolys,
                                                 time=tm,
                                                 filter=fltr,
                                                 properties='tags')
    else:
        response = client.elements.geometry.post(bpolys=bpolys,
                                                 time=tm,
                                                 filter=fltr)

    response_df = response.as_dataframe()
    post = datetime.now()
    time_diff = pre - post
    print(f'Downloading the OSM data took {round(abs(time_diff.total_seconds()))} seconds')
    return response_df


def get_distance(cluster_info, reponse_df):
    lon, lat = cluster_info.lon, cluster_info.lat
    point = gpd.points_from_xy(x=[lon], y=[lat], crs="EPSG:4326")
    proj_point = point.to_crs('EPSG:3857').copy()
    # proj_geom = reponse_df['geometry'].to_crs('EPSG:3857')
    # dist = []
    # for i in range(len(proj_geom)):
    #   with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     try:
    #       d = proj_point.distance(proj_geom[i])[0]
    #     except RuntimeWarning:
    #       print(f"Warning in iteration {i}")
    #       d = np.nan
    #     dist.append(d)
    dist = np.array([i.distance(proj_point)[0] for i in reponse_df['geometry'].to_crs('EPSG:3857')])
    return dist
