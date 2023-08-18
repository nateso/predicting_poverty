import numpy as np


def describe_results(cluster_gpd):
    '''
    input: cluster_gpd geopandas dataset with columns 'geometry', 'new_point', 'no_recenter'
    '''
    proj_new_points = cluster_gpd.new_point.copy().to_crs('EPSG:3857')
    proj_old_points = cluster_gpd.geometry.copy().to_crs('EPSG:3857')

    distance_old_new = []
    # get distance between old and new points
    for i in range(len(proj_new_points)):
        dist = proj_new_points[i].distance(proj_old_points[i])
        distance_old_new.append(dist)
    distance_old_new = np.array(distance_old_new)
    urban_mask = cluster_gpd.rural == 0

    same_location_mask = cluster_gpd[['new_lat', 'new_lon']].duplicated(keep=False)

    print(f'The minimum distance between old and new points: {min(distance_old_new)}')
    print(f'The maximum distance between old and new points: {max(distance_old_new)}')
    print(f'Maximum distance if urban: {max(distance_old_new[urban_mask])}')
    print(f'Maximum distance if rural: {max(distance_old_new[~urban_mask])}')
    print(
        f'Number of clusters where displacement exceedes the max raidus (Urban): {sum(distance_old_new[urban_mask] > 2000)}')
    print(
        f'Number of clusters where displacement exceedes the max raidus (Rural): {sum(distance_old_new[~urban_mask] > 5000)}')
    print(f'Number of clusters that were not relocated: {sum(cluster_gpd.no_recenter)}')
    print(f"Number of clusters with same geo-locations: {sum(same_location_mask)}")
    print("Note that two clusters have the same geo-locations by default")
