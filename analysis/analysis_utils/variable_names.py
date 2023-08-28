# this script defines the set of variable names
# define variable sets

# Static variables
esa_lc_vars = ['tree', 'shrubland', 'grassland', 'cropland',
               'built_up', 'barren', 'water', 'wetland']

wsf_vars = ['wsf_mean', 'wsf_std']

# OSM variables (also static)
osm_count_vars = ['restaurant_count', 'bar_count', 'cafe_count', 'marketplace_count',
                  'school_count', 'university_count', 'library_count', 'fuel_count',
                  'pharmacy_count', 'hospital_count', 'clinic_count']

osm_dist_vars = ['school_dist', 'restaurant_dist', 'cafe_dist', 'fuel_dist', 'marketplace_dist',
                 'hospital_dist', 'pharmacy_dist', 'bar_dist', 'clinic_dist','university_dist',
                 'library_dist']

osm_road_vars = ['road_length_road', 'distance_paved_road', 'distance_primary_road']

osm_vars = osm_count_vars + osm_dist_vars + osm_road_vars

osm_count_pca_vars = ['osm_count_pc_1', 'osm_count_pc_2', 'osm_count_pc_3',
                      'osm_count_pc_4', 'osm_count_pc_5']

osm_dist_pca_vars = ['osm_dist_pc_1', 'osm_dist_pc_2', 'osm_dist_pc_3',
                     'osm_dist_pc_4', 'osm_dist_pc_5']

# mean and standard deviation of the mean dynamic images
rs_mean_vars = ['avg_nl_mean', 'avg_nl_std', 'avg_ndvi_mean', 'avg_ndvi_std', 'avg_ndwi_gao_mean',
                'avg_ndwi_gao_std', 'avg_ndwi_mcf_mean', 'avg_ndwi_mcf_std',]

# mean and standard deviation of the dynamic images
rs_dyn_vars = ['dyn_nl_mean', 'dyn_nl_std', 'dyn_ndvi_mean', 'dyn_ndvi_std',
               'dyn_ndwi_gao_mean', 'dyn_ndwi_gao_std', 'dyn_ndwi_mcf_mean',
               'dyn_ndwi_mcf_std']

# precipitation
precipitation = ['precipitation']


# store variable names that should not be normalised
exclude_norm_cols = ['start_day', 'start_month', 'start_year', 'end_day',
                     'end_month', 'end_year', 'wave', 'rural', 'lsms_lat', 'lsms_lon',
                     'lat', 'lon', 'n_households', 'flagged', 'log_mean_pc_cons_usd_2017',
                     'log_median_pc_cons_usd_2017', 'mean_asset_index_yeh']




