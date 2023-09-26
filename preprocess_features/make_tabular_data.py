import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from features_utils.reduce_dimensions import reduce_dims
import os
#************************************************************
# Set the global parameters for the location of data
#************************************************************

root_data_dir = "../../Data"
sat_img_dir = f"{root_data_dir}/satellite_imgs"
lsms_path = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"
figures_dir = "../figures/tabular_features"

# static data
esa_lc_pth = f"{sat_img_dir}/RS_v2/esa_lc_decomp.csv"

# OSM data
osm_amenity_count_pth = f"{root_data_dir}/OSM/osm_amenity_count.csv"
osm_amenity_dist_pth = f"{root_data_dir}/OSM/osm_amenity_distances.csv"
osm_road_features_pth = f"{root_data_dir}/OSM/osm_road_features.csv"

# for the WSF data load the static image stats
static_img_stats_pth = f"{sat_img_dir}/RS_v2/RS_v2_static_img_stats.pkl"
rs_v2_mean_imgs_stats_pth = f"{sat_img_dir}/RS_v2/RS_v2_mean_img_stats.pkl"

# dynamic data
precip_pth = f"{root_data_dir}/precipitation.csv"
dynamic_img_stats_pth = f"{sat_img_dir}/RS_v2/RS_v2_dynamic_img_stats.pkl"

# Features of RGB images
ls_rgb_median_feats_pth = f"{sat_img_dir}/LS/median_rgb_feats.csv"
ls_rgb_dyn_feats_pth = f"{sat_img_dir}/LS/dyn_rgb_feats.csv"

# set the band name dictionaries
mean_name_dict = dict(zip(list(range(4)),['nl','ndvi','ndwi_gao','ndwi_mcf']))

#************************************************************
# Load the data
#************************************************************

lsms_df = pd.read_csv(lsms_path)

#### static data
esa_lc_df = pd.read_csv(esa_lc_pth)
esa_lc_df['wetland'] = esa_lc_df['wetland'] + esa_lc_df['mangroves']
esa_lc_df = esa_lc_df.drop(columns=['snow_ice','moss','mangroves'])

# load the wsf data
with open(static_img_stats_pth, 'rb') as f:
    static_img_stats = pickle.load(f)

wsf_df = static_img_stats[0][['mean','std','cluster_id']]
wsf_df = wsf_df.rename(columns={'mean':'wsf_mean','std':'wsf_std'})

## OSM data
# amenity count
osm_amenity_count = pd.read_csv(osm_amenity_count_pth)
column_mapping = {col: col + '_count' for col in osm_amenity_count.columns if col != 'cluster_id'}
osm_amenity_count = osm_amenity_count.rename(columns=column_mapping)

# amenity distance
osm_amenity_dist = pd.read_csv(osm_amenity_dist_pth)
column_mapping = {col: col + '_dist' for col in osm_amenity_dist.columns if col != 'cluster_id'}
osm_amenity_dist = osm_amenity_dist.rename(columns=column_mapping)

# road features
osm_road_features = pd.read_csv(osm_road_features_pth)
column_mapping = {col: col + '_road' for col in osm_road_features.columns if col != 'cluster_id'}
osm_road_features = osm_road_features.rename(columns=column_mapping)

# merge the osm data
osm_df = pd.merge(osm_amenity_count, osm_amenity_dist, on='cluster_id', how='left')
osm_df = pd.merge(osm_df, osm_road_features, on='cluster_id', how='left')

# for the mean images, extract the mean and std for each channel
with open(rs_v2_mean_imgs_stats_pth, 'rb') as f:
    rs_v2_mean_imgs_stats = pickle.load(f)

mean_nl_ndvi_ndwi_df = rs_v2_mean_imgs_stats[0][['cluster_id']]
for band, stats in rs_v2_mean_imgs_stats.items():
    band_name = mean_name_dict[band]
    band_stats = stats[['cluster_id','mean','std']]
    band_stats = band_stats.rename(columns={'mean':f'avg_{band_name}_mean','std':f'avg_{band_name}_std'})
    mean_nl_ndvi_ndwi_df = pd.merge(mean_nl_ndvi_ndwi_df, band_stats, on='cluster_id', how='left')

#### dynamic data
# load the precipitation data
precip_df = pd.read_csv(precip_pth)
precip_df = pd.merge(precip_df, lsms_df[['cluster_id','unique_id']], on='unique_id', how='left')
precip_df['avg_precipitation'] = precip_df.groupby('cluster_id')['precipitation'].transform('mean')
precip_df = precip_df.drop(columns = ['cluster_id'])

# load the dynamic image stats and extract the mean and std for each channel
with open(dynamic_img_stats_pth, 'rb') as f:
    dynamic_img_stats = pickle.load(f)

dynamic_df = dynamic_img_stats[0]['unique_id']
for band, stats in dynamic_img_stats.items():
    band_name = mean_name_dict[band]
    band_stats = stats[['mean','std','unique_id']]
    band_stats = band_stats.rename(columns={'mean':f'dyn_{band_name}_mean','std':f'dyn_{band_name}_std'})
    dynamic_df = pd.merge(dynamic_df, band_stats, on='unique_id', how='left')

#### load the LAndSAT RGB features
ls_median_df = pd.read_csv(ls_rgb_median_feats_pth)
ls_dyn_df = pd.read_csv(ls_rgb_dyn_feats_pth)

#************************************************************
# Merge the data
#************************************************************
feats_df = lsms_df[['cluster_id','unique_id']]
feats_df = pd.merge(feats_df, esa_lc_df, on='cluster_id', how='left')
feats_df = pd.merge(feats_df, wsf_df, on = 'cluster_id', how = 'left')
feats_df = pd.merge(feats_df, osm_df, on = 'cluster_id', how = 'left')
feats_df = pd.merge(feats_df, mean_nl_ndvi_ndwi_df, on='cluster_id', how='left')
feats_df = pd.merge(feats_df, precip_df, on = 'unique_id', how = 'left')
feats_df = pd.merge(feats_df, dynamic_df, on = 'unique_id', how = 'left')
feats_df = pd.merge(feats_df, ls_median_df, on = 'cluster_id', how = 'left')
feats_df = pd.merge(feats_df, ls_dyn_df, on = 'unique_id', how = 'left')

#************************************************************
# Plot the correlation matrix
#************************************************************
# check correlations among variables in the data

# Compute the correlation matrix
corr_matrix = feats_df.drop(columns = ['cluster_id','unique_id']).corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'correlation_matrix.png'), dpi=300)
#plt.show()

#************************************************************
# Feature engineering
#************************************************************
print('RUN PCA TO REDUCE DIMENSIONS OF OSM FEATURES')
# run PCA on OSM counts and dists seperately
print("\nCount variables")
osm_count_vars = [i for i in feats_df.columns if (('count' in i) and ('country' not in i))]
reduced_counts = reduce_dims(feats_df, osm_count_vars, n_comp = 5, col_prefix = 'osm_count_')
feats_df = pd.merge(feats_df, reduced_counts, on = 'unique_id', how = 'left')

print(f"\nDistance variables")
osm_dist_vars = [i for i in feats_df.columns if (('dist' in i) and (i not in ['distance_paved','distance_primary']))]
reduced_dists = reduce_dims(feats_df, osm_dist_vars, n_comp = 5, col_prefix = 'osm_dist_')
feats_df = pd.merge(feats_df, reduced_dists, on = 'unique_id', how = 'left')

#************************************************************
# Save the data
#************************************************************
feats_df.to_csv(f"{root_data_dir}/feature_data/tabular_data.csv", index=False)











