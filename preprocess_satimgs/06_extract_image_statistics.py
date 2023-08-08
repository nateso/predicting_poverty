import pickle
from satimg_utils.img_proc_utils import *

# Extract statistics for every image type (LS_raw, LS_median_cluster, RS_v2_raw,
# RS_v2_mean_cluster, RS_v2_dynamic_processed, RS_v2_static_processed)

# set the root data directory
root_data_dir = "../../Data"
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# Load the LSMS data
lsms_path = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"
lsms_df = pd.read_csv(lsms_path)
cluster_ids = lsms_df['cluster_id'].unique()

#*******************************************
#................ Landsat ..................
#*******************************************

#### RAW LANDSAT IMAGES ####
data_type = 'LS'
id_name = 'unique_id'
raw_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_raw"
ls_file_names = [f'{data_type}_{i}.tif' for i in lsms_df['unique_id']]
ls_file_paths = [
    f'{raw_img_dir}/{data_type}_{lsms_df.country[i]}_{lsms_df.series[i]}_{lsms_df.start_year[i]}/{ls_file_names[i]}' for i
    in range(len(lsms_df))]

uid_pth_dict = dict(zip(lsms_df[id_name], ls_file_paths))

print("\nPROCESS RAW LANDSAT IMAGES")
ls_raw_stats = extract_image_statistics(uid_pth_dict, data_type, id_name)

# save the img stats
pth = f"{sat_img_dir}/{data_type}/{data_type}_raw_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(ls_raw_stats, f)

#### MEDIAN CLUSTER LANDSAT IMAGES ####
id_name = 'cluster_id'
img_dir = f"{sat_img_dir}/{data_type}/{data_type}_median_cluster"
ls_file_names = [f'{data_type}_{i}.npy' for i in cluster_ids]
ls_file_paths = [f'{img_dir}/{i}' for i in ls_file_names]
cid_pth_dict = dict(zip(cluster_ids, ls_file_paths))

print("\nPROCESS MEDIAN LANDSAT IMAGES")
ls_median_stats = extract_image_statistics(cid_pth_dict, data_type, id_name)

# save the img stats
pth = f"{sat_img_dir}/{data_type}/{data_type}_median_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(ls_median_stats, f)

#*******************************************
#................ RS_V2 ..................
#*******************************************

#### RAW RS_V2 IMAGES ####
data_type = 'RS_v2'
id_name = 'unique_id'
raw_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_raw"
rs_file_names = [f'{data_type}_{i}.tif' for i in lsms_df['unique_id']]
rs_file_paths = [
    f'{raw_img_dir}/{data_type}_{lsms_df.country[i]}_{lsms_df.series[i]}_{lsms_df.start_year[i]}/{rs_file_names[i]}' for i
    in range(len(lsms_df))]

uid_pth_dict = dict(zip(lsms_df[id_name], rs_file_paths))

print("\nPROCESS RAW RS V2 IMAGES")
rs_raw_stats = extract_image_statistics(uid_pth_dict, data_type, id_name)

# save the img stats
pth = f"{sat_img_dir}/{data_type}/{data_type}_raw_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(rs_raw_stats, f)

#### RS_V2 dynamic images ####
data_type = 'RS_v2'
id_name = 'unique_id'
img_dir = f"{sat_img_dir}/{data_type}/{data_type}_dynamic_processed"
file_names = [f'{data_type}_{i}.npy' for i in lsms_df['unique_id']]
file_paths = [f'{img_dir}/{i}' for i in file_names]
uid_pth_dict = dict(zip(lsms_df[id_name], file_paths))

print("\nPROCESS RS V2 DYNAMIC IMAGES")
rs_dynamic_stats = extract_image_statistics(uid_pth_dict, data_type, id_name)

# save the img stats
pth = f"{sat_img_dir}/{data_type}/{data_type}_dynamic_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(rs_dynamic_stats, f)


#### RS_V2 static images ####
data_type = 'RS_v2'
id_name = 'cluster_id'
img_dir = f"{sat_img_dir}/{data_type}/{data_type}_static_processed"
file_names = [f'{data_type}_{i}.npy' for i in cluster_ids]
file_paths = [f'{img_dir}/{i}' for i in file_names]
cid_pth_dict = dict(zip(cluster_ids, file_paths))

print("\nPROCESS RS V2 STATIC IMAGES")
rs_static_stats = extract_image_statistics(cid_pth_dict, data_type, id_name)

# save the img stats
pth = f"{sat_img_dir}/{data_type}/{data_type}_static_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(rs_static_stats, f)


#### RS_V2 mean images ####
data_type = 'RS_v2'
id_name = 'cluster_id'
img_dir = f"{sat_img_dir}/{data_type}/{data_type}_mean_cluster"
file_names = [f'{data_type}_{i}.npy' for i in cluster_ids]
file_paths = [f'{img_dir}/{i}' for i in file_names]
cid_pth_dict = dict(zip(cluster_ids, file_paths))

print("\nPROCESS RS V2 MEAN IMAGES")
rs_mean_stats = extract_image_statistics(cid_pth_dict, data_type, id_name)

# save the img stats
pth = f"{sat_img_dir}/{data_type}/{data_type}_mean_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(rs_mean_stats, f)











