import os
from satimg_utils.img_proc_utils import *

# set the root data directory
root_data_dir = "../../Data"
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# Load the LSMS data
lsms_path = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"
lsms_df = pd.read_csv(lsms_path)

# create dictionary of cluster_id to unique_ids
cid_uid_dict = {}
for idx, row in lsms_df.iterrows():
    cid = row['cluster_id']
    uid = row['unique_id']

    if cid in cid_uid_dict:
        cid_uid_dict[cid].append(uid)
    else:
        cid_uid_dict[cid] = [uid]

#*******************************************
#................ Landsat ..................
#*******************************************
data_type = 'LS'

# check whether folder exists and create it if not
ls_proc_pth = f"{sat_img_dir}/{data_type}/{data_type}_median_cluster"
ls_rgb_pth = f"{sat_img_dir}/{data_type}/{data_type}_rgb"
ls_rgb_median_pth = f"{sat_img_dir}/{data_type}/{data_type}_rgb_median_cluster"
if not os.path.isdir(ls_proc_pth):
    os.makedirs(ls_proc_pth)
else:
    print(f"Warning: LS_median_cluster Folder already exists. Files might be overwritten.")
    if input("Do You Want To Continue? [y/n]") == 'n':
        exit("Process aborted.")

if not os.path.isdir(ls_rgb_pth):
    os.makedirs(ls_rgb_pth)

if not os.path.isdir(ls_rgb_median_pth):
    os.makedirs(ls_rgb_median_pth)


raw_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_raw"

ls_file_names = [f'{data_type}_{i}.tif' for i in lsms_df['unique_id']]
ls_file_paths = [
    f'{raw_img_dir}/{data_type}_{lsms_df.country[i]}_{lsms_df.series[i]}_{lsms_df.start_year[i]}/{ls_file_names[i]}' for i
    in range(len(lsms_df))]

# create an id-path dictionary
uid_pth_dict = dict(zip(lsms_df['unique_id'], ls_file_paths))

# load images aggregate them by cluster and save them
print("Processing Landsat images...")
cid_list = []
uid_list = []
ls_raw_img_stats = {}
ls_median_img_stats = {}

for cid, uids in tqdm(cid_uid_dict.items()):
    cid_list.append(cid)
    # load all images of a given cluster
    imgs = []
    for uid in uids:
        uid_list.append(uid)
        img_pth = uid_pth_dict[uid]
        img = load_img(img_pth)
        img = reorder_rgb(img)
        proc_img = np.delete(img, 5, axis=2)  # delete band 5 due to a lot of missing values
        imgs.append(proc_img)

        # save RGB images
        rgb_img = img[:, :, :3]
        rgb_file_pth = f'{sat_img_dir}/{data_type}/LS_rgb/{data_type}_{uid}.npy'
        np.save(rgb_file_pth, rgb_img)

        # extract image statistics
        for i in range(img.shape[2]):
            if i in ls_raw_img_stats.keys():
                ls_raw_img_stats[i].append(get_basic_band_stats(img[:, :, i]))
            else:
                ls_raw_img_stats[i] = [get_basic_band_stats(img[:, :, i])]

    # take the median over all images for that cluster
    imgs = np.array(imgs)
    median_img = np.nanmedian(imgs, axis=0)  # ignore nan values

    # extract image statistics
    for i in range(median_img.shape[2]):
        if i in ls_median_img_stats.keys():
            ls_median_img_stats[i].append(get_basic_band_stats(median_img[:, :, i]))
        else:
            ls_median_img_stats[i] = [get_basic_band_stats(median_img[:, :, i])]

    # save the median image for each cluster
    new_file_path = f'{sat_img_dir}/{data_type}/LS_median_cluster/{data_type}_{cid}.npy'
    np.save(new_file_path, median_img)

    # save the median rgb image for each cluster
    new_file_path = f'{sat_img_dir}/{data_type}/LS_rgb_median_cluster/{data_type}_{cid}.npy'
    np.save(new_file_path, median_img[:, :, :3])

# convert the image statistics dictionary values to pandas dataframes
for band_key in ls_raw_img_stats.keys():
    ls_raw_img_stats[band_key] = pd.json_normalize(ls_raw_img_stats[band_key])
    ls_raw_img_stats[band_key]['unique_id'] = uid_list

for band_key in ls_median_img_stats.keys():
    ls_median_img_stats[band_key] = pd.json_normalize(ls_median_img_stats[band_key])
    ls_median_img_stats[band_key]['cluster_id'] = cid_list

