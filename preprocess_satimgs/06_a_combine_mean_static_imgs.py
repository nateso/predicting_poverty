import numpy as np
import pandas as pd
import os
import pickle
from satimg_utils.img_proc_utils import *
from tqdm.auto import tqdm


# set the root data directory
root_data_dir = "../../Data"
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# Load the LSMS data
lsms_path = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"
lsms_df = pd.read_csv(lsms_path)

# create folder to save the demaned image
between_img_dir = f"{sat_img_dir}/RS_v2/RS_v2_between"
if not os.path.isdir(between_img_dir):
    os.makedirs(between_img_dir)
else:
    print(f"Warning: RS_v2_between Folder already exists. Files might be overwritten.")
    if input("Do You Want To Continue? [y/n]") == 'n':
        exit("Process aborted.")

# set the image directories
data_type = 'RS_v2'
mean_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_mean_cluster"
static_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_static_processed"

#****************************************************************
# combine mean images and static images
#****************************************************************
between_img_stats = {}
cid_list = []
for cid in tqdm(np.unique(lsms_df.cluster_id)):
    # load the mean image
    mean_img = np.load(f"{mean_img_dir}/{data_type}_{cid}.npy")
    # load the static image
    static_img = np.load(f"{static_img_dir}/{data_type}_{cid}.npy")
    # combine the images
    between_img = np.concatenate((mean_img, static_img), axis=2)
    # save the demeaned image
    np.save(f"{between_img_dir}/{data_type}_{cid}.npy", between_img)

    # extract image stats
    for i in range(between_img.shape[2]):
        if i in between_img_stats.keys():
            between_img_stats[i].append(get_basic_band_stats(between_img[:, :, i]))
        else:
            between_img_stats[i] = [get_basic_band_stats(between_img[:, :, i])]
    # add the cluster id to the list
    cid_list.append(cid)


for band_key in between_img_stats.keys():
    between_img_stats[band_key] = pd.json_normalize(between_img_stats[band_key])
    between_img_stats[band_key]['cluster_id'] = cid_list


# save the image stats
with open(f"{sat_img_dir}/{data_type}/{data_type}_between_img_stats.pkl", 'wb') as f:
    pickle.dump(between_img_stats, f)






