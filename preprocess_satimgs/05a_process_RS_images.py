import os
import pickle

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
#................ RS_V2 ..................
#*******************************************

data_type = 'RS_v2'
raw_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_raw"

rs_proc_pth = f"{sat_img_dir}/RS_v2/RS_v2_mean_cluster"
if not os.path.isdir(rs_proc_pth):
    os.makedirs(rs_proc_pth)
else:
    print(f"Warning: RS_v2_mean_cluster Folder already exists. Files might be overwritten.")
    if input("Do You Want To Continue? [y/n]") == 'n':
        exit("Process aborted.")

# ensure that all folders to save the processed images exist
pth = f"{sat_img_dir}/{data_type}/{data_type}_dynamic_processed"
if not os.path.isdir(pth):
  os.makedirs(pth)

pth = f"{sat_img_dir}/{data_type}/{data_type}_static_processed"
if not os.path.isdir(pth):
  os.makedirs(pth)

# set the file names
rs_file_names = [f'{data_type}_{i}.tif' for i in lsms_df['unique_id']]
rs_file_paths = [f'{raw_img_dir}/{data_type}_{lsms_df.country[i]}_{lsms_df.series[i]}_{lsms_df.start_year[i]}/{rs_file_names[i]}' for i in range(len(lsms_df))]

# create an id-path dictionary
uid_pth_dict = dict(zip(lsms_df['unique_id'], rs_file_paths))

# load images aggregate them by cluster and save them
print("Processing RS images...")
esa_lc_decomp = []
dynamic_img_stats = {}
static_img_stats = {}
rs_v2_mean_img_stats = {}
rs_v2_raw_img_stats = {}
cid_list = []
uid_list = []

for cid, uids in tqdm(cid_uid_dict.items()):
    cid_list.append(cid)
    # load all images of a given cluster
    imgs = []
    for uid in uids:
        uid_list.append(uid)
        img_pth = uid_pth_dict[uid]
        img = load_img(img_pth)

        # extract the statistics from the processed dynamic images
        # channel 2 - WSF has missing values, but ignore since not used in the analysis.
        for i in range(img.shape[2]):
            if i in rs_v2_raw_img_stats.keys():
                rs_v2_raw_img_stats[i].append(get_basic_band_stats(img[:, :, i]))
            else:
                rs_v2_raw_img_stats[i] = [get_basic_band_stats(img[:, :, i])]

        # set aside the ESA LC img (this is static)
        esa_lc_img = img[:,:,[-1]]

        # remove wsf image, ndwi mcf, modis LC, ESA LC image (index 2 and 4, 5, 6)
        proc_img = np.delete(img, [2,4,5,6], axis = 2)

        # rescale the nightlights image to 0 1 range.
        proc_img[:, :, 0] = np.clip(proc_img[:, :, 0], 0, 100)
        proc_img[:,:,0] = proc_img[:,:,0]/100 # min-max normalisation (min = 0, max = 100).

        # append the image to the list
        imgs.append(proc_img)

        # extract the statistics from the processed dynamic images
        for i in range(proc_img.shape[2]):
            if i in dynamic_img_stats.keys():
                dynamic_img_stats[i].append(get_basic_band_stats(proc_img[:, :, i]))
            else:
                dynamic_img_stats[i] = [get_basic_band_stats(proc_img[:, :, i])]

        # save the image to folder
        new_file_path = f'{sat_img_dir}/{data_type}/{data_type}_dynamic_processed/{data_type}_{uid}.npy'
        np.save(new_file_path, proc_img)

    # take the mean over NL, NDVI, NDWI
    imgs = np.array(imgs)
    mean_img = np.nanmean(imgs, axis = 0)

    # extract statistics from the mean image
    for i in range(mean_img.shape[2]):
        if i in rs_v2_mean_img_stats.keys():
            rs_v2_mean_img_stats[i].append(get_basic_band_stats(mean_img[:, :, i]))
        else:
            rs_v2_mean_img_stats[i] = [get_basic_band_stats(mean_img[:, :, i])]

    # save the mean image
    new_file_path = f'{sat_img_dir}/{data_type}/{data_type}_mean_cluster/{data_type}_{cid}.npy'
    np.save(new_file_path, mean_img)

    # decompose the lc image into its categories
    esa_decomp_lc = decompose_lc(esa_lc_img, lc_idx = 0, lc_type = 'esa')
    esa_decomp_lc['cluster_id'] = cid
    esa_lc_decomp.append(esa_decomp_lc)

    # modify the ESA LC image (categories 70 and 100 do not exist) -- snow_ice and moss
    # categories wetland (90) and mangroves(95) occur rarely --> merge together to 70
    esa_lc_img[(esa_lc_img == 95) | (esa_lc_img == 90)] = 70

    # assign values between 0 and 1 to the categories (values will range from 0.1, to 0.8)
    # DONE: check the values that are assigned to the categories --> getting weird min max values atm.
    esa_lc_img = np.trunc(esa_lc_img / 10)/10

    # load the WSF image
    wsf_pth = f'{sat_img_dir}/{data_type}/WSF_raw/WSF_{cid}.tif'
    wsf_img = load_img(wsf_pth)
    wsf_img[wsf_img == 255] = 1 # replace the values 255 with 1

    # concatenate the wsf and lc
    wsf_lc_img = np.concatenate((wsf_img, esa_lc_img), axis = 2)

    # compute the basic image statistics for the static image
    for i in range(wsf_lc_img.shape[2]):
        if i in static_img_stats.keys():
            static_img_stats[i].append(get_basic_band_stats(wsf_lc_img[:, :, i]))
        else:
            static_img_stats[i] = [get_basic_band_stats(wsf_lc_img[:, :, i])]

    # save the new images
    new_file_path = f'{sat_img_dir}/{data_type}/{data_type}_static_processed/{data_type}_{cid}.npy'
    np.save(new_file_path, wsf_lc_img)

# save the ESA landcover categories
esa_lc_df = pd.DataFrame(esa_lc_decomp)
esa_lc_df.to_csv(f'{sat_img_dir}/{data_type}/esa_lc_decomp.csv', index = False)

# convert the image statistics dictionary values to pandas dataframes
for band_key in rs_v2_raw_img_stats.keys():
    rs_v2_raw_img_stats[band_key] = pd.json_normalize(rs_v2_raw_img_stats[band_key])
    rs_v2_raw_img_stats[band_key]['unique_id'] = uid_list

for band_key in dynamic_img_stats.keys():
    dynamic_img_stats[band_key] = pd.json_normalize(dynamic_img_stats[band_key])
    dynamic_img_stats[band_key]['unique_id'] = uid_list

for band_key in rs_v2_mean_img_stats.keys():
    rs_v2_mean_img_stats[band_key] = pd.json_normalize(rs_v2_mean_img_stats[band_key])
    rs_v2_mean_img_stats[band_key]['cluster_id'] = cid_list

for band_key in static_img_stats.keys():
    static_img_stats[band_key] = pd.json_normalize(static_img_stats[band_key])
    static_img_stats[band_key]['cluster_id'] = cid_list

# save the image stats
pth = f"{sat_img_dir}/RS_v2/RS_v2_raw_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(rs_v2_raw_img_stats, f)

pth = f"{sat_img_dir}/RS_v2/RS_v2_dynamic_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(dynamic_img_stats, f)

pth = f"{sat_img_dir}/RS_v2/RS_v2_mean_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(rs_v2_mean_img_stats, f)

pth = f"{sat_img_dir}/RS_v2/RS_v2_static_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(static_img_stats, f)

