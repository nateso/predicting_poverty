from tqdm.auto import tqdm
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

#### Start with Landsat images
data_type = 'LS'

# check whether folder exists and create it if not
ls_proc_pth = f"{sat_img_dir}/{data_type}/{data_type}_median_cluster"
if not os.path.isdir(ls_proc_pth):
    os.makedirs(ls_proc_pth)
else:
    print(f"Warning: LS_median_cluster Folder already exists. Files might be overwritten.")
    if input("Do You Want To Continue? [y/n]") == 'n':
        exit("Process aborted.")


raw_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_raw"

ls_file_names = [f'{data_type}_{i}.tif' for i in lsms_df['unique_id']]
ls_file_paths = [
    f'{raw_img_dir}/{data_type}_{lsms_df.country[i]}_{lsms_df.series[i]}_{lsms_df.start_year[i]}/{ls_file_names[i]}' for i
    in range(len(lsms_df))]

# create an id-path dictionary
uid_pth_dict = dict(zip(lsms_df['unique_id'], ls_file_paths))

# load images aggregate them by cluster and save them
print("Processing Landsat images...")
for cid, uids in tqdm(cid_uid_dict.items()):
    # load all images of a given cluster
    imgs = []
    for uid in uids:
        img_pth = uid_pth_dict[uid]
        img = load_img(img_pth)
        img = reorder_rgb(img)
        img = np.delete(img, 5, axis=2)  # delete band 5 due to a lot of missing values
        imgs.append(img)

    # take the median over all images for that cluster
    imgs = np.array(imgs)
    median_img = np.nanmedian(imgs, axis=0)  # ignore nan values
    # save the median image for each cluster
    new_file_path = f'{sat_img_dir}/{data_type}/LS_median_cluster/{data_type}_{cid}.npy'
    np.save(new_file_path, median_img)

#### Process RS images
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
for cid, uids in tqdm(cid_uid_dict.items()):
    # load all images of a given cluster
    imgs = []
    for uid in uids:
        img_pth = uid_pth_dict[uid]
        img = load_img(img_pth)

        # set aside the ESA LC img (this is static)
        esa_lc_img = img[:,:,[-1]]

        # remove wsf image, modis LC, ESA LC image (index 2 and 5, 6)
        img = np.delete(img, [2,5,6], axis = 2)

        # append the image to the list
        imgs.append(img)

        # save the image to folder
        new_file_path = f'{sat_img_dir}/{data_type}/{data_type}_dynamic_processed/{data_type}_{uid}.npy'
        np.save(new_file_path, img)

    # take the mean over NL, NDVI, NDWI
    imgs = np.array(imgs)
    mean_img = np.nanmean(imgs, axis = 0)

    # save the mean image
    new_file_path = f'{sat_img_dir}/{data_type}/{data_type}_mean_cluster/{data_type}_{cid}.npy'
    np.save(new_file_path, mean_img)

    # load the WSF image
    wsf_pth = f'{sat_img_dir}/{data_type}/WSF_raw/WSF_{cid}.tif'
    wsf_img = load_img(wsf_pth)
    wsf_img[wsf_img == 255] = 1 # replace the values 255 with 1

    # concatenate the wsf and lc
    wsf_lc_img = np.concatenate((wsf_img, esa_lc_img), axis = 2)

    # save the new images
    new_file_path = f'{sat_img_dir}/{data_type}/{data_type}_static_processed/{data_type}_{cid}.npy'
    np.save(new_file_path, wsf_lc_img)


