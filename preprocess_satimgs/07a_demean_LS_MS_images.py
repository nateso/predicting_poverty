from satimg_utils.img_proc_utils import *
import os
import pickle

# set the root data directory
root_data_dir = "../../Data"
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# Load the LSMS data
lsms_path = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"
lsms_df = pd.read_csv(lsms_path)


# create folder to store the image deltas
delta_ms_img_dir = f"{sat_img_dir}/LS/LS_ms_delta"
if not os.path.isdir(delta_ms_img_dir):
    os.makedirs(delta_ms_img_dir)
else:
    print(f"Warning: LS_ms_delta Folder already exists. Files might be overwritten.")
    if input("Do You Want To Continue? [y/n]") == 'n':
        exit("Process aborted.")

# create dictionary of cluster_id to unique_ids
cid_uid_dict = {}
for idx, row in lsms_df.iterrows():
    cid = row['cluster_id']
    uid = row['unique_id']

    if cid in cid_uid_dict:
        cid_uid_dict[cid].append(uid)
    else:
        cid_uid_dict[cid] = [uid]

# sort the uids in each cluster
for cid, uids in cid_uid_dict.items():
    cid_uid_dict[cid] = sorted(uids)


# set the image directories
data_type = 'LS'
mean_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_median_cluster"
dynamic_img_dir = f"{sat_img_dir}/{data_type}/{data_type}_ms"

#****************************************************************
#........... create image deltas + demean images ................
#****************************************************************
demeaned_img_stats = {}
delta_img_stats = {}
uid_list = []
delta_id_list = []
print("PROCESS DATA...")
# iterate over all clusters
for cid, uids in tqdm(cid_uid_dict.items()):
    #print(f"Cluster: {cid}")
    # load the mean image
    mean_img = np.load(f"{mean_img_dir}/{data_type}_{cid}.npy")

    # load the dynamic images
    dynamic_imgs = []
    for uid in uids:
        uid_list.append(uid)
        dynamic_imgs.append(np.load(f"{dynamic_img_dir}/{data_type}_{uid}.npy"))
    # substract the mean image from the dyanmic images
    demeaned_imgs = [dynamic_img - mean_img for dynamic_img in dynamic_imgs]

    # save the demeaned images
    for i, uid in enumerate(uids):
        # save it to the delta dir
        np.save(f"{delta_ms_img_dir}/{data_type}_{uid}.npy", demeaned_imgs[i])

    # substract the dynamic images from each other to create deltas
    delta_imgs = []
    for i in range(len(dynamic_imgs) - 1):
        year_1 = uids[i][-4:]
        for j in range(i + 1, len(dynamic_imgs)):
            year_2 = uids[j][-4:]
            #print(f"{year_1}_{year_2}")
            # substract the older image from the newer image
            delta_img = dynamic_imgs[j] - dynamic_imgs[i]
            delta_imgs.append(delta_img)
            delta_id_list.append(f"{cid}_{year_1}_{year_2}")
            np.save(f"{delta_ms_img_dir}/{data_type}_{cid}_{year_1}_{year_2}.npy", delta_img)

    # extract image statistics for both images
    for img in demeaned_imgs:
        for i in range(img.shape[2]):
            if i in demeaned_img_stats.keys():
                demeaned_img_stats[i].append(get_basic_band_stats(img[:, :, i]))
            else:
                demeaned_img_stats[i] = [get_basic_band_stats(img[:, :, i])]

    for img in delta_imgs:
        for i in range(img.shape[2]):
            if i in delta_img_stats.keys():
                delta_img_stats[i].append(get_basic_band_stats(img[:, :, i]))
            else:
                delta_img_stats[i] = [get_basic_band_stats(img[:, :, i])]

#****************************************************************
#save image statistics as pickle
#****************************************************************


# convert the image statistics dictionary values to pandas dataframes
for band_key in demeaned_img_stats.keys():
    demeaned_img_stats[band_key] = pd.json_normalize(demeaned_img_stats[band_key])
    demeaned_img_stats[band_key]['unique_id'] = uid_list

for band_key in delta_img_stats.keys():
    delta_img_stats[band_key] = pd.json_normalize(delta_img_stats[band_key])
    delta_img_stats[band_key]['delta_id'] = delta_id_list

# append the demeaned images stats to the delta img stats
for k in delta_img_stats.keys():
    # rename the id variable in the demeaned_img_stats
    aux = demeaned_img_stats[k].copy().rename(columns={'unique_id': 'delta_id'})

    # append both dfs
    delta_img_stats[k] = pd.concat([delta_img_stats[k], aux]).reset_index(drop=True)

# save the image statistics as pickle

pth = f"{sat_img_dir}/LS/LS_ms_delta_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(delta_img_stats, f)




