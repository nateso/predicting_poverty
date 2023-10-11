
import pickle

# only Relevant for LS images
# extract statistics for subimages that have already been computed for different types of images
sat_img_dir = "../../Data/satellite_imgs"
data_type = "LS"

#######################################################################################################################
# extract the image statistics for the median RGB landsat images
#######################################################################################################################

median_img_stats_pth = f"{sat_img_dir}/{data_type}/{data_type}_median_img_stats.pkl"

with open(median_img_stats_pth, 'rb') as f:
    median_img_stats = pickle.load(f)

# get image statistics for RGB bands
median_rgb_img_stats = {}
for i in range(3):
    median_rgb_img_stats[i] = median_img_stats[i]

# save the image statistics
pth = f"{sat_img_dir}/{data_type}/{data_type}_rgb_median_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(median_rgb_img_stats, f)

#######################################################################################################################
# extract the image statistics for the MS landsat images
#######################################################################################################################

# extract the image statistics for the ms image from the raw statistics (which include band 5 which we don't want)
raw_img_stats_pth = f"{sat_img_dir}/{data_type}/{data_type}_raw_img_stats.pkl"

# load the raw image statistics
with open(raw_img_stats_pth, 'rb') as f:
    raw_img_stats = pickle.load(f)

ms_img_stats = {}
for i in [0,1,2,3,4,6]:
    ms_img_stats[i] = raw_img_stats[i]

# save the image statistics
pth = f"{sat_img_dir}/{data_type}/{data_type}_ms_img_stats.pkl"
with open(pth, 'wb') as f:
    pickle.dump(ms_img_stats, f)



