# import packages
import sys
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# load the functions to do spatial CV
from analysis_utils.spatial_CV import split_lsms_spatial

# import the torch_framework package
from analysis_utils.torch_framework.CrossValidator import CrossValidator
from analysis_utils.torch_framework.ResNet18 import ResNet18
from analysis_utils.torch_framework.SatDataset import SatDataset
from analysis_utils.torch_framework.torch_helpers import get_agg_img_stats, get_feat_stats, get_target_stats
from analysis_utils.torch_framework.torch_helpers import standardise

print("Hello!")
print("Initialising Training for the Between Model using Remote Sensing images")
print(f"Target variable: {sys.argv[3]}")
print("\n")

####################################################################################################
# Accept the command line arguments
####################################################################################################

# check if the number of command line arguments is correct
if len(sys.argv) != 3:
    print("Please provide the values for the following variables:")
    print("model_name, cv_object_name, between_target_var")
    print("Usage: python my_script.py model_name cv_object_name between_target_var")
    ValueError("Incorrect number of command line arguments")

# get the command line arguments
model_name = sys.argv[1]
cv_object_name = sys.argv[2]
between_target_var = sys.argv[3]

####################################################################################################
# Set the hyper-parameters
####################################################################################################

# These parameters are set in the command line
# model name etc to save the model
# model_name = 'between_cons_LS'
# cv_object_name = 'between_cons_LS_cv'
# define the target variable
# between_target_var = 'avg_log_mean_pc_cons_usd_2017'

data_type = 'RS_v2'
id_var = 'cluster_id'

# set the random seed
spatial_cv_random_seed = 348 # this ensures that the validation sets are constant across models
random_seed = 49832

# set the number of folds for k-fold CV
n_folds = 5

# share of data used
max_obs = 1000000

# set hyper-parameters
hyper_params = {
    'lr': [1e-1, 1e-2, 1e-3],
    'batch_size': [128],
    'alpha': [1e-1, 1e-2, 1e-3],
    'step_size': [1],
    'gamma': [0.96],
    'n_epochs': [200],
    'patience': [50]
}

# training device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device} \n")

####################################################################################################
# Set the data paths
####################################################################################################

# set the global file paths
root_data_dir = "/scratch/users/nschmid5/data_analysis/Data"

# the lsms data
lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

# set the image directories
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# the RS v2 images at the cluster level
RS_v2_between_img_dir = f"{sat_img_dir}/RS_v2/RS_v2_between"
RS_v2_between_stats_pth = f"{sat_img_dir}/RS_v2/RS_v2_between_img_stats.pkl"

####################################################################################################
# Load the data
####################################################################################################

# load the LSMS data and the feature data (OSM and precipitation)
lsms_df = pd.read_csv(lsms_pth).iloc[:max_obs, :]

# add the mean variable at the cluster level
lsms_df['avg_log_mean_pc_cons_usd_2017'] = lsms_df.groupby('cluster_id')['log_mean_pc_cons_usd_2017'].transform('mean')
lsms_df['avg_mean_asset_index_yeh'] = lsms_df.groupby('cluster_id')['mean_asset_index_yeh'].transform('mean')

# define the mean cluster dataset
between_df = lsms_df[['cluster_id', 'lat', 'lon', 'country', between_target_var]].drop_duplicates().reset_index(drop=True)

# divide the data into k different folds
print("Dividing the data into k different folds using spatial CV")
fold_ids = split_lsms_spatial(lsms_df, n_folds=n_folds, random_seed=spatial_cv_random_seed)

# get the image statistics for the Landsat images for each band
RS_img_stats = get_agg_img_stats(RS_v2_between_stats_pth, between_df, id_var = 'cluster_id')

# extract the relevant statistics for each band (i.e. the mean, std, min, max) and get them as a list
RS_feat_stats = get_feat_stats(RS_img_stats)

# For the RS feat stats, alter the mean and std of the last two channels (WSF and ESA LC)
# For these two channels normalisation does not introduce any advantage or yields meaningless numbers
# Thus just set mean and std for both channels to 0 and 1 (which effectively avoids normalisation)
RS_feat_stats['mean'][-2:] = [0,0]
RS_feat_stats['std'][-2:] = [1,1]

# get the stats for the target variable
between_target_stats = get_target_stats(between_df, between_target_var)

# get the data transforms for the target --> is used in the DataLoader object
target_transform = transforms.Compose([
    torchvision.transforms.Lambda(lambda t: standardise(t, between_target_stats['mean'], between_target_stats['std'])),
])

# get the data transform for the Landsat image (normalisation and random horizontal + vertical flips)
RS_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.RandomVerticalFlip(.5),
     torchvision.transforms.RandomHorizontalFlip(.5),
     transforms.Normalize(RS_feat_stats['mean'], RS_feat_stats['std'])]
)

####################################################################################################
# Train the model
####################################################################################################
print("\n\n")
print("=====================================================================")
print("Initialise CROSS VALIDATION")
print("=====================================================================")

# load the data into RAM first
# this increases training times
_dat = SatDataset(between_df,
                  RS_v2_between_img_dir,
                  data_type,
                  between_target_var,
                  id_var,
                  RS_transforms,
                  target_transform,
                  random_seed)
_loader = DataLoader(_dat, batch_size=hyper_params['batch_size'][0], shuffle=False)
_, _ = next(iter(_loader))

# run model training
# initialise the model and the CrossValidator object
resnet18 = ResNet18(
    input_channels=6,
    pretrained_weights=False,
    scaled_weight_init=False,
    random_seed=random_seed
)

cv = CrossValidator(
    model_class=resnet18,
    lsms_df=between_df,
    fold_ids=fold_ids,
    img_dir=RS_v2_between_img_dir,
    data_type=data_type,
    target_var=between_target_var,
    id_var=id_var,
    feat_transform=RS_transforms,
    target_transform=target_transform,
    device=device,
    model_name=model_name,
    random_seed=random_seed
)

# run k-fold-cv
cv.run_cv(hyper_params=hyper_params, tune_hyper_params=True)

# save the cv object
cv.save_object(name=cv_object_name)

# output the overall performance of the model
print("\n")
print('=' * 100)
print('Cross-Validated performance:')
print('=' * 100)
print(cv.compute_overall_performance(use_fold_weights=True))
print('\n\n')
