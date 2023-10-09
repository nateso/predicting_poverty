# import packages
import sys
import json
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
from analysis_utils.torch_framework.torch_helpers import standardise as torch_standardise

# import the flagged uids
from analysis_utils.flagged_uids import flagged_uids

# check if the number of command line arguments is correct
if len(sys.argv) != 9:
    print("Please provide the values for the following variables:")
    print("model_name, cv_object_name, target_var")
    print("Usage: python my_script.py model_name cv_object_name target_var")
    raise(ValueError("Incorrect number of command line arguments"))

print("Hello!")
print("Initialising Training for the Within model using Remote sensing images")
print(f"Model name: {sys.argv[1]}")
print(f"CV object name: {sys.argv[2]}")
print(f"Target variable: {sys.argv[3]}")
print(f"Data type: {sys.argv[4]}")
print(f"ID variable: {sys.argv[5]}")
print(f"Image folder: {sys.argv[6]}")
print(f"Stats file: {sys.argv[7]}")
print(f"ResNet parameters: {sys.argv[8]}")
print("\n")

####################################################################################################
# Accept the command line arguments
####################################################################################################

# get the command line arguments
model_name = sys.argv[1]
cv_object_name = sys.argv[2]
target_var = sys.argv[3]

# add more command line arguments
data_type = sys.argv[4]
id_var = sys.argv[5]
img_folder = sys.argv[6]
stats_file = sys.argv[7]

# pass the resnet parameters
resnet_params = json.loads(sys.argv[8])

####################################################################################################
# Set the hyper-parameters
####################################################################################################
# set the random seed
spatial_cv_random_seed = 348
random_seed = 4309

# set the number of folds for k-fold CV
n_folds = 5

# share of data used
max_obs = 1000000

# set hyper-parameters
hyper_params = {
    'lr': [1e-2, 1e-3],
    'batch_size': [64],
    'alpha': [1e-2, 1e-3],
    'step_size': [1],
    'gamma': [0.98],
    'n_epochs': [200],
    'patience': [40]
}

# training device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device} \n")

####################################################################################################
# Set the data paths
####################################################################################################

# set the global file paths
root_data_dir = "/scratch/users/nschmid5/data_analysis/Data"
#root_data_dir = "../../Data"

# the lsms data
lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

# set the image directories
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# median LS images at the cluster level
img_dir = f"{sat_img_dir}/{data_type}/{img_folder}"
stats_pth = f"{sat_img_dir}/{data_type}/{stats_file}"

####################################################################################################
# Load the data
####################################################################################################
# load the LSMS data and the feature data (OSM and precipitation)
lsms_df = pd.read_csv(lsms_pth).iloc[:max_obs, :]

# exclude all ids that are flagged
lsms_df = lsms_df[~lsms_df['unique_id'].isin(flagged_uids)].reset_index(drop=True)

# add the mean variable at the cluster level
lsms_df['avg_log_mean_pc_cons_usd_2017'] = lsms_df.groupby('cluster_id')['log_mean_pc_cons_usd_2017'].transform('mean')
lsms_df['avg_mean_asset_index_yeh'] = lsms_df.groupby('cluster_id')['mean_asset_index_yeh'].transform('mean')

# define the mean cluster dataset
between_df = lsms_df[['cluster_id', 'lat', 'lon', 'country', target_var]].copy().drop_duplicates().reset_index(drop=True)

# divide the data into k different folds
print("Dividing the data into k different folds using spatial CV")
fold_ids = split_lsms_spatial(lsms_df, n_folds=n_folds, random_seed=spatial_cv_random_seed)

# extract the image statistics for the demeaned images
img_stats = get_agg_img_stats(stats_pth, between_df, id_var='cluster_id')
feat_stats = get_feat_stats(img_stats)

# get the target stats
target_stats = get_target_stats(between_df, target_var)

# define the transforms
# get the data transforms for the target --> is used in the DataLoader object
target_transform = transforms.Compose([
    torchvision.transforms.Lambda(
        lambda t: torch_standardise(t, target_stats['mean'], target_stats['std'])),
])

# get the data transform for the Landsat image (normalisation and random horizontal + vertical flips)
feat_transform = torchvision.transforms.Compose(
    [torchvision.transforms.RandomVerticalFlip(.5),
     torchvision.transforms.RandomHorizontalFlip(.5),
     transforms.Normalize(feat_stats['mean'], feat_stats['std'])]
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
                  img_dir,
                  data_type,
                  target_var,
                  id_var,
                  feat_transform,
                  target_transform,
                  random_seed)
_loader = DataLoader(_dat, batch_size=hyper_params['batch_size'][0], shuffle=False)
_, _ = next(iter(_loader))

# run model training
# initialise the model and the CrossValidator object
resnet18 = ResNet18(
    input_channels=resnet_params['input_channels'],
    use_pretrained_weights=resnet_params['use_pretrained_weights'],
    scaled_weight_init=resnet_params['scaled_weight_init'],
    random_seed=random_seed
)

cv = CrossValidator(
    model_class=resnet18,
    lsms_df=between_df,
    fold_ids=fold_ids,
    img_dir=img_dir,
    data_type=data_type,
    target_var=target_var,
    id_var=id_var,
    feat_transform=feat_transform,
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

print('BYE BYE')










