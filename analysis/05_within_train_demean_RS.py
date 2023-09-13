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
from analysis_utils.torch_framework.torch_helpers import standardise as torch_standardise

# import the helpers to demean the data and get the deltas etc.
from analysis_utils.analysis_helpers import demean_df

# check if the number of command line arguments is correct
if len(sys.argv) != 4:
    print("Please provide the values for the following variables:")
    print("model_name, cv_object_name, target_var")
    print("Usage: python my_script.py model_name cv_object_name target_var")
    raise(ValueError("Incorrect number of command line arguments"))

print("Hello!")
print("Initialising Training for the Within model using Remote sensing images")
print(f"Target variable: {sys.argv[3]}")
print("This is the delta version of the target variable. Just keeps the same name.")
print("\n")

####################################################################################################
# Accept the command line arguments
####################################################################################################

# get the command line arguments
model_name = sys.argv[1]
cv_object_name = sys.argv[2]
target_var = sys.argv[3]

####################################################################################################
# Set the hyper-parameters
####################################################################################################
data_type = 'RS_v2'
id_var = 'delta_id'

# set the random seed
spatial_cv_random_seed = 348
random_seed = 230897

# set the number of folds for k-fold CV
n_folds = 5

# share of data used
max_obs = 1000000

# set hyper-parameters
hyper_params = {
    'lr': [1e-2, 1e-3],
    'batch_size': [256],
    'alpha': [1e-1, 1e-2, 1e-3],
    'step_size': [1],
    'gamma': [0.96],
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
#root_data_dir = "/scratch/users/nschmid5/data_analysis/Data"
root_data_dir = "../../Data"

# the lsms data
lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

# set the image directories
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# median LS images at the cluster level
img_dir = f"{sat_img_dir}/{data_type}/RS_v2_delta"
stats_pth = f"{sat_img_dir}/{data_type}/RS_v2_delta_img_stats.pkl"

####################################################################################################
# Load the data
####################################################################################################
# load the LSMS data
lsms_df = pd.read_csv(lsms_pth).iloc[:max_obs, :]

cl_df = lsms_df[['cluster_id', 'country', 'lat', 'lon']].copy().drop_duplicates().reset_index(drop=True)
within_df = lsms_df[['cluster_id', 'unique_id', target_var]]

# demean the df
demeaned_df = demean_df(within_df)
demeaned_df = demeaned_df.rename(columns={'unique_id': 'delta_id'})
demeaned_df = pd.merge(demeaned_df, cl_df, on='cluster_id', how='left')

# get the fold ids from spatial CV
print("\nDividing the data into k different folds using spatial CV")
fold_ids = split_lsms_spatial(lsms_df, n_folds=n_folds, random_seed=spatial_cv_random_seed)

# extract the image statistics for the demeaned images
img_stats = get_agg_img_stats(stats_pth, demeaned_df, id_var='delta_id')
feat_stats = get_feat_stats(img_stats)

# get the target stats
target_stats = get_target_stats(demeaned_df, target_var)

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
_dat = SatDataset(demeaned_df,
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
    input_channels=4,
    use_pretrained_weights=False,
    scaled_weight_init=False,
    random_seed=random_seed
)

cv = CrossValidator(
    model_class=resnet18,
    lsms_df=demeaned_df,
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










