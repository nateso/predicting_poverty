import pandas as pd
import sys
import torch
from analysis_utils.torch_framework.LsRsCombinedModel import LsRsCombinedModel

# load the variable names of the tabular feature data
from analysis_utils.variable_names import *

# load the flagged ids
from analysis_utils.flagged_uids import flagged_uids

# import the spatial CV function
from analysis_utils.spatial_CV import split_lsms_spatial

# check if the number of command line arguments is correct
from analysis_utils.analysis_helpers import standardise_df

from analysis_utils.analysis_helpers import demean_df


if len(sys.argv) != 5:
    print(len(sys.argv))
    raise(ValueError("Incorrect number of command line arguments"))

print("Hello!")
print("Initialising Training for the Between Model - combining LS and RS")
print(f"Object name: {sys.argv[1]}")
print(f"Target variable: {sys.argv[2]}")
print(f"LS CV path: {sys.argv[3]}")
print(f"RS CV path: {sys.argv[4]}")
print("\n")

####################################################################################################
# Accept the command line arguments
####################################################################################################

# get the command line arguments
object_name = sys.argv[1]
target_var = sys.argv[2]
ls_cv_pth = sys.argv[3]
rs_cv_pth = sys.argv[4]

####################################################################################################
# Set the data paths
####################################################################################################

# set id var and x_vars
id_var = 'delta_id'
x_vars = precipitation

# set the random seed
spatial_cv_random_seed = 348 # this ensures that the validation sets are constant across models
random_seed = 534

# set the number of folds for k-fold CV
n_folds = 5

# share of data used
max_obs = 1000000

# set hyper-parameters
hyper_params = {
    'min_samples_leaf':15,
    'n_components':25
}

# training device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
print(f"Training device: {device} \n")

####################################################################################################
# Set the data paths
####################################################################################################

# set the global file paths
root_data_dir = "/scratch/users/nschmid5/data_analysis/Data"
#root_data_dir = "../../Data"

# the lsms data
lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

# the feature data (OSM + precipitation)
feat_data_pth = f"{root_data_dir}/feature_data/tabular_data.csv"

####################################################################################################
# Load the data
####################################################################################################

# load the LSMS data and the feature data (OSM and precipitation)
lsms_df = pd.read_csv(lsms_pth).iloc[:max_obs, :]

# drop the flagged ids
lsms_df = lsms_df[~lsms_df['unique_id'].isin(flagged_uids)].reset_index(drop=True)

# load the feature data
feat_df = pd.read_csv(feat_data_pth)

# merge the lsms_df and the feat_df
df = pd.merge(lsms_df, feat_df, on = ('unique_id','cluster_id'), how = 'left')

cl_df = lsms_df[['cluster_id', 'country', 'lat', 'lon']].copy().drop_duplicates().reset_index(drop=True)
within_df = df[['cluster_id','unique_id', target_var] + x_vars]

# demean the data and standardise the variables
demeaned_df = demean_df(within_df)
demeaned_df = demeaned_df.rename(columns={'unique_id': 'delta_id'})
demeaned_df_norm = standardise_df(demeaned_df, exclude_cols = [target_var])

# divide the data into k different folds
print("Dividing the data into k different folds using spatial CV")
fold_ids = split_lsms_spatial(lsms_df, n_folds=n_folds, random_seed=spatial_cv_random_seed)

####################################################################################################
# Train the model
####################################################################################################
combined_model = LsRsCombinedModel(
    LS_cv_pth=ls_cv_pth,
    RS_cv_pth=rs_cv_pth,
    lsms_df=demeaned_df_norm,
    target_var=target_var,
    x_vars=x_vars,
    fold_ids=fold_ids,
    device=device,
    random_seed=random_seed
)

# run the combined model
combined_model.train(
    min_samples_leaf=hyper_params['min_samples_leaf'],
    n_components=hyper_params['n_components']
)

combined_model.save_object(object_name)


# output the overall performance of the model
print("\n")
print('=' * 100)
print('Cross-Validated performance:')
print('=' * 100)
print(combined_model.compute_overall_performance(use_fold_weights=True))
print('-'*50)
print('Training Performance per fold')
for i in range(n_folds):
    print(f"fold{i}")
    print(f"\tTrain MSE: {combined_model.res_mse['train'][i]} - Train R2: {combined_model.res_r2['train'][i]}")
    print(f"\tVal MSE: {combined_model.res_mse['val'][i]} - Val R2: {combined_model.res_r2['val'][i]}")
print('\n\n')
print("BYE BYE")