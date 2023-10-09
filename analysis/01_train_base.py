import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# load the variable names, this allows to access the variables in the feature data in a compact way
from analysis_utils.variable_names import *

# load flagged ids
from analysis_utils.flagged_uids import *

# load the functions to do spatial k-fold CV
from analysis_utils.spatial_CV import *

# load the helper functions
from analysis_utils.analysis_helpers import *

# load the random forest trainer and cross_validator
import analysis_utils.RandomForest as rf

# load the combined model
from analysis_utils.CombinedModel import CombinedModel


####################################################################################################
# accept command line arguments
####################################################################################################

print("Hello!")
print("Initialising the training of the base model")
print(f"Between target variable: {sys.argv[1]}")
print(f"Within target variable: {sys.argv[2]}")
print(f"Use LS variables: {sys.argv[3]}")
print(f"Remove Ethiopia: {sys.argv[4]}")
print(f"file output path: {sys.argv[5]}")

print("\n")

between_target_var = sys.argv[1]
within_target_var = sys.argv[2]
use_ls_vars = sys.argv[3] == 'True'
remove_eth = sys.argv[4] == 'True'
file_pth = sys.argv[5]

# set the id variables
between_id_var = 'cluster_id'
within_id_var = 'unique_id'

# set the x variables
avg_rs_vars = avg_ndvi_vars + avg_ndwi_gao_vars + avg_nl_vars
osm_vars = osm_dist_vars + osm_count_vars + osm_road_vars
dyn_rs_vars = dyn_ndvi_vars + dyn_ndwi_gao_vars + dyn_nl_vars

between_x_vars = osm_vars + esa_lc_vars + wsf_vars + avg_rs_vars + avg_preciptiation
within_x_vars = dyn_rs_vars + precipitation

if use_ls_vars:
    print("Using LS variables")
    between_x_vars += median_rgb_vars
    within_x_vars += dyn_rgb_vars

####################################################################################################
# set the global file paths
####################################################################################################

root_data_dir = "/scratch/users/nschmid5/data_analysis/Data"
#root_data_dir = "../../Data"

# the lsms data
lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

# asset index excluding ethiopia
eth_idx_pth = f"{root_data_dir}/lsms/processed/asset_index_no_eth.csv"

# the feature data
feat_data_pth = f"{root_data_dir}/feature_data/tabular_data.csv"

####################################################################################################
# Global Hyper-parameters
####################################################################################################
# set the random seed
random_seed = 534
spatial_cv_random_seed = 348

# set the number of folds for k-fold CV
n_folds = 5

####################################################################################################
# Load the datasets
####################################################################################################

lsms_df = pd.read_csv(lsms_pth)

# exclude all ids that are flagged
lsms_df = lsms_df[~lsms_df['unique_id'].isin(flagged_uids)].reset_index(drop=True)

# load the asset index excluding ethiopia
eth_idx = pd.read_csv(eth_idx_pth)

# merge the no eth asset index with the lsms data
lsms_df = pd.merge(lsms_df, eth_idx, on='unique_id', how='left')

# normalise the asset indices
lsms_df['mean_asset_index_yeh'] = (lsms_df['mean_asset_index_yeh'] - lsms_df['mean_asset_index_yeh'].mean()) / lsms_df['mean_asset_index_yeh'].std()
lsms_df['mean_asset_index_yeh_no_eth'] = (lsms_df['mean_asset_index_yeh_no_eth'] - lsms_df['mean_asset_index_yeh_no_eth'].mean()) / lsms_df['mean_asset_index_yeh_no_eth'].std()

# add between variables to the lsms data
lsms_df['avg_log_mean_pc_cons_usd_2017'] = lsms_df.groupby('cluster_id')['log_mean_pc_cons_usd_2017'].transform('mean')
lsms_df['avg_mean_asset_index_yeh'] = lsms_df.groupby('cluster_id')['mean_asset_index_yeh'].transform('mean')
lsms_df['avg_mean_asset_index_yeh_no_eth'] = lsms_df.groupby('cluster_id')['mean_asset_index_yeh_no_eth'].transform('mean')

if remove_eth:
    lsms_df = lsms_df[lsms_df.country != 'eth'].reset_index(drop=True)

# load the feature data
feat_df = pd.read_csv(feat_data_pth)

# merge the label and the feature data to one dataset
lsms_vars = ['unique_id', 'n_households',
             'log_mean_pc_cons_usd_2017', 'avg_log_mean_pc_cons_usd_2017',
             'mean_asset_index_yeh', 'avg_mean_asset_index_yeh',
             'mean_asset_index_yeh_no_eth', 'avg_mean_asset_index_yeh_no_eth']

df = pd.merge(lsms_df[lsms_vars], feat_df, on = 'unique_id', how = 'left')

# describe the training data broadly
print(f"Number of observations {len(lsms_df)}")
print(f"Number of clusters {len(np.unique(lsms_df.cluster_id))}")
print(f"Number of x vars {len(feat_df.columns)-2}")


####################################################################################################
# Train the model
####################################################################################################
between_df = df[[between_id_var, between_target_var] + between_x_vars].drop_duplicates().reset_index(drop = True)
between_df_norm = standardise_df(between_df, exclude_cols = [between_target_var])

within_df = df[['cluster_id','unique_id', within_target_var] + within_x_vars]
demeaned_df = demean_df(within_df)
demeaned_df_norm = standardise_df(demeaned_df, exclude_cols = [within_target_var])

rep_cv_res = {
    'between_r2': [],
    'within_r2': [],
    'overall_r2': []
}

for j in range(10):
    print("=" * 100)
    print(f"Iteration {j}")
    print("=" * 100)
    rep_seed = random_seed + j

    # divide the data into k different folds
    fold_ids = split_lsms_spatial(lsms_df, n_folds=n_folds, random_seed=spatial_cv_random_seed)

    # run the bewtween training
    print('Between training')
    between_cv_trainer = rf.CrossValidator(between_df_norm,
                                           fold_ids,
                                           between_target_var,
                                           between_x_vars,
                                           id_var=between_id_var,
                                           random_seed=rep_seed)
    between_cv_trainer.run_cv_training(min_samples_leaf=1)

    # run the within training
    print("\nWithin training")
    within_cv_trainer = rf.CrossValidator(demeaned_df_norm,
                                          fold_ids,
                                          within_target_var,
                                          within_x_vars,
                                          id_var=within_id_var,
                                          random_seed=rep_seed)
    within_cv_trainer.run_cv_training(min_samples_leaf=15)

    # combine both models
    combined_model = CombinedModel(lsms_df, between_cv_trainer, within_cv_trainer)
    combined_model.evaluate()
    combined_results = combined_model.compute_overall_performance(use_fold_weights=True)

    # store the results
    rep_cv_res['between_r2'].append(combined_results['r2']['between'])
    rep_cv_res['within_r2'].append(combined_results['r2']['within'])
    rep_cv_res['overall_r2'].append(combined_results['r2']['overall'])

    # print the results
    print("." * 100)
    print(combined_results)
    print("." * 100)

####################################################################################################
# Save the model results
####################################################################################################

# ensure that there is a folder to save the results
if not os.path.isdir("results/baseline/"):
    os.mkdir("results/baseline/")

if not os.path.isdir("results/baseline_ls/"):
    os.mkdir("results/baseline_ls/")

with open(file_pth, 'wb') as f:
    pickle.dump(rep_cv_res, f)


