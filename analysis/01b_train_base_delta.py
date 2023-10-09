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
print("Initialising the training of the base delta")
print(f"Between target variable: {sys.argv[1]}")
print(f"Within target variable: {sys.argv[2]}")
print(f"Use LS variables: {sys.argv[3]}")
print(f"file output path: {sys.argv[4]}")


between_target_var = sys.argv[1]
within_target_var = sys.argv[2]
use_ls_vars = sys.argv[3] == 'True'
file_pth = sys.argv[4]

# set the id variables
between_id_var = 'cluster_id'
within_id_var = 'delta_id'

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

lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

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
lsms_df['delta_id'] = lsms_df.unique_id

# exclude all ids that are flagged
lsms_df = lsms_df[~lsms_df['unique_id'].isin(flagged_uids)].reset_index(drop=True)

# normalise the asset index
lsms_df['mean_asset_index_yeh'] = (lsms_df['mean_asset_index_yeh'] - lsms_df['mean_asset_index_yeh'].mean()) / lsms_df['mean_asset_index_yeh'].std()

# add between variables to the lsms data
lsms_df['avg_log_mean_pc_cons_usd_2017'] = lsms_df.groupby('cluster_id')['log_mean_pc_cons_usd_2017'].transform('mean')
lsms_df['avg_mean_asset_index_yeh'] = lsms_df.groupby('cluster_id')['mean_asset_index_yeh'].transform('mean')

# load the feature data
feat_df = pd.read_csv(feat_data_pth)

# merge the label and the feature data to one dataset
lsms_vars = ['unique_id', 'n_households',
             'log_mean_pc_cons_usd_2017', 'avg_log_mean_pc_cons_usd_2017',
             'mean_asset_index_yeh', 'avg_mean_asset_index_yeh']
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

# create a delta df
demeaned_df = demean_df(within_df)
print("Creating Delta df")
delta_df = make_delta_df(within_df)

# combine the delta df, with the demeaned df
demeaned_df = demeaned_df.rename(columns = {'unique_id': 'delta_id'})
delta_dmn_df = pd.concat([delta_df, demeaned_df]).reset_index(drop = True)
delta_dmn_df_norm = standardise_df(delta_dmn_df, exclude_cols = [within_target_var])

# subset the normalised dataframe into the demeaned data (used in validation) and the delta data (used in training)
demeaned_df_norm = delta_dmn_df_norm[delta_dmn_df_norm.delta_id.isin(demeaned_df.delta_id)].copy().reset_index(drop = True)
delta_df_norm = delta_dmn_df_norm[delta_dmn_df_norm.delta_id.isin(delta_df.delta_id)].copy().reset_index(drop = True)

print(f"Number of observations in delta df: {len(delta_df_norm)}")


# run repeated Cross-validation
rep_cv_res = {
    'between_r2': [],
    'within_r2': [],
    'delta_r2': [],
    'overall_r2': []
}

for j in range(10):
    print("=" * 100)
    print(f"Iteration {j}")
    print("=" * 100)
    rep_seed = random_seed + j

    # divide the data into k different folds
    fold_ids = split_lsms_spatial(lsms_df, n_folds=n_folds, random_seed=spatial_cv_random_seed + j)

    # run the bewtween training
    print('Between training')
    between_cv_trainer = rf.CrossValidator(between_df_norm,
                                           fold_ids,
                                           between_target_var,
                                           between_x_vars,
                                           id_var='cluster_id',
                                           random_seed=rep_seed)
    between_cv_trainer.run_cv_training(min_samples_leaf=1)

    # run the within training
    print("\nWithin training")
    delta_trainer = rf.CrossValidator(delta_df_norm,
                                      fold_ids,
                                      within_target_var,
                                      within_x_vars,
                                      id_var='delta_id',
                                      random_seed=rep_seed)
    delta_trainer.run_cv_training(min_samples_leaf=15)

    # get results of the delta predictions
    delta_res = delta_trainer.compute_overall_performance(use_fold_weights=True)

    # evaluate the delta model on the demeaned variables
    delta_evaluator = rf.CV_Evaluator(demeaned_df_norm, fold_ids, delta_trainer, id_var='delta_id')
    delta_evaluator.evaluate()
    delta_evaluator.compute_overall_performance()

    # add the predictions to the delta_trainer
    delta_trainer.predictions = delta_evaluator.predictions

    # combine both models
    combined_model = CombinedModel(lsms_df, between_cv_trainer, delta_trainer)
    combined_model.evaluate()
    combined_results = combined_model.compute_overall_performance(use_fold_weights=True)

    # store the results
    rep_cv_res['between_r2'].append(combined_results['r2']['between'])
    rep_cv_res['within_r2'].append(combined_results['r2']['within'])
    rep_cv_res['delta_r2'].append(delta_res['val_r2'])
    rep_cv_res['overall_r2'].append(combined_results['r2']['overall'])

    # print the results
    print("." * 100)
    print(combined_results)
    print(delta_res['val_r2'])
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


