import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import haversine_distances
import copy


# function that splits the lsms data according to a list of validation ids:
def sample_cross_section(lsms_df, random_seed=None):
    cross_section_df = lsms_df.groupby('cluster_id').sample(n=1, random_state=random_seed).copy().reset_index(drop=True)
    return cross_section_df


def split_lsms_ids(lsms_df, val_ids):
    val_df = lsms_df[lsms_df['cluster_id'].isin(val_ids)].copy().reset_index(drop=True)
    train_df = lsms_df[~lsms_df['cluster_id'].isin(val_ids)].copy().reset_index(drop=True)
    return train_df, val_df


def split_lsms_basic(lsms_df, n_folds=5, random_seed=None):
    # use only cluster ids to split the data to ensure that the same cluster cannot
    # belong to the training and test set for different years.
    dat = lsms_df[['country', 'cluster_id', 'lon', 'lat']].copy().drop_duplicates().reset_index(drop=True)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # split the data and return the training and validation ids as dictionary
    id_dict = dict(zip(['val_ids', 'train_ids'], [[], []]))
    fold_ids = {fold: copy.deepcopy(id_dict) for fold in range(n_folds)}

    for fold, (train_ids, val_ids) in enumerate(k_fold.split(dat)):
        fold_ids[fold]['train_ids'] = list(dat.iloc[train_ids].cluster_id)
        fold_ids[fold]['val_ids'] = list(dat.iloc[val_ids].cluster_id)

    return fold_ids


def split_lsms_spatial(lsms_df, test_ratio=None, n_folds=None, cluster=False, random_seed=None, verbose=True):
    if all([n_folds is not None, test_ratio is not None]):
        raise ValueError("Either provide test ratio or n_folds, not both")
    elif n_folds is not None:
        test_ratio = 1 / n_folds
    elif test_ratio is not None:
        n_folds = 1

    if random_seed is not None:
        np.random.seed(random_seed)

    dat = lsms_df[['country', 'cluster_id', 'lon', 'lat']].copy().drop_duplicates().reset_index(drop=True)

    # for each country get the number of observations for test sample
    country_val_obs = dat['country'].value_counts() * test_ratio
    country_val_obs = dict(country_val_obs.astype(int))
    country_counts = dict(dat['country'].value_counts())

    # Calculate pairwise haversine distances between coordinates
    coords = np.deg2rad(dat[['lat', 'lon']])
    dist_matrix = haversine_distances(coords, coords) * 6371000 / 1000  # Multiply by Earth's radius to get kilometers
    dist_df = pd.DataFrame(dist_matrix, index=dat['cluster_id'], columns=dat['cluster_id'])

    # initialise the fold_ids dictionary
    id_dict = dict(zip(['val_ids', 'train_ids'], [[], []]))
    fold_ids = {fold: copy.deepcopy(id_dict) for fold in range(n_folds)}

    # ids to sample
    ids_to_sample = dat[['country', 'cluster_id']]
    id_dist = dist_df.copy()

    # for each fold sample 10 ids that are not used in previous validation sample
    for fold in fold_ids.keys():
        # print(f"Fold: {fold}")

        for cntry, val_obs in country_val_obs.items():
            cntry_ids = [i for i in id_dist.columns.values if cntry in i]
            val_obs_to_go = val_obs
            cntry_val_ids = []
            # print(f"country: {cntry}")
            # print(f"cntry observations to sample: {val_obs_to_go}")
            while ((val_obs_to_go > 0) and (len(cntry_ids) > 0)):
                # print(val_obs_to_go)
                if cluster == True:
                    k = val_obs_to_go + 1
                else:
                    k = 4

                # sample a random cluster from all ids that have not yet been sampled in that country
                # cid = random.sample(cntry_ids,1)[0]
                cid = np.random.choice(cntry_ids, size=1, replace=False)[0]

                # sample k nearest clusters to the randomly sampled cluster
                cntry_val_ids += get_k_nearest_clusters(id_dist, cid, k=k)
                cntry_val_ids = list(np.unique(cntry_val_ids))

                # define the temporary training ids
                tmp_train_ids = [i for i in cntry_ids if i not in cntry_val_ids]

                # ensure no overlap between the training ids and the validation ids
                _, cntry_val_ids = ensure_no_overlap(dist_df, tmp_train_ids, cntry_val_ids)

                # exclude the sampled ids from next sampling step
                cntry_ids = [i for i in cntry_ids if i not in cntry_val_ids]

                # get the number of observations that still need to be added to the validation set
                if cluster:
                    val_obs_to_go = 0
                else:
                    val_obs_to_go = max(0, val_obs - len(cntry_val_ids))

                # print(f"val_obs_to_go: {val_obs_to_go}")
                # print(f"len cntry ids: {len(cntry_ids)}")

            # add the sampled ids for the country to the fold_ids
            fold_ids[fold]['val_ids'] += cntry_val_ids

        # get the training ids
        fold_ids[fold]['train_ids'] += [j for j in dat['cluster_id'] if j not in fold_ids[fold]['val_ids']]

        # remove the selected validation ids from id_dist to ensure that they cannot be slected several times
        id_dist = id_dist.drop(index=fold_ids[fold]['val_ids'], columns=fold_ids[fold]['val_ids'])

    if verbose:
        for i in range(n_folds):
            print(
                f"Fold {i}, specified test ratio: {test_ratio} - Actual test ratio {len(fold_ids[i]['val_ids']) / len(dat):.2f}")

    return fold_ids


def get_k_nearest_clusters(dist_df, cid, k):
    # nearest_ids = []
    cntry = cid[:3]

    # ensure that I use only clusters within the same country
    cntry_mask = [cntry in i for i in dist_df.index.values]

    # distance of the selected cluster to all other clusters of the same country
    id_dist = dist_df[cid][cntry_mask].sort_values().reset_index()

    # get the k nearest clusters
    nearest_ids = list(id_dist.iloc[0:k].cluster_id)
    return nearest_ids


def calc_distance_mtrx(coord_df1, coord_df2, df1_names=None, df2_names=None):
    # Calculate pairwise haversine distances between coordinates
    dist_matrix = haversine_distances(coord_df1,
                                      coord_df2) * 6371000 / 1000  # Multiply by Earth's radius to get kilometers
    dist_df = pd.DataFrame(dist_matrix, index=df1_names, columns=df2_names)
    return dist_df


def ensure_no_overlap(dist_df, train_clusters, val_clusters):
    cond = True
    i = 0
    while cond:
        prob_train_ids = evaluate_overlap(dist_df, train_clusters, val_clusters)
        if len(prob_train_ids) == 0:
            cond = False
        else:
            # add the problematic train ids to the validation data
            val_clusters = val_clusters + [i for i in train_clusters if i in prob_train_ids]
            train_clusters = [i for i in train_clusters if (i in val_clusters) == False]
            i += 1
            # print(f"No overlap iteration {i}")

    return train_clusters, val_clusters


def evaluate_overlap(dist_df, train_clusters, val_clusters):
    dist_val_train = dist_df.loc[val_clusters, train_clusters]
    mask_2d = dist_val_train < 10
    mask_any_10 = (np.sum(mask_2d, axis=1) > 0)
    prob_train_ids = mask_2d.columns.values[np.sum(mask_2d[mask_any_10], axis=0) > 0]
    return prob_train_ids
