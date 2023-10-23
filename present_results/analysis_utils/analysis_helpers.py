import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# define functions to standardise the data

def standardise(var):
    return (var - np.mean(var)) / np.std(var)


def standardise_df(df, exclude_cols=[]):
    """
    standardises all numeric columns in the data, except for columns in exclude_cols
    """
    cp_df = df.copy()
    numeric_cols = cp_df.select_dtypes(include=[float, int]).columns
    numeric_cols = [i for i in numeric_cols if i not in exclude_cols]

    for col in numeric_cols:
        cp_df[col] = standardise(cp_df[col])

    return cp_df


# define a function to make the delta df
def make_delta_df(df):
    id_columns = ['cluster_id', 'unique_id']
    df = df.sort_values(by=id_columns, ascending=[True, True])

    # create dictionary of cluster_id to unique_ids
    cid_uid_dict = {}
    for idx, row in df.iterrows():
        cid = row['cluster_id']
        uid = row['unique_id']

        if cid in cid_uid_dict:
            cid_uid_dict[cid].append(uid)
        else:
            cid_uid_dict[cid] = [uid]

    # iterate over all clusters creating deltas
    delta_list = []
    for cid, uids in tqdm(cid_uid_dict.items()):
        # subset the data to the cluster
        cid_df = df[df['cluster_id'] == cid].reset_index(drop=True)
        # iterate over all years in the cluster
        for i in range(len(cid_df) - 1):
            year_1 = cid_df['unique_id'][i][-4:]
            for j in range(i + 1, len(cid_df)):
                year_2 = cid_df['unique_id'][j][-4:]
                x_0 = cid_df.iloc[i, :].drop(id_columns)
                x_1 = cid_df.iloc[j, :].drop(id_columns)
                # substract the two years (the most recent year minus the older year)
                delta = x_1 - x_0
                delta['cluster_id'] = cid
                delta['delta_id'] = cid + "_" + year_1 + "_" + year_2
                delta_list.append(delta)

    delta_df = pd.DataFrame(delta_list)
    return delta_df


def demean_df(df):
    id_columns = ['cluster_id', 'unique_id']
    # Group the DataFrame by the cluster ID variable and calculate the mean of each group
    cluster_means = df.groupby('cluster_id').mean(numeric_only=True)

    # merge it back to a data frame with a unique_id
    mean_df = pd.merge(df[id_columns], cluster_means, on='cluster_id', how='left')

    # substract the mean from the original data
    demeaned_df = df.drop(columns=id_columns) - mean_df.drop(columns=id_columns)
    demeaned_df[id_columns] = df[id_columns]

    return demeaned_df
