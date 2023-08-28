import torch
import torchvision
from torchvision import transforms
import numpy as np
import pickle


def min_max_norm(img, min, max):
    mins = torch.tensor(min, dtype=img.dtype)
    maxs = torch.tensor(max, dtype=img.dtype)
    return (img - mins.view(img.shape)) / (maxs.view(img.shape) - mins.view(img.shape))


def standardise(val, mean, std):
    mean = torch.tensor(mean, dtype=val.dtype)
    std = torch.tensor(std, dtype=val.dtype)
    return (val - mean) / std


def get_normalisation_transforms(feat_stats, target_stats):
    # define the transformations (i.e. the standardisation of the data)
    feat_transform = transforms.Compose([
        transforms.Normalize(feat_stats['mean'], feat_stats['std'])
    ])

    target_transform = transforms.Compose([
        torchvision.transforms.Lambda(lambda t: standardise(t, target_stats['mean'], target_stats['std'])),
    ])

    return feat_transform, target_transform


def get_feat_stats(agg_band_stats: dict, exclude: list = [], include: list = []):
    if len(include) > 0 and len(exclude) > 0:
        raise ValueError("Provide either include or exclude, not both")

    # invert include
    if len(include) > 0:
        exclude = [i for i in agg_band_stats.keys() if i not in include]

    feat_stats = {
        'mean': [v['mean'] for k, v in agg_band_stats.items() if (k not in exclude)],
        'std': [v['std'] for k, v in agg_band_stats.items() if (k not in exclude)],
        'min': [v['min'] for k, v in agg_band_stats.items() if (k not in exclude)],
        'max': [v['max'] for k, v in agg_band_stats.items() if (k not in exclude)]
    }
    return feat_stats


def get_target_stats(lsms_df, target_var):
    target_stats = {'mean': [np.mean(lsms_df[target_var])],
                    'std': [np.std(lsms_df[target_var])],
                    'min': [np.min(lsms_df[target_var])],
                    'max': [np.max(lsms_df[target_var])]}
    return target_stats


def get_agg_band_stats(band_stat_pth, lsms_df):
    with open(band_stat_pth, 'rb') as f:
        band_stats = pickle.load(f)

    all_ids = list(band_stats[0].unique_id)
    df_ids = list(lsms_df.unique_id)
    flagged_ids = [i for i in all_ids if i not in df_ids]

    agg_band_stats = aggregate_band_stats(band_stats, flagged_ids=flagged_ids)
    return agg_band_stats


def aggregate_band_stats(band_stats, flagged_ids=[]):
    summary_band_stats = {}
    for band in band_stats.keys():
        summary_band_stats[band] = aggregate_band(band_stats[band], flagged_ids)
    return summary_band_stats


def aggregate_band(band_stats_band, flagged_ids):
    res = {}
    if len(flagged_ids) == 0:
        aux = band_stats_band
    else:
        mask = np.array([i in flagged_ids for i in band_stats_band.unique_id])
        aux = band_stats_band.loc[~mask].reset_index(drop=True)
    res['min'] = np.min(aux['min'])
    res['max'] = np.max(aux['max'])
    res['mean'] = np.mean(aux['mean'])
    res['median'] = np.median(aux['median'])
    res['sum'] = np.sum(aux['sum'])
    res['sum_of_squares'] = np.sum(aux['sum_of_squares'])
    res['N'] = sum(aux['n'])
    res['N_na'] = sum(aux['n_na'])
    res['std'] = calc_std(res['sum'], res['sum_of_squares'], res['N'])
    return res


def calc_std(sm, ss, n):
    vr = ss / n - (sm / n) ** 2
    return np.sqrt(vr)
