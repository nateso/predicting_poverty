import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ***********************************************************************************************************************
# Functions to to load satellite images and basic processing
# ***********************************************************************************************************************

def load_img(file_path):
    # load the image
    img = import_tif_file(file_path)

    # center crop the image and return
    return center_crop_img(img)


def import_tif_file(geotiff_file_path, is_ls=False):
    with rasterio.open(geotiff_file_path) as src:
        # Read the image data.
        src_img = src.read()
    img = src_img.transpose(1, 2, 0)  # rearrange the dimensions of the np array
    return img


# centers and crops the image to a 224 x 224 pixel image
def center_crop_img(img, img_width=224, img_height=224):
    act_h = img.shape[0]
    act_w = img.shape[1]

    h_diff = act_h - img_height
    w_diff = act_w - img_width

    # sets the pixel value where to start
    h_start = int(h_diff / 2)
    w_start = int(w_diff / 2)

    # sets the end of the pixel where to end
    h_end = h_start + img_height
    w_end = w_start + img_width

    cropped_img = img[h_start:h_end, w_start:w_end, :]
    return cropped_img


def preprocess_img(img, data_type, start_year=None):
    if data_type == 'LS':
        return reorder_rgb(img)
    elif data_type == 'RS':
        if start_year is None:
            raise ValueError("Need to provide a start year for data_type = RS")
        return fix_wsf(img, start_year, wsf_idx=5)
    elif data_type == 'RS_v2':
        if start_year is None:
            raise ValueError("Need to provide a start year for data_type = RS_v2")
        return fix_wsf(img, start_year, wsf_idx=2)
    elif data_type == 'WP':
        # Water is NA -> recode as 0. 
        img[:, :, 0] = np.nan_to_num(img[:, :, 0], nan=0)
        return img


def reorder_rgb(img):
    '''
    The order of the channels in GEE is Blue, Green, Red
    Reorder the imgage to RGB.
    '''
    n_channels = img.shape[2]
    rgb = np.stack([img[:, :, 2], img[:, :, 1], img[:, :, 0]], axis=2)
    if n_channels > 3:
        oth = img[:, :, 3:]
        img = np.concatenate((rgb, oth), axis=2)
    else:
        img = rgb
    return img


def fix_wsf(img, start_year, wsf_idx):
    img[:, :, wsf_idx] = np.nan_to_num(img[:, :, wsf_idx], nan=0)
    is_pop = (img[:, :, wsf_idx] < (start_year + 1)) & (img[:, :, wsf_idx] > 1)
    not_yet_pop = img[:, :, wsf_idx] > start_year
    img[:, :, wsf_idx][is_pop] = 1
    img[:, :, wsf_idx][not_yet_pop] = 0
    return img


def replace_pixel_values(img_channel, old_px_value, new_px_value):
    img_channel[img_channel == old_px_value] = new_px_value
    return img_channel


def recode_water_cropland(img):
    img[:, :, 0] = replace_pixel_values(img[:, :, 0], -999, -0.01)  # in nightlights images replace water with -0.1
    img[:, :, 1] = replace_pixel_values(img[:, :, 1], -999, -1.01)  # NDVI mean
    img[:, :, 2] = replace_pixel_values(img[:, :, 2], -999, -1.01)  # NDVI median
    img[:, :, 3] = replace_pixel_values(img[:, :, 3], -999, -1.01)  # NDVi cropland mean
    img[:, :, 3] = replace_pixel_values(img[:, :, 3], -888, -1.02)  # NDVI cropland mean no cropland mask values
    img[:, :, 4] = replace_pixel_values(img[:, :, 4], -999, -1.01)  # NDVi cropland median water
    img[:, :, 4] = replace_pixel_values(img[:, :, 4], -888, -1.02)  # NDVi cropland median no cropland
    img[:, :, 5] = replace_pixel_values(img[:, :, 5], -999, -0.01)  # WSF water
    return img


def decompose_lc(img, lc_idx, lc_type='esa'):
    lc_band = img[:, :, lc_idx]
    lc_counts = {}
    if lc_type == 'esa':
        lc_counts['tree'] = np.sum(lc_band == 10)
        lc_counts['shrubland'] = np.sum(lc_band == 20)
        lc_counts['grassland'] = np.sum(lc_band == 30)
        lc_counts['cropland'] = np.sum(lc_band == 40)
        lc_counts['built_up'] = np.sum(lc_band == 50)
        lc_counts['barren'] = np.sum(lc_band == 60)
        lc_counts['snow_ice'] = np.sum(lc_band == 70)
        lc_counts['water'] = np.sum(lc_band == 80)
        lc_counts['wetland'] = np.sum(lc_band == 90)
        lc_counts['mangroves'] = np.sum(lc_band == 95)
        lc_counts['moss'] = np.sum(lc_band == 100)
    if lc_type == 'modis':
        lc_counts['tree'] = np.sum(lc_band < 6)
        lc_counts['woody_shrubland'] = np.sum(lc_band == 6)
        lc_counts['shrubland'] = np.sum(lc_band == 7)
        lc_counts['grassland'] = np.sum(lc_band == 10)
        lc_counts['cropland'] = np.sum((lc_band == 12) | (lc_band == 14))
        lc_counts['built_up'] = np.sum(lc_band == 13)
        lc_counts['barren'] = np.sum(lc_band == 16)
        lc_counts['snow_ice'] = np.sum(lc_band == 15)
        lc_counts['water'] = np.sum(lc_band == 17)
        lc_counts['wetland'] = np.sum(lc_band == 11)
        lc_counts['woody_savanna'] = np.sum(lc_band == 8)
        lc_counts['savanna'] = np.sum(lc_band == 9)

    return lc_counts


# ***********************************************************************************************************************
# Functions to compute statistics on the images
# ***********************************************************************************************************************
def extract_image_statistics(id_pth_dict, data_type, id_name):
    file_extension = list(id_pth_dict.values())[0].split('.')[-1]
    image_stats = {}
    ids = []
    for id, pth in tqdm(id_pth_dict.items()):
        ids.append(id)
        # load the image
        if file_extension == 'npy':
            img = np.load(pth)
        elif file_extension == 'tif':
            img = load_img(pth)
            if data_type == 'LS':
                img = reorder_rgb(img)

        # extract the statistics for each band
        for i in range(img.shape[2]):
            if i in image_stats.keys():
                image_stats[i].append(get_basic_band_stats(img[:, :, i]))
            else:
                image_stats[i] = [get_basic_band_stats(img[:, :, i])]

    # convert the dictionary values to pandas dataframes
    for band_key in image_stats.keys():
        image_stats[band_key] = pd.json_normalize(image_stats[band_key])
        image_stats[band_key][id_name] = ids

    return image_stats


def compute_band_stats(img, data_type):
    n_bands = img.shape[2]
    band_stats = {}
    for i in range(n_bands):
        band = img[:, :, i]
        if data_type == "LS":
            band_stats[i] = get_basic_band_stats(band)
            band_stats[i]['n_negative'] = np.nansum(band < 0)
            band_stats[i]['n_over_1'] = np.nansum(band > 1)
        elif data_type == 'RS':
            masked_band = band[
                ((band > -888) | (np.isnan(band)))].flatten()  # -888 indicates 'no cropland' and -999 indicates 'water'
            band_stats[i] = get_basic_band_stats(masked_band)
        elif data_type == 'RS_v2':
            band_stats[i] = get_basic_band_stats(band)
        else:
            raise ValueError("No known data type provided")
    return band_stats


def get_basic_band_stats(band):
    basic_stats = {}
    basic_stats['min'] = np.nanmin(band)
    basic_stats['max'] = np.nanmax(band)
    basic_stats['mean'] = np.nanmean(band)
    basic_stats['median'] = np.nanmedian(band)
    basic_stats['sum'] = np.nansum(band)
    basic_stats['sum_of_squares'] = np.nansum(band ** 2)
    basic_stats['std'] = np.nanstd(band)
    basic_stats['n'] = len(band.flatten())
    basic_stats['n_na'] = np.count_nonzero(np.isnan(band))
    return basic_stats


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


# ***********************************************************************************************************************
# Functions to process the LS images
# ***********************************************************************************************************************


def normalise_img_min_max(img, mins, maxs):
    # function to rescale the image to a 0 - 1 range
    n_bands = img.shape[2]
    norm_img = []
    for band_idx in range(n_bands):
        norm_img.append((img[:, :, band_idx] - mins[band_idx]) / (maxs[band_idx] - mins[band_idx]))
    norm_img = np.array(norm_img)
    norm_img = norm_img.transpose(1, 2, 0)
    return norm_img


def standardise_img(img, means, stds):
    # function to standard normalise the image
    n_bands = img.shape[2]
    norm_img = []
    for band_idx in range(n_bands):
        norm_img.append((img[:, :, band_idx] - means[band_idx]) / stds[band_idx])
    norm_img = np.array(norm_img)
    norm_img = norm_img.transpose(1, 2, 0)
    return norm_img


def impute_pixels(img, band_means):
    # sets the pixel values to the overall mean of the image in order to not NAs in the image
    n_bands = img.shape[2]
    imp_img = []
    for band_idx in range(n_bands):
        img_band = img[:, :, band_idx].copy()
        na_mask = np.isnan(img_band)
        img_band[na_mask] = band_means[band_idx]
        imp_img.append(img_band)
    imp_img = np.array(imp_img).transpose(1, 2, 0)
    return imp_img


def count_bad_pixels(img):
    # a bad pixel is a pixel that is NA, or greater than 1 (only for RGB channels)
    n_bands = img.shape[2]
    bad_pixel_count = 0
    for band_idx in range(n_bands):
        if not band_idx == 5:
            na_count = np.sum(np.isnan(img[:, :, band_idx].flatten()))
            if band_idx in [0, 1, 2]:
                over_1_count = np.sum(img[:, :, band_idx].flatten() > 1)
                bad_pixel_count += na_count + over_1_count
            else:
                bad_pixel_count += na_count
    return bad_pixel_count


# ***********************************************************************************************************************
# Functions to print statistics on the images
# ***********************************************************************************************************************

def print_band_summary(summary_band_stats, band_name_dict):
    for band, stats in summary_band_stats.items():
        print(
            f"{band}:\t min:{stats['min']:.5f}\t max:{stats['max']:.5f}\t mean:{stats['mean']:.5f}\t std:{stats['std']:.5f}\t{band_name_dict[band]}")


def print_min_max(img):
    n_channels = img.shape[2]
    for i in range(n_channels):
        flat_img = img[:, :, i].flatten()
        img_min = min(flat_img)
        img_max = max(flat_img)
        print(f"\n Channel {i}")
        print(img_min)
        print(img_max)


def print_band_quality(band_stats, band_name_dict):
    for band in band_stats.keys():
        aux = band_stats[band]['n_na']
        print(f"\nBand {band_name_dict[band]} sum of NA pixels: {sum(aux)}")
        print(f"Band {band_name_dict[band]} number of images with NA pixels: {np.sum(aux > 0)}")
        print(f"Band {band_name_dict[band]} mean number of NA pixels: {np.mean(aux)}")


# ***********************************************************************************************************************
# Functions to plot statistics on the images
# ***********************************************************************************************************************

def plot_most_affected_ls_imgs(lsms_df, band_stats, band_nr, by='n_na', lower=0, upper=100):
    most_affected = band_stats[band_nr].sort_values(by=by, ascending=False).reset_index(drop=True).iloc[
                    lower:upper].copy()
    most_affected = pd.merge(most_affected, lsms_df[['unique_id', 'file_path']], on='unique_id')
    most_affected_ids = list(most_affected.unique_id)
    most_affected_Na_count = list(most_affected.n_na)
    most_affected_paths = list(most_affected.file_path)

    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(50, 50))
    for i, ax in enumerate(axs.flat):
        img = load_img(most_affected_paths[i])
        img = preprocess_img(img, data_type='LS')
        ax.imshow(img[:, :, band_nr], cmap='gray')
        ax.set_title(f'{most_affected_ids[i]}-count:{most_affected_Na_count[i]}')
    plt.show()


def plot_ls_img(img, title=None):
    '''
      plots all 6 bands of a processed Landsat image
      '''
    # the first three channels are the RGB image
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    ax[0].imshow(img[:, :, :3])
    ax[0].set_title('RGB')
    ax[1].imshow(img[:, :, 3])
    ax[1].set_title("NIR")
    ax[2].imshow(img[:, :, 4])
    ax[2].set_title("SWIR1")
    ax[3].imshow(img[:, :, 5])
    ax[3].set_title("TEMP1")
    for i in range(4):
        ax[i].axis('off')
    fig.suptitle(title)
    plt.show()

# ***********************************************************************************************************************
# Old functions
# ***********************************************************************************************************************
#
#
# def load_img_old(file_path):
#     # extract the file name
#     file_name = file_path.split("/")[-1].split('.')[0]
#
#     # if it is a Landsat Image reorder the first three channels
#     is_ls = 'LS' in file_name
#     if is_ls:
#         img = reorder_rgb(img)
#
#     # if it is a RS image, fix the WSF image
#     is_rs = ('RS' in file_name) and ('RS_v2_' not in file_name)
#     if is_rs:
#         start_year = int(file_name[-4:])
#         img = fix_wsf(img, start_year, wsf_idx=5)
#         img[:, :, 0] = replace_pixel_values(img[:, :, 0], -999, -0.01)  # in nightlights images replace water with -0.1
#         img[:, :, 1] = replace_pixel_values(img[:, :, 1], -999, -1.01)  # NDVI mean
#         img[:, :, 2] = replace_pixel_values(img[:, :, 2], -999, -1.01)  # NDVI median
#         img[:, :, 3] = replace_pixel_values(img[:, :, 3], -999, -1.01)  # NDVi cropland mean
#         img[:, :, 3] = replace_pixel_values(img[:, :, 3], -888, -1.02)  # NDVI cropland mean no cropland mask values
#         img[:, :, 4] = replace_pixel_values(img[:, :, 4], -999, -1.01)  # NDVi cropland median water
#         img[:, :, 4] = replace_pixel_values(img[:, :, 4], -888, -1.02)  # NDVi cropland median no cropland
#         img[:, :, 5] = replace_pixel_values(img[:, :, 5], -999, -0.01)  # WSF water
#     is_rs_v2 = 'RS_v2_' in file_name
#     if is_rs_v2:
#         start_year = int(file_name[-4:])
#         img = fix_wsf(img, start_year, wsf_idx=2)
#
#     return img

# def count_na_pixels(img):
#     '''
#     count the number of NA pixels in each image.
#     Returns a dictionary with the number of NA pixels per image band
#     '''
#     n_bands = img.shape[2]
#     na_pixels = {}
#     na_pixels['n_pixels'] = len(img[:, :, 0].flatten())
#     for i in range(n_bands):
#         na_pixels[i] = sum(np.isnan(img[:, :, i].flatten()))
#     return na_pixels


# def print_band_quality(band_stats, band_name_dict, data_type):
#     for band in band_stats.keys():
#         aux = band_stats[band]['n_na']
#         print(f"\nBand {band_name_dict[band]} sum of NA pixels: {sum(aux)}")
#         print(f"Band {band_name_dict[band]} number of images with NA pixels: {np.sum(aux > 0)}")
#         print(f"Band {band_name_dict[band]} mean number of NA pixels: {np.mean(aux)}")
#         if data_type == 'LS':
#             aux_neg = band_stats[band]['n_negative']
#             aux_pos = band_stats[band]['n_over_1']
#             print(f"Band {band_name_dict[band]} number of images with negative pixels: {np.sum(aux_neg > 0)}")
#             print(f"Band {band_name_dict[band]} number of negative pixels: {np.sum(aux_neg)}")
#             print(f"Band {band_name_dict[band]} mean number of negative pixels: {np.mean(aux_neg)}")
#             print(f"Band {band_name_dict[band]} median number of negative pixels: {np.median(aux_neg)}")
#             print(f"Band {band_name_dict[band]} max number of negative pixels: {np.max(aux_neg)}")
#             print(f"Band {band_name_dict[band]} number of images with pixels > 1: {np.sum(aux_pos > 0)}")
