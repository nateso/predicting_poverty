import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

#***********************************************************************************************************************
# Functions to to load satellite images and basic processing
#***********************************************************************************************************************

def load_img(file_path):
    # extract the file name
    file_name = file_path.split("/")[-1].split('.')[0]

    # load the image
    img = import_tif_file(file_path)

    # center crop the image
    img = center_crop_img(img)

    # if it is a Landsat Image reorder the first three channels
    is_ls = 'LS' in file_name
    if is_ls:
        img = reorder_rgb(img)

    # if it is a RS image, fix the WSF image
    is_rs = ('RS' in file_name) and ('RS_v2_' not in file_name)
    if is_rs:
        start_year = int(file_name[-4:])
        img = fix_wsf(img, start_year, wsf_idx = 5)
        img[:,:,0] = replace_pixel_values(img[:,:,0], -999, -0.01) # in nightlights images replace water with -0.1
        img[:,:,1] = replace_pixel_values(img[:,:,1], -999, -1.01) # NDVI mean
        img[:,:,2] = replace_pixel_values(img[:,:,2], -999, -1.01) # NDVI median
        img[:,:,3] = replace_pixel_values(img[:,:,3], -999, -1.01) # NDVi cropland mean
        img[:,:,3] = replace_pixel_values(img[:,:,3], -888, -1.02) # NDVI cropland mean no cropland mask values
        img[:,:,4] = replace_pixel_values(img[:,:,4], -999, -1.01) # NDVi cropland median water
        img[:,:,4] = replace_pixel_values(img[:,:,4], -888, -1.02) # NDVi cropland median no cropland
        img[:,:,5] = replace_pixel_values(img[:,:,5], -999, -0.01) # WSF water
    is_rs_v2 = 'RS_v2_' in file_name
    if is_rs_v2:
        start_year = int(file_name[-4:])
        img = fix_wsf(img, start_year, wsf_idx = 2)

    return img

def import_tif_file(geotiff_file_path, is_ls=False):
    with rasterio.open(geotiff_file_path) as src:
        # Read the image data.
        src_img = src.read()
    img = src_img.transpose(1, 2, 0)  # rearrange the dimensions of the np array
    return img


def reorder_rgb(img):
    '''
    The order of the channels in GEE is Blue, Green, Red
    Reorder the imgage to RGB.
    '''
    n_channels = img.shape[2]
    bgr = img[:, :, :3]
    rgb = np.stack([img[:, :, 2], img[:, :, 1], img[:, :, 0]], axis=2)
    if n_channels > 3:
        oth = img[:, :, 3:]
        img = np.concatenate((rgb, oth), axis=2)
    else:
        img = rgb
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


def count_na_pixels(img):
    '''
    count the number of NA pixels in each image.
    Returns a dictionary with the number of NA pixels per image band
    '''
    n_bands = img.shape[2]
    na_pixels = {}
    na_pixels['n_pixels'] = len(img[:, :, 0].flatten())
    for i in range(n_bands):
        na_pixels[i] = sum(np.isnan(img[:, :, i].flatten()))
    return na_pixels


def fix_wsf(img, start_year, wsf_idx = 5):
    img[:, :, wsf_idx] = np.nan_to_num(img[:, :, wsf_idx], nan=0)
    is_pop = (img[:, :, wsf_idx] < (start_year + 1)) & (img[:, :, wsf_idx] > 1)
    not_yet_pop = img[:, :, wsf_idx] > start_year
    img[:, :, wsf_idx][is_pop] = 1
    img[:, :, wsf_idx][not_yet_pop] = 0
    return img

def replace_pixel_values(img_channel, old_px_value, new_px_value):
    img_channel[img_channel == old_px_value] = new_px_value
    return img_channel

#***********************************************************************************************************************
# Functions to compute statistics on the images
#***********************************************************************************************************************

def compute_stats(img, data_type = "LS"):
    n_bands = img.shape[2]
    basic_stats = {}
    for i in range(n_bands):
        if data_type == "LS":
            basic_stats[i] = get_basic_stats_ls(img[:,:,i])
        else:
            basic_stats[i] = get_basic_stats_rs(img[:,:,i])
    return basic_stats

def get_basic_stats_ls(band):
    stats = {}
    stats['min'] = np.nanmin(band)
    stats['max'] = np.nanmax(band)
    stats['mean'] = np.nanmean(band)
    stats['median'] = np.nanmedian(band)
    stats['sum'] = np.nansum(band)
    stats['sum_of_squares'] = np.nansum(band**2)
    stats['std'] = np.nanstd(band)
    stats['n_negative'] = np.nansum(band < 0)
    stats['n_over_1'] = np.nansum(band > 1)
    stats['n_not_na'] = np.count_nonzero(~np.isnan(band))
    stats['n_na'] = np.count_nonzero(np.isnan(band))
    stats['n'] = len(band.flatten())
    return stats

def get_basic_stats_rs(band):
    stats = {}
    masked_band = band[((band > -888) | (np.isnan(band)))].flatten() # -888 indicates 'no cropland' and -999 indicates 'water'
    stats['min'] = np.nanmin(masked_band)
    stats['max'] = np.nanmax(masked_band)
    stats['mean'] = np.nanmean(masked_band)
    stats['median'] = np.nanmedian(masked_band)
    stats['sum'] = np.nansum(masked_band)
    stats['sum_of_squares'] = np.nansum(masked_band**2)
    stats['std'] = np.nanstd(masked_band)
    stats['n_negative'] = np.nansum(masked_band < 0)
    stats['n_over_1'] = np.nansum(masked_band > 1)
    stats['n_not_na'] = np.count_nonzero(~np.isnan(band))
    stats['n_na'] = np.count_nonzero(np.isnan(band))
    stats['n'] = band.shape[0] * band.shape[1]
    stats['n_water'] = np.sum(band == -999)
    stats['n_no_cropland'] = np.sum(band == -888)
    return stats

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
    res['N_not_na'] = sum(aux['n_not_na'])
    res['std'] = calc_std(res['sum'], res['sum_of_squares'], res['N'])
    return res


def calc_std(sm, ss, n):
    vr = ss / n - (sm / n) ** 2
    return np.sqrt(vr)

#***********************************************************************************************************************
# Functions to print statistics on the images
#***********************************************************************************************************************

def print_summary(summary_band_stats, band_name_dict):
    for band, stats in summary_band_stats.items():
      print(f"{band}:\t min:{stats['min']:.5f}\t max:{stats['max']:.5f}\t mean:{stats['mean']:.5f}\t std:{stats['std']:.5f}\t{band_name_dict[band]}")

def print_min_max(img):
    n_channels = img.shape[2]
    for i in range(n_channels):
        flat_img = img[:,:,i].flatten()
        img_min = min(flat_img)
        img_max = max(flat_img)
        print(f"\n Channel {i}")
        print(img_min)
        print(img_max)

def print_band_stats(band_stats, band_name_dict):
  for band in band_stats.keys():
    aux = band_stats[band]['n_na']
    aux_neg = band_stats[band]['n_negative']
    aux_pos = band_stats[band]['n_over_1']
    print(f"\nBand {band_name_dict[band]} sum of NA pixels: {sum(aux)}")
    print(f"Band {band_name_dict[band]} number of images with NA pixels: {np.sum(aux > 0)}")
    print(f"Band {band_name_dict[band]} mean number of NA pixels: {np.mean(aux)}")
    print(f"Band {band_name_dict[band]} number of images with negative pixels: {np.sum(aux_neg > 0)}")
    print(f"Band {band_name_dict[band]} number of negative pixels: {np.sum(aux_neg)}")
    print(f"Band {band_name_dict[band]} mean number of negative pixels: {np.mean(aux_neg)}")
    print(f"Band {band_name_dict[band]} median number of negative pixels: {np.median(aux_neg)}")
    print(f"Band {band_name_dict[band]} max number of negative pixels: {np.max(aux_neg)}")
    print(f"Band {band_name_dict[band]} number of images with pixels > 1: {np.sum(aux_pos > 0)}")

#***********************************************************************************************************************
# Functions to plot statistics on the images
#***********************************************************************************************************************

def plot_most_affected_imgs(lsms_df, band_stats, band_nr, by = 'n_na', lower = 0, upper = 100):
  most_affected = band_stats[band_nr].sort_values(by = by, ascending = False).reset_index(drop = True).iloc[lower:upper].copy()
  most_affected = pd.merge(most_affected, lsms_df[['unique_id','file_path']], on = 'unique_id')
  most_affected_ids = list(most_affected.unique_id)
  most_affected_Na_count = list(most_affected.n_na)
  most_affected_paths = list(most_affected.file_path)

  fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(50, 50))
  for i, ax in enumerate(axs.flat):
    img = load_img(most_affected_paths[i])
    ax.imshow(img[:,:,band_nr], cmap = 'gray')
    ax.set_title(f'{most_affected_ids[i]}-count:{most_affected_Na_count[i]}')
  plt.show()


def plot_ls_img(img, title = None):
      '''
      plots all 6 bands of a processed Landsat image
      '''
      # the first three channels are the RGB image
      fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (20,5))
      ax[0].imshow(img[:,:,:3])
      ax[0].set_title('RGB')
      ax[1].imshow(img[:,:,3])
      ax[1].set_title("NIR")
      ax[2].imshow(img[:,:,4])
      ax[2].set_title("SWIR1")
      ax[3].imshow(img[:,:,5])
      ax[3].set_title("TEMP1")
      for i in range(4):
        ax[i].axis('off')
      fig.suptitle(title)
      plt.show()
