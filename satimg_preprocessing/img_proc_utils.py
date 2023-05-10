import numpy as np
import rasterio

#***********************************************************************************************************************
# Functions to to load satellite images and basic processing
#***********************************************************************************************************************

def load_img(file_path):
    # extract the file name
    file_name = file_path.split("/")[-1]

    # load the image
    img = import_tif_file(file_path)

    # center crop the image
    img = center_crop_img(img)

    # if it is a Landsat Image reorder the first three channels
    is_ls = 'LS' in file_name
    if is_ls:
        img = reorder_rgb(img)

    # if it is a RS image, fix the WSF image
    is_rs = 'RS' in file_name
    if is_rs:
        img = fix_wsf(img, info)

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


# centers and crops the image to a 255 x 255 pixel image
def center_crop_img(img, img_width=255, img_height=255):
    act_h = img.shape[0]
    act_w = img.shape[1]
    n_channles = img.shape[2]

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


def fix_wsf(img, info):
    img[:, :, 5] = np.nan_to_num(img[:, :, 5], nan=0)
    is_pop = (img[:, :, 5] < (info.start_year + 1)) & (img[:, :, 5] > 1)
    not_yet_pop = img[:, :, 5] > info.start_year
    img[:, :, 5][is_pop] = 1
    img[:, :, 5][not_yet_pop] = 0
    return img

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