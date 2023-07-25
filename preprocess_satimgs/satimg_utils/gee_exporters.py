from typing import Dict, Tuple, Any
from .gee_classes import *
from .gee_helpers import *

# Export Landsat images
# Export them as one image
def export_ls_images(
        df: pd.DataFrame,
        country: str,
        series: str,
        unique_id: str,
        start_ts: str,
) -> Dict[Tuple[Any], ee.batch.Task]:
    '''
    Args
    - df: pd.DataFrame, contains columns ['lat', 'lon', 'country', 'year']
    - country: str, together with `year` determines the survey to export
    - year: int, together with `country` determines the survey to export
    - export_folder: str, name of folder for export

    Returns: dict, maps task name tuple (export_folder, country, year, chunk) to ee.batch.Task
    '''
    subset_df = df[(df['unique_id'] == unique_id)].reset_index(drop=True)
    fcPoint = df_to_fc(subset_df)

    start_year = datetime.strptime(start_ts, '%Y-%m-%d').date().year

    # tabular features
    tab_feats = {}
    tab_feats['unique_id'] = unique_id

    # get daytime satellite images (1 year median)
    ms_bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
    ls = LandsatSR(fcPoint.geometry(), start_ts=start_ts)
    ls_img_col = ls.merged_cr  # these are all LS5-LS8 images with clouds masked that are available for the given timestamp
    ls_img = ls_img_col.select(ms_bands).median().toFloat()  # convert to float 32 numbers

    roi = fcPoint.geometry().buffer(128 * 30).bounds()

    # # add image quality attributes
    # for name, val in count_na_pixels(ls_img, roi, 30).items():
    #     ky = 'na_count_'+name
    #     tab_feats[ky] = val

    # add number of images used for merged image
    tab_feats['n_img'] = ls.nImg

    # export the ls scenes and the other RS feats seperately
    task = {}

    ls_export_folder = f'LS_{country}_{series}_{start_year}'
    ls_fname = f'LS_{unique_id}'

    task[(ls_export_folder, country, ls_fname)] = geotiffexporter(img=ls_img,
                                                                  fcPoint=fcPoint,
                                                                  scale=30,
                                                                  roi=roi,
                                                                  prefix=ls_export_folder,
                                                                  fname=ls_fname)

    return task, tab_feats


# Export other Remote sensing images
def export_rs_images(
        df: pd.DataFrame,
        country: str,
        series: str,
        unique_id: str,
        start_ts: str,
        ) -> Dict[Tuple[Any], ee.batch.Task]:
    '''
    Args
    - df: pd.DataFrame, contains columns ['lat', 'lon', 'country', 'year']
    - country: str, together with `year` determines the survey to export
    - year: int, together with `country` determines the survey to export
    - export_folder: str, name of folder for export
    - chunk_size: int, optionally set a limit to the # of images exported per TFRecord file
        - set to a small number (<= 50) if Google Earth Engine reports memory errors

    Returns: dict, maps task name tuple (export_folder, country, year, chunk) to ee.batch.Task
    '''
    subset_df = df[(df['unique_id'] == unique_id)].reset_index(drop = True)
    fcPoint = df_to_fc(subset_df)

    start_year = datetime.strptime(start_ts, '%Y-%m-%d').date().year\

    # define the region of interest
    roi = fcPoint.geometry().buffer(128 * 30).bounds()

    # tabular features
    tab_feats = {}
    tab_feats['unique_id'] = unique_id

    # get the RS derived datasets
    rs = RS_Feats(fcPoint.geometry(), start_ts = start_ts)

    # nightlights
    nl_img = rs.init_nl_harm().img

    # wsf
    wsf_img = rs.init_wsf().img

    # Modis landcover
    modis_lc_img = rs.init_modis_lc().img

    # ESA landcover
    esa_lc = rs.init_esa_lc()

    esa_lc_img = esa_lc.img

    # get the Modis data
    modis = ModisSR(fcPoint.geometry(), start_ts = start_ts)

    ndvi = modis.init_SR_250().compute_NDVI()
    ndvi_mean = ndvi.mean().rename('ndvi_mean')

    modis_SR_500 = modis.init_SR_500()
    ndwi_gao = modis_SR_500.compute_NDWI_Gao()
    ndwi_gao_mean = ndwi_gao.mean()

    ndwi_McF = modis_SR_500.compute_NDWI_McFeeters()
    ndwi_McF_mean = ndwi_McF.mean()

    rs_img = ee.Image.cat([nl_img, ndvi_mean, wsf_img, ndwi_gao_mean, ndwi_McF_mean, modis_lc_img, esa_lc_img])
    rs_img = rs_img.toFloat()

    task = {}

    rs_export_folder = f'RS_v2_{country}_{series}_{start_year}'
    rs_fname = f'RS_v2_{unique_id}'

    task[(rs_export_folder, country, rs_fname)] = geotiffexporter(img = rs_img,
                                                            fcPoint = fcPoint,
                                                            scale = 30,
                                                            roi = roi,
                                                            prefix = rs_export_folder,
                                                            fname = rs_fname)

    return task, tab_feats


# export world Pop data
def export_wp_imgs(
    df: pd.DataFrame,
    country: str,
    series: str,
    unique_id: str,
    start_ts: str,
    ) -> Dict[Tuple[Any], ee.batch.Task]:
    '''
    Args
    - df: pd.DataFrame, contains columns ['lat', 'lon', 'country', 'year']
    - country: str, together with `year` determines the survey to export
    - year: int, together with `country` determines the survey to export
    - export_folder: str, name of folder for export
    - chunk_size: int, optionally set a limit to the # of images exported per TFRecord file
        - set to a small number (<= 50) if Google Earth Engine reports memory errors

    Returns: dict, maps task name tuple (export_folder, country, year, chunk) to ee.batch.Task
    '''
    subset_df = df[(df['unique_id'] == unique_id)].reset_index(drop = True)
    fcPoint = df_to_fc(subset_df)

    start_year = datetime.strptime(start_ts, '%Y-%m-%d').date().year\

    # define the region of interest
    roi = fcPoint.geometry().buffer(128 * 30).bounds()


    # initialise the image
    wp_img = WorldPop(fcPoint, start_ts, country).img.clip(roi)

    task = {}

    export_folder = f'WP_{country}_{series}_{start_year}'
    fname = f'WP_{unique_id}'

    task[(export_folder, country, fname)] = geotiffexporter(img = wp_img,
                                                            fcPoint = fcPoint,
                                                            scale = 30,
                                                            roi = roi,
                                                            prefix = export_folder,
                                                            fname = fname)

    return task


# Export WSF images one for every cluster
def export_wsf_images(
        df: pd.DataFrame,
        country: str,
        cluster_id: str,
) -> Dict[Tuple[Any], ee.batch.Task]:
    subset_df = df[(df['cluster_id'] == cluster_id)].reset_index(drop=True)
    fcPoint = df_to_fc(subset_df)

    # define the region of interest
    roi = fcPoint.geometry().buffer(128 * 30).bounds()

    # get the RS derived datasets
    rs = RS_Feats(fcPoint.geometry(), start_ts='2011-09-01')  # just use a generic start date, as it won't be used...

    # wsf
    wsf_img = rs.init_wsf().img

    task = {}

    export_folder = 'WSF_raw'
    fname = f'WSF_{cluster_id}'

    task[(export_folder, country, fname)] = geotiffexporter(img=wsf_img,
                                                            fcPoint=fcPoint,
                                                            scale=30,
                                                            roi=roi,
                                                            prefix=export_folder,
                                                            fname=fname)

    return task
