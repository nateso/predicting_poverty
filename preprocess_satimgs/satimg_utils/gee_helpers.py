import ee
from typing import Union, Mapping, Any, Dict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
import time

Numeric = Union[int, float]

# geotiff exporter function
def geotiffexporter(img: ee.Image,
                    fcPoint: ee.Geometry,
                    scale: Numeric,
                    roi: ee.Geometry,
                    prefix: str,
                    fname: str) -> ee.batch.Task:
    '''Creates and starts a task to export a ee.FeatureCollection to a GeoTIFF
    file in Google Drive.

    Drive: prefix/fname.tif

    Args
    - img: ee.Image
    - prefix: str, folder name in Drive no trailing '/'
    - fname: str, filename

    Returns
    - task: ee.batch.Task
    '''
    task = ee.batch.Export.image.toDrive(
        image = img,
        region = roi,
        scale = scale,
        description = fname,
        folder = prefix,
        fileNamePrefix = fname,
        fileFormat = 'GeoTIFF',
        crs = 'EPSG:3857'
        )
    task.start()
    return task

# wait on task function
def wait_on_tasks(tasks: Mapping[Any, ee.batch.Task],
                  poll_interval: int = 20,
                  ) -> None:
    '''Displays a progress bar of task progress.

    Args
    - tasks: dict, maps task ID to a ee.batch.Task
    - show_progbar: bool, whether to display progress bar
    - poll_interval: int, # of seconds between each refresh
    '''
    remaining_tasks = list(tasks.keys())
    done_states = {ee.batch.Task.State.COMPLETED,
                   ee.batch.Task.State.FAILED,
                   ee.batch.Task.State.CANCEL_REQUESTED,
                   ee.batch.Task.State.CANCELLED}

    progbar = tqdm(total=len(remaining_tasks))
    while len(remaining_tasks) > 0:
        new_remaining_tasks = []
        for taskID in remaining_tasks:
            status = tasks[taskID].status()
            state = status['state']

            if state in done_states:
                progbar.update(1)

                if state == ee.batch.Task.State.FAILED:
                    state = (state, status['error_message'])
                elapsed_ms = status['update_timestamp_ms'] - status['creation_timestamp_ms']
                elapsed_min = int((elapsed_ms / 1000) / 60)
                progbar.write(f'Task {taskID} finished in {elapsed_min} min with state: {state}')
            else:
                new_remaining_tasks.append(taskID)
        remaining_tasks = new_remaining_tasks
        time.sleep(poll_interval)
    progbar.close()


# general helpers
def df_to_fc(df: pd.DataFrame, lat_colname: str = 'lat',
             lon_colname: str = 'lon') -> ee.FeatureCollection:
    '''
    Args
    - csv_path: str, path to CSV file that includes at least two columns for
        latitude and longitude coordinates
    - lat_colname: str, name of latitude column
    - lon_colname: str, name of longitude column

    Returns: ee.FeatureCollection, contains one feature per row in the CSV file
    '''
    # convert values to Python native types
    # see https://stackoverflow.com/a/47424340
    df = df.astype('object')

    ee_features = []
    for i in range(len(df)):
        props = df.iloc[i].to_dict()

        # oddly EE wants (lon, lat) instead of (lat, lon)
        _geometry = ee.Geometry.Point([
            props[lon_colname],
            props[lat_colname],
        ], proj = 'EPSG:4326')
        ee_feat = ee.Feature(_geometry, props)
        ee_features.append(ee_feat)

    return ee.FeatureCollection(ee_features)

def get_roi(point: ee.Geometry, pixel_radius: Numeric):
    buffer_size = pixel_radius * 30 # Landsat images are 30m / px resolution
    roi_coords = point.buffer(buffer_size).bounds().getInfo()['coordinates']
    y_coords = [i[1] for i in roi_coords[0]]
    x_coords = [i[0] for i in roi_coords[0]]
    coords = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    roi = ee.Geometry.Rectangle(coords)
    return roi

def add_latlon(img: ee.Image) -> ee.Image:
    '''Creates a new ee.Image with 2 added bands of longitude and latitude
    coordinates named 'LON' and 'LAT', respectively
    '''
    latlon = ee.Image.pixelLonLat().select(
        opt_selectors=['longitude', 'latitude'],
        opt_names=['LON', 'LAT'])
    return img.addBands(latlon)

def to_array(img, scale, roi):
    '''
    Turns a single image with one channel into a numpy array
    '''
    band_name = img.bandNames().getInfo()[0]
    img = img.reduceRegion(
        reducer = ee.Reducer.toList(),
        geometry = roi,
        scale = scale,
        maxPixels = 1e9
    ).get(band_name).getInfo()
    img_arr = np.array(img)
    return img_arr

def count_na_pixels(image, roi, scale):
    # Create a mask for NA pixels
    na_mask = image.mask().Not()

    # Compute the number of NA pixels
    num_na_pixels = na_mask.reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = roi,
        scale=scale
    )
    # the output of this function is a ee.dictionary.Dictionary. It does not preserve
    # the input order of the image bands. The dictionary instead is ordered alphabetically.
    return num_na_pixels.getInfo()

def check_downloaded_files(prefix = 'RS', source_dir = "../"):
  folders = os.listdir(source_dir)

  mask = np.array([folder.startswith(prefix) for folder in folders])
  folders = np.array(folders)[mask]

  folder_paths = [source_dir + folder for folder in folders]
  downloaded_files = []
  for i in range(len(folder_paths)):
    downloaded_files += os.listdir(folder_paths[i])

  return downloaded_files

def apply_mask(img, mask, fill_val, invert = False):
    if invert:
      masked_img = img.where(mask.Not(), fill_val)
    else:
      masked_img = img.where(mask, fill_val)
    return masked_img