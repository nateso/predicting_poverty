
## These functions are masked from the earthpy package and adapted to the needs of this project

import numpy.ma as ma
from .img_proc_utils import *

def plot_rgb(
    arr,
    rgb=(0, 1, 2),
    figsize=(10, 10),
    ax=None,
    extent=None,
    title=""
):
    """Plot three bands in a numpy array as a composite RGB image.

    Parameters
    ----------
    arr : numpy array
        An n-dimensional array in rasterio band order (bands, rows, columns)
        containing the layers to plot.
    rgb : list (default = (0, 1, 2))
        Indices of the three bands to be plotted.
    figsize : tuple (default = (10, 10)
        The x and y integer dimensions of the output plot.
    str_clip: int (default = 2)
        The percentage of clip to apply to the stretch. Default = 2 (2 and 98).
    ax : object (optional)
        The axes object where the ax element should be plotted.
    extent : tuple (optional)
        The extent object that matplotlib expects (left, right, bottom, top).
    title : string (optional)
        The intended title of the plot.
    stretch : Boolean (optional)
        Application of a linear stretch. If set to True, a linear stretch will
        be applied.

    Returns
    ----------
    ax : axes object
        The axes object associated with the 3 band image.

    """

    if len(arr.shape) != 3:
        raise ValueError(
            "Input needs to be 3 dimensions and in rasterio "
            "order with bands first"
        )

    # Index bands for plotting and clean up data for matplotlib
    rgb_bands = arr[:, :, rgb]

    nan_check = np.isnan(rgb_bands)

    if np.any(nan_check):
        rgb_bands = np.ma.masked_array(rgb_bands, nan_check)

    # If type is masked array - add alpha channel for plotting
    if ma.is_masked(rgb_bands):
        # Build alpha channel
        mask = ~(np.ma.getmask(rgb_bands[0])) * 255

        # Add the mask to the array & swap the axes order from (bands,
        # rows, columns) to (rows, columns, bands) for plotting
        rgb_bands = np.vstack(
            (bytescale(rgb_bands), np.expand_dims(mask, axis=0))
        )
    else:
        # Index bands for plotting and clean up data for matplotlib
        rgb_bands = bytescale(rgb_bands)

    # Then plot. Define ax if it's undefined
    show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True

    ax.imshow(rgb_bands, extent=extent)
    ax.set_title(title)
    ax.set(xticks=[], yticks=[])

    # Multipanel won't work if plt.show is called prior to second plot def
    if show:
        plt.show()
    return ax

def bytescale(data, high=255, low=0, cmin=None, cmax=None):
    """Byte scales an array (image).

    Byte scaling converts the input image to uint8 dtype, and rescales
    the data range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    Source code adapted from scipy.misc.bytescale (deprecated in scipy-1.0.0)

    Parameters
    ----------
    data : numpy array
        image data array.
    high : int (default=255)
        Scale max value to `high`.
    low : int (default=0)
        Scale min value to `low`.
    cmin : int (optional)
        Bias scaling of small values. Default is ``data.min()``.
    cmax : int (optional)
        Bias scaling of large values. Default is ``data.max()``.

    Returns
    -------
    img_array : uint8 numpy array
        The byte-scaled array.
    """
    if cmin is None or (cmin < data.min()):
        cmin = float(data.min())

    if (cmax is None) or (cmax > data.max()):
        cmax = float(data.max())

    # Calculate range of values
    crange = cmax - cmin
    if crange < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif crange == 0:
        raise ValueError(
            "`cmax` and `cmin` should not be the same value. Please specify "
            "`cmax` > `cmin`"
        )

    scale = float(high - low) / crange

    # If cmax is less than the data max, then this scale parameter will create
    # data > 1.0. clip the data to cmax first.
    data[data > cmax] = cmax
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype("uint8")

def plot_most_affected_rgb(uid_pth_dict, img_stats, lower=0, upper=25):
    most_affected = img_stats[0].sort_values(by='n_na', ascending=False).reset_index(drop=True)
    most_affected = most_affected.copy().iloc[lower:upper]
    most_affected['pth'] = [uid_pth_dict[uid] for uid in most_affected['unique_id']]

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(50, 50))
    for i, ax in enumerate(axs.flat):
        img = load_img(most_affected['pth'][i])
        img = reorder_rgb(img)

        # title = f"{most_affected['unique_id'][i]}-count:{most_affected['n_na'][i]}"
        plot_rgb(img[:, :, :3], ax=ax)
        ax.set_title(f"{most_affected['unique_id'][i]}-count:{most_affected['n_na'][i]}")

    plt.tight_layout()
    plt.show()