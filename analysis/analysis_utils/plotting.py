import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
#******************************************************************************
#........ Functions to plot Landsat images ........
#******************************************************************************

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
    ax : object (optional)
        The axes object where the ax element should be plotted.
    extent : tuple (optional)
        The extent object that matplotlib expects (left, right, bottom, top).
    title : string (optional)
        The intended title of the plot.
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

def plot_ls(cluster_id, fname = None, title = None):
    data_type = 'LS'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/{data_type}/LS_median_cluster"
    img_pth = f"{img_dir}/{data_type}_{cluster_id}.npy"

    ls_img = np.load(img_pth)#.transpose(2,0,1)

    fig, ax = plt.subplots()
    plot_rgb(ls_img[:,:,:3], ax = ax, figsize = (5,5))
    plt.title(title)
    if fname is not None:
      pth = f"../figures/sat_imgs/{fname}"
      plt.savefig(pth, dpi = 300, bbox_inches = 'tight')
    plt.show()


#******************************************************************************
#........ Functions to plot static RS images ........
#******************************************************************************

def plot_wsf(cluster_id, fname = None, title = None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_static_processed"
    img_pth = f"{img_dir}/{data_type}_{cluster_id}.npy"

    static_img = np.load(img_pth)
    plt.imshow(static_img[:,:,0], cmap = 'gray')
    plt.axis('off')
    plt.title(title)
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi = 300, bbox_inches = 'tight')
    plt.show()

def plot_lc_esa(cluster_id, fname = None, title = None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_static_processed"

    img_pth = f"{img_dir}/{data_type}_{cluster_id}.npy"
    static_img = np.load(img_pth)

    # LC img
    colors = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000',
              '#b4b4b4', '#0096a0', '#0064c8']

    labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up',
              'Barren', 'Wetland', 'Water']

    values = [.1,.2,.3,.4,.5,.6,.7,.8]

    # Create a custom colormap with discrete colors
    cmap = plt.cm.colors.ListedColormap(colors)

    # Normalize the image data for mapping the values to the colormap
    norm = plt.cm.colors.Normalize(vmin=.1, vmax=.8)

    # Plot the image with the custom color map
    image = plt.imshow(static_img[:,:,1], cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title(title)

    # Create colorbar with labels
    tick_values = np.linspace(.15,.75,8)
    cbar = plt.colorbar(image, ticks=tick_values)#, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(labels)

    # save the image
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi = 300, bbox_inches = 'tight')

    plt.show()


def plot_lc_modis(unique_id, fname = None, title = None):
    data_type = 'NL_WSF_LC'
    root_data_dir = "../Data"
    img_dir = f"{root_data_dir}/RS_v2/{data_type}_processed"

    img_pth = f"{img_dir}/{data_type}_{unique_id}.npy"
    nl_wsf_lc_img = np.load(img_pth)


    colors = ['#05450a', '#086a10', '#54a708', '#78d203', '#009900',
              '#c6b044', '#dcd159', '#dade48', '#fbff13', '#b6ff05',
              '#27ff87', '#c24f44', '#a5a5a5', '#ff6d4c', '#69fff8',
              '#f9ffa4', '#1c0dff']

    labels = ['Evergreen Needleleaf Forests', 'Evergreen Broadleaf Forests',
              'Deciduous Needleleaf Forests', 'Deciduous Broadleaf Forests',
              'Mixed Forests', 'Closed Shrublands', 'Open Shrublands', 'Woody Savannas',
              'Savannas', 'Grasslands', 'Permanent Wetlands', 'Croplands', 'Built-up',
              'Natural Vegetation', 'Snow and Ice', 'Barren', 'Water']

    # Create a custom colormap with discrete colors
    cmap = plt.cm.colors.ListedColormap(colors)

    # Normalize the image data for mapping the values to the colormap
    norm = plt.cm.colors.Normalize(vmin=1, vmax=len(labels))

    # Plot the image with the custom color map
    image = plt.imshow(nl_wsf_lc_img[:, :, 2], cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title(title)

    # Create colorbar with labels
    tick_values = np.linspace(1.5,16.5,17)
    cbar = plt.colorbar(image, ticks=tick_values, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(labels)

    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi = 300, bbox_inches = 'tight')
    plt.show()


#******************************************************************************
#........ Functions to plot dynamic RS images ........
#******************************************************************************
def plot_mean_nl(cluster_id, fname=None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_mean_cluster"
    img_pth = f"{img_dir}/{data_type}_{cluster_id}.npy"

    mean_img = np.load(img_pth)

    # plot the mean nl image
    min_val = np.min(mean_img[:, :, 0])
    max_val = np.max(mean_img[:, :, 0])
    plt.imshow(mean_img[:, :, 0], cmap='gray', vmin=min_val, vmax=max_val)
    plt.axis('off')
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean_ndvi(cluster_id, fname=None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_mean_cluster"
    img_pth = f"{img_dir}/{data_type}_{cluster_id}.npy"

    mean_img = np.load(img_pth)
    color_palette = [
        '#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
        # '#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
        '#012e01', '#011d01', '#011301']

    # Create a custom colormap using the defined color palette
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', color_palette, N=256)

    # Plot the image
    plt.imshow(mean_img[:, :, 1], cmap=cmap, vmin=-1, vmax=1)
    plt.axis('off')

    # Display the plot
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')

    plt.show()


def plot_mean_ndwi(cluster_id, fname=None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_mean_cluster"
    img_pth = f"{img_dir}/{data_type}_{cluster_id}.npy"

    mean_img = np.load(img_pth)

    color_palette = ['#F4F4F6', '#000E75']

    # Create a custom colormap using the defined color palette
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', color_palette, N=256)

    # Plot the image
    min_val = np.min(mean_img[:, :, 2])
    max_val = np.max(mean_img[:, :, 2])
    plt.imshow(mean_img[:, :, 2], cmap=cmap, vmin=min_val, vmax=max_val)
    plt.axis('off')

    # Display the plot
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')

    plt.show()

def plot_nl(unique_id, fname = None, title = None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_dynamic_processed"
    img_pth = f"{img_dir}/{data_type}_{unique_id}.npy"

    dynamic_img = np.load(img_pth)

    # plot the nl img
    plt.imshow(dynamic_img[:,:,0], cmap = 'gray', vmin = 0, vmax = 1)
    plt.title(title)
    plt.axis('off')
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi = 300, bbox_inches = 'tight')
    plt.show()

def plot_ndvi(unique_id, fname = None, title = None):
    data_type = 'RS_V2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/RS_v2_dynamic_processed"
    img_pth = f"{img_dir}/{data_type}_{unique_id}.npy"

    dynamic_img = np.load(img_pth)
    color_palette = [
        '#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
        #'#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
        '#012e01', '#011d01', '#011301']

    # Create a custom colormap using the defined color palette
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', color_palette, N=256)

    # Plot the image
    plt.imshow(dynamic_img[:,:,1], cmap=cmap, vmin=-1, vmax=1)
    plt.axis('off')
    plt.title(title)

    # Display the plot
    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi = 300, bbox_inches = 'tight')

    plt.show()


def plot_nl_cluster(cluster_id, fname = None, title = None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_dynamic_processed"

    # list all files in image directory
    files = np.array(os.listdir(img_dir))
    mask = [cluster_id in i for i in files]
    files = np.sort(files[mask])

    # plot all images in one plot
    # Plot all images in one plot
    num_images = len(files)
    rows = 1
    cols = num_images

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, file in enumerate(files):
        year = file[-8:-4]
        img_pth = os.path.join(img_dir, file)
        img = np.load(img_pth)[:,:,0]
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')

        ax.set_title(f"{year}")
    plt.tight_layout()

    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')

    plt.show()


def plot_ndvi_cluster(cluster_id, fname=None, title=None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_dynamic_processed"

    # list all files in image directory
    files = np.array(os.listdir(img_dir))
    mask = [cluster_id in i for i in files]
    files = np.sort(files[mask])

    # plot all images in one plot
    # Plot all images in one plot
    num_images = len(files)
    rows = 1
    cols = num_images

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    color_palette = [
        '#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
        # '#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
        '#012e01', '#011d01', '#011301']

    # Create a custom colormap using the defined color palette
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', color_palette, N=256)

    # Plot the image
    for i, file in enumerate(files):
        year = file[-8:-4]
        img_pth = os.path.join(img_dir, file)
        img = np.load(img_pth)[:, :, 1]
        ax = axes[i]
        ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.axis('off')

        ax.set_title(f"{year}")
    plt.tight_layout()

    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')

    plt.show()

def plot_ndwi_gao_cluster(cluster_id, fname=None, title=None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_dynamic_processed"

    # list all files in image directory
    files = np.array(os.listdir(img_dir))
    mask = [cluster_id in i for i in files]
    files = np.sort(files[mask])

    # plot all images in one plot
    # Plot all images in one plot
    num_images = len(files)
    rows = 1
    cols = num_images

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    color_palette = ['#F4F4F6', '#000E75']

    # Create a custom colormap using the defined color palette
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', color_palette, N=256)

    # Plot the image
    for i, file in enumerate(files):
        year = file[-8:-4]
        img_pth = os.path.join(img_dir, file)
        img = np.load(img_pth)[:, :, 2]
        ax = axes[i]
        ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.axis('off')

        ax.set_title(f"{year}")
    plt.tight_layout()

    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')

    plt.show()


def plot_ndwi_mcf_cluster(cluster_id, fname=None, title=None):
    data_type = 'RS_v2'
    sat_img_dir = "../../Data/satellite_imgs"
    img_dir = f"{sat_img_dir}/RS_v2/{data_type}_dynamic_processed"

    # list all files in image directory
    files = np.array(os.listdir(img_dir))
    mask = [cluster_id in i for i in files]
    files = np.sort(files[mask])

    # plot all images in one plot
    # Plot all images in one plot
    num_images = len(files)
    rows = 1
    cols = num_images

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    color_palette = ['#F4F4F6', '#000E75']

    # Create a custom colormap using the defined color palette
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', color_palette, N=256)

    # Plot the image
    for i, file in enumerate(files):
        year = file[-8:-4]
        img_pth = os.path.join(img_dir, file)
        img = np.load(img_pth)[:, :, 3]
        ax = axes[i]
        ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.axis('off')

        ax.set_title(f"{year}")
    plt.tight_layout()

    if fname is not None:
        pth = f"../figures/sat_imgs/{fname}"
        plt.savefig(pth, dpi=300, bbox_inches='tight')

    plt.show()


#******************************************************************************
#........ Functions to plot demeaned / delta images ........
#******************************************************************************
    def plot_demeaned_imgs(cluster_id, fname=None, title=None, channel=0):
        data_type = 'RS_v2'
        sat_img_dir = "../../Data/satellite_imgs"
        img_dir = f"{sat_img_dir}/RS_v2/{data_type}_demeaned"

        # list all files in image directory
        files = np.array(os.listdir(img_dir))
        mask = [cluster_id in i for i in files]
        files = np.sort(files[mask])

        # plot all images in one plot
        num_images = len(files)
        rows = 1
        cols = num_images

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

        for i, file in enumerate(files):
            year = file[-8:-4]
            img_pth = os.path.join(img_dir, file)
            img = np.load(img_pth)[:, :, channel]
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')

            ax.set_title(f"{year}")
        plt.tight_layout()
        # plt.legend()

        if fname is not None:
            pth = f"../figures/sat_imgs/{fname}"
            plt.savefig(pth, dpi=300, bbox_inches='tight')

        plt.show()


#******************************************************************************
#........ Functions to plot target variables ........
#******************************************************************************

def plot_target_by_cntry(lsms_df, target_var, fname = None):
    plt.figure(figsize=(5, 5))
    for cntry in np.unique(lsms_df.country):
        # subset the data
        sub_df = lsms_df[lsms_df['country'] == cntry].groupby('start_year')[target_var].mean()
        plt.plot(sub_df.index, sub_df.values, label = cntry)
        plt.scatter(sub_df.index, sub_df.values)
    x_ticks = range(min(sub_df.index)-1, max(sub_df.index)+2)
    plt.xticks(x_ticks, rotation=45)
    plt.xlabel("Year")
    plt.ylabel(target_var)
    plt.legend()
    if fname is not None:
        plt.savefig(f"../figures/target_vars/{fname}", dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_target_cluster(cluster_id, lsms_df, target_var, fname = None):
    plt.figure(figsize = (6,6))
    # subset the data
    sub_df = lsms_df[lsms_df['cluster_id'] == cluster_id].reset_index(drop = True).sort_values('start_year')

    plt.plot(sub_df['start_year'], sub_df[target_var])
    plt.scatter(sub_df['start_year'], sub_df[target_var])
    x_ticks = range(min(sub_df['start_year']), max(sub_df['start_year'])+1)
    plt.xticks(x_ticks)
    plt.xlabel("Year")
    plt.ylabel(target_var)
    if fname is not None:
        plt.savefig(f"../figures/target_vars/{fname}", bbox_inches = 'tight', dpi = 300)
    plt.show()







