import numpy as np


def decomp_SST(lsms_df, spatial_cluster_id, target_var, verbose=True):
    group_avg = lsms_df.groupby(spatial_cluster_id)[target_var].mean()
    group_counts = lsms_df.groupby(spatial_cluster_id).size()

    n = len(lsms_df)
    k = len(group_avg)

    # Calculate SSW
    SSW = 0
    for group in group_avg.index:
        sq_diff = (lsms_df[lsms_df[spatial_cluster_id] == group][target_var] - group_avg[group]) ** 2
        SSW += np.sum(sq_diff)

    # Calculate SSB
    avg = lsms_df[target_var].mean()
    SSB = np.sum((group_avg - avg) ** 2 * group_counts)

    # Calculate SST
    SST = SSW + SSB
    SST_2 = np.sum((lsms_df[target_var] - avg) ** 2)

    if verbose:
        print(f"SSW = {SSW:.4f} - SSB = {SSB:.4f} - SST = {SST:.4f} - SST directly calc = {SST_2:.4f}")
        print(f"Share of SSW = {SSW / SST:.4f} - Share of SSB = {SSB / SST:.4f}")
        print("\n Latex ouput (TSS, BSS, WSS, BSS (share), WSS (share))")
        print(f"{target_var} & {SST:.0f} & {SSB:.2f} & {SSW:.2f} & {SSB / SST:.2f} & {SSW / SST:.2f}")

    return SSW, SSB, SST


def describe_target_cntry(lsms_df, target_var):
    cntry_name_dict = {'eth': "Ethiopia", 'mwi': 'Malawi', 'nga': "Nigeria", 'uga': 'Uganda', 'tza': 'Tanzania'}
    for cntry in np.unique(lsms_df.country):
        sub_df = lsms_df[lsms_df.country == cntry]
        mn = np.mean(sub_df[target_var])
        med = np.median(sub_df[target_var])
        mini = np.min(sub_df[target_var])
        maxi = np.max(sub_df[target_var])
        sd = np.std(sub_df[target_var])
        n = len(sub_df)
        str = f'{cntry_name_dict[cntry]} & {mn:.2f} & {med:.2f} & {mini:.2f} & {maxi:.2f} & {sd:.2f} & {n} \\\\'
        print("\\hspace{0.5cm} " + str)

    total_output = f'''
    {np.mean(lsms_df[target_var]):.2f} &
    {np.median(lsms_df[target_var]):.2f} &
    {np.min(lsms_df[target_var]):.2f} &
    {np.max(lsms_df[target_var]):.2f} &
    {np.std(lsms_df[target_var]):.2f} &
    {len(lsms_df[target_var])} \\\\'''
    total_output = total_output.replace('\n', '').replace('\t', '').replace('    ', ' ')
    print('\\hspace{0.5cm} Overall &' + total_output)


# Table 4 - descriptives for the target of the spatial, temporal and overall model
def describe_target(lsms_df, target_var, var_name = "w"):
    output = f'''
        {np.mean(lsms_df[target_var]):.2f} &
        {np.median(lsms_df[target_var]):.2f} &
        {np.min(lsms_df[target_var]):.2f} &
        {np.max(lsms_df[target_var]):.2f} &
        {np.std(lsms_df[target_var]):.2f} &
        {len(lsms_df[target_var])} \\\\'''
    output = output.replace('\n', '').replace('\t', '').replace('    ', ' ')
    print('\t\\hspace{0.5cm} ' + var_name + ' & ' + output)

def describe_var(lsms_df, var, var_name="w"):
    n = len(lsms_df)
    if len(lsms_df.columns) == 1:
        c = n
    else:
        c = len(np.unique(lsms_df.cluster_id))

    if n != c:
        SSW, SSB, SST = decomp_SST(lsms_df, 'cluster_id', var, verbose=False)
        sd_b = np.round(np.sqrt(SSB / (c - 1)),2)
        sd_w = np.round(np.sqrt(SSW / (n - c)),2)
        sd = np.sqrt(SST / (n - 1))

    else:
        sd_b = " "
        sd_w = " "
        sd = np.std(lsms_df[var])

    output = f''' 
        {np.mean(lsms_df[var]):.2f} & 
        {np.min(lsms_df[var]):.2f} & 
        {np.max(lsms_df[var]):.2f} & 
        {sd_b} & 
        {sd_w} & 
        {sd:.2f} & 
        {len(lsms_df[var])} \\\\'''

    output = output.replace('\n', '').replace('\t', '').replace('    ', '')
    print('\t\\hspace{0.5cm}' + var_name + ' &' + output)

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of the Earth in kilometers (mean value)
    earth_radius = 6371.0

    # Calculate the distance
    distance = earth_radius * c
    return distance