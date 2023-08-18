import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def reduce_dims(dat, vars, n_comp=5, col_prefix=None):
    # scale the variables
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dat[vars])

    # Create an instance of PCA
    pca = PCA(n_components=n_comp)

    # Perform PCA on the standardized data
    pca.fit(scaled_df)

    # Print the explained variance ratio
    print(f"Number of variables: {len(vars)}")
    print(f"Number of components: {n_comp}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

    # transform the data
    column_names = [f"{col_prefix}pc_{i + 1}" for i in range(n_comp)]
    transformed_df = pd.DataFrame(pca.transform(scaled_df), columns=column_names)
    transformed_df['unique_id'] = dat['unique_id']
    return transformed_df
