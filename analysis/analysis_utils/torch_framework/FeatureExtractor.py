import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from tqdm.auto import tqdm


def reduce_dimensions(extracted_feats, n_components):
    pca = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca.fit_transform(extracted_feats)

    # normalise the reduced features
    X_reduced = (X_reduced - np.mean(X_reduced, axis = 0)) / np.std(X_reduced, axis = 0)

    # Get the explained variance ratios of the first 100 components
    explained_variance_ratios = pca.explained_variance_ratio_
    total_variance_explained = np.sum(explained_variance_ratios)

    print(f"\t\tTotal variance explained by first {n_components} components: {total_variance_explained:.4f}")

    return X_reduced


class FeatureExtractor:
    def __init__(self, model, device):
        self.model = copy.deepcopy(model)
        self.device = device

        # replace the last layer in the network with an identity to directly output the penultimate layer.
        self.model.fc = nn.Identity()

    def extract_feats(self, dat_loader, reduced=True, n_components=25):
        print('\t\tExtracting Features')
        self.model.to(self.device)
        self.model.eval()  # initialise validation mode
        extracted_feats = []
        with torch.no_grad():  # disable gradient tracking
            for x, _ in tqdm(dat_loader):
                # forward pass
                feats = self.model(x.to(self.device))
                extracted_feats.append(feats.cpu().numpy())
        extracted_feats = np.concatenate(extracted_feats, axis=0)
        if reduced:
            return reduce_dimensions(extracted_feats, n_components)
        else:
            return np.array(extracted_feats)
