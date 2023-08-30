import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

from torch.utils.data import DataLoader
from .SatDataset import SatDataset
import torchvision


def reduce_dimensions(extracted_feats, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(extracted_feats)

    # Get the explained variance ratios of the first 100 components
    explained_variance_ratios = pca.explained_variance_ratio_
    total_variance_explained = np.sum(explained_variance_ratios)

    print(f"\tTotal variance explained by first {n_components} components: {total_variance_explained:.4f}")

    return X_reduced


class FeatureExtractor:
    def __init__(self, model, device):
        self.model = copy.deepcopy(model)
        self.device = device

        # replace the last layer in the network with an identity to directly output the penultimate layer.
        self.model.fc = nn.Identity()

    def extract_feats(self, dat_loader, reduced=True, n_components=50):
        print('\tExtracting Features')
        self.model.to(self.device)
        self.model.eval()  # initialise validation mode
        extracted_feats = []
        with torch.no_grad():  # disable gradient tracking
            for x, _ in dat_loader:
                # forward pass
                feats = self.model(x.to(self.device))
                extracted_feats.append(feats.cpu().numpy())
        extracted_feats = np.concatenate(extracted_feats, axis=0)
        if reduced:
            return reduce_dimensions(extracted_feats, n_components)
        else:
            return np.array(extracted_feats)
