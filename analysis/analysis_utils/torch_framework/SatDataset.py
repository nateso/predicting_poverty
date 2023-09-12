import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SatDataset(Dataset):
    def __init__(self,
                 labels_df,
                 img_dir,
                 data_type,
                 target_var,
                 id_var,
                 feat_transform=None,
                 target_transform=None,
                 random_seed=None):

        self.img_labels = labels_df
        self.img_dir = img_dir
        self.id_var = id_var
        self.data_type = data_type
        self.target_var = target_var
        self.feat_transform = feat_transform
        self.target_transform = target_transform
        self.random_seed = random_seed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file_name = f"{self.data_type}_{self.img_labels[self.id_var][idx]}.npy"
        img_path = os.path.join(self.img_dir, file_name)

        # load the image data
        image = torch.from_numpy(np.load(img_path).astype(np.float32).transpose(2, 1, 0))

        # apply the feature transform
        if self.feat_transform:
            if self.random_seed:
                torch.manual_seed(self.random_seed)
                np.random.seed(self.random_seed)
            image = self.feat_transform(image)

        # load the label data
        label = torch.from_numpy(np.array(self.img_labels[self.target_var][idx], dtype=np.float32))

        # apply the label transform
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
