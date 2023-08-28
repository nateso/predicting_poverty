import torch
import os
import numpy as np
from torch.utils.data import Dataset


class SatDataset(Dataset):
    def __init__(self, labels_df, img_dir, data_type, target_var,
                 id_var, feat_transform=None, target_transform=None):
        self.img_labels = labels_df
        self.img_dir = img_dir
        self.id_var = id_var
        self.data_type = data_type
        self.target_var = target_var
        self.feat_transform = feat_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file_name = f"{self.data_type}_{self.img_labels[self.id_var][idx]}.npy"
        img_path = os.path.join(self.img_dir, file_name)

        # load the image data
        image = torch.from_numpy(np.load(img_path).astype(np.float32).transpose(2, 1, 0))

        # load the label data
        label = torch.from_numpy(np.array(self.img_labels[self.target_var][idx], dtype=np.float32))
        if self.feat_transform:
            image = self.feat_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
