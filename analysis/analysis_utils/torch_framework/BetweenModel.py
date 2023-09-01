import os
import time

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .FeatureExtractor import FeatureExtractor
from .SatDataset import SatDataset
from .torch_helpers import *
from .. import RandomForest as rf
from ..spatial_CV import *


def load_cv_object(pth):
    with open(pth, 'rb') as f:
        return pickle.load(f)


class BetweenModel:
    def __init__(self, LS_cv_pth, RS_cv_pth, df, target_var, x_vars, fold_ids, device, random_seed = None):
        self.cv_ls = load_cv_object(LS_cv_pth)
        self.cv_rs = load_cv_object(RS_cv_pth)
        self.id_var = self.cv_ls.id_var
        self.target_var = target_var
        self.x_vars = x_vars
        self.df = df
        self.fold_ids = fold_ids
        self.device = device
        self.random_seed = random_seed

        # get the target transform
        #self.target_transform = self.get_target_transform()

        # initialise the objects to store the results
        self.r2 = {'train': [], 'val': []}
        self.mse = {'train': [], 'val': []}
        self.predictions = {self.id_var: [], 'y': [], 'y_hat': []}
        self.models = {}
        self.feat_names = []

    def train(self, min_samples_leaf = 10, n_components = 50):
        print('Initialising training')
        start_time = time.time()

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        for fold, splits in tqdm(self.fold_ids.items(), total = len(self.fold_ids)):
            # load the best landsat and rs models
            ls_state_dict = self.cv_ls.best_model_paths[fold]
            ls_model = self.cv_ls.model
            ls_model.load_state_dict(torch.load(ls_state_dict, map_location=self.device))

            rs_state_dict = self.cv_rs.best_model_paths[fold]
            rs_model = self.cv_rs.model
            rs_model.load_state_dict(torch.load(rs_state_dict, map_location=self.device))

            # get the training and validation data for this fold
            train_df, val_df = split_lsms_ids(lsms_df=self.df, val_ids=splits['val_ids'])

            # get the train and val loader for the LS and RS images
            ls_train_loader, ls_val_loader = self.get_dataloaders(self.cv_ls, train_df, val_df, 128)
            rs_train_loader, rs_val_loader = self.get_dataloaders(self.cv_rs, train_df, val_df, 128)

            # extract features
            ls_feat_extractor = FeatureExtractor(ls_model, self.device)
            rs_feat_extractor = FeatureExtractor(rs_model, self.device)

            print("Landsat Feature Extraction - Train, Val")
            ls_train_feats = ls_feat_extractor.extract_feats(ls_train_loader, reduced=True, n_components=n_components)
            ls_val_feats = ls_feat_extractor.extract_feats(ls_val_loader, reduced=True, n_components=n_components)

            print("RS Feature Extraction - Train, Val")
            rs_train_feats = rs_feat_extractor.extract_feats(rs_train_loader, reduced=True, n_components=n_components)
            rs_val_feats = rs_feat_extractor.extract_feats(rs_val_loader, reduced=True, n_components=n_components)
            print("\n")

            # concatenate the rs feats, the ls feats, the OSM feats and the precip feats
            X_train = np.concatenate([ls_train_feats, rs_train_feats, train_df[self.x_vars].values], axis=1)
            X_val = np.concatenate([ls_val_feats, rs_val_feats, val_df[self.x_vars].values], axis=1)

            # store the feature names
            ls_feat_names = ["ls_feat_" + str(i) for i in range(ls_train_feats.shape[1])]
            rs_feat_names = ["rs_feat_" + str(i) for i in range(rs_train_feats.shape[1])]
            self.feat_names = ls_feat_names + rs_feat_names + self.x_vars

            y_train = train_df[self.target_var].values
            y_val = val_df[self.target_var].values

            # train the between model on the concatenated features (Random Forest)
            if self.random_seed is not None:
                random_seed = self.random_seed + fold
            forest_trainer = rf.Trainer(X_train, y_train, X_val, y_val, random_seed)
            forest_trainer.train(min_samples_leaf=min_samples_leaf)
            forest_trainer.validate()

            # store the trained model
            self.models[fold] = forest_trainer.model

            # store the models predictions
            self.predictions[self.id_var] += list(val_df[self.id_var])
            self.predictions['y'] += list(y_val)
            self.predictions['y_hat'] += list(forest_trainer.y_hat_val)

            # store the models results
            self.r2['train'].append(forest_trainer.r2['train'])
            self.mse['train'].append(forest_trainer.mse['train'])
            self.r2['val'].append(forest_trainer.r2['val'])
            self.mse['val'].append(forest_trainer.mse['val'])

        end_time = time.time()
        time_elapsed = np.round(end_time - start_time, 0).astype(int)
        print(f"Finished training after {time_elapsed} seconds")

    # def get_target_transform(self):
    #     # get the target transform:
    #     # get the stats for the target variable
    #     target_stats = get_target_stats(self.df, self.target_var)
    #
    #     # get the data transforms for the target --> is used in the DataLoader object
    #     target_transform = transforms.Compose([
    #         torchvision.transforms.Lambda(
    #             lambda t: standardise(t, target_stats['mean'], target_stats['std'])),
    #     ])
    #
    #     return target_transform

    def get_dataloaders(self, cv_object, train_df, val_df, batch_size):
        # only take the normalisation from the feature transforms
        feat_transform = torchvision.transforms.Compose([cv_object.feat_transform.transforms[-1]])

        # initialise the Landsat data
        dat_train = SatDataset(train_df, cv_object.img_dir, cv_object.data_type, cv_object.target_var, cv_object.id_var,
                               feat_transform = feat_transform, target_transform = None)
        dat_val = SatDataset(val_df, cv_object.img_dir, cv_object.data_type, cv_object.target_var, cv_object.id_var,
                             feat_transform = feat_transform, target_transform = None)

        train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dat_val, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def compute_overall_performance(self, use_fold_weights = True):
        if use_fold_weights:
            fold_weights = [len(v['val_ids'])/(len(v['val_ids']) + len(v['train_ids'])) for v in self.fold_ids.values()]
            r2 = np.average(self.r2['val'], weights = fold_weights)
            mse = np.average(self.mse['val'], weights = fold_weights)
            return {'r2':r2, 'mse':mse}
        else:
            r2 = np.mean(self.r2['val'])
            mse = np.mean(self.mse['val'])
            return {'r2':r2, 'mse':mse}

    def get_feature_importance(self):
        # add feature importance
        feat_importance = self.models[0].feature_importances_
        for fold in self.models.keys():
            if fold > 0:
                feat_importance = np.vstack([feat_importance, self.models[fold].feature_importances_])
        feat_importance = np.mean(feat_importance, axis=0)
        importance_df = pd.DataFrame({'importance':feat_importance, 'feature':self.feat_names})
        importance_df = importance_df.sort_values(by='importance', ascending=True)
        return importance_df

    def plot_feature_importance(self, variable_labels:dict = None):
        feat_imp = self.get_feature_importance()
        if variable_labels is not None:
            feat_imp['feature'] = [variable_labels[f] for f in self.feat_names]
        plt.figure(figsize=(7, 7))
        plt.barh(feat_imp['feature'], feat_imp['importance'])
        plt.ylabel("Feature Importance")
        plt.xlabel("Relative Feature Importance")
        plt.show()

    def save_object(self, name):
        folder = f'../results/model_objects'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pth = f"{folder}/{name}.pkl"
        with open(pth, 'wb') as f:
            pickle.dump(self, f)

