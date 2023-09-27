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


def get_dataloaders(cv_object, train_df, val_df, batch_size):
    # only take the normalisation from the feature transform
    feat_transform = cv_object.feat_transform_val_test

    # initialise the Landsat data
    dat_train = SatDataset(train_df,
                           cv_object.img_dir,
                           cv_object.data_type,
                           cv_object.target_var,
                           cv_object.id_var,
                           feat_transform=feat_transform,
                           target_transform=None)  # no need to transform target variable

    dat_val = SatDataset(val_df,
                         cv_object.img_dir,
                         cv_object.data_type,
                         cv_object.target_var,
                         cv_object.id_var,
                         feat_transform=feat_transform,
                         target_transform=None)  # no need to transform target variable

    train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dat_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class BetweenModel:
    def __init__(self,
                 LS_cv_pth,
                 RS_cv_pth,
                 lsms_df,
                 target_var,
                 x_vars,
                 fold_ids,
                 device,
                 random_seed=None):

        self.cv_ls = load_cv_object(LS_cv_pth)
        self.cv_rs = load_cv_object(RS_cv_pth)
        self.id_var = self.cv_ls.id_var
        self.lsms_df = lsms_df
        self.target_var = target_var
        self.x_vars = x_vars
        self.fold_ids = fold_ids
        self.device = device
        self.random_seed = random_seed

        # initialise the objects to store the results
        self.res_r2 = {'train': [], 'val': []}
        self.res_mse = {'train': [], 'val': []}
        self.predictions = {self.id_var: [], 'y': [], 'y_hat': []}
        self.models = {}
        self.feat_names = []

    def train(self, min_samples_leaf=10, n_components=50):
        print('Initialising training')
        start_time = time.time()

        for fold, splits in tqdm(self.fold_ids.items(), total=len(self.fold_ids)):
            print('-'*50)
            print(f"Training and Evaluating on fold {fold}")
            # set the random seed for each fold
            if self.random_seed is not None:
                np.random.seed(self.random_seed + fold)
                torch.manual_seed(self.random_seed + fold)

            # load the trained LS model on that fold
            ls_state_dict = self.cv_ls.best_model_paths[fold]
            ls_model = self.cv_ls.model_class.model
            ls_model.load_state_dict(torch.load(ls_state_dict, map_location=self.device))

            # load the trained RS model on that fold
            rs_state_dict = self.cv_rs.best_model_paths[fold]
            rs_model = self.cv_rs.model_class.model
            rs_model.load_state_dict(torch.load(rs_state_dict, map_location=self.device))

            # get the training and validation data for this fold
            train_df, val_df = split_lsms_ids(lsms_df=self.lsms_df, val_ids=splits['val_ids'])

            # get the train and val loader for the LS and RS images
            ls_train_loader, ls_val_loader = get_dataloaders(self.cv_ls, train_df, val_df, 128)
            rs_train_loader, rs_val_loader = get_dataloaders(self.cv_rs, train_df, val_df, 128)

            # extract features
            ls_feat_extractor = FeatureExtractor(ls_model, self.device)
            rs_feat_extractor = FeatureExtractor(rs_model, self.device)

            print("\tLandsat Feature Extraction - Train")
            ls_train_feats = ls_feat_extractor.extract_feats(ls_train_loader, reduced=True,
                                                             n_components=n_components)
            print("\tLandsat Feature Extraction - Val")
            ls_val_feats = ls_feat_extractor.extract_feats(ls_val_loader, reduced=True,
                                                           n_components=n_components)

            print("\tRS Feature Extraction - Train")
            rs_train_feats = rs_feat_extractor.extract_feats(rs_train_loader, reduced=True,
                                                             n_components=n_components)
            print("\tRS Feature Extraction - Val")
            rs_val_feats = rs_feat_extractor.extract_feats(rs_val_loader, reduced=True,
                                                           n_components=n_components)

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

            # initialise the Random Froest trainer
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
            self.res_r2['train'].append(forest_trainer.r2['train'])
            self.res_mse['train'].append(forest_trainer.mse['train'])
            self.res_r2['val'].append(forest_trainer.r2['val'])
            self.res_mse['val'].append(forest_trainer.mse['val'])

        end_time = time.time()
        time_elapsed = np.round(end_time - start_time, 0).astype(int)
        print(f"Finished training after {time_elapsed} seconds")


    def get_fold_weights(self):
        '''
        Fold weights differ when running the delta or demeaned model as compared to the between model
        In the between models, the fold weights are only defined by the number of clusters in each fold
        In the within model, fold weights are defined by the number of observations in each fold
        :return:
        '''
        n = len(self.lsms_df)
        val_weights = []
        for split in self.fold_ids.values():
            # subset the lsms df to the clusters in the validation split
            val_cids = split['val_ids']
            mask = self.lsms_df.cluster_id.isin(val_cids)
            sub_df = self.lsms_df[mask]
            val_weights.append(len(sub_df) / n)

        train_weights = [1 - w for w in val_weights]
        train_weights = [w / sum(train_weights) for w in train_weights]

        return val_weights, train_weights

    def compute_overall_performance(self, use_fold_weights=True):
        if use_fold_weights:
            # compute the fold weights
            val_fold_weights, train_fold_weights = self.get_fold_weights()
            # compute the weighted average of the performance metrics
            train_r2 = np.average(self.r2['train'], weights=train_fold_weights)
            train_mse = np.average(self.mse['train'], weights=train_fold_weights)
            val_r2 = np.average(self.r2['val'], weights=val_fold_weights)
            val_mse = np.average(self.mse['val'], weights=val_fold_weights)
        else:
            train_r2 = np.mean(self.r2['train'])
            train_mse = np.mean(self.mse['train'])
            val_r2 = np.mean(self.r2['val'])
            val_mse = np.mean(self.mse['val'])
        performance = {'train_r2': train_r2, 'train_mse': train_mse, 'val_r2': val_r2, 'val_mse': val_mse}
        return performance

    def get_feature_importance(self):
        feat_imp = []
        for fold in range(len(self.fold_ids.keys())):
            feat_imp.append(self.models[fold].feature_importances_)

        mean_feat_imp = pd.DataFrame({
            'variable_name': self.models[0].feature_names_in_,
            'feat_importance': np.mean(np.vstack(feat_imp).T, axis=1)
        })

        mean_feat_imp = mean_feat_imp.sort_values(by='feat_importance', ascending=True)
        return mean_feat_imp

    def plot_feature_importance(self, fname=None, varnames=None):
        feat_imp = self.get_feature_importance()
        if not varnames:
            varnames = feat_imp['variable_name']
        fig, ax = plt.subplots(figsize=(8, 10))
        plt.barh(y=varnames, width=feat_imp['feat_importance'], height=0.8)
        ax.set_xlabel("Mean Relative Feature Importance")
        if fname is not None:
            pth = f"../figures/results/{fname}"
            plt.savefig(pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

    def save_object(self, name):
        folder = f'results/model_objects'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pth = f"{folder}/{name}.pkl"

        aux = copy.deepcopy(self)
        # move all models to cpu
        aux.cv_ls.model_class.model.to('cpu')
        aux.cv_rs.model_class.model.to('cpu')

        # aux.feat_transform = None
        with open(pth, 'wb') as f:
            pickle.dump(aux, f)
