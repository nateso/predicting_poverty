
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# import the k-fold cross validation functions
from .spatial_CV import *


# define the Trainer class
class Trainer():
    def __init__(self, X_train, y_train, X_val, y_val, random_seed = None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_hat_train = None

        self.X_val = X_val
        self.y_val = y_val
        self.y_hat_val = None

        self.random_seed = random_seed
        self.model = None

        self.r2 = {'train':None, 'val':None}
        self.mse = {'train': None, 'val': None}

    def train(self, min_samples_leaf = 5):
        forest = RandomForestRegressor(n_estimators = 3000,
                                       min_samples_leaf = min_samples_leaf,
                                       max_features = 'sqrt',
                                       random_state = self.random_seed)

        forest.fit(self.X_train, self.y_train)

        # get the training performance
        self.y_hat_train = forest.predict(self.X_train)
        self.r2['train'] = forest.score(self.X_train, self.y_train)
        self.mse['train'] = mean_squared_error(self.y_train, self.y_hat_train)

        # store the model the the Trainer object
        self.model = forest

    def validate(self):
        # validate the model
        self.y_hat_val = self.model.predict(self.X_val)
        self.r2['val'] = self.model.score(self.X_val, self.y_val)
        self.mse['val'] = mean_squared_error(self.y_val, self.y_hat_val)

    def get_feature_importance(self):
        # add feature importance
        importance = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        importance_df = pd.DataFrame({'importance':importance, 'std':std}, index = self.x_vars)
        return importance_df

    def plot_feature_importance(self):
        fig, ax = plt.subplots()
        feat_imp = self.get_feature_importance()
        feat_imp.plot.bar(y = 'importance', yerr = 'std', ax=ax)
        ax.set_xlab("Feature")
        ax.set_ylab("Feature Importance")
        plt.show()


class CrossValidator():
    def __init__(self, lsms_df, fold_ids, target_var, x_vars, id_var = 'unique_id', random_seed = None):
        self.lsms_df = lsms_df
        self.fold_ids = fold_ids
        self.target_var = target_var
        self.id_var = id_var
        self.x_vars = x_vars
        self.random_seed = random_seed

        self.r2 = {'train': [], 'val': []}
        self.mse = {'train': [], 'val': []}
        self.predictions = {id_var: [], 'y': [], 'y_hat': []}
        self.models = {}

    def run_cv_training(self, DL_feats = False, dl_feat_dir = None, min_samples_leaf = 5):
        print('Initialising training')
        start_time = time.time()
        for fold, splits  in tqdm(self.fold_ids.items(), total = len(self.fold_ids)):
            if DL_feats: # TODO: implement this
                # # load the Extracted Features for the fold, training df and target
                # if 'asset' in self.target_var: target_type = 'asset'
                # else: target_type = 'cons'
                # pth = f"{dl_feat_dir}/feats_fold_{fold}_{self.lsms_df.name[0]}_{target_type}.csv"
                #
                # dl_feats = pd.read_csv(pth)
                #
                # # subset to those observations that are in the lsms data frame
                # dl_feats = dl_feats[dl_feats[self.id_var].isin(self.lsms_df[self.id_var])].reset_index(drop = True)
                # ids = dl_feats[self.id_var].copy()
                # # normalise the dl_features
                # dl_feats = (dl_feats - dl_feats.mean(numeric_only = True))/dl_feats.std(numeric_only = True)
                # dl_feats[self.id_var] = ids
                #
                # # merge to the lsms data
                # dat = pd.merge(self.lsms_df, dl_feats, on = self.id_var, how = 'left')
                print("not yet implemented")
            else:
                dat = self.lsms_df

            # split actual data into training and validation sample based on the k-fold split
            train_df, val_df = split_lsms_ids(lsms_df = dat, val_ids = splits['val_ids'])

            # split data into X and y
            X_train, y_train = train_df.loc[:,self.x_vars], np.array(train_df[[self.target_var]]).ravel()
            X_val, y_val = val_df.loc[:,self.x_vars], np.array(val_df[[self.target_var]]).ravel()

            # run the training
            if self.random_seed is not None:
                random_seed = self.random_seed + fold
            forest_trainer = Trainer(X_train, y_train, X_val, y_val, random_seed)
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

    def plot_true_vs_preds(self):
        ## plots the true observations vs the predicted observations
        y = np.array(self.predictions['y'])
        y_hat = np.array(self.predictions['y_hat'])
        plt.figure(figsize = (5,5))
        plt.scatter(y, y_hat, s = 2)
        plt.plot([min(y), max(y)], [min(y_hat), max(y_hat)], color='red', linestyle='--')  # Line of perfect correlation
        plt.xlabel('True outcome values')
        plt.ylabel('Predicted outcome values')
        plt.title('True vs Predicted Values')
        plt.show()



class CV_Evaluator():
    def __init__(self, lsms_df, fold_ids, cv_trainer, id_var):
        self.lsms_df = lsms_df
        self.fold_ids = fold_ids
        self.id_var = id_var

        self.cv_trainer = cv_trainer
        self.predictions = {self.id_var:[], 'y':[], 'y_hat':[]}

        self.cv_r2 = []
        self.cv_mse = []

    def evaluate(self, DL_feats = False, dl_feat_dir = None):
        for fold, splits in self.fold_ids.items():
              x_vars = self.cv_trainer.x_vars
              target_var = self.cv_trainer.target_var

              if DL_feats: #TODO: implement this
                # # load the Extracted Features for the fold, training df and target
                # if 'asset' in target_var: target_type = 'asset'
                # else: target_type = 'cons'
                # pth = f"{dl_feat_dir}/feats_fold_{fold}_{train_df_type}_{target_type}.csv"
                #
                # dl_feats = pd.read_csv(pth)
                #
                # # subset to those observations that are in the lsms data frame
                # dl_feats = dl_feats[dl_feats['unique_id'].isin(self.lsms_df.unique_id)].reset_index(drop = True)
                # unique_ids = dl_feats['unique_id'].copy()
                # # normalise the dl_features
                # dl_feats = (dl_feats - dl_feats.mean(numeric_only = True))/dl_feats.std(numeric_only = True)
                # dl_feats['unique_id'] = unique_ids
                #
                # # merge to the lsms data
                # dat = pd.merge(self.lsms_df, dl_feats, on = 'unique_id', how = 'left')
                print("not yet implemented")
              else:
                  dat = self.lsms_df

              _, val_df = split_lsms_ids(lsms_df = dat, val_ids = splits['val_ids'])
              X_val, y_val = val_df.loc[:, x_vars], np.array(val_df[[target_var]]).ravel()

              model = self.cv_trainer.models[fold]
              y_hat_val = model.predict(X_val)

              # store the models predictions
              self.predictions[self.id_var] += list(val_df[self.id_var])
              self.predictions['y'] += list(y_val)
              self.predictions['y_hat'] += list(y_hat_val)

              self.cv_r2.append(model.score(X_val, y_val))
              self.cv_mse.append(mean_squared_error(y_val, y_hat_val))

    def compute_overall_performance(self, use_fold_weights = True):
        if use_fold_weights:
            fold_weights = [len(v['val_ids'])/(len(v['val_ids']) + len(v['train_ids'])) for v in self.fold_ids.values()]
            r2 = np.average(self.cv_r2, weights = fold_weights)
            mse = np.average(self.cv_mse, weights = fold_weights)
            return {'r2':r2, 'mse':mse}
        else:
            r2 = np.mean(self.cv_r2)
            mse = np.mean(self.cv_mse)
            return {'r2':r2, 'mse':mse}





