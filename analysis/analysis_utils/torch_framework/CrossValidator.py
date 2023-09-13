import copy
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .ParamTuner import ParamTuner
from .SatDataset import SatDataset
from .Evaluator import Evaluator
from .Trainer import Trainer

from ..spatial_CV import split_lsms_ids, split_lsms_spatial


class CrossValidator():
    def __init__(self,
                 model_class,
                 lsms_df,
                 fold_ids,
                 img_dir,
                 data_type,
                 target_var,
                 id_var,
                 feat_transform,
                 target_transform,
                 device,
                 model_name=None,
                 random_seed=None):

        self.model_class = copy.deepcopy(model_class)

        self.lsms_df = lsms_df
        self.fold_ids = fold_ids

        self.img_dir = img_dir
        self.data_type = data_type
        self.target_var = target_var
        self.id_var = id_var
        self.feat_transform_train = feat_transform
        self.feat_transform_val_test = torchvision.transforms.Compose(
            [feat_transform.transforms[-1]]  # avoids the random rotation and flipping on the test set
        )
        self.target_transform = target_transform
        self.device = device
        self.random_seed = random_seed
        self.model_name = model_name

        self.res_mse = {'train': [], 'val': []}
        self.res_r2 = {'train': [], 'val': []}
        self.predictions = {id_var: [], 'y': [], 'y_hat': []}
        self.best_model_paths = []

        # get the fold weights (depending on the size of each fold)
        self.train_fold_weights = [len(v['train_ids']) / (len(v['val_ids']) + len(v['train_ids'])) for v in
                                   fold_ids.values()]
        self.val_fold_weights = [len(v['val_ids']) / (len(v['val_ids']) + len(v['train_ids'])) for v in
                                 fold_ids.values()]

    def run_cv(self, hyper_params, tune_hyper_params=False):
        start_time = time.time()

        for fold, split in tqdm(self.fold_ids.items()):
            print('\n')
            print(
                '=====================================================================================================')
            print(f"Training on fold {fold}")

            if self.model_name is not None:
                model_fold_name = f"{self.model_name}_f{fold}"

            if self.random_seed is None:
                fold_seed = None
            else:
                fold_seed = self.random_seed + fold
                np.random.seed(fold_seed)
                torch.manual_seed(fold_seed)

            # prepare the training data
            train_df, val_df, test_df = self.split_data_train_val_test(split['val_ids'])
            train_loader, val_loader, test_loader = self.get_dataloaders(train_df, val_df, test_df,
                                                                         batch_size=hyper_params['batch_size'][0])

            # if wished for, tune the hyper-parameters
            if tune_hyper_params:
                # initialise the param tuner
                param_tuner = ParamTuner(model_class=self.model_class,
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         hyper_params=hyper_params,
                                         device=self.device,
                                         random_seed=fold_seed)

                # tune the hyper-parameters
                param_tuner.grid_search()

                # get the best hyper-parameters
                best_params = param_tuner.best_params
            else:
                # convert all the hyper-parameters to values using list comprehension
                best_params = {k: v[0] if isinstance(v, list) else v for k, v in hyper_params.items()}

            # train the model using the best hyper-parameters
            print("\nTrain the model using the best hyper-parameters")

            # split data into training and test fold
            train_df, test_df = split_lsms_ids(self.lsms_df, val_ids=split['val_ids'])
            train_loader, test_loader = self.get_dataloaders(train_df, test_df, batch_size=best_params['batch_size'])

            # reset the model weights
            self.model_class.reset_weights(random_seed=fold_seed)

            # train the model
            self.train_fold(train_loader, best_params, model_fold_name)

            # evaluate the model on the test set (using the just trained model)
            self.evaluate_fold(self.best_model_paths[fold], test_loader, split)

            # print the results of the fold
            print(f"\nResults of fold {fold}:")
            print(f"\tTrain MSE: {self.res_mse['train'][-1]}, Val MSE: {self.res_mse['val'][-1]}")
            print(f"\tTrain R2: {self.res_r2['train'][-1]}, Val R2: {self.res_r2['val'][-1]}")

        end_time = time.time()
        time_elapsed = np.round(end_time - start_time, 0).astype(int)
        print("=" * 100)
        print(f"\nFinished Cross-validation after {time_elapsed} seconds")

    def train_fold(self, train_loader, params, model_fold_name=None):
        # train model
        loss_fn = nn.MSELoss()
        optimiser = optim.Adam(self.model_class.model.parameters(),
                               lr=params['lr'],
                               weight_decay=params['alpha'])
        scheduler = optim.lr_scheduler.StepLR(optimiser,
                                              step_size=params['step_size'],
                                              gamma=params['gamma'])

        trainer = Trainer(model=self.model_class.model,
                          train_loader=train_loader,
                          val_loader=None,
                          optimiser=optimiser,
                          loss_fn=loss_fn,
                          device=self.device,
                          scheduler=scheduler,
                          early_stopper=None,
                          model_folder=self.model_name,
                          model_name=model_fold_name)

        trainer.run_training(params['n_epochs'])

        # append the model results to the list of results (the results of the last epoch)
        self.res_mse['train'].append(trainer.mse['train'][-1])
        self.res_r2['train'].append(trainer.r2['train'][-1])

        # append the model path to the list of best model paths
        self.best_model_paths.append(trainer.best_model_path)

    def evaluate_fold(self, model_pth, test_loader, split):

        # use the best model to initialise the evaluator on the test set
        evaluator = Evaluator(model=self.model_class.model,
                              state_dict_pth=model_pth,
                              test_loader=test_loader,
                              device=self.device)
        evaluator.predict()

        # save the predictions
        self.predictions[self.id_var] += split['val_ids']
        self.predictions['y'] += evaluator.predictions['y']
        self.predictions['y_hat'] += evaluator.predictions['y_hat']

        # store the test results (i.e. the results on that fold)
        self.res_mse['val'].append(evaluator.calc_mse())
        self.res_r2['val'].append(evaluator.calc_r2())

    def compute_overall_performance(self, use_fold_weights=True):
        if use_fold_weights:
            train_r2 = np.average(self.res_r2['train'], weights=self.val_fold_weights)
            train_mse = np.average(self.res_mse['train'], weights=self.val_fold_weights)
            val_r2 = np.average(self.res_r2['val'], weights=self.val_fold_weights)
            val_mse = np.average(self.res_mse['val'], weights=self.val_fold_weights)
        else:
            train_r2 = np.mean(self.res_r2['train'])
            train_mse = np.mean(self.res_mse['train'])
            val_r2 = np.mean(self.res_r2['val'])
            val_mse = np.mean(self.res_mse['val'])
        performance = {'train_r2': train_r2, 'train_mse': train_mse, 'val_r2': val_r2, 'val_mse': val_mse}
        return performance

    def split_data_train_val_test(self, val_ids):
        # split the data into training and test dataframes
        train_df, test_df = split_lsms_ids(self.lsms_df, val_ids=val_ids)

        # split the training data further into a training and a validation set for hyper-parameter tuning
        # for now just do hyper-parameter tuning using a simple validation set approach
        train_val_folds = split_lsms_spatial(train_df,
                                             test_ratio=0.2,
                                             random_seed=self.random_seed,
                                             verbose=False)
        train_df, val_df = split_lsms_ids(train_df,
                                          val_ids=train_val_folds[0][
                                              'val_ids'])  # there is only one fold in train_val_folds
        return train_df, val_df, test_df

    def get_dataloaders(self, train_df, val_df, test_df=None, batch_size=128):
        # initialise the datasets
        dat_train = SatDataset(
            labels_df=train_df,
            img_dir=self.img_dir,
            data_type=self.data_type,
            target_var=self.target_var,
            id_var=self.id_var,
            feat_transform=self.feat_transform_train,
            target_transform=self.target_transform,
            random_seed=self.random_seed
        )

        dat_val = SatDataset(
            labels_df=val_df,
            img_dir=self.img_dir,
            data_type=self.data_type,
            target_var=self.target_var,
            id_var=self.id_var,
            feat_transform=self.feat_transform_val_test,
            target_transform=self.target_transform,
            random_seed=None
        )

        # initialise the data loader objects
        if self.random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_seed)
            train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True, generator=generator)
        else:
            train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True)

        # validation data loader
        val_loader = DataLoader(dat_val, batch_size=batch_size, shuffle=False)

        if test_df is not None:
            dat_test = SatDataset(test_df, self.img_dir, self.data_type, self.target_var, self.id_var,
                                  self.feat_transform_val_test, self.target_transform)
            test_loader = DataLoader(dat_test, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader

    def plot_true_vs_preds(self, xlabel="Predicted outcome values", ylabel='True outcome values'):
        # plots the true observations vs the predicted observations
        y_hat = np.array(self.predictions['y_hat'])
        y = np.array(self.predictions['y'])
        plt.figure(figsize=(5, 5))
        plt.scatter(y, y_hat)
        plt.plot([min(y), max(y)], [min(y_hat), max(y_hat)], color='red', linestyle='--')  # Line of perfect correlation
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('True vs Predicted Values')
        plt.show()

    def save_object(self, name):
        folder = f'results/model_objects'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pth = f"{folder}/{name}.pkl"
        with open(pth, 'wb') as f:
            aux = copy.deepcopy(self)
            aux.target_transform = None  # remove the target transforms as it cannot be saved as pickle
            # aux.feat_transform = None
            pickle.dump(aux, f)
