import os
import time

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .SatDataset import SatDataset
from .Trainer import Evaluator
from .Trainer import Trainer
from .torch_helpers import *
from ..analysis_utils.spatial_CV import *


class CrossValidator():
    def __init__(self, model, lsms_df, fold_ids, img_dir,
                 data_type,
                 target_var,
                 feat_transform,
                 target_transform,
                 device,
                 model_name=None,
                 random_seed=None):

        self.model = model
        self.orig_state_dict = copy.deepcopy(model.state_dict())

        self.lsms_df = lsms_df
        self.fold_ids = fold_ids

        self.img_dir = img_dir
        self.data_type = data_type
        self.target_var = target_var
        self.feat_transform = feat_transform
        self.target_transform = target_transform
        self.device = device
        self.random_seed = random_seed
        self.model_name = model_name

        self.training_r2 = {'train': [], 'val': []}
        self.training_mse = {'train': [], 'val': []}
        self.cv_r2 = []
        self.cv_mse = []
        self.predictions = {'unique_id': [], 'y': [], 'y_hat': []}
        self.best_model_paths = []

    def run_cv(self, hyper_params, tune_params=False):
        start_time = time.time()

        for fold, split in tqdm(self.fold_ids.items()):
            print(f"\nTraining on fold {fold}")
            if self.model_name is not None:
                model_fold_name = f"{self.model_name}_f{fold}"

            if self.random_seed is not None:
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

            # prepare the data
            train_df, val_df, test_df = self.split_data_train_val_test(self.lsms_df, split['val_ids'])
            train_loader, val_loader, test_loader = self.get_dataloaders(train_df, val_df, test_df,
                                                                         batch_size=hyper_params['batch_size'])
            # print(val_df.cluster_id)
            # initialise the weights of the model
            self.model.load_state_dict(self.orig_state_dict)

            # train model
            loss_fn = nn.MSELoss()
            optimiser = optim.Adam(self.model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['alpha'])
            scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=hyper_params['step_size'],
                                                  gamma=hyper_params['gamma'])
            trainer = Trainer(self.model, train_loader, val_loader, optimiser, loss_fn,
                              self.device, scheduler, self.model_name, model_fold_name)

            trainer.run_training(hyper_params['n_epochs'])

            # store the training results for later
            self.training_r2['train'].append(trainer.r2['train'])
            self.training_r2['val'].append(trainer.r2['val'])
            self.training_mse['train'].append(trainer.mse['train'])
            self.training_mse['val'].append(trainer.mse['val'])

            # get the best model
            trainer.get_best_model()
            self.best_model_paths.append(trainer.best_model_path)

            # use the best model to initialise the evaluator on the test set
            evaluator = Evaluator(model=self.model, state_dict_pth=trainer.best_model_path,
                                  test_loader=test_loader, device=self.device)

            # save the predictions
            self.predictions['unique_id'] += split['val_ids']
            self.predictions['y'] += evaluator.predictions['y']
            self.predictions['y_hat'] += evaluator.predictions['y_hat']

            # store the test results (i.e. the results on that fold)
            self.cv_r2.append(evaluator.calc_r2())
            self.cv_mse.append(evaluator.calc_mse())

        end_time = time.time()
        time_elapsed = np.round(end_time - start_time, 0).astype(int)
        print(f"Finished Cross-validation after {time_elapsed} seconds")

    def split_data_train_val_test(self, val_ids):
        # split the data into training and test dataframes
        train_df, test_df = split_lsms_ids(self.lsms_df, val_ids=val_ids)

        # split the training data further into a training and a validation set for hyper-parameter tuning
        # for now just do hyper-parameter tuning using a simple validation set approach
        train_val_folds = split_lsms_spatial(train_df, test_ratio=0.2, random_seed=self.random_seed, verbose=False)
        train_df, val_df = split_lsms_ids(train_df, val_ids=train_val_folds[0][
            'val_ids'])  # there is only one fold in train_val_folds
        return train_df, val_df, test_df

    def get_dataloaders(self, train_df, val_df, test_df, batch_size):
        # initialise the Landsat data
        dat_train = SatDataset(train_df, self.img_dir, self.data_type,
                               self.target_var, self.feat_transform, self.target_transform)
        dat_val = SatDataset(val_df, self.img_dir, self.data_type,
                             self.target_var, self.feat_transform, self.target_transform)
        dat_test = SatDataset(test_df, self.img_dir, self.data_type,
                              self.target_var, self.feat_transform, self.target_transform)

        # initialise the data loader objects
        if self.random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_seed)
            train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True, generator=generator)
        else:
            train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dat_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dat_test, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def compute_overall_performance(self, use_fold_weights=True):
        if use_fold_weights:
            fold_weights = [len(v['val_ids']) / (len(v['val_ids']) + len(v['train_ids'])) for v in
                            self.fold_ids.values()]
            r2 = np.average(self.cv_r2, weights=fold_weights)
            mse = np.average(self.cv_mse, weights=fold_weights)
            return {'r2': r2, 'mse': mse}
        else:
            r2 = np.mean(self.cv_r2)
            mse = np.mean(self.cv_mse)
            return {'r2': r2, 'mse': mse}

    def plot_true_vs_preds(self, xlabel="Predicted outcome values", ylabel='True outcome values'):
        ## plots the true observations vs the predicted observations
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
        folder = f"model_results"
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pth = f"model_results/{name}.pkl"
        with open(pth, 'wb') as f:
            aux = copy.deepcopy(self)
            aux.target_transform = None  # remove transforms as they often cannot be saved as pickles
            aux.feat_transform = None
            aux.model = None  # remove the model, the best model is saved somewhere else.
            pickle.dump(aux, f)
