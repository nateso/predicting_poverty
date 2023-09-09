import time
import numpy as np
import copy
from itertools import product

import torch.nn as nn
import torch.optim as optim

from .Trainer import Trainer


class ParamTuner():
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 hyper_params,
                 device):

        self.model = model
        self.orig_state_dict = copy.deepcopy(model.state_dict())

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hyper_params = hyper_params

        self.results = {'hyper_params': [], 'min_loss': [], 'min_loss_epoch': [], 'max_r2': [], 'max_r2_epoch': []}
        self.best_params = None

        self.device = device

    def grid_search(self):
        start_time = time.time()
        # Expand the grid
        # Generate all possible combinations of hyperparameters
        hyperparameter_combinations = list(product(*self.hyper_params.values()))
        hyper_param_combs = []
        for combination in hyperparameter_combinations:  # append the hyper-parameters to a list as dictionary
            hyper_param_combs.append(dict(zip(self.hyper_params.keys(), combination)))

        # Loop over all hyperparameter combinations
        for comb_nr, params in enumerate(hyper_param_combs):
            print('\n\n')
            print('\t-----------------------------------')
            print(f'\tCombination {comb_nr + 1} of {len(hyper_param_combs)}  -- Hyperparameters: {params}')
            print('\n')

            # train the model
            min_loss, min_loss_epoch, max_r2, max_r2_epoch = self.train(params)

            # store the number of epochs that gave the best results
            params['n_epochs'] = max_r2_epoch

            # store the results
            self.results['hyper_params'].append(params)
            self.results['min_loss'].append(min_loss)
            self.results['min_loss_epoch'].append(min_loss_epoch)
            self.results['max_r2'].append(max_r2)
            self.results['max_r2_epoch'].append(max_r2_epoch)

        # get the index of the best hyper-parameters
        best_index = np.argmax(self.results['max_r2'])
        # get the best hyper-parameters
        self.best_params = self.results['hyper_params'][best_index]

        # get the end time for the fold
        end_time = time.time()
        time_elapsed = np.round(end_time - start_time, 0).astype(int)
        print(f"Best Hyper-parameter combination: {best_index + 1} --- {self.best_params}")
        print(f"Finished hyper-parameter tuning after {time_elapsed} seconds")

    def train(self, params):
        # initialise the weights of the model
        self.model.load_state_dict(self.orig_state_dict)

        # train model
        loss_fn = nn.MSELoss()

        optimiser = optim.Adam(self.model.parameters(),
                               lr=params['lr'],
                               weight_decay=params['alpha'])

        scheduler = optim.lr_scheduler.StepLR(optimiser,
                                              step_size=params['step_size'],
                                              gamma=params['gamma'])

        trainer = Trainer(self.model,
                          self.train_loader,
                          self.val_loader,
                          optimiser,
                          loss_fn,
                          self.device,
                          scheduler)

        trainer.run_training(params['n_epochs'])

        return trainer.return_best_results()


