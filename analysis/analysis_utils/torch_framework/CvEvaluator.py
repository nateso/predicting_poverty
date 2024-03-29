from ..spatial_CV import split_lsms_ids
from .SatDataset import SatDataset
from torch.utils.data import DataLoader
from .Evaluator import Evaluator

import numpy as np
import os
import copy
import pickle


class CvEvaluator:
    def __init__(self, cv_object, lsms_df, fold_ids, device):
        self.lsms_df = lsms_df
        self.cv_object = cv_object
        self.device = device
        self.fold_ids = fold_ids

        # store some of the cv_object attributes
        self.model_class = copy.deepcopy(cv_object.model_class)
        self.id_var = copy.deepcopy(cv_object.id_var)

        # store the results
        self.res_mse = {'val': []}
        self.res_r2 = {'val': []}
        self.predictions = {self.id_var: [], 'y': [], 'y_hat': []}

    def evaluate(self, feat_transform, target_transform):
        for fold, split in self.fold_ids.items():
            print('\n')
            print(
                '=====================================================================================================')
            print(f"Evaluate on fold {fold}")

            # get the model parameters
            model_pth = self.cv_object.best_model_paths[fold]

            # load split the data into training and test for panel and cross section
            _, test_df = split_lsms_ids(self.lsms_df, val_ids=split['val_ids'])

            # initalise the Sat_Dataset
            dat_test = SatDataset(
                labels_df=test_df,
                img_dir=self.cv_object.img_dir,
                data_type=self.cv_object.data_type,
                target_var=self.cv_object.target_var,
                id_var=self.id_var,
                feat_transform=feat_transform,
                target_transform=target_transform,
                random_seed=None
            )

            # get the data loader object
            test_loader = DataLoader(dat_test, batch_size=128, shuffle=False)

            # load the model evaluator
            evaluator = Evaluator(model=self.model_class.model,
                                  state_dict_pth=model_pth,
                                  test_loader=test_loader,
                                  device=self.device)

            evaluator.predict()

            # save the predictions
            self.predictions[self.id_var] += split['val_ids']
            self.predictions['y'] += evaluator.predictions['y']
            self.predictions['y_hat'] += evaluator.predictions['y_hat']

            # append the results
            self.res_r2['val'].append(evaluator.calc_r2())
            self.res_mse['val'].append(evaluator.calc_mse())

            print(f"\nResults of fold {fold}:")
            print(f"\tVal MSE: {self.res_mse['val'][-1]}")
            print(f"\tVal R2: {self.res_r2['val'][-1]}")

    def get_fold_weights(self, ids='val_ids'):
        '''
        Fold weights differ when running the delta or demeaned model as compared to the between model
        In the between models, the fold weights are only defined by the number of clusters in each fold
        In the within model, fold weights are defined by the number of observations in each fold
        :return: list
        '''
        n = len(self.lsms_df)
        weights = []
        for split in self.fold_ids.values():
            # subset the lsms df to the clusters in the validation split
            val_cids = split[ids]
            mask = self.lsms_df.cluster_id.isin(val_cids)
            sub_df = self.lsms_df[mask]
            weights.append(len(sub_df) / n)
        return weights

    def compute_overall_performance(self, use_fold_weights=True):
        if use_fold_weights:
            fold_weights = self.get_fold_weights(ids='val_ids')
            r2 = np.average(self.res_r2['val'], weights=fold_weights)
            mse = np.average(self.res_mse['val'], weights=fold_weights)
            return {'r2': r2, 'mse': mse}
        else:
            r2 = np.mean(self.cv_r2)
            mse = np.mean(self.cv_mse)

        performance = {'val_r2': r2, 'val_mse': mse}

        return performance

    def save_object(self, name):
        folder = f'results/model_objects'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pth = f"{folder}/{name}.pkl"
        with open(pth, 'wb') as f:
            aux = copy.deepcopy(self)
            aux.cv_object = None  # remove the target transforms as it cannot be saved as pickle
            # aux.feat_transform = None
            pickle.dump(aux, f)
