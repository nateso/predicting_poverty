import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# load the functions to do spatial k-fold CV
from .spatial_CV import split_lsms_ids


class CombinedModel:
    def __init__(self, lsms_df, between_cv_object, within_cv_object):
        self.lsms_df = lsms_df
        self.between_cv_object = between_cv_object
        self.within_cv_object = within_cv_object
        self.target_var = within_cv_object.target_var
        self.fold_ids = within_cv_object.fold_ids

        self.pred_df = self.get_preds()
        self.res_r2 = {'combined': [], 'between': [], 'within': []}
        self.res_mse = {'combined': [], 'between': [], 'within': []}

    def get_preds(self):
        # get the id variables of both models
        within_id_var = self.within_cv_object.id_var
        between_id_var = self.between_cv_object.id_var

        # get the within predictions
        within_preds = pd.DataFrame(self.within_cv_object.predictions)
        within_preds = pd.merge(within_preds, self.lsms_df[[between_id_var, within_id_var]], on=within_id_var)
        within_preds = within_preds.rename(columns={'y_hat': 'y_hat_change', 'y': 'y_change'})

        # get the between predictions
        between_preds = pd.DataFrame(self.between_cv_object.predictions)
        between_preds = between_preds.rename(columns={'y_hat': 'y_hat_mn', 'y': 'y_mn'})

        # merge the predictions
        preds = pd.merge(within_preds, between_preds, on=between_id_var, how='left')

        # get the overall prediction
        preds['y_hat'] = preds['y_hat_change'] + preds['y_hat_mn']

        # to double check also combine the targets (should be the same as in lsms_df)
        preds['y'] = preds['y_change'] + preds['y_mn']

        # add the target variable from the lsms
        preds = pd.merge(preds, self.lsms_df[[within_id_var, self.target_var]],
                         on=within_id_var, how='left')

        # add the validation fold to the prediction dataframe
        preds['fold'] = np.nan
        for fold, splits in self.fold_ids.items():
            preds.loc[preds['cluster_id'].isin(splits['val_ids']), 'fold'] = fold

        return preds

    def evaluate(self):
        # add the R2 from the between and within model
        self.res_r2['between'] = self.between_cv_object.r2['val']
        self.res_r2['within'] = self.within_cv_object.r2['val']

        # add MSE from the between and within model
        self.res_mse['between'] = self.between_cv_object.mse['val']
        self.res_mse['within'] = self.within_cv_object.mse['val']

        # calculate the results for each fold in the combined model
        for fold, splits in self.fold_ids.items():
            # get the training and validation sample
            train_df, val_df = split_lsms_ids(self.pred_df, splits['val_ids'])

            # get the validation error
            y_hat_val = val_df['y_hat']
            y_val = val_df[self.target_var]

            # calculate the performance on the validation sample
            self.res_r2['combined'].append(r2_score(y_val, y_hat_val))
            self.res_mse['combined'].append(mean_squared_error(y_val, y_hat_val))

    def calculate_fold_weights(self):
        # calculate within fold weights
        within_df = self.within_cv_object.lsms_df
        between_df = self.between_cv_object.lsms_df

        within_n = len(within_df)
        within_fold_weights = []

        between_n = len(between_df)
        between_fold_weights = []

        for split in self.fold_ids.values():
            # subset the lsms df to the clusters in the validation split
            val_cids = split['val_ids']

            # calculate between fold weights
            between_mask = between_df.cluster_id.isin(val_cids)
            b_sub_df = between_df[between_mask]
            between_fold_weights.append(len(b_sub_df)/between_n)

            # calculate within fold weights
            within_mask = within_df.cluster_id.isin(val_cids)
            w_sub_df = within_df[within_mask]
            within_fold_weights.append(len(w_sub_df)/within_n)

        return between_fold_weights, within_fold_weights

    def compute_overall_performance(self, use_fold_weights=True):
        if use_fold_weights:
            between_fold_weights, within_fold_weights = self.calculate_fold_weights()

            comb_r2 = np.average(self.res_r2['combined'], weights=within_fold_weights)
            comb_mse = np.average(self.res_mse['combined'], weights=within_fold_weights)

            between_r2 = np.average(self.res_r2['between'], weights=between_fold_weights)
            between_mse = np.average(self.res_mse['between'], weights=between_fold_weights)

            within_r2 = np.average(self.res_r2['within'])
            within_mse = np.average(self.res_mse['within'])
        else:
            comb_r2 = np.mean(self.res_r2['combined'])
            comb_mse = np.mean(self.res_mse['combined'])

            between_r2 = np.mean(self.res_r2['between'])
            between_mse = np.mean(self.res_mse['between'])

            within_r2 = np.mean(self.res_r2['within'])
            within_mse = np.mean(self.res_mse['within'])

        r2_perf = {'combined': comb_r2, 'between': between_r2, 'within': within_r2}
        mse_perf = {'combined': comb_mse, 'between': between_mse, 'within': within_mse}
        performance = {'r2': r2_perf, 'mse': mse_perf}

        return performance

    def print_tex(self, metric='r2'):
        overall_perf = self.compute_overall_performance(use_fold_weights=True)
        metric_perf = overall_perf[metric]
        print(f"& {metric_perf['between']:.4f} & {metric_perf['within']:.4f} & {metric_perf['combined']:.4f}")

    def plot_true_vs_preds(self,
                           xlabel="True outcome values",
                           ylabel='Predicted outcome values',
                           title=None,
                           fname=None):
        # plots the true observations vs the predicted observations
        y = np.array(self.pred_df['y'])
        y_hat = np.array(self.pred_df['y_hat'])
        plt.figure(figsize=(5, 5))
        plt.scatter(y, y_hat, s=2)
        plt.plot([min(y), max(y)], [min(y_hat), max(y_hat)], color='red', linestyle='--')  # Line of perfect correlation
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if fname is not None:
            pth = f"../figures/results/{fname}"
            plt.savefig(pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

