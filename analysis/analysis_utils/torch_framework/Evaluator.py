
import copy
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class Evaluator():
    def __init__(self, model, state_dict_pth, test_loader, device):
        self.device = device
        self.state_dict_pth = state_dict_pth
        self.model = copy.deepcopy(model)
        self.test_loader = test_loader
        self.predictions = {'y': [], 'y_hat': []}

        self.load_state_dict()

    def load_state_dict(self):
        self.model.load_state_dict(torch.load(self.state_dict_pth, map_location=self.device))

    def predict(self):
        # takes the model and predicts the values on the test loader
        self.model.to(self.device)
        self.model.eval()
        print("\t\tPredicting values")
        with torch.no_grad():
            for x, t in tqdm(self.test_loader):
                y_hat = list(self.model(x.to(self.device)).cpu().numpy().squeeze(1))
                y = list(t.cpu().numpy().squeeze(1))
                self.predictions['y'] += y
                self.predictions['y_hat'] += y_hat

    def calc_mse(self):
        return mean_squared_error(self.predictions['y'], self.predictions['y_hat'])

    def calc_r2(self):
        return r2_score(self.predictions['y'], self.predictions['y_hat'])

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

    def plot_residuals(self, xlabel="Predicted outcome values", ylabel='Residuals'):
        # plots the residuals
        y_hat = np.array(self.predictions['y_hat'])
        y = np.array(self.predictions['y'])
        plt.figure(figsize=(5, 5))
        plt.scatter(y_hat, y - y_hat)
        plt.plot([min(y_hat), max(y_hat)], [0, 0], color='red', linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Residuals')
        plt.show()
