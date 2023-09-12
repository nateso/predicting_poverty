import numpy as np

class EarlyStopper:
    '''Implements an early stopper, which stops trainings if the validation R2 does not increase'''

    def __init__(self, patience=40):
        '''

        :param path: path where the best model is saved, default: checkpoint.pt
        :param patience: number of epochs to wait before terminating training
        '''
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf
        self.best_r2 = -np.Inf

    @property
    def early_stop(self):
        return self.counter >= self.patience

    def update(self, val_loss, val_r2):
        if val_r2 > self.best_r2:
            self.counter = 0
            self.best_loss = val_loss
            self.best_r2 = val_r2
        else:
            self.counter += 1



