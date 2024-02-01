import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#from tqdm.auto import tqdm
import time
import os


class Trainer():
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 optimiser,
                 loss_fn,
                 device,
                 scheduler=None,
                 early_stopper=None,
                 model_folder=None,
                 model_name=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.model_folder = model_folder
        self.model_name = model_name

        self.res_mse = {'train': [], 'val': []}
        self.res_r2 = {'train': [], 'val': []}
        self.best_model_path = None

    def train(self):
        '''
        Trains one epoch of a neural network.
        :param dataloader: dataloader object containing the training data
        :param model: initialised Torch nn (nn.Module) to train
        :param optimiser: Torch optimiser object
        :param loss_fn: Torch loss function
        :param device: device to use for training
        :param classification: Boolean, set true if classification
        :return: mean of epoch loss, if classification also accuracy
        '''
        epoch_loss = []
        epoch_y = []
        epoch_yhat = []
        for x, y in self.train_loader:
            # initialise training mode
            self.optimiser.zero_grad()
            self.model.train()
            # forward pass
            y_hat = self.model(x.to(self.device))
            # loss
            loss = self.loss_fn(y_hat, y.to(self.device))
            # Backpropagation
            loss.backward()
            # Update weights
            self.optimiser.step()
            # store batch loss
            batch_loss = loss.detach().cpu().numpy()  # move loss back to CPU
            epoch_loss.append(batch_loss)
            # save the model's predictions
            epoch_y += list(y.detach().cpu().numpy())
            epoch_yhat += list(y_hat.detach().cpu().numpy())
        # save results
        self.res_mse['train'].append(np.mean(epoch_loss))
        self.res_r2['train'].append(r2_score(epoch_y, epoch_yhat))

    def validate(self):
        '''
        Calculates validation error for validation dataset

        :param dataloader: dataloader object containing the validation data
        :param model: initialised Torch nn (nn.Module) to train
        :param loss_fn: Torch loss function
        :param device: device to use for training
        :param classification: Boolean set true if doing classification
        :return: validation loss for one epoch, if classification returns acc too
        '''
        epoch_y = []
        epoch_yhat = []
        epoch_loss = []
        self.model.eval()  # initialise validation mode
        with torch.no_grad():  # disable gradient tracking
            for x, y in self.val_loader:
                # forward pass
                y_hat = self.model(x.to(self.device))
                # loss
                loss = self.loss_fn(y_hat, y.to(self.device))
                batch_loss = loss.detach().cpu().numpy()
                epoch_loss.append(batch_loss)
                # save the model's predictions
                epoch_y += list(y.detach().cpu().numpy())
                epoch_yhat += list(y_hat.detach().cpu().numpy())
        # save results
        self.res_mse['val'].append(np.mean(epoch_loss))
        self.res_r2['val'].append(r2_score(epoch_y, epoch_yhat))

    def run_training(self, n_epochs):
        '''
        Wrapper for training and validation
        '''
        print('\t\tInitialising training')
        start_time = time.time()

        self.model.to(self.device)

        for epoch in range(n_epochs):
            self.train()
            if self.val_loader is not None:
                self.validate()

            # print the epoch result
            if self.val_loader is not None:
                print(
                    f"\t\tEPOCH {epoch+1} - Train MSE: {self.res_mse['train'][-1]:.4f} - Train R2 {self.res_r2['train'][-1]:.4f} - Val MSE: {self.res_mse['val'][-1]:.4f} - Val R2 {self.res_r2['val'][-1]:.4f}")
            else:
                print(f"\t\tEPOCH {epoch+1} - Train MSE: {self.res_mse['train'][-1]:.4f} - Train R2 {self.res_r2['train'][-1]:.4f}")

            if self.early_stopper is not None:
                self.early_stopper.update(
                    val_loss=self.res_mse['val'][-1],
                    val_r2=self.res_r2['val'][-1]
                )
                if self.early_stopper.early_stop:
                    print('\n')
                    print(f"\t\tPatience exhausted. Stopping early...")
                    print(f'\t\tValidation loss {self.early_stopper.best_loss:4f}')
                    print(f'\t\tValidation R2 {self.early_stopper.best_r2:4f}')
                    print('\n')
                    break

            # update the learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # save the model's parameters:
            if self.model_name is not None:
                # save the parameters of the currently best model only if model name provided and validation set used
                if self.val_loader is not None:
                    # only keep the best model (to reduce memory usage)
                    current_val_r2 = self.res_r2['val'][-1]
                    best_val_r2 = max(self.res_r2['val'])
                    if current_val_r2 == best_val_r2:
                        self.save_model_params(suffix='best')

        # if model name provided and no validation set used, save the parameters of the last epoch
        if self.val_loader is None:
            if self.model_name is not None:
                # save model parameters
                self.save_model_params(suffix='last_epoch')
                # store the path to the final model
                self.best_model_path = f"results/model_checkpoints/{self.model_folder}/{self.model_name}_last_epoch.pth"

        end_time = time.time()
        time_elapsed = np.round(end_time - start_time, 0).astype(int)
        print(f"\t\tFinished training after {time_elapsed} seconds")

    def save_model_params(self, suffix):
        folder = f'results/model_checkpoints/{self.model_folder}'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        checkpoint_pth = f"{folder}/{self.model_name}_{suffix}.pth"
        torch.save(self.model.state_dict(), checkpoint_pth)

    def save_performance_metrics(self, folder_pth):
        if len(self.res_mse['train']) != 0:
            val_pth = f"{folder_pth}/val_loss.npy"
            np.save(val_pth, self.res_mse['val'])
            train_pth = f"{folder_pth}/train_loss.npy"
            np.save(train_pth, self.res_mse['train'])
            print(f'Training results saved to {folder_pth}')
        else:
            print("Model not yet trained")

    def return_best_results(self, smooth_epochs = 1):
        if self.val_loader is not None:
            if smooth_epochs < 2:
                min_loss = np.min(self.res_mse['val'])
                min_loss_epoch = np.argmin(self.res_mse['val']) + 1
                max_r2 = np.max(self.res_r2['val'])
                max_r2_epoch = np.argmax(self.res_r2['val']) + 1

                print(f"\t\tSmoothing factor {smooth_epochs} - No smoothing")
                print(f"\t\tLowest loss on validation set in epoch {min_loss_epoch}: {min_loss:.6f}")
                print(f"\t\tMaximum R2 on validation set in epoch {max_r2_epoch}: {max_r2:.6f}")
            else:
                # smooth the results
                moving_avg_filter = np.ones(smooth_epochs)/smooth_epochs

                # create a new array with the smoothed results
                loss = np.convolve(self.res_mse['val'], moving_avg_filter, mode='valid')
                r2 = np.convolve(self.res_r2['val'], moving_avg_filter, mode='valid')

                # get the highest smoothed R2 and min MSE
                max_r2 = np.max(r2)
                min_loss = np.min(loss)

                # get the epoch for each of these
                # the epoch is always the median epoch of the smoothing window
                epoch_adjustment = int(np.ceil(np.median(range(1, smooth_epochs+1))))
                max_r2_epoch = np.argmax(r2) + epoch_adjustment
                min_loss_epoch = np.argmin(loss) + epoch_adjustment

                print(f"\t\tSmoothing factor {smooth_epochs}")
                print(f"\t\tLowest smoothed loss on validation set in epoch {min_loss_epoch}: {min_loss:.6f}")
                print(f"\t\tMaximum smoothed R2 on validation set in epoch {max_r2_epoch}: {max_r2:.6f}")

            return min_loss, min_loss_epoch, max_r2, max_r2_epoch

    def get_best_model(self):
        if len(self.res_mse['train']) != 0:
            min_loss = np.min(self.res_mse['val'])
            min_loss_epoch = np.argmin(self.res_mse['val']) + 1
            max_r2 = np.max(self.res_r2['val'])
            max_r2_epoch = np.argmax(self.res_r2['val']) + 1
            print(f"\t\tLowest loss on validation set in epoch {min_loss_epoch}: {min_loss:.6f}")
            print(f"\t\tMaximum R2 on validation set in epoch {max_r2_epoch}: {max_r2:.6f}")
            self.best_model_path = f"results/model_checkpoints/{self.model_folder}/{self.model_name}_best.pth"
        else:
            print("Model not yet trained")

    def plot_loss_mse_r2(self):
        if len(self.res_mse['val']) != 0:
            fig, ax1 = plt.subplots(figsize=(5, 5))
            ax1.plot(self.res_mse['val'], label='Validation MSE', color='red')
            ax1.plot(self.res_mse['train'], label='Training MSE', color='red', linestyle='--')
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax2 = ax1.twinx()
            ax2.plot(self.res_r2['val'], label='Validation R2', color='blue')
            ax2.plot(self.res_r2['train'], label='Training R2', color='blue', linestyle='--')
            ax2.set_ylabel("R2", color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            plt.tight_layout()
            plt.legend()
            plt.show()
        else:
            print("Model not yet trained")
