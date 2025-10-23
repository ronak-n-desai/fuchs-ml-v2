import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet, NeuralNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.auto import tqdm, trange
from sklearn.base import BaseEstimator, TransformerMixin
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import Callback
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_transform(X): # transforms first column of data.  Alternativly, this could be implemented via operations on DataFrames
    X1 = X.copy()
    X1[:, 0] = np.log(X1[:, 0])
    return X1
def log_inverse(X):
    X1 = X.copy()
    X1[:, 0] = np.exp(X1[:,0])
    return X1

class OutputLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = StandardScaler()
    def fit(self, y):
        y_copy = y.copy()
        y_copy = np.log(y_copy)
        self._estimator.fit(y_copy)
        
        return self
    def transform(self, y):
        y_copy = y.copy()
        y_copy = np.log(y_copy)
        return self._estimator.transform(y_copy)
    def inverse_transform(self, y):
        y_copy = y.copy()
        y_reverse = np.exp(self._estimator.inverse_transform(y_copy))
        
        return y_reverse

class InputLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = StandardScaler()
    def fit(self, X):
        X_copy = X.copy()
        X_copy = log_transform(X_copy)
        self._estimator.fit(X_copy)
        
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy = log_transform(X_copy)
        return self._estimator.transform(X_copy)
    def inverse_transform(self, X):
        X_copy = X.copy()
        X_reverse = log_inverse(self._estimator.inverse_transform(X_copy))
        
        return X_reverse
    
class LDIAModel(nn.Module):
    '''
    Perceptron Model of variable architecture for hyperparameter tuning
    '''
    def __init__(self, n_hidden = 1,n_neurons=64,activation=nn.LeakyReLU()):
        super().__init__()
        self.norms = []
        self.layers = []
        self.acts = []
        self.norm0 = nn.BatchNorm1d(3)
        self.layer0 = nn.Linear(3,n_neurons)
        for i in range(1,n_hidden+1):
            self.norms.append(nn.BatchNorm1d(n_neurons))
            self.acts.append(activation)
            self.add_module(f"norm{i}", self.norms[-1])
            self.add_module(f"act{i}", self.acts[-1])
            if (i != n_hidden):
                self.layers.append(nn.Linear(n_neurons, n_neurons))
                self.add_module(f"layer{i}", self.layers[-1])
        self.output = nn.Linear(n_neurons, 3)

    def forward(self, x):
        '''
          Forward pass
        '''
        x = self.layer0(self.norm0(x))
        for norm, layer, act in zip(self.norms, self.layers, self.acts):
            x = act(layer(norm(x)))
        return self.output(x)
    
def build_neural_network(max_epochs=50, n_hidden=3, n_neurons=32, activation=nn.LeakyReLU(), device=torch.device('cuda'),loss_fn=nn.MSELoss(), optimizer=optim.Adam, lr=1e-2, shuffled=True, batch_size=1024, patience=5, gamma=0.85,valid_ds=None,compiled=False):
    return NeuralNetRegressor(
    module=LDIAModel,
    max_epochs = max_epochs,
    module__n_hidden=n_hidden,
    module__n_neurons = n_neurons,
    module__activation=activation,
    device=device,
    criterion = loss_fn,
    optimizer=optimizer,
    optimizer__lr = lr,
    iterator_train__shuffle=shuffled,
    batch_size=batch_size,
    callbacks=[('early_stopping', EarlyStopping(patience=patience,monitor='valid_loss')),
    ('lr_scheduler', LRScheduler(policy='ExponentialLR',gamma=0.85)), ('memory_logger', GPUMemoryLogger(verbose=True))],
    train_split=predefined_split(valid_ds),
    compile = compiled
)

def make_datasets(X, y, *, train_size=0.8, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size,random_state=random_state)
    input_transformer = InputLogTransformer()
    output_transformer = OutputLogTransformer()
    X_train = input_transformer.fit_transform(X_train)
    X_val = input_transformer.transform(X_val)
    y_train = output_transformer.fit_transform(y_train)
    y_val = output_transformer.transform(y_val)
    #train_ds = Dataset(X_train, y_train)
    #valid_ds = Dataset(X_val, y_val)
    return X_train, y_train, X_val, y_val, input_transformer, output_transformer

class GPUMemoryLogger(Callback):
    def __init__(self, verbose=False):
        self.gpu_memory = []
        self.verbose = verbose

    def on_epoch_end(self, net, **kwargs):
        memory_stats = torch.cuda.memory_stats()
        # Calculate available GPU memory
        total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 **2)
        gpu_memory_usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        available_memory = total_memory - gpu_memory_usage
        self.gpu_memory.append(gpu_memory_usage)
        if self.verbose:
            print(f"GPU memory usage: {gpu_memory_usage:.2f} MB")
            print(f"Available memory: {available_memory:.2f} MB")
            #print("Available memory: {:.2f} MB".format(available_memory))
    def get_memory_usage(self):
        return self.gpu_memory
    