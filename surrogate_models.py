from typing import Optional
import numpy as np
import numpy as np
import torch
import torch.utils.data as Data
from util import impute_nan

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import BayesianRidge
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


class BaseModel(object):
    def __init__(self):
        self.is_trained = False
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        self.model.fit(X, y) # squeeze y?
        self.is_trained = True

    def predict(self, X: np.ndarray, **kwargs):
        if len(X.shape) == 1:
           X = X[np.newaxis, :]
        if not self.is_trained:
            raise ValueError("model needs to be trained first!")
        
        X = impute_nan(X)

        return self._predict(X, **kwargs)

    def _predict(self, X: np.ndarray, **kwargs):
        raise NotImplementedError


class GaussianProcess(BaseModel):
    def __init__(self):
        super(GaussianProcess, self).__init__()
        #kernel = RBF(length_scale=0.01)
        kernel = Matern(length_scale=1.0) + WhiteKernel()
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)

    def _predict(self, X):
        mu, std = self.model.predict(X, return_std=True)
        var = std*std

        return mu, var


class RandomForest(BaseModel):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.model = RandomForestRegressor(n_estimators=10)
    
    def _predict(self, X):
        predictions = []
        for estimator in self.model.estimators_:
            predictions.append(estimator.predict(X))

        predictions = np.array(predictions)
        predictions = np.transpose(predictions)

        mean = np.mean(predictions, axis=1)
        var = np.var(predictions, axis=1)

        return mean, var


class MLP(torch.nn.Module):
    def __init__(self, num_hidden=200, epochs=200, dropout=0.):
        super(MLP, self).__init__()

        self.epochs = epochs
        self.num_hidden = num_hidden
        self.architecture = False
        self.dropout = dropout

    def _define_architecture(self, in_features, out_features):
        self.fc1 = torch.nn.Linear(in_features, self.num_hidden)
        self.fc2 = torch.nn.Linear(self.num_hidden, self.num_hidden)
        self.fc3 = torch.nn.Linear(self.num_hidden, self.num_hidden)
        self.fc4 = torch.nn.Linear(self.num_hidden, out_features)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def fit(self, x_train, y_train):
        x_train = torch.Tensor(np.array(x_train)) # (num_configs, num_hyperparameters)
        y_train = torch.Tensor(np.array(y_train)) # (num_configs, )
        #x_train = x_train[:, :, np.newaxis,] # Add batch dimension
        y_train = y_train[:, np.newaxis,]

        if not self.architecture:
            in_features, out_features = x_train.shape[1], y_train.shape[1]
            self._define_architecture(in_features, out_features)
            self.architecture = True

        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=25)

        for epoch in range(self.epochs):
            losses = []
            for x, y in dataloader:

                self.optimizer.zero_grad()
                y_pred = self.forward(x)

                # Compute Loss
                loss = self.criterion(y_pred, y)
                losses.append(loss.item())
            
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            #print('Epoch {}: train loss: {}'.format(epoch, np.mean(np.array(losses))))


    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x


class BayesianNeuralNetworkBayesLinear(BaseModel):
    def __init__(self, num_features):
        super(BayesianNeuralNetworkBayesLinear, self).__init__()
        self.model = MLP(num_features=num_features)
    
    def _predict(self, x_test):
        return _predict(self.model, x_test)


class BayesianNeuralNetworkDropOut(BaseModel):
    def __init__(self, num_hidden, dropout=0.25):
        super(BayesianNeuralNetworkDropOut, self).__init__()
        self.model = MLP(num_hidden, dropout=dropout)
    
    def _predict(self, x_test):
        return _predict(self.model, x_test)


def _predict(model, x_test):
    with torch.no_grad():
        predictions = []
        for i in range(len(x_test)):

            x_test = torch.Tensor(np.array(x_test))
            prediction = model(x_test)
            prediction = prediction.detach().numpy()
            predictions.append(prediction)
            
        predictions = np.array(predictions)
        predictions = np.transpose(predictions, (1, 0, 2))

        mean = np.mean(predictions, axis=1)
        var = np.var(predictions, axis=1)

        return mean, var


