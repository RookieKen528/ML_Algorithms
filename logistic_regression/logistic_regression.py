# by Ken Liang, last update: 2021.12.8

import numpy as np # version: 1.20.1

class logistic_regression():
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter  # maximum number of iterations
        self.batch_size = batch_size  # batch size of the gradient descent. If 1, it is SGD
        self.w_history = []  # store weights at each iteration during the training
        self.cost_history = []  # store cost at each iteration during the training

    def fit(self, xtrain, ytrain):
        # train the model with the input training data, accept ndarray
        xtrain = self.augmentation(xtrain)  # add bias column to X

        # batch gradient descent
        self.w_history, self.cost_history = self.gredient_descent(xtrain, ytrain)

        return self

    def gredient_descent(self, x, t):
        # x is training data, t is target
        n, m = x.shape
        w = np.zeros(m)  # initialize w

        w_history = []  # store history weights in the processing of training
        cost_history = []
        for i in range(self.max_iter):
            sample_index = np.random.randint(0, n, self.batch_size)  # stochastic selection of samples
            xx = x[sample_index]
            tt = t[sample_index]

            z = np.dot(w, xx.T)  # calculate X*w, shape=(n,1)
            y = self.sigmoid(z)  # calculate sigmoid(X*w), shape=(n,1)

            gd = 1 / n * np.dot(xx.T, (y - tt))  # calculate gredient

            w = w - self.learning_rate * gd  # update w

            w_history.append(w)  # record each w

            self.w = w  # update w

            # based on the updated weights compute cost for the training data: log loss or cross entropy loss
            z = np.dot(self.w, x.T)
            y = self.sigmoid(z)
            train_loss = self.cross_entropy_loss(t, y)
            train_cost = train_loss.sum() / len(y)

            cost_history.append(train_cost)

        return w_history, cost_history

    def predict(self, x, threshold=0.5, add_bias=True, return_proba=False):
        # convert to ndarray
        if not isinstance(x, np.ndarray):
            x = x.values
        # add bias column to x
        if add_bias:
            x = self.augmentation(x)
        z = np.dot(x, self.w)
        yhat = self.sigmoid(z).ravel()
        if return_proba:
            return yhat
        else:
            return (yhat >= threshold) * 1

    def augmentation(self, x):
        # add bias column to x
        n, m = x.shape
        ones = np.ones((n, 1))
        return np.hstack((ones, x))

    def sigmoid(self, x):
        # sigmoid function
        return 1 / (1 + np.exp(-x))

    def cross_entropy_loss(self, y, yhat):
        # calculate cross entropy loss : -y*log(yhat) - (1-y)*log(1-yhat)

        loss = -y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
        return loss