import numpy as np


def mae(y_pr, y):
    return abs(y_pr - y).sum() / len(y)


def mse1(y_pr, y):
    return ((y_pr - y) ** 2).sum() / len(y)


def rmse(y_pr, y):
    return (((y_pr - y) ** 2).sum() / len(y)) ** 0.5


def mape(y_pr, y):
    return 100 * abs(((y_pr - y) / y)).sum() / len(y)


def r2(y_pr, y):
    numerator = np.sum((y - y_pr) ** 2)
    denominator = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (numerator / denominator)
    return r_squared


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
    #   mae mse rmse mape r2

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X, y, verbose=False):
        global metric_calc
        X.insert(loc=0, column='w0', value=1)
        self.weights = np.array(X.shape[1] * [1])
        for i in range(self.n_iter):
            y_pr = X @ self.weights
            mse = ((y_pr - y) ** 2).sum() / len(y)
            grad = (2 / len(y)) * (y_pr - y) @ X
            self.weights = self.weights - self.learning_rate * grad
            if verbose and i % verbose == 0:
                print(f'{i} | loss: {mse} | metric_name: {self.calc_metrics(y_pr, y)}')
        final_y_pr = X @ self.weights
        self.best_score = self.calc_metrics(final_y_pr, y)

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X.insert(loc=0, column='w0', value=1)
        return X @ self.weights

    def get_best_score(self):
        return self.best_score

    def calc_metrics(self, pred: np.ndarray, y: np.ndarray) -> float:
        if self.metric == 'mae':
            return np.sum(np.abs(pred - y)) / len(pred)
        elif self.metric == 'mse':
            return np.sum(np.power(pred - y, 2)) / len(pred)
        elif self.metric == 'rmse':
            return np.sqrt(np.sum(np.power(pred - y, 2)) / len(pred))
        elif self.metric == 'r2':
            return 1 - (np.sum(np.power(y - pred, 2)) / np.sum(np.power(y - y.mean(), 2)))
        elif self.metric == 'mape':
            return np.sum(np.abs((y - pred) / y)) / len(pred) * 100
        else:
            return None