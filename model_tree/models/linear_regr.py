"""

 linear_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


class linear_regr:

    def __init__(self, fit_intercept=False, model=None):
        self.model_raw = LinearRegression(fit_intercept=fit_intercept)
        self.model = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.model_raw.fit(X, y)
        self.model = self.model_raw.coef_ + [self.model_raw.intercept_]

    def predict(self, X):
        return self.model_raw.predict(X)

    def loss(self, X, y, y_pred):
        if y.shape != y_pred.shape:
            import pdb
            pdb.set_trace()
        return mean_squared_error(y, y_pred)

    def combine_loss(self, left_loss, right_loss):
        return left_loss + right_loss

    def to_string(self, header, d=2):
        if self.fit_intercept:
            header = header[:-1] + ["1"]
        string = " + ".join(["{}*{}".format(round(self.model[c], d), header[c])
                             for c in range(len(self.model))
                             if round(self.model[c], d) != 0
                             ])
        return string, "mean square error"
