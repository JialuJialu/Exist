"""

 linear_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")


class linear_regr:

    def __init__(self, fit_intercept=False):
        from sklearn.linear_model import LinearRegression
        # forced intercept to be zero now
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        if y.shape != y_pred.shape:
            import pdb
            pdb.set_trace()
        return mean_squared_error(y, y_pred)

    def to_string(self, header):
        string = " + ".join(["{}*{}".format(round(self.model.coef_[c], 3), header[c])
                             for c in range(len(self.model.coef_))
                             if round(self.model.coef_[c], 1) != 0
                             ])
        string += " + ({})".format(round(self.model.intercept_, 1))
        string += "\n mean_squared_error"
        return string
