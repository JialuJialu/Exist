"""

 linear_regr.py  (author: Anson Wong / git: ankonzoid)

"""
import warnings
import numpy as np
import pdb
import cvxpy as cp
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.norm import norm
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

pnorm = 2


class linear_regr_pnorm:

    def __init__(self, fit_intercept=False):
        from sklearn.linear_model import LinearRegression
        # forced intercept to be zero now
        self.model = None

    def fit(self, X, y):
        try:
            _, d = X.shape
        except ValueError:
            pdb.set_trace()
        c = cp.Variable(d)
        cost = norm(X@c - y, p=pnorm)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        self.model = c
        return

    def predict(self, X):
        X = np.array(X)
        y_pred = np.array([i.item(0) for i in np.dot(X, self.model.value)])
        return y_pred

    def loss(self, X, y, y_pred):
        if y.shape != y_pred.shape:
            import pdb
            pdb.set_trace()
        return norm(y_pred - y, p=pnorm).value
        # return np.max(np.abs(np.subtract(y, y_pred)))

    def to_string(self, header):
        coef = self.model.value
        string = " + ".join(["{}*{}".format(round(coef.item(c), 1), header[c])
                             for c in range(len(coef))
                             if round(coef.item(c), 1) != 0])
        string += "\n {}-norm".format(pnorm)
        return string
