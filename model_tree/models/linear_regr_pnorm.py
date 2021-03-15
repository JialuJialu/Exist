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


def _expandby1(X):
    dummybias = np.ones((len(X), 1))
    newX = np.concatenate((np.array(X), dummybias), axis=1)
    return newX


class linear_regr_pnorm:

    def __init__(self, pnorm=2, fit_intercept=False, model=None):
        # forced intercept to be zero by default
        self.model = model
        self.pnorm = pnorm
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = _expandby1(X)
        try:
            _, d = X.shape
        except ValueError:
            pdb.set_trace()
        c = cp.Variable(d)
        cost = norm(X@c - y, p=self.pnorm)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        self.model = [c.value.item(i) for i in range(len(c.value))]
        return

    def predict(self, X):
        X = np.array(X)
        if self.fit_intercept:
            X = _expandby1(X)
        y_pred = np.array([i.item(0) for i in np.dot(X, self.model)])
        return y_pred

    def loss(self, X, y, y_pred):
        if y.shape != y_pred.shape:
            import pdb
            pdb.set_trace()
        return norm(y_pred - y, p=self.pnorm).value

    def to_string(self, header, d=2):
        if self.fit_intercept:
            header = header[:-1] + ["1"]
        coef = self.model
        try:
            string = " + ".join(["{}*{}".format(round(coef[c], d), header[c])
                                 for c in range(len(coef))
                                 if round(coef[c], d) != 0])
        except IndexError:
            import pdb
            pdb.set_trace()
        return string, "{}-norm".format(self.pnorm)


class linear_regr_2norm(linear_regr_pnorm):
    def __init__(self, fit_intercept=False, model=None):
        # forced intercept to be zero by default
        self.model = model
        self.pnorm = 2
        self.fit_intercept = fit_intercept
