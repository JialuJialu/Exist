
import warnings
import numpy as np
import pdb
import cvxpy as cp
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.norm import norm
# from cvxpy.atoms.elementwise.maximum import maximum
# from cvxpy import *
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")
pnorm = 2


def loss_fn(X, Y, beta):
    return norm(X - Y, p=pnorm)
    # return cp.norm2(X @ beta - Y)**2


def L1regularizer(beta):
    return cp.norm1(beta)


def L2regularizer(beta):
    return cp.norm2(beta)


def objective_fn(X, Y, beta, lambd, regularizer):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


regularizer = L1regularizer


class linear_regr_constraint:

    def __init__(self, fit_intercept=False, lambd=1, model=None):
        # forced intercept to be zero now
        self.model = model
        self.lambd = lambd

    def fitwithConstraint(self, X, y, consX, consY, use_inv):
        try:
            _, d = X.shape
        except ValueError:
            pdb.set_trace()
        c = cp.Variable(d)
        pre_cons, post_cons = consY
        if use_inv == "inv_only":
            constraint = [consX@c <= post_cons] + [consX@c >= pre_cons]
        elif use_inv == 'both':
            constraint = [consX@c <= post_cons]
        else:
            constraint = []

        cost = norm(X@c - y, p=pnorm)
        prob = cp.Problem(cp.Minimize(cost), constraint)
        try:
            prob.solve()
        except cp.error.SolverError:
            pdb.set_trace()
        # prob.solve(solver=cp.OSQP)
        copy = c
        try:
            self.model = [c.value.item(i) for i in range(len(c.value))]
        except TypeError:
            pdb.set_trace()
        return

    def fit(self, X, y):
        try:
            _, d = X.shape
        except ValueError:
            pdb.set_trace()
        c = cp.Variable(d)
        cost = norm(X@c - y, p=pnorm)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        self.model = [c.value.item(i) for i in range(len(c.value))]
        return

    def predict(self, X):
        X = np.array(X)
        y_pred = np.array([i.item(0) for i in np.dot(X, self.model)])
        return y_pred

    def loss(self, X, y, y_pred):
        if y.shape != y_pred.shape:
            import pdb
            pdb.set_trace()
        return norm(y_pred - y, p=pnorm).value
        # return np.max(np.abs(np.subtract(y, y_pred)))

    def to_string(self, header, d=2):
        coef = self.model
        string = " + ".join(["{}*{}".format(round(coef[c], d), header[c])
                             for c in range(len(coef))
                             if round(coef[c], d) != 0])
        return string, "{}-norm".format(pnorm)


class linear_regr_constraint_L1(linear_regr_constraint):
    def fitwithConstraint(self, X, y, consX, consY):
        try:
            _, d = X.shape
        except ValueError:
            pdb.set_trace()
        c = cp.Variable(d)
        constraint = [consX@c <= consY]
        prob = cp.Problem(cp.Minimize(objective_fn(
            X@c, y, c, self.lambd, L1regularizer)), constraint)
        prob.solve()
        self.model = [c.value.item(i) for i in range(len(c.value))]
        return


class linear_regr_constraint_L2(linear_regr_constraint):
    def fitwithConstraint(self, X, y, consX, consY):
        try:
            _, d = X.shape
        except ValueError:
            pdb.set_trace()
        c = cp.Variable(d)
        constraint = [consX@c <= consY]
        prob = cp.Problem(cp.Minimize(objective_fn(
            X@c, y, c, self.lambd, L2regularizer)), constraint)
        prob.solve()
        self.model = [c.value.item(i) for i in range(len(c.value))]
        return


def _concat0(first, second, axis):
    if first.shape[axis] == 0:
        return second
    if second.shape[axis] == 0:
        return first
    try:
        y = np.concatenate((first, second), axis=axis)
    except ValueError:
        pdb.set_trace()
    return y
