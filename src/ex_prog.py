import numpy as np
import random
import pdb
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import bernoulli
from src.data_utils import RecordStat
'''
Programs are instrumented with codes to record data. 
Every program takes:
- [inpt]: a tuple used for initializing probabilities used in the program
- [init_tuple]: a tuple used for initializing variables other than
probability in the program.
- [hists]: 
    [None] if it is the first run. 
    A [RecordState] object, otherwise, which records of past runs of the program. 
and returns 
- updated [hists] 

The instrumention is currently done manually. 
See Geo0 for explanations for how to instrument a plain program. 
'''


class NormNameError(Exception):
    """
    Raised when the initial program state does not satisfy assertion invariant
    """

    def __init__(self, message):
        self.message = message


class model_maker:
    def __init__(self, func_array):
        func_array = np.array(func_array)
        self.func_array = func_array
        self.linear_model = lambda x: np.dot(func_array.T, np.array(x))
        self.leafmodelname = ""

    def predict(self, X):
        return np.array([self.linear_model(s) for s in X])

    def loss(self, X, y, y_pred, leafmodelname, normp=2):
        if y.shape != y_pred.shape:
            pdb.set_trace()
        if leafmodelname == "MSE" or "2norm":
            self.leafmodelname = leafmodelname
            return mean_squared_error(y, y_pred)
        elif leafmodelname == "pnorm":
            from cvxpy.atoms.axis_atom import AxisAtom
            from cvxpy.atoms.norm import norm
            self.leafmodelname = "{}_{}".format(leafmodelname, normp)
            return norm(y_pred - y, p=normp).value
        else:
            raise NormNameError("unrecognized norm names")

    def to_string(self, header):
        if len(header) != len(self.func_array) + 1:  # header include y
            import pdb
            pdb.set_trace()
        string = " + ".join(["{}*{}".format(round(self.func_array[i], 1), header[i])
                             for i in range(len(self.func_array))
                             if round(self.func_array[i], 1) != 0
                             ])
        string += "\n{}".format(self.leafmodelname)
        return string, "{}-norm".format(self.leafmodelname)


class VarInfo:
    def __init__(self, progname, const, var, ninput, bools=[]):
        self.const = const
        self.var = var
        self.preds_str_lst = const + var
        self.preds_str = ",".join(self.preds_str_lst)
        self.init_var_indices = np.arange(len(const), len(const) + len(var))
        self.ninput = ninput
        filename = os.path.join("feature", "{}.txt".format(progname))
        with open(filename, "w") as f:
            f.write("benchmark: {}\n".format(progname))
            f.write("\n")
            f.write("branching_variables: \n")
            for feature in var:
                if feature in bools:
                    f.write("{}: bool\n".format(feature))
                else:
                    f.write("{}: int\n".format(feature))
            f.write("\n")
            f.write("feature_expressions: \n")
            for i in range(ninput):
                f.write("f{}: prob{}\n".format(i, i+1))
            for feature in const:
                f.write("f{}: {}\n".format(
                    const.index(feature)+ninput, feature))
            for feature in var:
                f.write("f{}: {}\n".format(
                    var.index(feature)+len(const)+ninput, feature))
            f.write("\n")
        f.close()

    def index_in_whole(self, name):
        if name in self.var:
            return self.ninput + len(self.const) + self.var.index(name)
        if name in self.const:
            return self.ninput + self.const.index(name)
        if name == "1":
            return self.ninput + len(self.const) + len(self.var)
        if name.startswith("p"):  # HACKY
            return int(name[-1]) - 1
        raise Exception

    def linear_func(self, *args):
        lst = np.zeros(self.ninput + len(self.const) + len(self.var) + 1)
        for arg in args:
            if len(arg) == 2:
                name, coef = arg
                lst[self.index_in_whole(name)] = coef
            else:
                lst[self.index_in_whole(arg)] = 1
        return list(lst)


# -------------------------------------------------------------------------------
# Geometric distributions
# Defn.
# The probability distribution of the number X of Bernoulli trials needed to get one success, supported on the set { 1, 2, 3, ... }
# or
# The probability distribution of the number Y = X − 1 of failures before the first success, supported on the set { 0, 1, 2, 3, ... }
# -------------------------------------------------------------------------------

'''
The number of iteration before [flip] turns to 1. 

Program variables: flip, z
Guard variables: flip
Augmented wpfeatures: (1 - p)/p
wp(Geo 0, z) = [flip != 0] * z + [flip == 0] * (z + (1 - p)/p)
'''


def Geo0(progname, inpt, hists,  init_tuple):

    # we initialize [hists] if it was [None]
    if hists is None:
        # Construct VarInfo(progname, constants to record, variables to record, ninput=...)
        # While which variables to record depends on our domain knowledge of the program,
        # by default we add a 0-1 variable indicates whether the guard is true,
        # each variables that need to be initialized before the loop,
        # and the probability expression appeared in invariants.
        vI = VarInfo(progname, ["(1 - p1)/p1"],
                     ["flip", "z"], ninput=1, bools=["flip"])
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + (1 - p1)/p1)"
        # If TESTING_KNOWN_MODEL is True, then we need to hard code [known_inv_model];
        # Otherwise, we could leave it to None
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("flip"),
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        # this encodes the function: z + (1 - p1)/p1
                        "model": model_maker(vI.linear_func("z", "(1 - p1)/p1"))},
            "root_>": {"split": False,
                       # this encodes the function: z
                       # If you want to encode the function 2z, model_maker(vI.linear_func((2,"z")))
                       "model": model_maker(vI.linear_func("z"))},
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    z = init_tuple["int"][0]
    flip = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate(
        [(1 - p1)/p1, flip, z])
    # The original loop
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=p1)[0]
        if d:
            flip = 1
        else:
            z = z + 1
    # Record the realized post
    hists.record_predicate_end(z)
    return hists


'''
Geo0 with extra variable x creating noise. 
x gets doubled whenever z gets increased by 1.
Program variables: flip, z, p1
Guard variables: flip
Augmented features: (1 - p)/p
wp(Geo 0, z) = [flip != 0] * z + [flip == 0] * (z + (1 - p1)/p1)
'''


def Geo1(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1 - p1)/p1"],
                     ["flip", "x", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + (1 - p1)/p1)"
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("flip"),
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        "model": model_maker(vI.linear_func("z", "(1 - p1)/p1"))},
            "root_>": {"split": False,
                       "model": model_maker(vI.linear_func("z"))},
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    z = init_tuple["int"][0]
    x = init_tuple["int"][1]
    flip = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate(
        [(1 - p1)/p1, flip, x, z])
    # The original loop
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=p1)[0]
        if d:
            flip = 1
        else:
            x = x * 2
            z = z + 1
    # Record the realized post
    hists.record_predicate_end(z)
    return hists


'''
Geo1 with extra variable x creating noise such that 
x gets increased by 1 whenever z gets increased by 1.

wp(Geo2, z) = [flip != 0] * z + [flip == 0] * (z + (1 - p1)/p1)
'''


def Geo2(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1 - p1)/p1"],
                     ["flip", "x", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + (1 - p1)/p1)"
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("flip"),
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        "model": model_maker(vI.linear_func("z", "(1 - p1)/p1"))},
            "root_>": {"split": False,
                       "model": model_maker(vI.linear_func("z"))},
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    z = init_tuple["int"][0]
    x = init_tuple["int"][1]
    flip = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate(
        [(1 - p1)/p1, flip, x, z])
    # The original loop
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=p1)[0]
        if d:
            flip = 1
        else:
            x = x + 1
            z = z + 1
    # Record the realized post
    hists.record_predicate_end(z)
    return hists


'''
This is a program tossing 2 fair coins in one while loop.
Program variables: p1, p2, count, c1, c2
Guard variables: c1 or c2
Augmented features: (p1 + p2)/(p1 + p2 - p1 * p2)
wp(Fair, count) = count + [not (c1 or c2)] * (p1 + p2) / (p1 + p2 - p1 * p2)
'''


def Fair(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(p1 + p2)/(p1 + p2 - p1 * p2)"],
                     ["c1orc2", "count"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[c1 or c2] * count + [not (c1 or c2)]* (count + (p1 + p2)/ (p1 + p2 - p1 * p2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("c1orc2"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("count", "(p1 + p2)/(p1 + p2 - p1 * p2)"))},  # count + (p1 + p2)/ (p1 + p2 - p1 * p2)
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("count"))},  # count
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    p2 = inpt[1]
    count = init_tuple["int"][0]
    c1 = init_tuple["bool"][0]
    c2 = init_tuple["bool"][1]
    # Record initial program states
    hists.record_predicate(
        [(p1 + p2)/(p1 + p2 - p1 * p2), (c1 or c2), count])
    # The original loop
    while not (c1 or c2):
        c1 = bernoulli.rvs(size=1, p=p1)[0]
        if c1:
            count = count + 1
        c2 = bernoulli.rvs(size=1, p=p2)[0]
        if c2:
            count = count + 1
    # Record the realized post
    hists.record_predicate_end(count)
    return hists


''' 
Martingale. It is from the page 202 of Prinsys paper. 
We can think the program as simulating a betting strategy, where b is the bet, 
and c is the accumulated capital. In iterations that we do not win, the amount 
of the current bet b is getting deducted from the captial c, and we will then 
double our bet for the next round. In the iteration that we win, we win all the 
current bet b, and get to keep all the captial c. 

Program variables: c, b, rounds, p
Guard variables: b
Augmented features: (1 - p1)/ p1
wp(Mart, count) = rounds + [b > 0] * (1-p1)/(p1)
'''


def Mart(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1-p1)/p1"],
                     ["b", "c", "rounds"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[b <= 0] * rounds + [b > 0] * (rounds + (1-p1)/p1)"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("b"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("rounds"))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("rounds", "(1-p1)/p1"))},
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    c = init_tuple["int"][0]
    b = init_tuple["int"][1]
    rounds = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate(
        [(1-p)/p,  b, c, rounds])
    # The original loop
    while b > 0:
        d = bernoulli.rvs(size=1, p=p)
        if d:
            c = c+b
            b = 0
        else:
            c = c-b
            b = 2*b
        rounds += 1
    # Record the realized post
    hists.record_predicate_end(rounds)
    return hists


'''
Gambler's ruin problem / Random walk on a line with equal probability to go 
left or right
From Lagrange Interpolation (Chen et. al.) Paper
Assumed p = 0.5 
(We can compute the invariant when p is not 0.5 but its form is complicated. )
Program variables: x, y, z
Guard variable: 0 < x < y
Constants: None
Augmented features: x * (y - x)
Invariant: z + [0 < x and x < y] * x * (y - x)
'''


def Gambler0(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "z", "0 < x < y", "x * (y - x)"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "z + [0 < x and x < y] * x * (y - x)"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("0 < x < y"),  # index for b
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("z", "x * (y - x)"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z"))},
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = 0.5
    x = init_tuple["int"][0]
    y = init_tuple["int"][1]
    z = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate(
        [x, y, z, 0 < x and x < y, x * (y - x)])
    # The original loop
    while 0 < x and x < y:
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            x = x + 1
        else:
            x = x - 1
        z = z + 1
    # Record the realized post
    hists.record_predicate_end(z)
    return hists


'''
Lagrange Interpolation paper:
A mix of Geometric Distribution and Arithmetic progression. 
It's like a geometric distribution in the sense that in each iteration, 
[z] has probability [p] to become 0, and probability 1-[p] to stay the same, 
and the loop ends when [z] equals 0.
It's also like an arithmetic distribution in the sense that 
x = 1 + 2 + 3 + ... + n, where n is the number of iterations it runs before 
it exits the loop. 

Program variables: z, x, y, p
Guard variable: z
Augmented feature: p/((1-p)**2)
 wp(GeoAr0, x) =  [z = 0]* x + [z != 0] (x + y(1-p)/p + (1-p)/(p**2))
'''


def GeoAr0(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["(1-p)/(p^2)"], ["x", "y(1-p)/p", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[z = 0]* x + [z != 0] (x + y(1-p)/p + (1-p)/(p**2))"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    y = init_tuple["int"][1]
    z = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate(
        [(1-p)/(p**2), x, y*(1-p)/p, z])
    # The original loop
    while not (z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Same as GeoAr0 but fix the initial value of p and y
y = 0, p = 0.25
Program variables: z, x, y
Guard variable: z
Augmented feature: None
wp(GeoAr1, x) = [z = 0]* x + [z != 0] (x + 12)
'''


def GeoAr1(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "I= [z = 0]* x + [z != 0] (x + 12)"
        known_inv_model = None
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = 0.25
    x = init_tuple["int"][0]
    y = 0
    z = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate([x, y, z])
    # The original loop
    while not(z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Same as GeoAr0 but fix the initial value of p and y
p = 0.25

Program variables: z, x
Guard variable: z
Augmented feature: None
wp(GeoAr2, x) = [z = 0]* x + [z != 0] (x + 3y + (12)
'''


def GeoAr2(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[z = 0]* x + [z != 0] (x + 3y + 12)"
        known_inv_model = None
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = 0.25
    x = init_tuple["int"][0]
    y = init_tuple["int"][1]
    z = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate([x, y, z])
    # The original loop
    while not (z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Same as GeoAr0 but fix the initial value of p and yp
y = 0

Program variables: z, x, p
Guard variable: z
Augmented feature: (1-p)/(p^2)
wp(GeoAr3, x) = [z = 0]* x + [z != 0] (x + (1-p)/(p^2))
'''


def GeoAr3(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["(1-p)/(p^2)"], ["x", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[z = 0]* x + [z != 0] (x + (1-p)/(p^2))"
        known_inv_model = None
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    y = 0
    z = init_tuple["bool"][0]
    # Record initial program states
    hists.record_predicate([(1-p)/(p**2), x, z])
    # The original loop
    while not(z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


# -------------------------------------------------------------------------------
# Binomial distributions
# usually have non-linear invariant
# -------------------------------------------------------------------------------
'''
Binomial Distribution in Lagrange Interpolation paper (Chen et. al.): 
n iterations in total, and in each iteration, increase x by y with probability p. 

Program variables: x, n, y, p
Guard variable: n
Augmented features: p * n * y
wp(Bin0, x) = [n > 0] * (x + p * n * y) + [n <= 0] * x
'''


def Bin0(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "n", "p * n * y"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n > 0] * (x + p * n * y) + [n <= 0] * x"
        known_inv_model = None
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    y = init_tuple["int"][1]
    n = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate([x, y, n, p * n * y])
    # The original loop
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + y
        n = n-1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Algorithm which generates a sample x distributed binomially with parameters p and M.
src: http://www-i2.informatik.rwth-aachen.de/pub/index.php?type=download&pub_id=1274
pg-120 Fig 5.9

Program variables: n, x, M, p
Guard variables: n - M
Augmented features: p*M, p*n
wp(Bin1, x) = [n − M < 0] * (x - p*n + p*M) + [n − M >= 0] * x
'''


def Bin1(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p*M"],
                     ["x", "n", "n-M", "p*n"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("n-M"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func(("x", 1), ("p*n", -1), ("p*M", 1)))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func())},
                           }
        known_inv = "[n − M < 0] * (x - p*n + p*M) + [n − M >= 0] * x"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    n = init_tuple["int"][0]
    x = init_tuple["int"][1]
    M = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate([M * p, x, n, n-M, n*p])
    # The original loop
    while n - M < 0:
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            x = x + 1
        n = n + 1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Binomial Distribution 2: 
we keep a counter n, enter an iteration only if n>0, and decrease the counter in 
each iteration. In each iteration, with probability p, x gets increased by some 
y (for some preset y fixed throughout the whole program); with probability 1-p, 
x gets increased by n. 
It is bin2 from Lagrange Interpolation paper by Chen et al. (their program bin2)

Program variables: x, n, y, p
Guard variable: n
Augmented features: pn(n+1), (1-p)ny
wp(Bin2,x) = [n > 0] * (0.5pn(n+1) + (1-p)ny)
'''


def Bin2(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "pn(n+1)", "(1-p)ny"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[y >= 0 and n − 1 >= 0] * (0.5pn(n+1) + (1-p)ny)"
        known_inv_model = None
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    y = init_tuple["int"][1]
    n = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate([x, y, p * n * (n+1), (1-p)*n*y])
    # The original loop
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        else:
            x = x + y
        n = n-1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Same as Bin2 except set p=0.25

Program variables: x, n, y
Guard variable: n
Augmented features: n(n+1), ny
wp(Bin3,x) = [n > 0] * 0.125n(n+1) + 0.75ny
'''


def Bin3(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "n(n+1)", "ny"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[y >= 0 and n > 0] * 0.125n(n+1) + 0.75ny)"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = 0.25
    x = init_tuple["int"][0]
    y = init_tuple["int"][1]
    n = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate([x, y, n * (n+1), n*y])
    # The original loop
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        else:
            x = x + y
        n = n-1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
A binomial distribution such that in each iteration, [count] increases by a sum 
of three random variables' realized values, whose expected value can be calculated 
by the linearity of expectation. 

Program variables: count, n, x1, x2, x3
Guard variable: n
Augmented features: None
wp(LinExp, count) = [n<=0] * count + [n>0] * (count + n*(7*3/8))
'''


def LinExp(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x1 ", "x2", "x3", "n", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = None
        known_inv = "[n<=0] * count + [n>0] * (count + n*(7*3/8))"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    n = init_tuple["int"][0]
    count = init_tuple["int"][1]
    x1, x2, x3 = init_tuple["bool"][0], init_tuple["bool"][1], init_tuple["bool"][2]
    # Record initial program states
    hists.record_predicate([x1, x2, x3,  n, count])
    # The original loop
    while(n > 0):
        x1 = bernoulli.rvs(size=1, p=0.5)[0]
        x2 = bernoulli.rvs(size=1, p=0.5)[0]
        x3 = bernoulli.rvs(size=1, p=0.5)[0]
        n = n - 1
        c1 = x1 or x2 or x3
        c2 = (not x1) or x2 or x3
        c3 = x1 or (not x2) or x3
        count = count + c1 + c2 + c3
    # Record the realized post
    hists.record_predicate_end(count)
    return hists


# -------------------------------------------------------------------------------
# Others

'''
Learn the preexpectation w.r.t. two sequencing loops at once

Program variables: p1, p2, x, flip1, flip2
Guard variable: flip1, flip2
Augmented features: p1/(1-p1), p2/(1-p2) 
wp(Seq0, x) = x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (p2/(1-p2))
'''


def Seq0(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)"],
                     ["x", "flip1", "flip2"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (p2/(1-p2))"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    p2 = inpt[1]
    x = init_tuple["int"][0]
    flip1 = init_tuple["bool"][0]
    flip2 = init_tuple["bool"][1]
    # Record initial program states
    hists.record_predicate([p1/(1-p1), p2/(1-p2), x, flip1, flip2])
    # The original loop
    while (flip1 == 0):
        d1 = bernoulli.rvs(size=1, p=p1)
        if d1:
            x = x + 1
        else:
            flip1 = 1
    while (flip2 == 0):
        d2 = bernoulli.rvs(size=1, p=p2)
        if d2:
            x = x + 1
        else:
            flip2 = 1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Seq0 but one loop increases x, another loop decreases x. 

Program variables: p1, p2, x, flip1, flip2
Guard variable: flip1, flip2
Augmented features: p1/(1-p1), p2/(1-p2) 
wp(Seq1, x) = x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (-p2/(1-p2))
'''


def Seq1(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)"],
                     ["x", "flip1", "flip2"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (-p2/(1-p2))"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    p2 = inpt[1]
    x = init_tuple["int"][0]
    flip1 = init_tuple["bool"][0]
    flip2 = init_tuple["bool"][1]
    # Record initial program states
    hists.record_predicate([p1/(1-p1), p2/(1-p2), x, flip1, flip2])
    # The original loop
    while (flip1 == 0):
        d1 = bernoulli.rvs(size=1, p=p1)
        if d1:
            x = x + 1
        else:
            flip1 = 1
    while (flip2 == 0):
        d2 = bernoulli.rvs(size=1, p=p2)
        if d2:
            x = x - 1
        else:
            flip2 = 1
    # p4
    hists.record_predicate_end(x)
    return hists


'''
Learn the invariant of nested loops

Program variables: p1, p2, x, flip1, flip2
Guard variable: flip1, flip2
Augmented features: p1/(1-p1), p2/(1-p2) 
Known Invariant: x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) + [flip1 = 0 and flip2 != 0] * p1 * p1 /(1-p1) * p2/(1-p2)
'''


def Nest(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1) * p2/(1-p2)", "p1* p1/(1-p1) * p2/(1-p2)"],
                     ["x", "flip1", "flip2"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) + [flip1 = 0 and flip2 != 0] * p1* p1/(1-p1) * p2/(1-p2)"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    p2 = inpt[1]
    x = init_tuple["int"][0]
    flip1 = init_tuple["bool"][0]
    flip2 = init_tuple["bool"][1]
    # Record initial program states
    hists.record_predicate(
        [(p1/(1-p1)) * p2/(1-p2), (p1 * p1/(1-p1)) * p2/(1-p2), x, flip1, flip2])
    # The original loop
    while (flip1 == 0):
        d1 = bernoulli.rvs(size=1, p=p1)
        if d1:
            while (flip2 == 0):
                d2 = bernoulli.rvs(size=1, p=p2)
                if d2:
                    x = x + 1
                else:
                    flip2 = 1
            flip2 = 0
        else:
            flip1 = 1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
wp(body, x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) + 
         [flip1 = 0 and flip2 != 0] * (2p1-1)/(1-p1) * p2/(1-p2))
= p1*wp(block1, x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) 
       + [flip1 = 0 and flip2 != 0] * (2p1-1)/(1-p1) * p2/(1-p2))) 
+(1-p1)*x
= p1*wp(loop1, x+ [flip1 = 0 and 0 = 0] * p1/(1-p1) * p2/(1-p2))) 
+(1-p1)*x
= p1* (x + [flip2 == 0] p2/(1-p2) + [flip1 = 0] * p1/(1-p1) * p2/(1-p2))) 
+(1-p1)*x
= x + [flip2 == 0] p2/(1-p2) * p1 + [flip1 = 0] * p1/(1-p1) * p2/(1-p2) * p1
'''

'''
Probabilistic sum of arithmetic series
Adapted from the Lagrange Interpolation paper by Chen et. al. 

Program variables: x, n, p
Guard variables: n
Augmented features: n(n+1)
wp(Sum0, x) = [n > 0] * (x + p * (0.5n(n+1))) + [n <= 0] * x
'''


def Sum0(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "n", "pn", "pn^2"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n > 0] * (x + 0.5pn(n+1)) + [n <= 0] * x"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    n = init_tuple["int"][1]
    # Record initial program states
    hists.record_predicate([x, n, p * n, p * n * n])
    # The original loop
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        n = n - 1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Sum0 with specialized p

Program variables: x, n
Guard variables: n
Augmented features: n(n+1)
wp(Sum1, x) = [n > 0] * (x + (0.25n(n+1))) + [n <= 0] * x
'''


def Sum1(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "n", "n^2"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n > 0] * (x + 0.25 * n(n+1)) + [n <= 0] * x"
        known_inv_model = None

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = 0.5
    x = init_tuple["int"][0]
    n = init_tuple["int"][1]
    # Record initial program states
    hists.record_predicate([x, n, n * n])
    # The original loop
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        n = n - 1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
Product of dependent random variables
Adapted from the Lagrange Interpolation paper by Chen et. al. 
Program variables: x, n, y
Guard: n
Augmented features: n^2, nx, ny, xy
wp(DepRV, xy) ([n>0]*(1/4(n^2 + 2nx + 2ny + 4xy - n))+ [n<=0]*(xy)
'''


def DepRV(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "n", "y",
                                    "n^2", "nx", "ny", "xy"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n>0]*(1/4(n^2 + 2nx + 2ny + 4xy - n))+ [n<=0]*(xy)"
        known_inv_model = None
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = 0.5
    x = init_tuple["int"][0]
    n = init_tuple["int"][1]
    y = init_tuple["int"][2]
    # Record initial program states
    hists.record_predicate([x, n, y, n**2, n*x, n*y, x*y])
    # The original loop
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + 1
        else:
            y = y + 1
        n = n - 1
    # Record the realized post
    hists.record_predicate_end(x*y)
    return hists


'''
Simulation of fair coin with biased coin: opposite of Hurd’s algorithm
Adpated from Prinsys Paper's Lst 7

Program variables: x, y, p1
Guard variables: x - y == =
Augmented features: None
wp(Bias0Prinsys, [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0])
= [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]
= [not x and y] - [x and not y]
= [x = 0]*(y) + [x = 1]*([y = 0]*(-1) + [y = 1] * 0)
'''


def Bias0Prinsys(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "y", "x-y"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": True,
                                       "j_feature": vI.index_in_whole("y"),
                                       "threshold": 0},
                           "root_>": {"split": True,
                                      "j_feature": vI.index_in_whole("y"),
                                      "threshold": 0},
                           "root_>_<=": {"split": False,
                                         "model": model_maker(vI.linear_func(("1", -1)))},
                           "root_>_>": {"split": False,
                                        "model": model_maker(vI.linear_func())},
                           "root_<=_<=": {"split": False,
                                          "model": model_maker(vI.linear_func())},
                           "root_<=_>": {"split": False,
                                         "model": model_maker(vI.linear_func(("1", 1)))}
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    x = init_tuple["bool"][0]
    y = init_tuple["bool"][1]
    # Record initial program states
    hists.record_predicate([x, y, x-y])
    # The original loop
    while(x-y == 0):
        d1 = bernoulli.rvs(size=1, p=p1)[0]
        if(d1):
            x = 0
        else:
            x = 1
        d2 = bernoulli.rvs(size=1, p=p1)[0]
        if(d2):
            y = 0
        else:
            y = 1
    # Record the realized post
    hists.record_predicate_end(
        int(x == 0 and y - 1 == 0) - int(x - 1 == 0 and y == 0))
    return hists


'''
A variation of 11 where we let post be x.
Adpated from Prinsys Paper's Lst 7
 
Program variables: x, y, p1
Guard variables: x - y == 0
Augmented features: None 
wp(Bias0direct, x) = [x != y] * x + [x == y] * 1/2
'''


def Bias0direct(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "y", "x=y"], ninput=1)
        preds_str = vI.preds_str
        known_inv = " [x - y != 0] * x + [x - y == 0] * 1/2"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x=y"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": True,
                                       "model": model_maker(vI.linear_func(("1", 0.5)))},
                           "root_>": {"split": True,
                                      "model": model_maker(vI.linear_func(("x", 1)))}
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1 = inpt[0]
    x = init_tuple["bool"][0]
    y = init_tuple["bool"][1]
    # Record initial program states
    hists.record_predicate([x, y, int(x == y)])
    # The original loop
    while(x-y == 0):
        d1 = bernoulli.rvs(size=1, p=p1)[0]
        if(d1):
            x = 0
        else:
            x = 1
        d2 = bernoulli.rvs(size=1, p=p1)[0]
        if(d2):
            y = 0
        else:
            y = 1
    # Record the realized post
    hists.record_predicate_end(x)
    return hists


'''
An example where one probability does not matter
Adapted from the Prinsys paper's Lst. 2

Program variables: p1, p2, x
Guard variables: x
Augmented features: None
wp(Prinsys, x==1) = [x = 0] * (1-p2) + [x != 0] * [x = 1]
'''


def Prinsys(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[x = 0] * (1-p2) + [x != 0] * [x = 1]"
        known_inv_model = {"root": {"split": True,
                                    "j_feature": vI.index_in_whole("x"),
                                    "threshold": 0
                                    },
                           "root_<=": {"split": True,
                                       "j_feature": vI.index_in_whole("x"),
                                       "threshold": -1},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func(("1", 1)))},
                           "root_<=_>": {"split": False,
                                         "model": model_maker(vI.linear_func(("p2", -1), ("1", 1)))},
                           "root_<=_<=": {"split": False,
                                          "model": model_maker(vI.linear_func())},
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p1, p2 = inpt[0], inpt[1]
    x = init_tuple["int"][0]
    # Record initial program states
    hists.record_predicate([x])
    # The original loop
    while(x == 0):
        d1 = bernoulli.rvs(size=1, p=p1)[0]
        if(d1):
            x = 0
        else:
            d2 = bernoulli.rvs(size=1, p=p2)[0]
            if(d2):
                x = -1
            else:
                x = 1
    # Record the realized post
    hists.record_predicate_end(x == 1)
    return hists


'''
Duelling cowboys: interested in the probability such that player A wins
Adapted from https://moves.rwth-aachen.de/wp-content/uploads/WS1819/probprog/prob-prog-2018-lec89.pdf

Program variables: p1, p2, t, c
Guard variables: c == 1
Augmented features: p1/(p1 + p2 - p1 * p2), (1 - p2)/(p1 + p2 - p1 * p2)
wp(Duel, t) = [t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 [t = B and c = 1] * ((1 - p2)/(p1 + p2 - p1 * p2))
'''


def Duel(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["p1/(p1 + p2 - p1 * p2)", "(1 - p2)p1/(p1 + p2 - p1 * p2)"], ["t", "c"], ninput=2)
        preds_str = vI.preds_str
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("c"),
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        "model": model_maker(vI.linear_func("t"))},
            "root_>": {"split": True,
                       "threshold": 0,
                       "j_feature": vI.index_in_whole("t")},
            "root_>_<=": {"split": False,
                          "model": model_maker(vI.linear_func("(1 - p2)p1/(p1 + p2 - p1 * p2)"))},
            "root_>_>": {"split": False,
                         "model": model_maker(vI.linear_func("p1/(p1 + p2 - p1 * p2)"))},
        }
        known_inv = "[t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 [t = B and c = 1] * ((1 - p2)p1/(p1 + p2 - p1 * p2))"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)

    # Initialize program variables
    p1, p2 = inpt[0], inpt[1]
    t = init_tuple["bool"][1]  # t = True if t = A, t = False if t = B
    c = init_tuple["bool"][0]  # whether the game has ended
    # Record initial program states
    hists.record_predicate(
        [p1/(p1 + p2 - p1*p2), (1-p2)*p1/(p1 + p2 - p1*p2), t, c])
    # The original loop
    while(c == 1):
        if t:
            d1 = bernoulli.rvs(size=1, p=p1)[0]
            if d1:
                c = 0
            else:
                t = not t
        else:
            d2 = bernoulli.rvs(size=1, p=p2)[0]
            if d2:
                c = 0
            else:
                t = not t
    # Record the realized post
    hists.record_predicate_end(t)  # t == A
    return hists


'''
Sampling from uniform distribution
Program variables: p, x, count
Guard variable: x
Augmented features: None
wp(Unif, count) = count + [x <= 10]*(10-x+1)
'''


def Unif(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = None
        known_inv = "count + [x <= 10]*(10-x+1)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    count = init_tuple["int"][1]
    # Record initial program states
    hists.record_predicate([x, count])
    # The original loop
    while(x <= 10):
        x = x + random.choice(list(np.arange(0.1, 2, 0.1)))
        count = count + 1
    # Record the realized post
    hists.record_predicate_end(count)
    return hists


'''
Similar to Unif but no sampling

Program variables: p, x, count
Guard variable: x
Augmented features: None
Known invariant: count + [x <= 10]*(10-x+1) 
'''


def Detm(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = None
        known_inv = "count + [x <= 10]*(10-x+1)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    count = init_tuple["int"][1]
    # Record initial program states
    hists.record_predicate([x, count])
    # The original loop
    while(x <= 10):
        x = x + 1
        count = count + 1
    # Record the realized post
    hists.record_predicate_end(count)
    return hists


'''
A program that has a diverging path with p 0 and (positively) almost surely terminate;
and conceptually like inverse binonmial distribution. 
Adapted from the example on page 125 of Kaminski's thesis 
Known invariant: [x > 0] * (z + x*(1/p)) + [x <= 0](z)
'''
'''
TODO
wp(body, [x > 0] * (z + x*(1/p)) + [x <= 0](z))
= wp(probabilistic branch, [x > 0] * (z + 1 + x*(1/p)) + [x <= 0](z + 1))
= p * ([x-1 > 0] * (z + 1 + (x-1)*(1/p)) + [x - 1 <= 0](z + 1)) 
+ (1-p) * ([x > 0] * (z + 1 + x*(1/p)) + [x <= 0](z + 1)) 
= [x <= 0](z + 1) + [x = 1](z + 1 + x *(1-p)/p) + [x > 1]*(z + 1 + x*(1/p) - p/p)
[G]*wp(body, [x > 0] * (z + x*(1/p)) + [x <= 0](z))
= [x > 0] ([x = 1](z + 1 + (1-p)/p) + [x > 1]*(z + x*(1/p)))
= [x > 0] ([x = 1](z + 1/p) + [x > 1]*(z + x*(1/p)))
= [x > 0] (z + x*(1/p))
'''


def RevBin(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "x*(1/p)", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z"))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("z", ("x*(1/p)", 1)))},
                           }
        known_inv = "[x >= 1] * (z + x*(1/p)) + [x <= 0](z)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)
    # Initialize program variables
    p = inpt[0]
    x = init_tuple["int"][0]
    z = init_tuple["int"][1]
    # Record initial program states
    hists.record_predicate([x, x/p, z])
    # The original loop
    while(x-1 >= 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x - 1
        z = z + 1
    # Record the realized post
    hists.record_predicate_end(z)
    return hists


# ----------------------------------------------------
# the following are not included now
'''
http://www-i2.informatik.rwth-aachen.de/pub/index.php?type=download&pub_id=1274
pg-120
generate biased coin from a fair coin.
The algorithm generates a sample x = 1 with probability p and
x = 0 with probability 1 − p by repeatedly fliping a fair coin when (p == 1/2),
which works because
    # 1. for p > 0.5: with p 0.5, always return x = 1;
    # with p 0.5, recurse with p' = p*2 - 1 =(p-0.5)*2,
    # by induction, returns x = 1 with p (p-0.5)*2 in this case. 
    # Thus, in expectation, it returns x = 1 with p 0.5 * 1 + 0.5 * (p-0.5)*2 = p
    # 2. for p = 0.5: with p 0.5, return x = 1, with 0.5, recurse with p' = 0,
    # which always returns x = 0.
    # 3. for p < 0.5: with p 0.5, always return x = 0;
    # with p 0.5, recurse with p' = 2 * p,
    # thus in expectation, it returns x = 1 with p 0.5 * 2 * p = p
Fig 5.8
I = [p >= 0 and p − 1 <= 0 and (b − 1 == 0 or p == 0 or p − 1 == 0)] * (p)
 (b − 1 == 0 or p == 0 or p− 1 == 0)] is assertion invariant. 
 
 TODO
'''


def ex17(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "b"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {
            "easy_side": "<="
        }
        known_inv = "[p >= 0 and p − 1 <= 0] * (p)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           variable_indices=vI.init_var_indices)

    p = init_tuple["float"][0]
    b = init_tuple["bool"][0]
    init_p, init_b = p, b
    hists.record_predicate([init_p, init_b, p, b])
    while b:
        b = bernoulli.rvs(size=1, p=0.5)[0]
        if b:  # if b==1, x = x*2 mod 1
            p = 2 * p
            if(p - 1 >= 0):
                p = p - 1
        elif(p - 0.5 >= 0):
            p = 1
        else:
            p = 0
    hists.record_predicate_end(p)
    return hists


'''
let I = [p >= 0 and  p − 1 <= 0] * (p)
wp(body, I) = [b] * wp(block, I) + [not b and p >= 0.5] * wp(p=1, I) + [not b and p < 0.5] * wp(p=0, I)
= [b] * wp(block, I) + [not b and p >= 0.5] * 1 + [not b and p < 0.5] * 0
= [b] * wp(p = 2p, [p>=1] * [p-1 >= 0 and  p − 2 <= 0 ] * (p-1) + [p<1] * I )+ [not b and p >= 0.5] * 1 + [not b and p < 0.5] * 0
= [b] * ([2p>=1] * [2p-1 >= 0 and  2p − 2 <= 0 ] * (2p-1) + [2p<1] * [2p >= 0 and  2p − 1 <= 0 ] * (2p) )+ [not b and p >= 0.5] * 1 + [not b and p < 0.5] * 0
= [b] * ([p>=0.5] * [ p <= 1] * (2p-1) + [p<0.5] * [2p >= 0] * (2p) )+ [not b and p >= 0.5] * 1 + [not b and p < 0.5] * 0
[G]*wp(body, I) =  [b] * ([p>=0.5] * [ p <= 1] * (2p-1) + [p<0.5] * [2p >= 0] * (2p) )

let I = (b − 1 == 0 or p == 0 or p− 1 == 0) * [p >= 0 and  p − 1 <= 0 ] * (p)
then wp(body, I) = [b] * wp(first block, I) + [not b and p >= 0.5] * 1 + [not b and p < 0.5] * 0
 wp(block, I)
= wp(p = 2p, [p - 1 >= 0] * (b − 1 == 0 or p-1 == 0 or p− 2 == 0) * [p-1 >= 0 and  p − 2 <= 0 ] * (p-1) + [p < 1] * I)
= [2p - 1 >= 0] * (b − 1 == 0 or 2p-1 == 0 or 2p− 2 == 0) * [2p-1 >= 0 and  2p − 2 <= 0 ] * (2p-1) + [2p < 1] * (b − 1 == 0 or 2p == 0 or 2p− 1 == 0) * [2p >= 0 and  2p − 1 <= 0 ] * (2p))
= [p == 1] + [p!= 0.5 and p!= 1 and b=1] ([0.5<p<=1] * (2p-1) + [0<=p<0.5] * (2p)])
very strange

wpe(body, p) = [b] * wp(first block, p) + [not b and p >= 0.5] * 1 + [not b and p < 0.5] * 0
'''


'''
SKIPPED for now
Coupon collector problem:
Kaminski
# col= number of coupons collected till now
I= 1 + Sum_(l=0 to w)[x>l] * (3 + 2 * Sum_(k=0 to w) ((#col+l)/N)^k) - \
   (2 * [cp[i] == 0]*[x>0] * Sum_(k=0 to w) (#col/N)^k)
'''


def ex16(progname, inpt, hists,  init_tuple, constraint):
    x = 10
    cp = np.zeros(20)
    i = 1
    z = 0
    p1, p2 = inpt[:2]
    while(x-1 >= 0):
        while not(cp[i] == 0):
            i = random.randint(1, 10)
        cp[i] = 1
        x = x - 1
        z = z + 1
    return z
