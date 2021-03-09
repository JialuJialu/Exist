import numpy as np
import random
import pdb
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import bernoulli
from src.data_utils import RecordStat
'''
Programs here are from example programs in benchmark.py and are instrumented
with codes to help analysis.
Every program takes:
- inpt: a tuple used for initializing probabilities used in the program
- hists: RecordState
    record of past runs of the program under the same probability initialization
    hists is None if it's the first run.
- NUM_ITR: used to set hists.NUM_ITR
- init_tuple: a tuple used for randomizing initialization of variables other than
probability in the program.

The instrumention is currently done manually. Ex.
Say in benchmark.py, we have
def ex(input):
    x <- something
    guard <- something
    while guard:
        loop body
    return x
we would instrument it to

def ex(input, hists,  init_tuple,constraint\):
    if hists is None: #we initialize hists
        wp = None # a function for the symbolic checker to compute the weakest precondition, not currently used
        vI = VarInfo(progname, constants to record, variables to record, ninput=...)
        preds_str = vI.preds_str
        known_inv = the string invariants
        # known_inv_model is used when we want to check how well the ground invairant does
        # For example, for example 0, it looks like
        known_inv_model = {
                           "easy_side": ">"  # currently, the sign is either <= or >, easy side is the side that we will fit the first. Either <= or > is fine
                            "root": {"j_feature": vI.index_in_whole("flip"),  # index for flip, the variable that we make the split on
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z", "(1 - prob1)/prob1"))},  # this encodes z + (1 - prob1)/prob1
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("z"))},  # this encodes z; model_maker(vI.linear_func((2,"z"))) encodes 2z}
                           }
    prob1,... = inpt[0], ... #get prob from inpt
   # generate hard constraints
    x = random.choice(np.linspace(-10, 10, 21))
    hists.record_hard_constraint(constants to record, variables to record, not guard, assertion invariant, post)
    # instrumented
    x = init_tuple["int"][0] or x = init_tuple["int"][1]
    guard <- init_tuple[j]
    init_x, init_guard = x, gaurd
    # record the state
    hists.record_variable_info()
    hists.record_predicate([init_x, init_guard, x, guard, ...])
    while guard:
        loop body
        hists.record_predicate([init_x, init_guard, x, guard, ...])
    hists.record_predicate_end(post)
    return hists

-----
Also see the example 0 below for how to instrument to a benchmark
'''
bool_space = np.linspace(0, 1, 2)
non_neg_int_space = np.linspace(0, 4, 5)
int_space = np.linspace(-10, 10, 21)
prob_space = np.linspace(0, 1, 11)


# class AssertionInvError(Exception):
#     """
#     Raised when the initial program state does not satisfy assertion invariant
#     """

#     def __init__(self, message):
#         self.message = message


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
        self.loss_func = ""

    def predict(self, X):
        return np.array([self.linear_model(s) for s in X])

    def loss(self, X, y, y_pred, loss_func, normp):
        if y.shape != y_pred.shape:
            pdb.set_trace()
        if loss_func == "MSE" or "MSEwithConstraint":
            self.loss_func = loss_func
            return mean_squared_error(y, y_pred)
        elif loss_func == "pnorm":
            from cvxpy.atoms.axis_atom import AxisAtom
            from cvxpy.atoms.norm import norm
            self.loss_func = "{}_{}".format(loss_func, normp)
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
        string += "\n{}".format(self.loss_func)
        return string, "{}-norm".format(self.loss_func)


class VarInfo:
    def __init__(self, progname, const, var, ninput, bools=[]):
        self.const = const
        self.var = var
        self.preds_str_lst = const + \
            ["init_{}".format(s) for s in var] + \
            ["cur_{}".format(s) for s in var]
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
        if name.startswith("prob"):  # HACKY
            return int(name[-1]) - 1
        if name == "1":
            return self.ninput + len(self.const) + len(self.var)
        raise Exception

    def linear_func(self, *args):
        lst = np.zeros(self.ninput + len(self.const) + len(self.var))
        for arg in args:
            if len(arg) == 2:
                name, coef = arg
                lst[self.index_in_whole(name)] = coef
            else:
                lst[self.index_in_whole(arg)] = 1
        return list(lst)


# -------------------------------------------------------------------------------
# Geometric distributions
# The probability distribution of the number X of Bernoulli trials needed to get one success, supported on the set { 1, 2, 3, ... }
# or
# The probability distribution of the number Y = X − 1 of failures before the first success, supported on the set { 0, 1, 2, 3, ... }
'''
Known invariant: [z >= 0] * z + [z >= 0 and flip == 0] * (1 - prob)/prob
Assertion invariant: z -> [0,+inf], flip -> [0,1]
program variables: flip, z
guard variables: flip
other vairables: (1 - prob)/prob
We are not expecting it to learn the assertion invariant, i.e., the [z >= 0] part
and instead just expect it to learn
[flip != 0] * z + [flip == 0] * (z + (1 - prob)/prob)
'''


def geo_0(progname, inpt, hists,  init_tuple):

    # we initialize hists
    if hists is None:
        # Construct VarInfo(progname, constants to record, variables to record, ninput=...)
        # While which variables to record depends on our domain knowledge of the program,
        # it's usually safe to include that a 0-1 variable indicates whether the guard is true,
        # all other variables initialized before the loop,
        # and the probability expression appeared in invariants.
        vI = VarInfo(progname, ["(1 - prob1)/prob1"],
                     ["flip", "z"], ninput=1, bools=["flip"])
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + (1 - prob1)/prob1)"
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("flip"),  # index for flip, the variable that we make the split on
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        "model": model_maker(vI.linear_func("z", "(1 - prob1)/prob1"))},  # this encodes z + (1 - prob1)/prob1
            "root_>": {"split": False,
                       "model": model_maker(vI.linear_func("z"))},  # this encodes z; model_maker(vI.linear_func((2,"z"))) encodes 2z}z
            "easy_side": ">"
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)  # init_flip
    prob1 = inpt[0]
    z = init_tuple["non_neg_int"][0]
    flip = init_tuple["bool"][0]
    init_z, init_flip = z, flip
    hists.record_predicate(
        [(1 - prob1)/prob1, init_flip, init_z, flip, z])
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=prob1)[0]
        if d:
            flip = 1
        else:
            z = z + 1
    hists.record_predicate_end(z)
    return hists


def exp0a(progname, inpt, hists,  init_tuple):
    # we initialize hists
    if hists is None:
        # Construct VarInfo(progname, constants to record, variables to record, ninput=...)
        # While which variables to record depends on our domain knowledge of the program,
        # it's usually safe to include that a 0-1 variable indicates whether the guard is true,
        # all other variables initialized before the loop,
        # and the probability expression appeared in invariants.
        vI = VarInfo(progname, ["(1 - prob1)/prob1", "prob1 * prob1",
                                "prob1/(1-prob1)"], ["flip", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + (1 - prob1)/prob1)"
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("flip"),  # index for flip, the variable that we make the split on
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        "model": model_maker(vI.linear_func("z", "(1 - prob1)/prob1"))},  # this encodes z + (1 - prob1)/prob1
            "root_>": {"split": False,
                       "model": model_maker(vI.linear_func("z"))},  # this encodes z; model_maker(vI.linear_func((2,"z"))) encodes 2z}z
            "easy_side": ">"
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)  # init_flip
    prob1 = inpt[0]
    z = init_tuple["non_neg_int"][0]
    flip = init_tuple["bool"][0]
    init_z, init_flip = z, flip
    hists.record_predicate(
        [(1 - prob1)/prob1, prob1 * prob1, prob1/(1-prob1), init_flip, init_z, flip, z])
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=prob1)[0]
        if d:
            flip = 1
        else:
            z = z + 1
    hists.record_predicate_end(z)
    return hists


'''
geo_0 with extra variable x creating noise
Previously we had x gets updated the same way as z does
and it was hard to distinguish through data-driven technique;
Now we made it easier by doubling x whenever increasing z by 1.
Known invariant: [z >= 0] * z + [z >= 0 and flip == 0] * (1 - prob)/prob
Assertion invariant: z -> [0,+inf], flip -> [0,1]
program variables: flip, z
guard variables: flip
other vairables: (1 - prob)/prob
'''


def geo_0a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1 - prob1)/prob1"],
                     ["flip", "x", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + (1 - prob1)/prob1)"
        known_inv_model = {
            "root": {"j_feature": vI.index_in_whole("flip"),  # index for flip, the variable that we make the split on
                     "threshold": 0,
                     "split": True
                     },
            "root_<=": {"split": False,
                        "model": model_maker(vI.linear_func("z", "(1 - prob1)/prob1"))},  # this encodes z + (1 - prob1)/prob1
            "root_>": {"split": False,
                       "model": model_maker(vI.linear_func("z"))},  # this encodes z; model_maker(vI.linear_func((2,"z"))) encodes 2z}
            "easy_side": ">"
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    z = init_tuple["non_neg_int"][0]
    x = init_tuple["non_neg_int"][1]
    flip = init_tuple["bool"][0]
    init_z, init_x, init_flip = z, x, flip
    hists.record_predicate(
        [(1 - prob1)/prob1, init_flip, init_x, init_z, flip, x, z])
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=prob1)[0]
        if d:
            flip = 1
        else:
            x = x * 2
            z = z + 1
    hists.record_predicate_end(z)
    return hists


'''
geo_0 with extra variable i such that
1. whenever i is even the WHILE loop does nothing except increase i by 1
2. whenever i is even the WHILE loop increases x by 2
Known Invariant: I = [z >= 0] * z + [z >= 0 and flip == 0] * 2 * (1 - prob)/prob
Assertion Invariant: TODO
program variables: flip, z, i
guard variables: flip
other vairables: (1 - prob)/prob
'''


def geo_0b(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1 - prob1)/prob1"],
                     ["flip", "i%2", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * z + [flip == 0] * (z + 2*(1 - prob1)/prob1)"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("flip"),  # index for flip
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z", "(1 - prob1)/prob1"))},  # z + (1 - prob1)/prob1
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("z"))},  # z
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    z = init_tuple["non_neg_int"][0]
    i = init_tuple["non_neg_int"][1]
    flip = init_tuple["bool"][0]
    init_z, init_i, init_flip = z, i, flip
    hists.record_predicate(
        [(1 - prob1)/prob1, init_flip, init_i % 2, init_z, flip, i % 2, z])
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=prob1)[0]
        i += 1
        if (i % 2) == 0:
            if d:
                flip = 1
            else:
                z = z + 2
    hists.record_predicate_end(z)
    return hists


'''
This is a program tossing 2 fair coins in one while loop.
Known invariant: count + [not (c1 or c2)] * (p_1 + p_2) / (p_1 + p_2 - p_1 * p_2)
Assertion Invariant: c1 -> [0,1]  c2 -> [0,1]  count ->[0, +inf]
'''


def ex1(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)"],
                     ["c1orc2", "count"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[c1 or c2] * count + [not (c1 or c2)]* (count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("c1orc2"),
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("count", "(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)"))},  # count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2)
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("count"))},  # count
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]

    count = init_tuple["non_neg_int"][0]
    c1 = init_tuple["bool"][0]
    c2 = init_tuple["bool"][1]
    init_c1orc2 = (c1 or c2)
    init_count = count
    hists.record_predicate(
        [(prob1 + prob2)/(prob1 + prob2 - prob1 * prob2), init_c1orc2, init_count, (c1 or c2), count])
    while not (c1 or c2):
        c1 = bernoulli.rvs(size=1, p=prob1)[0]
        if c1:
            count = count + 1
        c2 = bernoulli.rvs(size=1, p=prob2)[0]
        if c2:
            count = count + 1
    hists.record_predicate_end(count)
    return hists


def exp1a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)", "p1*p1", "p2*p2"],
                     ["c1orc2", "count"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[c1 or c2] * count + [not (c1 or c2)]* (count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("c1orc2"),
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("count", "(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)"))},  # count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2)
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("count"))},  # count
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]

    count = init_tuple["non_neg_int"][0]
    c1 = init_tuple["bool"][0]
    c2 = init_tuple["bool"][1]
    init_c1orc2 = (c1 or c2)
    init_count = count
    hists.record_predicate(
        [(prob1 + prob2)/(prob1 + prob2 - prob1 * prob2), prob1*prob1, prob2*prob2, init_c1orc2, init_count, (c1 or c2), count])
    while not (c1 or c2):
        c1 = bernoulli.rvs(size=1, p=prob1)[0]
        if c1:
            count = count + 1
        c2 = bernoulli.rvs(size=1, p=prob2)[0]
        if c2:
            count = count + 1
    hists.record_predicate_end(count)
    return hists


def exp1b(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)", "p1*p1", "p2*p2", "p1/(1-p1)", "(p2/(1-p2))"],
                     ["c1orc2", "count"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[c1 or c2] * count + [not (c1 or c2)]* (count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("c1orc2"),
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("count", "(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)"))},  # count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2)
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("count"))},  # count
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]
    # instrumented
    count = init_tuple["non_neg_int"][0]
    c1 = init_tuple["bool"][0]
    c2 = init_tuple["bool"][1]
    init_c1orc2 = (c1 or c2)
    init_count = count
    hists.record_predicate(
        [(prob1 + prob2)/(prob1 + prob2 - prob1 * prob2), prob1*prob1, prob2*prob2, prob1/(1-prob1), prob2/(1-prob2), init_c1orc2, init_count, (c1 or c2), count])
    while not (c1 or c2):
        c1 = bernoulli.rvs(size=1, p=prob1)[0]
        if c1:
            count = count + 1
        c2 = bernoulli.rvs(size=1, p=prob2)[0]
        if c2:
            count = count + 1
    hists.record_predicate_end(count)
    return hists


def exp1c(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)", "p1*p1", "p2*p2", "p1/(1-p1)", "(p2/(1-p2))", "p1+p2", "1/(p_1 + p_2 - p_1 * p_2)"],
                     ["c1orc2", "count"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[c1 or c2] * count + [not (c1 or c2)]* (count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("c1orc2"),
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("count", "(p_1 + p_2)/(p_1 + p_2 - p_1 * p_2)"))},  # count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2)
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("count"))},  # count
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]
    count = init_tuple["non_neg_int"][0]
    c1 = init_tuple["bool"][0]
    c2 = init_tuple["bool"][1]
    init_c1orc2 = (c1 or c2)
    init_count = count
    hists.record_predicate(
        [(prob1 + prob2)/(prob1 + prob2 - prob1 * prob2), prob1*prob1, prob2*prob2, prob1/(1-prob1), prob2/(1-prob2), prob1+prob2, 1/(prob1 + prob2 - prob1 * prob2), init_c1orc2, init_count, (c1 or c2), count])
    while not (c1 or c2):
        c1 = bernoulli.rvs(size=1, p=prob1)[0]
        if c1:
            count = count + 1
        c2 = bernoulli.rvs(size=1, p=prob2)[0]
        if c2:
            count = count + 1
    hists.record_predicate_end(count)
    return hists


''' Martingale. We can think it as a betting strategy, where b is the bet, and
c is the accumulated capital. It is from the page 202 of Prinsys paper.
Known invariant: rounds + [b > 0] * (1-p1)/(p1)
Assertion Invariant: b -> [ 0,inf ], rounds -> [0,inf]
program variables: c, b, rounds
guard variables: b
other variables: (1 - p1)/ p1
'''


def ex2(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1-prob1)/prob1"],
                     ["b", "c", "rounds"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[b <= 0] * rounds + [b > 0] * (rounds + (1-prob1)/prob1)"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("b"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("rounds"))},  # rounds
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("rounds", "(1-prob1)/prob1"))},  # (rounds + (1-p_1)/p_1)
                           "easy_side": "<="
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)  # the guard is b
    prob = inpt[0]
    c = init_tuple["non_neg_int"][0]
    b = init_tuple["non_neg_int"][1]
    rounds = init_tuple["non_neg_int"][2]
    init_c = c
    init_b = b
    init_rounds = rounds
    hists.record_predicate(
        [(1-prob)/prob, init_b, init_c, init_rounds, b, c, rounds])  # instrumented
    while b > 0:
        d = bernoulli.rvs(size=1, p=prob)
        if d:
            c = c+b
            b = 0
        else:
            c = c-b
            b = 2*b
        rounds += 1
    hists.record_predicate_end(rounds)
    return hists


'''
Non-linear
Gambler's ruin problem
From Lagrange Interpolation Paper
Invariant: z + [0 < x and x < y] * x * (y - x)
Assumed prob = 0.5
Assertion Invariant: z->[0,inf]
Program variables: x, y, z
Guard variables: 0 < x and x < y
Constants: None
Extra variables: x * (y - x)
(We can compute the invariant when prob is not 0.5 but its form is complicated. )
'''


def ex4(progname, inpt, hists,  init_tuple):
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
                           "easy_side": "<="
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = 0.5

    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    z = init_tuple["non_neg_int"][2]
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [init_x, init_y, init_z, 0 < init_x and init_x < init_y, init_x * (init_y - init_x), x, y, z, 0 < x and x < y, x * (y - x)])
    while 0 < x and x < y:
        d = bernoulli.rvs(size=1, p=prob)[0]
        if d:
            x = x + 1
        else:
            x = x - 1
        z = z + 1
    hists.record_predicate_end(z)
    return hists


'''
wp(body,  z + [x * (y - x) <= 1.000]( 1* 0 < x < y) + [x * (y - x) > 1.000](1* x * (y - x)) ) 
= wp(body,  z + 1 + [x * (y - x) <= 1.000]( 1* 0 < x < y) + [x * (y - x) > 1.000](1* x * (y - x)) ) 

wp(body, z + [0 < x < y] * x * (y - x))
= wp(...[p]..., z + 1 + [0 < x < y] * x * (y - x))
= p * (z + 1 + [0 < x + 1 < y] * (x + 1) * (y - x - 1))
+ (1-p) * (z + 1 + [0 < x - 1 < y] * (x - 1) * (y - x + 1))
= z + 1 + [0 < x + 1 < y] * p * (x + 1) * (y - x - 1) + [0 < x - 1 < y] * (1-p) * (x - 1) * (y - x + 1)
[G]* wp(body, z + [0 < x < y] * x * (y - x))
= [0 < x < y](z + 1 + [x != y-1] * p * (x + 1) * (y - x - 1) + [x != 1] * (1-p) * (x - 1) * (y - x + 1))
p = 0.5
when 1 < x = y - 1, (say such x exists)
[G] wp(body, z + [0 < x < y] * x * (y - x))) = [0 < x < y](z + 1 + (1-p) * (x - 1) * (y - x + 1)))
= [0 < x < y](z + 1 + 0.5* (y - 2) * 2))
= [0 < x < y](z + 1 + (y - 2))
= [0 < x < y](z + (y - 1))
When 1 = x = y-1, 
[G] wp(body, z + [0 < x < y] * x * (y - x))) = [0 < x < y](z + 1)
when 1 < x < y
[G] wp(body, z + [0 < x < y] * x * (y - x))) = [0 < x < y](z + xy - x^2 + p * (2y-4x) - y + 2x)
p = 0.5
works out

[x * (y - x) <= 1.000]( 1* z+ 1* 0 < x < y) + [x * (y - x) > 1.000]( 1* z+ 1* x * (y - x))
= z + [x * (y - x) <= 1.000]( 1* 0 < x < y) + [x * (y - x) > 1.000](1* x * (y - x))
= z + [x * (y - x) <= 1.000]( 1* 0 < x < y) + [x * (y - x) > 1.000](1* x * (y - x)) 
z + [0 < x and x < y] * x * (y - x)
'''

'''
similar as 4a but simpler features
'''


def ex4a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "z", "x < y", "x * (y - x)"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "z + [0 < x and x < y] * x * (y - x)"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x < y"),  # index for b
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("z", "x * (y - x)"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z"))},
                           "easy_side": "<="
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = 0.5
    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    z = init_tuple["non_neg_int"][2]
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [init_x, init_y, init_z, init_x < init_y, init_x * (init_y - init_x), x, y, z, x < y, x * (y - x)])
    while 0 < x and x < y:
        d = bernoulli.rvs(size=1, p=prob)[0]
        if d:
            x = x + 1
        else:
            x = x - 1
        z = z + 1
    hists.record_predicate_end(z)
    return hists


'''
Lagrange Interpolation paper:
Geometric Distribution
Assertion Invaraint: z -> [0,1], x -> [0,+inf], y -> [0,+inf]
Post-expectation:	x
Preexp Inv: I= [z = 0]* x + [z != 0] (x + (1-p)/(p^2))
Program variables: z, x, y
Guard variable: z
Other variable: (1-p)/(p^2)
'''


def ex5p(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["(1-p)/(p^2)"], ["x", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[z = 0]* x + [z != 0] (x + (1-p)/(p^2))"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    p = inpt[0]
    x = init_tuple["non_neg_int"][0]
    y = 0
    z = init_tuple["bool"][0]
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [(1-p)/(p**2), init_x, init_z, x, z])
    while not(z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    hists.record_predicate_end(x)
    return hists


'''
similar as ex5p but treat y as a variable too
Preexp Inv: TODO I= [z = 0]* x + [z != 0] (x + y(1-p)/p + (1-p)/(p**2))
Program variables: z, x, y
Guard variable: z
Other variable: p/((1-p)**2)
'''


def ex5yp(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["(1-p)/(p^2)"], ["x", "y(1-p)/p", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[z = 0]* x + [z != 0] (x + y(1-p)/p + (1-p)/(p**2))"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    p = inpt[0]
    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    z = init_tuple["bool"][0]
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [(1-p)/(p**2), init_x, init_y*(1-p)/p, init_z, x, y*(1-p)/p, z])
    while not (z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    hists.record_predicate_end(x)
    return hists


'''
I = [z = 0]* x + [z != 0] (x + y(1-p)/p + (1-p)/(p**2))
wp(body, I)
= wp(y=y+1, p * I[z/0] + (1-p) * I[x <- x + y])
= wp(y=y+1, p * x + (1-p) * ([z = 0]* (x + y) + [z != 0] (x +y+ y(1-p)/p + (1-p)/(p^2))) )
=  p * x + (1-p) * ([z = 0]* (x + y + 1) + [z != 0] (x + y +1 + y(1-p)/p + (1-p)/p + (1-p)/(p^2))) 
=  p * x + (1-p) * ([z = 0]* (x + y + 1) + [z != 0] (x + y +1 + y(1-p)/p  + (1-p^2)/(p^2))) 
[G] * I = wp(I, body) iff [G] * I = [G] * wp(I, body)
[G] * wp(I, body) = p * x + (1-p) * ([z != 0] (x + y +1 + y(1-p)/p  + (1-p^2)/(p^2))) 
= [z != 0] (x + y(1-p)/p + (1-p)/(p^2))

[not G] post  = [z = 0]* x
'''

'''
similar as ex5yp but treat p as a constant
Preexp Inv: I= [z = 0]* x + [z != 0] (x + 3y + (12)
Program variables: z, x, y
Guard variable: z
Other variable: p/((1-p)**2)
'''


def ex5y(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[z = 0]* x + [z != 0] (x + 3y + 12)"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    p = 0.25

    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    z = init_tuple["bool"][0]
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [init_x, init_y, init_z, x, y, z])
    while not (z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    hists.record_predicate_end(x)
    return hists


'''
Same as ex5p but treat p as a constant too
p = 0.25
Preexp Inv: I= [z = 0]* x + [z != 0] (x + 12)
Program variables: z, x, y
Guard variable: z
'''

# we should exclude y


def ex5(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "I= [z = 0]* x + [z != 0] (x + 12)"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    p = 0.25
    x = init_tuple["non_neg_int"][0]
    y = 0
    z = init_tuple["bool"][0]
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [init_x, init_y, init_z, x, y, z])
    while not(z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    hists.record_predicate_end(x)
    return hists


def ex5barebone(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "12"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    p = 0.25
    x = 0
    y = 0
    z = 1
    init_x, init_y, init_z = x, y, z
    hists.record_predicate(
        [init_x, x])
    while not(z == 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if(d):
            z = 0
        else:
            x = x + y
    hists.record_predicate_end(x)
    return hists


# -------------------------------------------------------------------------------
# Binomial distributions
# usually have non-linear invariant

'''
Lagrange Interpolation paper
Binomial Distribution
post: x
Invariant: I=[n > 0] * (x + p * n * y) + [n <= 0] * x
Assertion Invariant: x -> [0,+inf],    n -> [0,10]
Program variables: x, n, y
Guard variable: n
other variables: p * n * y
'''


def ex7(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "n", "p * n * y"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n > 0] * (x + p * n * y) + [n <= 0] * x"
        known_inv_model = {
            "easy_side": "<="
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = inpt[0]

    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    n = init_tuple["non_neg_int"][2]
    init_x, init_y, init_n = x, y, n
    hists.record_predicate(
        [init_x, init_y, init_n, p * init_n * init_y, x, y, n, p * n * y])
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + y
        n = n-1
    hists.record_predicate_end(x)
    return hists


'''
Lagrange Interpolation paper
Binomial Distribution 2 (bin2)
post: x
I = [y >= 0 and n − 1 >= 0] * (p * (1/2n(n+1)) + (1-p)ny)
when p=0.25
Assertion Invariant: x -> [0,+inf],    n -> [0,10],   [n-x <= 10]
Program variables: x, n, y
Guard variable: n
other variables: p * (1/2n(n+1)), (1-p)ny
'''


def ex8(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "n(n+1)", "ny"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[y >= 0 and n > 0] * 0.125n(n+1) + 0.75ny)"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = 0.25
    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    n = init_tuple["non_neg_int"][2]
    init_x, init_y, init_n = x, y, n
    hists.record_predicate(
        [init_x, init_y, init_n * (init_n+1), init_n*init_y, x, y, n * (n+1), n*y])
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        else:
            x = x + y
        n = n-1
    hists.record_predicate_end(x)
    return hists


'''
same as ex8 except general p
'''


def ex8p(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     [], ["x", "y", "pn(n+1)", "(1-p)ny"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[y >= 0 and n − 1 >= 0] * 0.5pn(n+1) + (1-p)ny)"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = inpt[0]
    x = init_tuple["non_neg_int"][0]
    y = init_tuple["non_neg_int"][1]
    n = init_tuple["non_neg_int"][2]
    init_x, init_y, init_n = x, y, n
    init1 = p * init_n * (init_n+1)
    init2 = (1-p)*init_n*init_y
    hists.record_predicate(
        [init_x, init_y, init1, init2, x, y, p * n * (n+1), (1-p)*n*y])
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        else:
            x = x + y
        n = n-1
    hists.record_predicate_end(x)
    return hists


'''
src: http://www-i2.informatik.rwth-aachen.de/pub/index.php?type=download&pub_id=1274
pg-120
Fig 5.9
Algorithm which generates a sample x distributed binomially with parameters p and M.
I = [x ≥ 0 and x − n ≤ 0 and n − M ≤ 0] * (x - prob*n + prob*M)
we aim to learn
[x − n ≤ 0 and n − M ≤ 0] * (x - prob*n + prob*M)
because x >= 0 would always be true and x - n > 0 would be not reachable
relationship between 2 variables
'''


def ex18(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["prob*M"],
                     ["x", "n", "n-M", "prob*n"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("n-M"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func(("x", 1), ("prob*n", -1), ("prob*M", 1)))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func())},
                           "easy_side": ">"
                           }
        known_inv = "[n − M < 0] * (x - prob*n + prob*M) + [n − M >= 0] * x"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    prob = inpt[0]
    n = init_tuple["non_neg_int"][0]
    x = init_tuple["non_neg_int"][1]
    M = init_tuple["non_neg_int"][2]
    init_x, init_n = x, n
    hists.record_predicate(
        [M * prob, init_x, init_n, init_n-M, init_n*prob, x, n, n-M, n*prob])
    while n - M < 0:
        d = bernoulli.rvs(size=1, p=prob)[0]
        if d:
            x = x + 1
        n = n + 1
    hists.record_predicate_end(x)
    return hists


'''
ex20 
Binomial and linearity of expectation
[n<=0] * count + [n>0] * (count + n*(7*3/8))
'''


def ex20(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x1 ", "x2", "x3", "n", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {
            "easy_side": ">"
        }
        known_inv = "[n<=0] * count + [n>0] * (count + n*(7*3/8))"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    n = init_tuple["non_neg_int"][0]
    count = init_tuple["non_neg_int"][1]
    x1, x2, x3 = init_tuple["bool"][0], init_tuple["bool"][1], init_tuple["bool"][2]
    init_n, init_count, init_x1, init_x2, init_x3 = n, count, x1, x2, x3
    hists.record_predicate(
        [init_x1, init_x2, init_x3, init_n, init_count, x1, x2, x3,  n, count])
    while(n > 0):
        x1 = bernoulli.rvs(size=1, p=0.5)[0]
        x2 = bernoulli.rvs(size=1, p=0.5)[0]
        x3 = bernoulli.rvs(size=1, p=0.5)[0]
        n = n - 1
        c1 = x1 or x2 or x3
        c2 = (not x1) or x2 or x3
        c3 = x1 or (not x2) or x3
        count = count + c1 + c2 + c3
    hists.record_predicate_end(count)
    return hists


# -------------------------------------------------------------------------------
# Others

'''
learn two sequencing loops at once
I =  x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (p2/(1-p2))
'''


def ex3(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)"],
                     ["x", "flip1", "flip2"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (p2/(1-p2))"
        known_inv_model = {
            "easy_side": ">"
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]

    x = init_tuple["non_neg_int"][0]
    flip = init_tuple["bool"][0]
    x = init_tuple["non_neg_int"][0]
    flip1 = init_tuple["bool"][0]
    flip2 = init_tuple["bool"][1]
    init_x = x
    init_flip1, init_flip2 = flip1, flip2
    hists.record_predicate(
        [prob1/(1-prob1), prob2/(1-prob2), init_x, init_flip1, init_flip2, x, flip1, flip2])
    while (flip1 == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)
        if d1:
            x = x + 1
        else:
            flip1 = 1
    while (flip2 == 0):
        d2 = bernoulli.rvs(size=1, p=prob2)
        if d2:
            x = x + 1
        else:
            flip2 = 1
    hists.record_predicate_end(x)
    return hists


'''
learn two sequencing loops at once
I =  x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (-p2/(1-p2))
'''


def ex3hard(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)"],
                     ["x", "flip1", "flip2"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "x + [flip1 = 0] * p1/(1-p1) + [flip2 = 0] * (-p2/(1-p2))"
        known_inv_model = {
            "easy_side": ">"
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]

    x = init_tuple["non_neg_int"][0]
    flip1 = init_tuple["bool"][0]
    flip2 = init_tuple["bool"][1]
    init_x = x
    init_flip1, init_flip2 = flip1, flip2
    hists.record_predicate(
        [prob1/(1-prob1), prob2/(1-prob2), init_x, init_flip1, init_flip2, x, flip1, flip2])
    while (flip1 == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)
        if d1:
            x = x + 1
        else:
            flip1 = 1
    while (flip2 == 0):
        d2 = bernoulli.rvs(size=1, p=prob2)
        if d2:
            x = x - 1
        else:
            flip2 = 1
    hists.record_predicate_end(x)
    return hists


'''
learn two nested loops at once
I =  x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) + [flip1 = 0 and flip2 != 0] * p1 * p1 /(1-p1) * p2/(1-p2)
'''


def ex3nest(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1) * p2/(1-p2)", "p1* p1/(1-p1) * p2/(1-p2)"],
                     ["x", "flip1", "flip2"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "x + [flip1 = 0 and flip2 = 0] * p1/(1-p1) * p2/(1-p2) + [flip1 = 0 and flip2 != 0] * p1* p1/(1-p1) * p2/(1-p2)"
        known_inv_model = {
            "easy_side": ">"
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]

    x = init_tuple["non_neg_int"][0]
    flip1 = init_tuple["bool"][0]
    flip2 = init_tuple["bool"][1]
    init_x = x
    init_flip1, init_flip2 = flip1, flip2
    hists.record_predicate(
        [(prob1/(1-prob1)) * prob2/(1-prob2), (prob1 * prob1/(1-prob1)) * prob2/(1-prob2), init_x, init_flip1, init_flip2, x, flip1, flip2])
    while (flip1 == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)
        if d1:
            while (flip2 == 0):
                d2 = bernoulli.rvs(size=1, p=prob2)
                if d2:
                    x = x + 1
                else:
                    flip2 = 1
            flip2 = 0
        else:
            flip1 = 1
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
Lagrange Interpolation paper
Sum of Random Series
Known invriant before the first loop: I= [n > 0] * (x + p * (0.5n(n+1))) + [n <= 0] * x
Assertion invariant: x -> [0,+inf],    n -> [0,10],  [n - x <= 10]
program variables: x, n
guard variables: n
other variables: n(n+1)
'''


def ex9(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "n", "n^2"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n > 0] * (x + 0.25 * n(n+1)) + [n <= 0] * x"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = 0.5

    x = init_tuple["non_neg_int"][0]
    n = init_tuple["non_neg_int"][1]
    init_x, init_n = x, n
    hists.record_predicate(
        [init_x, init_n, init_n * (init_n), x, n, n * n])
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        n = n - 1
    hists.record_predicate_end(x)
    return hists


def ex9p(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "n", "pn", "pn^2"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n > 0] * (x + 0.5pn(n+1)) + [n <= 0] * x"
        known_inv_model = {
            "easy_side": "<="
        }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = inpt[0]
    x = init_tuple["non_neg_int"][0]
    n = init_tuple["non_neg_int"][1]
    init_x, init_n = x, n
    hists.record_predicate(
        [init_x, init_n, p * init_n, p * init_n * init_n, x, n, p * n, p * n * n])
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + n
        n = n - 1
    hists.record_predicate_end(x)
    return hists


'''
Lagrange Interpolation paper
Product of dependent random variables
Assertion Invariant: x -> [0,+inf],    n -> [0,30],    y -> [0,+inf]
Lagrange output:
pre=n(n-1)
post=xy
invariant = (1/4)(n^2 + 2nx + 2ny + 4xy - n) when p = 1/2
Program variables: x, n, y
Guard: n
'''


def ex10(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "n", "y",
                                    "n^2", "nx", "ny", "xy"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[n>0]*(1/4(n^2 + 2nx + 2ny + 4xy - n))+ [n<=0]*(xy)"
        known_inv_model = {
            "easy_side": "<="
        }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = 0.5
    x = init_tuple["non_neg_int"][0]
    n = init_tuple["non_neg_int"][1]
    y = init_tuple["non_neg_int"][2]
    init_x, init_n, init_y = x, n, y
    hists.record_predicate(
        [init_x, init_n, init_y, init_n**2, init_n * init_x, init_n * init_y, init_x * init_y,
            x, n, y, n**2, n*x, n*y, x*y])
    while(n > 0):
        d = bernoulli.rvs(size=1, p=p)[0]
        if(d):
            x = x + 1
        else:
            y = y + 1
        n = n - 1
    hists.record_predicate_end(x*y)
    return hists


'''
Simulation of fair coin with biased coin: opposite of Hurd’s algorithm
Prinsys Paper
Lst 7
I = [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]
= [not x and y] - [x and not y]
= [x = 0]*([y = 1]*1+[y = 0]*0) + [x = 1]*([y = 0]*(-1) + [y = 1] * 0)
= [x <= 0]*([y > 0]*1+[y <= 0]*0) + [x > 0]*([y <= 0]*(-1) + [y > 0] * 0)
Post is x
'''


def ex11(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "y"], ninput=1)
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
                                         "model": model_maker(vI.linear_func(("1", 1)))},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]

    x = init_tuple["bool"][0]
    y = init_tuple["bool"][1]
    init_x = x
    init_y = y
    hists.record_predicate([init_x, init_y, x, y])
    while(x-y == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d1):
            x = 0
        else:
            x = 1
        d2 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d2):
            y = 0
        else:
            y = 1
    hists.record_predicate_end(
        int(x == 0 and y - 1 == 0) - int(x - 1 == 0 and y == 0))
    return hists


'''
check loop invariants:
let I = [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]
[G] * I = [x - y = 0] * I = 0 - 0 = 0
write prob1 as p
wp(body, I) = wp(...[p]..., p * -[x − 1 = 0 and 0 = 0] + (1-p) * [x = 0 and 1 -1 =0])
 = wp(...[p]..., p * -[x − 1 = 0 ] + (1-p) * [x = 0])
 = (1-p) * p * -[1 − 1 = 0] + p * (1-p) * [0 = 0]
 = p * (1-p) * ([0 = 0] - [0=0]) = 0
 
 I get that wp(loop,  [x = 0 and y − 1 = 0] − [x − 1 = 0 and y = 0]) = 0 
 but why is that a loop invariant. 
 
 how about I = [x!=y]*x+ [x = y]*1/2 for post = x
  wp(body, I) = ((p^2)+(1-p)^2)*1/2 + p(1-p)*1
= 1/2
'''

'''
A variation of 11
Invariant I = [x != y] * x + [x == y] * 1/2
different preds_str as ex11
'''


def ex11a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "x==y"], ninput=1)
        preds_str = vI.preds_str
        known_inv = " [x != y] * x + [x == y] * 1/2"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x==y"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": True,
                                       "model": model_maker(vI.linear_func(("1", 0.5)))},
                           "root_>": {"split": True,
                                      "model": model_maker(vI.linear_func(("x", 1)))},
                           "easy_side": ">"
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]

    x = init_tuple["bool"][0]
    y = init_tuple["bool"][1]
    init_x = x
    init_y = y
    hists.record_predicate(
        [init_x, init_x == init_y, x, np.abs(x - y)])
    while(x-y == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d1):
            x = 0
        else:
            x = 1
        d2 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d2):
            y = 0
        else:
            y = 1
    hists.record_predicate_end(x)
    return hists


def exp11a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["p1 * p1", "p1/(1-p1)"],
                     ["x", "|x-y|"], ninput=1)
        preds_str = vI.preds_str
        known_inv = " [x != y] * x + [x == y] * 1/2"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("|x-y|"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_<=": {"split": True,
                                       "model": model_maker(vI.linear_func(("1", 0.5)))},
                           "root_>": {"split": True,
                                      "model": model_maker(vI.linear_func(("x", 1)))},
                           "easy_side": ">"
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]

    x = init_tuple["bool"][0]
    y = init_tuple["bool"][1]
    init_x = x
    init_y = y
    hists.record_predicate(
        [prob1*prob1, prob1/(1-prob1), init_x, np.abs(init_x - init_y), x, np.abs(x - y)])
    while(x-y == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d1):
            x = 0
        else:
            x = 1
        d2 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d2):
            y = 0
        else:
            y = 1
    hists.record_predicate_end(x)
    return hists


'''
Assertion invariant:  x -> [0,2]
Prinsys paper
Lst. 2
I = [x = 0] * (1-prob2) + [x != 0] * [x = 1]
'''


def ex12(progname, inpt, hists,  init_tuple):
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
                                         "model": model_maker(vI.linear_func(("prob2", -1), ("1", 1)))},
                           "root_<=_<=": {"split": False,
                                          "model": model_maker(vI.linear_func())},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1, prob2 = inpt[0], inpt[1]

    x = init_tuple["non_neg_int"][0]
    init_x = x
    hists.record_predicate([init_x, x])
    while(x == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d1):
            x = 0
        else:
            d2 = bernoulli.rvs(size=1, p=prob2)[0]
            if(d2):
                x = -1
            else:
                x = 1
    hists.record_predicate_end(x == 1)
    return hists


'''
let I = [x = 0] * (1-p2) + [x != 0] * [x = 1]
write prob1, prob2 as p1, p2
wp(body, I) = p1 * (1-p2) + (1-p1) * (p2*0 + (1-p2)*1)
= 1 - p2
thus, [G] * wp(body, I) = [G] * I
is an invariant
'''


def exp12(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["prob1*prob1", "prob2*prob2"], ["x"], ninput=2)
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
                                         "model": model_maker(vI.linear_func(("prob2", -1), ("1", 1)))},
                           "root_<=_<=": {"split": False,
                                          "model": model_maker(vI.linear_func())},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1, prob2 = inpt[0], inpt[1]
    x = init_tuple["non_neg_int"][0] % 3 - 1  # -1, 0, 1 are all possible
    assert (x == -1) or (x == 0) or (x == 1)
    init_x = x
    hists.record_predicate([prob1*prob1, prob2*prob2, init_x, x])
    while(x == 0):
        d1 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d1):
            x = 0
        else:  # -d2 + (1-d2) = 1 - 2d2
            d2 = bernoulli.rvs(size=1, p=prob2)[0]
            if(d2):
                x = 2
            else:
                x = 1
    hists.record_predicate_end(x == 1)
    return hists


'''
http://www-i2.informatik.rwth-aachen.de/pub/index.php?type=download&pub_id=1274
pg-120
generate biased coin from a fair coin.
The algorithm generates a sample x = 1 with probability p and
x = 0 with probability 1 − p by repeatedly fliping a fair coin when (prob == 1/2),
which works because
    # 1. for p > 0.5: with prob 0.5, always return x = 1;
    # with prob 0.5, recurse with p' = p*2 - 1 =(p-0.5)*2,
    # by induction, returns x = 1 with prob (p-0.5)*2 in this case. 
    # Thus, in expectation, it returns x = 1 with prob 0.5 * 1 + 0.5 * (p-0.5)*2 = p
    # 2. for p = 0.5: with prob 0.5, return x = 1, with 0.5, recurse with p' = 0,
    # which always returns x = 0.
    # 3. for p < 0.5: with prob 0.5, always return x = 0;
    # with prob 0.5, recurse with p' = 2 * p,
    # thus in expectation, it returns x = 1 with prob 0.5 * 2 * p = p
Fig 5.8
I = [p >= 0 and p − 1 <= 0 and (b − 1 == 0 or p == 0 or p − 1 == 0)] * (p)
 (b − 1 == 0 or p == 0 or p− 1 == 0)] is assertion invariant. 
'''

# It learns p


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
                           qual_feature_indices=vI.init_var_indices)

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
Duelling cowboys: when does A win?
src: https://moves.rwth-aachen.de/wp-content/uploads/WS1819/probprog/prob-prog-2018-lec89.pdf
Joost-Pieter Katoen
I = [t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 [t = B and c = 1] * ((1 - p2)/(p1 + p2 - p1 * p2))
'''


def ex19(progname, inpt, hists,  init_tuple):
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
            "easy_side": "<="
        }
        known_inv = "[t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 [t = B and c = 1] * ((1 - p2)p1/(p1 + p2 - p1 * p2))"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    # the instrumented program
    p1, p2 = inpt[0], inpt[1]
    t = init_tuple["bool"][1]  # t = True if t = A, t = False if t = B
    c = init_tuple["bool"][0]  # whether the game has ended
    init_t, init_c = t, c
    hists.record_predicate(
        [p1/(p1 + p2 - p1*p2), (1-p2)*p1/(p1 + p2 - p1*p2), init_t, init_c, t, c])
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
    hists.record_predicate_end(t)  # t == A
    return hists


def exp19a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["p1/(p1 + p2 - p1 * p2)", "(1 - p2)p1/(p1 + p2 - p1 * p2)", "p1*p1"], ["t", "c"], ninput=2)
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
            "easy_side": "<="
        }
        known_inv = "[t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 [t = B and c = 1] * ((1 - p2)p1/(p1 + p2 - p1 * p2))"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    # the instrumented program
    p1, p2 = inpt[0], inpt[1]
    t = init_tuple["bool"][1]  # t = True if t = A, t = False if t = B
    c = init_tuple["bool"][0]  # whether the game has ended
    init_t, init_c = t, c
    hists.record_predicate(
        [p1/(p1 + p2 - p1*p2), (1-p2)*p1/(p1 + p2 - p1*p2), p1*p1, init_t, init_c, t, c])
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
    hists.record_predicate_end(t)  # t == A
    return hists


def exp19b(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname,
                     ["p1/(p1 + p2 - p1 * p2)", "(1 - p2)p1/(p1 + p2 - p1 * p2)", "p1 * p1", "p2 * p2", "p1 * p2"], ["t", "c"], ninput=2)
        preds_str = vI.preds_str
        known_inv_model = {
            "easy_side": ">"
        }
        known_inv = "[t = A and c = 0]+ [t = A and c = 1] * (p1/(p1 + p2 - p1 * p2))+ \
 [t = B and  = 1] * ((1 - p2)p1/(p1 + p2 - p1 * p2))"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    # the instrumented program
    p1, p2 = inpt[0], inpt[1]
    t = init_tuple["bool"][1]  # t = True if t = A, t = False if t = B
    c = init_tuple["bool"][0]  # whether the game has ended
    init_t, init_c = t, c
    hists.record_predicate(
        [p1/(p1 + p2 - p1*p2), (1-p2)*p1/(p1 + p2 - p1*p2), p1 * p1, p2*p2, p1*p2, init_t, init_c, t, c])
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
    hists.record_predicate_end(t)  # t == A
    return hists


'''
uniform distribution
'''


def ex15(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {
            "easy_side": "<="
        }
        known_inv = "count + [x <= 10]*(10-x+1)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = inpt[0]

    x = init_tuple["non_neg_int"][0]
    count = init_tuple["non_neg_int"][1]
    init_x, init_count = x, count
    hists.record_predicate([init_x, init_count, x, count])
    while(x <= 10):
        x = x + random.choice(list(np.arange(0.1, 2, 0.1)))
        count = count + 1
    hists.record_predicate_end(count)
    return hists


'''
ex15 but deterministic 
'''


def ex15a(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {
            "easy_side": "<="
        }
        known_inv = "count + [x <= 10]*(10-x+1)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = inpt[0]

    x = init_tuple["non_neg_int"][0]
    count = init_tuple["non_neg_int"][1]
    init_x, init_count = x, count
    hists.record_predicate([init_x, init_count, x, count])
    while(x <= 10):
        x = x + 1
        count = count + 1
    hists.record_predicate_end(count)
    return hists


'''
I = "count + [x <= 10]*(10-x+1)"
wp(body, I) = wp(x= x + randint(0,2), count+1 +[x <= 10]*(10-x+1))
= E_[v in [0,2]] count+1 + [x <= 10]*(10-x-v+1)
= count+1 + [x <= 10]*(10-x-1+1) 
= count+1 + [x <= 10]*(10-x) 
[x <= 10] * wp(body, I) = [x <= 10]*(count+1+10-x)
[x <= 10] * I = [x <= 10]*(count+10-x+1)

alternatively,
I' = "count + [x <= 10]*(10-x)
 wp(x= x + randint(0,2), count+1 +[x <= 10]*(10-x)) = count+1 + [x <= 10]*(10-x-1)
 [G] I' =  [x <= 10]*(count+10-x)
 [G] wp(body, I') =[x <= 10]*(count+10-x) 
 
 
In general, if I = [G] * wp(P,I) + [not G] * f
then [G] * wp(P,I) + [not G] * f = [G] * wp(P,[G] * wp(P,I) + [not G] * f) + [not G] * f
also if there's any I' such that I' = [G] * wp(P,I') + [not G] * f, 
it must I' = I by our proof. 
I' = [G] * wp(P,I') + [not G] * f follows form [G] * I = [G] * wp(P,I') and [notG] * I' = [not G] * f 

Note that 
[G] * I = [G] * wp(P,I) + [not G] * f = [G] * wp(P,[G] * wp(P,I) + [not G] * f) + [not G] * f
implies
[G] * wp(P,I) = [G] * wp(P,[G] * wp(P,I) + [not G] * f) 
= [G] * (wp(P,[G] * wp(P,I)) + wp(P, [not G] * f))
'''


'''
Kaminski pg 125
a program that has a diverging path with prob 0 and (positively) almost surely terminate;
it's like inverse binonmial distribution
I think Inv is [x > 0] * (z + x*(1/prob)) + [x <= 0](z)
'''
'''
wp(body, [x > 0] * (z + x*(1/p)) + [x <= 0](z))
= wp(probabilistic branch, [x > 0] * (z + 1 + x*(1/p)) + [x <= 0](z + 1))
= p * ([x-1 > 0] * (z + 1 + (x-1)*(1/p)) + [x - 1 <= 0](z + 1)) 
+ (1-p) * ([x > 0] * (z + 1 + x*(1/p)) + [x <= 0](z + 1)) 
[G]*wp(body, [x > 0] * (z + x*(1/p)) + [x <= 0](z))
= [x > 0] ([x > 1] (z + x/p ) + [x == 1](z + 1 + (1-p)x/p )
= [x > 0] ([x > 1] (z + x/p ) + [x == 1](z + 1/p )
'''

'''[x > 0] * (z + x*(1/prob)) + [x <= 0](z)'''


def ex22(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "x*(1/prob)", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z"))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("z", ("x*(1/prob)", 1)))},
                           "easy_side": "<="
                           }
        known_inv = "[x >= 1] * (z + x*(1/prob)) + [x <= 0](z)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = inpt[0]

    x = init_tuple["non_neg_int"][0]
    z = init_tuple["non_neg_int"][1]
    init_x, init_z = x, z
    hists.record_predicate(
        [init_x, init_x/prob, init_z, x, x/prob, z])

    while(x-1 >= 0):
        d = bernoulli.rvs(size=1, p=prob)[0]
        if(d):
            x = x - 1
        z = z + 1
    hists.record_predicate_end(z)
    return hists


'''
Guess: I = [count + [x <= 10]*(10-x+1) <= 100] (count + [x <= 10]*(10-x+1))
wp(body, I) = [count + 1 + [x + 1 <= 10]*(10-x-1+1) <= 100] (count + 1 + [x + 1 <= 10]*(10-x-1+1)) 
= [count + 1 + [x <= 9]*(10-x) <= 100] (count + 1 + [x <= 9]*(10-x) ) 
unclear what is an invariant
'''
# ------------------------------------------------------------------------------
'''
Kaminski pg 125
I = [x-1 >= 0] + [x-1 < 0] * z
my guess I = [x > 1] * (z + x/(2-prob)) + [x <= 0](z) + [x == 1](z + 1)
'''


def ex21(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, [], ["x", "x/(2-prob)", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z"))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func(("z", 1), "x/(2-prob)"))},
                           "easy_side": "<="
                           }
        known_inv = "[x > 1] * (z + x/(2-prob)) + [x <= 0](z) + [x == 1](z + 1)"
        hists = RecordStat(preds_str, wp, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = inpt[0]

    x = init_tuple["non_neg_int"][0]
    z = init_tuple["non_neg_int"][1]
    init_x, init_z = x, z
    hists.record_predicate(
        [init_x, init_x/(2-prob), init_z, x, x/(2-prob), z])
    while(x-1 >= 0):
        d = bernoulli.rvs(size=1, p=prob)[0]
        if(d):
            x = x - 1
        else:
            x = x - 2
        z = z + 1
    hists.record_predicate_end(z)
    return hists


def exp21(progname, inpt, hists,  init_tuple):
    if hists is None:
        def wp(inv):
            raise NotImplementedError
        vI = VarInfo(progname, ["prob/(1-prob)", "prob*prob"],
                     ["x", "x/(2-prob)", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("x"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("z"))},
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func(("z", 1), "x/(2-prob)"))},
                           "easy_side": "<="
                           }
        known_inv = "[x >= 1] * (z + x/(2-prob)) + [x <= 0](z) "
        hists = RecordStat(preds_str, wp, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = inpt[0]
    x = init_tuple["non_neg_int"][0]
    z = init_tuple["non_neg_int"][1]
    init_x, init_z = x, z
    hists.record_predicate(
        [prob/(1-prob), prob*prob, init_x, init_x/(2-prob), init_z, x, x/(2-prob), z])
    while(x-1 >= 0):
        d = bernoulli.rvs(size=1, p=prob)[0]
        if(d):
            x = x - 1
        else:
            x = x - 2
        z = z + 1
    hists.record_predicate_end(z)
    return hists


'''
geo_0 with extra variable i that we only increase z if i is odd
Assertion invariant: TODO
program variables: flip, z, i
guard variables: flip
other vairables: (1 - prob)/prob
'''


def geo_0c(progname, inpt, hists,  init_tuple):
    if hists is None:
        vI = VarInfo(progname, ["(1 - prob1)/prob1"],
                     ["flip", "i%2", "z"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "unclear"
        known_inv_model = {"root": {"model": model_maker(vI.linear_func("z")),  # affected sign
                                    "split": False},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob = inpt[0]

    z = init_tuple["non_neg_int"][0]
    i = init_tuple["non_neg_int"][1]
    flip = init_tuple["bool"][0]
    init_z, init_i, init_flip = z, i, flip
    hists.record_predicate(
        [(1 - prob)/prob, init_flip, init_i % 2, init_z, flip, i % 2, z])
    while (flip == 0):
        d = bernoulli.rvs(size=1, p=prob)[0]
        i += 1
        if d:
            flip = 1
        else:
            if (i % 2) == 0:
                z = z + 2
    hists.record_predicate_end(z)
    return hists


'''
Tortoise and Hare
Aleksandar Chakarov et al.
https://www.cs.colorado.edu/~srirams/papers/sas14-expectations.pdf
page 26
Trivial invariant: I = [h >= 0 and t >= 0 and count >= 0] * (1)
(no probabilistic component in the invariant)
'''


def ex13(progname, inpt, hists,  init_tuple, constraint):
    if hists is None:
        vI = VarInfo(progname, [], ["h", "t", "count"], ninput=1)
        preds_str = vI.preds_str
        known_inv = "[h >= 0 and t >= 0 and count >= 0] * (1)"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("h"),
                                    "threshold": -1,
                                    "split": True
                                    },
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func())},
                           "root_>": {"split": True,
                                      "j_feature": vI.index_in_whole("t"),
                                      "threshold": -1},
                           "root_>_<=": {"split": False,
                                         "model": model_maker(vI.linear_func())},
                           "root_>_>": {"split": True,
                                        "j_feature": vI.index_in_whole("count"),
                                        "threshold": -1},
                           "root_>_>_<=": {"split": False,
                                           "model": model_maker(vI.linear_func())},
                           "root_>_>_>": {"split": False,
                                          "model": model_maker(vI.linear_func())},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)

    prob = inpt[0]
    if prob < 0.3:
        prob = 0.35 + 2 * prob

    # takes too long to run for small probability(p<=0.31) according to notes
    t = 3 * init_tuple["non_neg_int"][0] + 10  # make t greater than h
    h = init_tuple["non_neg_int"][1]
    count = init_tuple["non_neg_int"][2]
    init_t = t
    init_h = h
    init_count = count
    hists.record_predicate(
        [init_h, init_t, init_count, h, t, count])
    while((t-h) >= 0):
        t = t + 1
        d = bernoulli.rvs(size=1, p=prob)[0]
        if d:
            h = h + 3
        count = count + 1
        if count >= 30:
            break  # prevent running too long
    hists.record_predicate_end(1)
    return hists


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
    prob1, prob2 = inpt[:2]
    while(x-1 >= 0):
        while not(cp[i] == 0):
            i = random.randint(1, 10)
        cp[i] = 1
        x = x - 1
        z = z + 1
    return z


'''
ex 17 but b = 1 - bernoulli.rvs(size=1, p=prob)[0] instead of drawing from bernoulli(0.5)
TODO: what is it's invariant.
    # 1. for p >= 0.5: with prob, return x = 1;
    # with 1-prob, recurse with p' = p*2 - 1 =(p-0.5)*2,
    # by induction, returns x = 1 with UNCLEAR.
    # Thus, in expectation, it returns x = 1 with UNCLEAR
    # 2. for p < 0.5: with prob, always return x = 0;
    # with 1 - prob, recurse with p' = 2 * p,
    # thus in expectation, it returns x = 1 with UNCLEAR
'''


def ex17b(progname, inpt, hists,  init_tuple, constraint):
    if hists is None:
        vI = VarInfo(
            [], ["x", "b", "(b − 1 == 0 or x == 0 or x − 1 == 0)"], ninput=2)
        preds_str = vI.preds_str
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("(b − 1 == 0 or x == 0 or x − 1 == 0)"),
                                    "threshold": 0,
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("x"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func())},
                           "easy_side": "<="
                           }
        known_inv = "(b − 1 == 0 or x == 0 or x − 1 == 0)] * (x)"
        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    p = init_tuple["prob"][0]
    prob = inpt[1]
    b = init_tuple["bool"][0]
    init_p, init_b = p, b
    hists.record_variable_info()
    hists.record_predicate([init_p, init_b, int((init_b == 1) or (
        p == 0) or (p == 1)), p, b, int((b == 1) or (p == 0) or (p == 1))])
    while b:
        b = 1 - bernoulli.rvs(size=1, p=prob)[0]
        if b:  # if b==1, x = x*2 mod 1
            p = 2 * p
            if(p - 1 >= 0):
                p = p - 1
        else:
            if(p - 0.5 >= 0):
                p = 1
            else:
                p = 0
    hists.record_predicate_end(p)
    return hists


'''
product.v2
Pre-expectation:	n*n
Post-expectation:	4*x*y
Lagrange Interpolation output:
Invariant:		n^2 + 2*n*y + 2*n*x + 4*x*y
'''


def ex25(progname, inpt, hists,  init_tuple, constraint):
    n = 0
    x = 1  # random
    y = 1  # random
    prob1, prob2 = inpt
    while(n > 0):
        d1 = bernoulli.rvs(size=1, p=prob1)[0]
        if(d1):
            x = x + 1
        d2 = bernoulli.rvs(size=1, p=prob2)[0]
        if(d1):
            y = y + 1
        n = n - 1
    return (4 * x * y)


'''
The following two are the same program, where we are interested in loop invariants
before different loops.
Known invariant before the first loop: I = x + [flip = 0] * p1/(1-p1))
Assertion invariant: flip -> [0,1], x -> [0,+inf]
program variables: x, flip
guard variables: flip
other variables: p1/(1-p1), p1/(1-p2)
'''


def ex3a(progname, inpt, hists,  init_tuple, constraint):
    if hists is None:
        def wp(inv):
            inv_true = inv.replace("x", "(x+1)")
            inv_true = "(" + inv_true + ") * p1"
            inv_false = inv.replace("flip", "1")
            inv_false = "( " + inv_false + ") * (1-p1)"
            return inv_true + " + " + inv_false

        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)"],
                     ["x", "flip"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * x +[flip = 0] * (x + p1/(1-p1))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("flip"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("x"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("x", "p1/(1-p1)"))},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, wp, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]
    if constraint:  # generate hard constraints
        x = init_tuple["int"][0]
        flip = init_tuple["bool"][0]
        hists.record_hard_constraint(
            [prob1/(1-prob1), prob2/(1-prob2)], [x, flip], not(flip == 0), True, x)
    else:  # instrumented programs
        x = init_tuple["non_neg_int"][0]
        flip = init_tuple["bool"][0]
        init_x = x
        init_flip = flip
        hists.record_variable_info()
        hists.record_predicate(
            [prob1/(1-prob1), prob2/(1-prob2), init_x, init_flip, x, flip])
        while (flip == 0):
            d1 = bernoulli.rvs(size=1, p=prob1)
            if d1:
                x = x + 1
            else:
                flip = 1
        hists.record_predicate_end(x)
        flip = 0
        while (flip == 0):
            d2 = bernoulli.rvs(size=1, p=prob2)
            if d2:
                x = x - 1
            else:
                flip = 1
    return hists


def exp3a(progname, inpt, hists,  init_tuple, constraint):
    if hists is None:
        def wp(inv):
            inv_true = inv.replace("x", "(x+1)")
            inv_true = "(" + inv_true + ") * p1"
            inv_false = inv.replace("flip", "1")
            inv_false = "( " + inv_false + ") * (1-p1)"
            return inv_true + " + " + inv_false

        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)", "p1*p1", "p2*p2", "p1*p2"],
                     ["x", "flip"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * x +[flip = 0] * (x + p1/(1-p1))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("flip"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("x"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("x", "p1/(1-p1)"))},
                           "easy_side": ">"
                           }
        hists = RecordStat(preds_str, wp, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]
    if constraint:  # generate hard constraints
        x = init_tuple["int"][0]
        flip = init_tuple["bool"][0]
        hists.record_hard_constraint(
            [prob1/(1-prob1), prob2/(1-prob2), prob1*prob1, prob2*prob2, prob1*prob2], [x, flip], not(flip == 0), True, x)
    else:  # instrumented programs
        x = init_tuple["non_neg_int"][0]
        flip = init_tuple["bool"][0]
        init_x = x
        init_flip = flip
        hists.record_variable_info()
        hists.record_predicate(
            [prob1/(1-prob1), prob2/(1-prob2), prob1*prob1, prob2*prob2, prob1*prob2, init_x, init_flip, x, flip])
        while (flip == 0):
            d1 = bernoulli.rvs(size=1, p=prob1)
            if d1:
                x = x + 1
            else:
                flip = 1
        hists.record_predicate_end(x)
        flip = 0
        while (flip == 0):
            d2 = bernoulli.rvs(size=1, p=prob2)
            if d2:
                x = x - 1
            else:
                flip = 1
    return hists


'''
The following two are the same program, where we are interested in loop invariants
before different loops.
Known invriant before the first loop: I =  x + [flip = 0] * (-p2/(1-p2))
Assertion invariant: flip -> [0,1], x -> [0,-inf] 
program variables: x, flip
guard variables: flip
other variables: p1/(1-p1), p2/(1-p2)
Test negative number
'''


def ex3b(progname, inpt, hists,  init_tuple, constraint):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)"],
                     ["x", "flip"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * x +[flip = 0] * (x-p2/(1-p2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("flip"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("x"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("x", ("p2/(1-p2)", -1)))},
                           "easy_side": ">"
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]
    if constraint:  # generate hard constraints
        x = init_tuple["int"][0]
        flip = init_tuple["bool"][0]
        hists.record_hard_constraint(
            [prob1/(1-prob1), prob2/(1-prob2)], [x, flip], not(flip == 0), True, x)
    else:  # instrumented programs
        x = init_tuple["non_neg_int"][0]
        flip = init_tuple["bool"][0]
        while (flip == 0):
            d1 = bernoulli.rvs(size=1, p=prob1)
            if d1:
                x = x + 1
            else:
                flip = 1
        x = init_tuple["non_neg_int"][0]
        flip = init_tuple["bool"][0]
        init_x = x
        init_flip = flip
        hists.record_variable_info()
        hists.record_predicate(
            [prob1/(1-prob1), prob2/(1-prob2), init_x, init_flip, x, flip])
        while (flip == 0):
            d2 = bernoulli.rvs(size=1, p=prob2)
            if d2:
                x = x - 1
            else:
                flip = 1
        hists.record_predicate_end(x)
    return hists


def exp3b(progname, inpt, hists,  init_tuple, constraint):
    if hists is None:
        vI = VarInfo(progname, ["p1/(1-p1)", "p2/(1-p2)", "p1*p1", "p2*p2", "p1*p2"],
                     ["x", "flip"], ninput=2)
        preds_str = vI.preds_str
        known_inv = "[flip != 0] * x +[flip = 0] * (x-p2/(1-p2))"
        known_inv_model = {"root": {"j_feature": vI.index_in_whole("flip"),  # index for b
                                    "threshold": 0,  # affected sign
                                    "split": True
                                    },
                           "root_>": {"split": False,
                                      "model": model_maker(vI.linear_func("x"))},
                           "root_<=": {"split": False,
                                       "model": model_maker(vI.linear_func("x", ("p2/(1-p2)", -1)))},
                           "easy_side": ">"
                           }

        hists = RecordStat(preds_str, known_inv, known_inv_model, ninput=vI.ninput,
                           qual_feature_indices=vI.init_var_indices)
    prob1 = inpt[0]
    prob2 = inpt[1]
    if constraint:  # generate hard constraints
        x = init_tuple["int"][0]
        flip = init_tuple["bool"][0]
        hists.record_hard_constraint(
            [prob1/(1-prob1), prob2/(1-prob2), prob1*prob1, prob2*prob2, prob1*prob2], [x, flip], not(flip == 0), True, x)
    else:  # instrumented programs
        x = init_tuple["non_neg_int"][0]
        flip = init_tuple["bool"][0]
        while (flip == 0):
            d1 = bernoulli.rvs(size=1, p=prob1)
            if d1:
                x = x + 1
            else:
                flip = 1
        x = init_tuple["non_neg_int"][0]
        flip = init_tuple["bool"][0]
        init_x = x
        init_flip = flip
        hists.record_variable_info()
        hists.record_predicate(
            [prob1/(1-prob1), prob2/(1-prob2), prob1*prob1, prob2*prob2, prob1*prob2, init_x, init_flip, x, flip])
        while (flip == 0):
            d2 = bernoulli.rvs(size=1, p=prob2)
            if d2:
                x = x - 1
            else:
                flip = 1
        hists.record_predicate_end(x)
    return hists
