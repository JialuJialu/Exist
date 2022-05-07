import math

from tensorflow.python.ops.special_math_ops import bessel_i0
from learners.utils import (
    load_expected_post,
    load_next_iter,
    log_no_nan,
)
from learners.abstract_learner import Learner
import tensorflow as tf
import pdb
from functools import reduce
import numpy as np
from collections import defaultdict
import copy

# ----- This part of the code is adapted from https://github.com/wOOL/DNDT -----
# -------- Documentations are added by us to explain our adapted version -------
"""
Given:
    [a]: an N by M matrix
    [b]: an N by L matrix
Return: 
    [res]: an N by M*L matrix such that 
    a_{ij} * b_{ik} = res_{il}, 
    where l = M*j + k

Example 1:
Given: 
a = [[1,0], [1,0]]
b = [[1,0], [0,1]]
Returns: 
tf_kron_prod(a, b) = [[1, 0, 0, 0], [0, 1, 0, 0]]

Example 2:
Given: 
a = [[1,0], [0,1]]
b = [[1,0,0], [0,1,0]]
Returns: 
tf_kron_prod(a, b) = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
"""


@tf.function
def tf_kron_prod(a, b):
    res = tf.einsum("ij,ik->ijk", a, b)
    res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
    return res


"""
[tf_bin] is a soft version of the binning function
For example, if x = [[0],[1],[2],[5]] and [cut_points] = [0,2,4], then 
the hard binning function will classify each of 0,1,2,5 into one of the 
following bins: numbers smaller than 0, numbers between 0 and 2, 
numbers between 2 and 4, and numbers of greater than 4

Given:
    [x]: a N-by-1 matrix (column vector)
    [cut_points]: a D-dim vector (D is the number of cut-points) with monotonically 
    increasing values.
    [temperature]: a parameter adjusting the softness of [tf_bin]. 
    The smaller the [temperature], the more [tf_bin] behaves like the hard-binning function
Return:
    [res]:a N-by-(D+1) matrix, where each row has only one element approximately one 
    and the rest are approximately zeros. 
"""


@tf.function
def tf_bin(x, cut_points, temperature):
    D = cut_points.get_shape().as_list()[0]
    W = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
    # cut_points = tf.sort(cut_points) 
    b = tf.cumsum(
        tf.concat([tf.constant(0.0, shape=[1]), tf.cast(-cut_points, tf.float32)], 0)
    )  
    h = tf.matmul(tf.cast(x, tf.float32), W) + b
    res = tf.nn.softmax(h / temperature)
    return res


"""
Given:
    [x]: N by M matrix, where N is the number of data entries, 
         and M is the number of features
    [cut_points_list]: variables that are getting trained. 
                       It contains the cut_points for each feature. 
    [leaf_score]: variables that are getting trained. 
                    It's a L by M matrix, where M is the number of features, and 
                    L = \Prod_i (number of bins for feature i)
    [temperature]: the temperature of the soft binning function [tf_bin]
Return:
    a N by 1 matrix that encodes a linear model
"""


@tf.function
def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.01):
    # TODO: maybe use tf.fold instead of reduce to speed up
    x = tf.convert_to_tensor(x)
    leaf = reduce(
        tf_kron_prod,
        map(
            lambda z: tf_bin(x[:, z[0] : z[0] + 1], z[1], temperature),
            enumerate(cut_points_list),
        ),
    )
    # enumerate(cut_points_list) is a list of (feature index number, trainable cutpoint for feature)
    # for each pair z in enumerate(cut_points_list)
    ### x[:, z[0]:z[0] + 1] selects out the z[0]'s feature of the data,
    ### z[1] is the cut point for that feature,
    # Thus, map(lambda z: tf_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list))
    # gives a list of M matrices, where the ith matrix is an N-by-(D_i+1) matrix with zero-one entries.
    # (M is the length of [cut_points_list] (and thus is the number of splittable features),
    # N is the size of [x], and D_i + 1 is the number of bins for the ith feature.
    res = tf.matmul(leaf, leaf_score)
    # leaf is a N by (number of classes)^M matrix, assuming the number of classes is the same for all features
    return res


def createVar(feature, size):
    return tf.Variable(
        tf.random.normal([size], dtype=tf.float32), name="cut_points".format(feature)
    )


# -------------------Encoding Model trees as Neural networks--------------------

"""
NNModelTree is a neural architecture for encoding model trees. 
"""

class NNModelTree(tf.Module):
    def __init__(self, features, splittables, fit_intercept, name=None):
        super().__init__(name=name)
        self.features = features
        self.splittables = splittables
        num_features = len(features)
        self.cut_points_list = [
            createVar(features[i], 1)
            if (i in splittables)
            else createVar(features[i], 0)
            for i in range(num_features)
        ]
        self.leaf_score = tf.Variable(
            tf.zeros([2 ** len(splittables), num_features], dtype=tf.float32),
            name="leaf_models",
        )

    def __call__(self, x):
        nn = nn_decision_tree(x, self.cut_points_list, self.leaf_score)
        x = tf.convert_to_tensor(x)
        y = tf.einsum("ij,ij->i", nn, x)
        return y

    def set_init(self, cut_points_list, leaf_score):
        self.cut_points_list = cut_points_list
        self.leaf_score = leaf_score
        return self

    def random_init(self):
        features = self.features
        splittables = self.splittables
        num_features = len(features)
        self.cut_points_list = [
            createVar(features[i], 1)
            if (i in splittables)
            else createVar(features[i], 0)
            for i in range(num_features)
        ]
        self.leaf_score = tf.Variable(
            tf.random.normal([2 ** len(splittables), num_features], dtype=tf.float32),
            name="leaf_models",
        )
        return self

    # assign currently used for debugging only
    def assign(self, lofdic):
        features = self.features
        new_leaf_score = []
        for leafidx in range(len(lofdic)):
            leaf_model = lofdic[leafidx]
            new_leaf_score.append(
                np.array(
                    [leaf_model[f] if f in leaf_model.keys() else 0 for f in features],
                    dtype=np.float32,
                )
            )
        self.leaf_score = tf.Variable(new_leaf_score, name="leaf_models")


"""
NNProduct is the same as NNModelTree except exponentiate the output before returning 
in __call__()
TODO: it's better to inherit from NNModelTree or use model composition, 
but I don't know how to do that and currently hack by copying
"""


class NNProduct(tf.Module):
    def __init__(self, features, splittables, fit_intercept, name=None):
        super().__init__(name=name)
        self.features = features
        self.splittables = splittables
        num_features = len(features)
        self.cut_points_list = [
            createVar(features[i], 1)
            if (i in splittables)
            else createVar(features[i], 0)
            for i in range(num_features)
        ]
        self.leaf_score = tf.Variable(
            tf.zeros([2 ** len(splittables), num_features], dtype=tf.float32),
            name="leaf_models",
        )

    def __call__(self, x):
        nn = nn_decision_tree(x, self.cut_points_list, self.leaf_score)
        x = tf.convert_to_tensor(x)
        y = tf.einsum("ij,ij->i", nn, x)
        return tf.math.exp(y)

    def set_init(self, cut_points_list, leaf_score):
        self.cut_points_list = cut_points_list
        self.leaf_score = leaf_score
        return self

    def random_init(self):
        features = self.features
        splittables = self.splittables
        num_features = len(features)
        self.cut_points_list = [
            createVar(features[i], 1)
            if (i in splittables)
            else createVar(features[i], 0)
            for i in range(num_features)
        ]
        self.leaf_score = tf.Variable(
            tf.random.normal([2 ** len(splittables), num_features], dtype=tf.float32),
            name="leaf_models",
        )
        return self


"""
This function roughly corresponds to [extractInv] described in the Fig. 2 of 
the paper. It returns a string representation of [model_tree_model] 
Given:
    [model_tree_model] has two sets of parameters: [leaf_score] that encodes the 
    linear models on leaves, and [cut_points_list] that specifies the threshold to 
    split on features. 
    [features]: a list of feature names
    [splittables]: a list of features we used when binning data to leaves
    [fit_logspace]: whether we are fitting the data in logspace. If yes, we need to 
    take exponentials when formulating the leaf model
    [digits]: the rounding digits 

Return:
[tree_str]: a string representation of the model tree encoded by [model_tree_model]

Assumption: 
In general, NNModelTree can also handle predicates that split to more than to children, 
although in this function, we currently assume that every feature has at most 
one cutpoint (and thus only two children). 

Example:
[features] = ["x", "y", "z"]
[splittables] = [0, 1]
Then we can split on "x" and "y" but not "z". 
[model_tree_model].cut_points_list = [[1], [6], []]
Then we have predicates "x <= 1" vs "x > 1", "y <= 6" vs "y > 6". 
Since the underlying binning function is soft, the predicates we formulate are 
approximations, and it is arbitrary whether to have "x <= 1" or "x < 1". 
Then we can formulate our tree 
      x<=1
      /  \
   y<=6   y<=6
   / \    / \
If [model_tree_model].leaf_score = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]], then 
we have the tree
                 x<=1
               /       \
            /             \
       y<=6               y<=6
     /      \           /       \
    /        \         /         \
1x+2y+3z  4x+5y+6z  7x+8y+9z  10x+11y+12z
"""


def makeModelTree(model_tree_model, splittables, fit_logspace, fit_intercept, digits):
    def round_d(num):
        if digits == 0:
            return int(round(num, 0))
        else:
            return round(num, digits)

    try:
        weights = model_tree_model.leaf_score.numpy()
    except AttributeError:
        pdb.set_trace()
    features = model_tree_model.features
    m, n = weights.shape
    assert n == len(features)
    cut_points_list = model_tree_model.cut_points_list
    str_lst = []
    for i in range(m):
        weight = weights[i]
        binary = "{0:b}".format(i)
        if len(binary) < len(splittables):
            binary = (len(splittables) - len(binary)) * "0" + binary
        if m == 1:
            predicates_str = ""
        else:
            predicates = []
            for j in range(len(binary)):
                c = int(binary[j])
                j_idx = splittables[j]
                cut_point = round_d(float(cut_points_list[j_idx]))
                predicates.append(
                    "[{} {} {}]".format(features[j_idx], ">=" if c else "<", cut_point)
                )
            predicates_str = "*".join(predicates) + "*"
        # if fit_intercept and fit_logspace, we would have a multiplying coefficient
        # e**c, we round 2.72**c, instead of round c and then exponentiate
        if fit_logspace:
            if not fit_intercept:
                leaf_str = "*".join(
                    [
                        "{}^{:2f}".format(features[c], int(round(weight[c])))
                        for c in range(n)
                        if int(round(weight[c])) != 0
                    ]
                )
            else:
                assert features[-1] == "2.72"
                leaf_str = "*".join(
                    [
                        "{}^{:2f}".format(features[c], int(round(weight[c])))
                        for c in range(n - 1)
                        if int(round(weight[c])) != 0
                    ]
                    + [str(round_d((2.72) ** weight[-1]))]
                )
        else:
            leaf_str = " + ".join(
                [
                    "{:2f}*{}".format(round_d(weight[c]), features[c])
                    for c in range(n)
                    if round_d(weight[c]) != 0
                ]
            )
        if leaf_str != "":
            str_lst.append("{} ({})".format(predicates_str, leaf_str))
    tree_str = "+".join(str_lst)
    if tree_str == "":
        if fit_logspace:
            tree_str = "1"
        else:
            tree_str = "0"
    return tree_str


# ------------------Our custom training of the Neural networks-------------------
# -------------------------------------------------------------------------------
"""
This is currently used for learning exact invariants. 

For an initial state [x], [post] is the expected value of the post-expectation 
when the [program] ends. [post] approximates [wp([program], post-expcatation)(x)]. 
We want to train model such that 
wp([program], postexp) = [guard] * model + postexp, 
so on states [x] that satisfies [guard], 
we want to minimize the 
so we define the loss to be mean_square_error between model(x) and post. 
"""


@tf.function
def exactInvLoss(model, data):
    x, post = data
    x_predict = model(x)
    loss_predict = tf.reduce_mean((x_predict - post) ** 2)
    return loss_predict


"""
Assume: 
[x_cur] all satisfies [G] (we filtered out states that does not satisfy [G] when sampling)
[pre], [post] are the values of pre and post on [x_cur]
[y_next]_i is a set of vectors, obtained by evaluating [features] 
after running the loop for one iteration from [x_cur]_i

We want [G] * model + post to behave like a subinvariant, which requires
1. [G] * ([G] * model + post) <= [G] * wp(body, [G] * model + post), 
which simplifies to [G] * (model + post) <= [G] * wp(body, [G] * model) + [G] * wp(body, post), 
Since [x_cur] all satisfies [G], we just check
(model + post) <= wp(body, [G] * model) +  wp(body, post)
We define [loss_predict] to capture this requirement. 

2. [not G]([G] * model + post) <= [not G] * post
This can be simplified into [not G] * post <= [not G] * post and is automatically satisfied

3. pre <= [G] * model + post 
for x satisfies G, we need pre(x) <= model(x) + post(x); 
for x satisfies not G, this is satisfied as long as pre <= post
we define [loss_preexp] to capture that. 
"""

@tf.function
def subInvLoss(model, data):
    x_cur, pre, post, y_next, G_next, post_next, w = data
    IplusPost = model(x_cur) + post
    wpbodyGI = tf.reduce_mean(tf.einsum("ij,ij->ij", G_next, tf.vectorized_map(model, y_next)), axis=0)
    wpbodypost = tf.reduce_mean(post_next, axis=1)
    loss_predict = tf.reduce_mean(
        tf.einsum("i,i->i", w, tf.nn.relu(IplusPost - wpbodyGI - wpbodypost))
    )
    loss_preexp = tf.reduce_mean(tf.einsum("i,i->i", w, tf.nn.relu(pre - IplusPost)))
    return loss_predict + loss_preexp



def grad(model, data, lossfunc):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss = lossfunc(model, data)
    # print("The loss is {:4f}".format(loss))
    return tape.gradient(loss, model.trainable_variables)


def NN_learn(
    nn,
    all_data,
    batches,
    loss_func,
    splittables,
    fit_logspace,
    fit_intercept,
    digits,
    sgd=False,
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    best_model = None
    best_loss = 10000000
    # print("     Start learning a new invariant")
    # Gradient descent iterations
    for iter in range(50):
        # print("Iteration {}".format(iter))
        # print(makeModelTree(nn, splittables, fit_logspace, fit_intercept, digits))
        # Perform Stochastic Gradient Descent if [sgd] set to True
        if sgd:
            # At each step, we calculate the gradient of one batch of the data
            # and update our model based on that gradient
            for batch in batches:
                grads = grad(nn, batch, loss_func)
                if np.any(np.isnan(grads[-1])):
                    pdb.set_trace()
                optimizer.apply_gradients(
                    zip(list(grads), list(nn.trainable_variables)))
                print(makeModelTree(nn, splittables, fit_logspace, fit_intercept, digits))
        grads = grad(nn, all_data, loss_func)
        # if there's numerical error that causing Nan in taking gradient
        # we stop the training process and return
        if np.any(np.isnan(grads[-1])):
            break
        optimizer.apply_gradients(zip(list(grads), list(nn.trainable_variables)))
        loss = loss_func(nn, all_data)
        if loss < best_loss:
            best_model = nn
            best_loss = loss
            if loss < 1e-3:
                break

    best_model_str = makeModelTree(
        best_model, splittables, fit_logspace, fit_intercept, digits
    )
    print("The learned NN model tree is \n {}".format(best_model_str))
    return nn, best_loss

'''
Some choices: 

1. We adapted the neural decision trees in https://arxiv.org/pdf/1806.06988.pdf 
to neural model trees. While the framework works for neural model trees of any 
depth, we observe that neural model trees tends to make superfulous splits when 
we learn subinvariants. Thus, in the current implementation, we disallow neural 
model trees to make splits on predicates; in other word, we are just using 
neural networks to represent linear model/multiplicative model. But it is still 
meaningful to use the neural architecture, because we want to use gradient 
descent to train the model. 

If you want the neural model trees to make split, when initializing NNLearner in 
main.py, uncomment the block of code below `splittables_log = [], 
                                            splittables_linear = []`, 
and change [tf_bin] to its state in 
https://github.com/wOOL/DNDT/blob/master/tensorflow/neural_network_decision_tree.py

'''
class NNLearner(Learner):
    def __init__(
        self,
        features,
        splittables,
        assumed_shape,
        subinv: bool,
        fit_logspace: bool,
        fit_intercept: bool,
    ):
        self.splittables = splittables
        self.fit_logspace = fit_logspace
        self.subinv = subinv
        self.assumed_shape = assumed_shape
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            if self.fit_logspace:
                self.features = features + ["2.72"]
            else:
                self.features = features + ["1"]
        else:
            self.features = features


    def learn_inv(self, data):
        invlist = []
        # prepare models differently depending on whether we learn exact invariants
        # or subinvariants
        if self.subinv:
            model_tree_nn, all_data, batches, loss_func = prepare_subinv(
                self.features,
                self.splittables,
                data,
                self.fit_logspace,
                self.fit_intercept,
            )
        else:
            model_tree_nn, all_data, batches, loss_func = prepare_exactinv(
                self.features,
                self.splittables,
                data,
                self.fit_logspace,
                self.fit_intercept,
            )
        # prepare a list of initializations [initial_models]
        m = model_tree_nn # m is a NNModelTree() whose variables initialized at 0
        initial_models = [m]
        for _ in range(0):
            m1 = copy.deepcopy(m)
            m1 = m1.random_init()
            initial_models.append(m1)
        # Train models using data [all_data], [batches] starting from initial 
        # models in [initial_models]
        for m in initial_models:
            model, loss = NN_learn(
                m,
                all_data,
                batches,
                loss_func,
                self.splittables,
                self.fit_logspace,
                self.fit_intercept,
                2,
            )
            if self.fit_logspace:
                digits_lst = [0]
            else:
                digits_lst = [0, 1, 2]
            for digits in digits_lst:
                inv = makeModelTree(
                    model,
                    self.splittables,
                    self.fit_logspace,
                    self.fit_intercept,
                    digits,
                )
                invlist.append((inv, loss * digits, model))
        print(invlist)
        return invlist


# -------------------------Helper functions-------------------------------------
# ------------------------------------------------------------------------------
def makebatch(lst, batchsize):
    datasize = len(lst)
    if datasize <= batchsize:
        return [lst]
    else:
        batchnum = math.floor(datasize / float(batchsize))
        all = [lst[num * batchsize : (num + 1) * batchsize] for num in range(batchnum)]
        if datasize % batchsize > 0:
            all += [lst[batchsize * batchnum :]]
        return all


def makeSubInvData(
    X_cur, pre_G, post_cur, Y_next, G_next, post_next, weight, batchsize=96
):
    batched_cur = makebatch(X_cur, batchsize)
    batched_pre_G = makebatch(pre_G, batchsize)
    batched_post_cur = makebatch(post_cur, batchsize)
    batched_next = makebatch(Y_next, batchsize)
    batched_G_next = makebatch(G_next, batchsize)
    batched_G_next = [tf.transpose(tf.convert_to_tensor(e)) for e in batched_G_next]
    batched_post_next = makebatch(post_next, batchsize)
    batched_weight = makebatch(weight, batchsize)
    assert len(batched_cur) == len(batched_pre_G)
    assert len(batched_pre_G) == len(batched_next)
    batches = list(
        zip(
            batched_cur,
            batched_pre_G,
            batched_post_cur,
            batched_next,
            batched_G_next,
            batched_post_next,
            batched_weight,
        )
    )
    return batches


def makeSubInvData_pre(X, pre, post, G, Y, pre_next, weight, batchsize=96):
    batched_cur = makebatch(X, batchsize)
    batched_pre = makebatch(pre, batchsize)
    batched_post_cur = makebatch(post, batchsize)
    batched_G = makebatch(G, batchsize)
    batched_next = makebatch(Y, batchsize)
    batched_pre_next = makebatch(pre_next, batchsize)
    batched_weight = makebatch(weight, batchsize)
    assert len(batched_cur) == len(batched_pre)
    assert len(batched_pre) == len(batched_next)
    batches = list(
        zip(
            batched_cur,
            batched_pre,
            batched_post_cur,
            batched_G,
            batched_next,
            batched_pre_next,
            batched_weight,
        )
    )
    return batches


def makeInvData(X_init, Y_post, batchsize=96):
    batched_X_init = makebatch(X_init, batchsize)
    batched_Y_post = makebatch(Y_post, batchsize)
    batches = list(zip(batched_X_init, batched_Y_post))
    return batches


def prepare_subinv(features, splittables, data, fit_logspace, fit_intercept):
    df_G_init, df_G_next, weight = data
    X_cur, pre_G, post_cur, Y_next, G_next, post_next = load_next_iter(
        df_G_init, df_G_next, features
    )
    if fit_logspace:
        X_cur = log_no_nan(X_cur)
        Y_next = [log_no_nan(e) for e in Y_next]
        model_tree_nn = NNProduct(
            features, splittables, fit_intercept, "model_tree_subinvariant"
        )
    else:
        model_tree_nn = NNModelTree(
            features, splittables, fit_intercept, "model_tree_subinvariant"
        )
    # Prepare data
    batches = makeSubInvData(X_cur, pre_G, post_cur, Y_next, G_next, post_next, weight)
    all_data = (
        X_cur,
        pre_G,
        post_cur,
        Y_next,
        tf.transpose(tf.convert_to_tensor(G_next)),
        post_next,
        weight,
    )
    # Define model and the loss function
    loss_func = subInvLoss
    return model_tree_nn, all_data, batches, loss_func


def prepare_exactinv(features, splittables, data, fit_logspace, fit_intercept):
    # Load data
    X_init, y_post = load_expected_post(data, features)
    if fit_logspace:
        X_init = log_no_nan(X_init)
        y_post = log_no_nan(y_post)
    # Prepare data
    batches = makeInvData(X_init, y_post)
    all_data = (X_init, y_post)
    # Define model and the loss function
    model_tree_nn = NNModelTree(features, splittables, "model_tree_exact_wpe")
    loss_func = exactInvLoss
    return model_tree_nn, all_data, batches, loss_func

