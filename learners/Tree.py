import numpy as np
from numpy.core.function_base import logspace
from learners.abstract_learner import Learner
from learners.utils import load_expected_post, log_no_nan, exp_no_nan, exp_int_no_nan
from copy import deepcopy
import pdb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

THRESHOLD = 1e-4
'''
Some choices: 

1. We assumed that predicates are in the form of [v == c], [v != c], [v <= c], 
[v > c], where [v] is a splittable variable, and [c] is a value that [v] takes 
in data 

2. When deciding whether to make a leaf node branching on a predicate, we branch 
as long as the loss after branching is more than 1e-4 lower than the loss before 
branching. By that choice, the training algorithm would likely keep branching 
the model tree until there is only one data on each leaf. That model tree would 
terribly overfit the data, so we restrict the maximum tree depth to 2. 
    Alternatively, we can also set maximum tree depth to higher while setting 
also THRESHOLD to a larger number. 

3. If the mean square loss of the learned model's prediction is too big, then 
we do not consider that learned model as plausible and would not verify it. 
Currently, we check whether the loss is less than 1/5 the variance of prediction 
targets, with the intuition that the learned model should use the features to 
explain the majority of the variance. 

4. For the learned model that pass the 1/5 variance criteria, we associate them 
with a final loss that favors less mean square loss and less complexity. 
Specifically, we let 
final loss = 
mean squared loss * (1 + number of digits we keep) * (1 + the depth of the tree). 

Later, in `cegis.py`, [verify_cand] will choose to five candidate invariants 
with the lowest final loss to verify. 

5. We specifies the rounding digits we try in the function learn_inv. 
For all examples except LinExp, we only need to try rounding with 0,1,2 digits, 
for LinExp, we need rounding with 3 digits too. 

'''
#---------------------------------Leaf Model------------------------------------
'''
[linear_model] wraps up `sklearn.linear_model.LinearRegression` to perform 
standard linear regression. 

The method [to_string] outputs a string representation of the learned linear 
model.
'''
class linear_model:

    def __init__(self, fit_intercept):
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, y, y_pred):
        error = mean_squared_error(y, y_pred)
        return error
    
    def combine_loss(self, left_loss, right_loss, len_left, len_right):
        return (left_loss * len_left + right_loss * len_right)/(len_left + len_right)
    
    def adjustedR2(self, x,y ):
        r2 = self.model.score(x,y)
        n = x.shape[0]
        p = x.shape[1]
        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        return adjusted_r2

    def to_string(self, features, logspace, d):
        coefs = self.model.coef_
        def round_d(num):
            if d == 0: 
                return int(round(num,0))
            else: 
                return round(num, d)
        if not logspace:
            lst = ["({})*{}".format(round_d(coefs[c]), features[c])
                                for c in range(len(coefs))
                                if round_d(coefs[c]) != 0
                                ]
            if len(lst) != 0: 
                string = " + ".join(lst)
            else: 
                string = "0"
        else:
            lst = [f"({features[c]})**({round_d(coefs[c])})"
                                for c in range(len(coefs))
                                if round_d(coefs[c]) != 0
                                ]
            if len(lst) != 0:
                string = " * ".join(lst)
            else:
                string = "1"
        if self.fit_intercept:
            if logspace:
                string += "*{}".format(str(round_d(eval(f"exp_int_no_nan({self.model.intercept_})"))))
            else: 
                string += "+{}".format(str(round_d(self.model.intercept_)))
        return string


#------------------Helper Function for Training the Model Tree------------------

'''
Given: 
    [X_init], [y_post], [model], [logspace] passed down from [_build_tree]

Returns: 
    a node that will be used as the root of a model tree
'''
def _create_root(X_init, y_post, model, logspace):
    loss_node, model_node = _fit_model(X_init, y_post, model, logspace)
    node = {"name": "root",
            "loss": loss_node,
            "model": model_node,
            "data": (X_init, y_post),
            "n_samples": len(X_init),
            "j_feature": None,
            "threshold": None,
            "sign": None,
            "children": {"left": None, "right": None},
            "depth": 0}
    return node

'''
Create a node when we already knows the best model fitting to the data on it. 
Given: 
    [X_init], [y_post]: the data to be associated with the created node
    [depth]: the depth of of the node we are creating, the depth of root is 0, 
    the depth of child is the depth of parent plus one, 
    [loss]: the loss of [learned_model] on ([X_init], [y_post])
    [learned_model]: a fitted model
    [parent]: the parent of the node we are creating; 
    [direction]: it's either "l" or "r", indicating whether it's a left child or 
    the right child
    [model]: the regression model we use to fit data at the leaf model

Returns: 
    a node that is child at the [direction] of [parent], whose field [model] is 
    set to [learned_model] and field [loss] is set to the input [loss]
''' 
def _create_node_fitted(X_init, y_post, depth, loss, learned_model, parent, direction):
    node_name = "{}_{}".format(parent, direction)
    node = {"name": node_name,
            "loss": loss,
            "model": learned_model,
            "data": (X_init, y_post),
            "n_samples": len(X_init),
            "j_feature": None,
            "threshold": None,
            "sign": None, 
            "children": {"left": None, "right": None},
            "direction": direction,
            "depth": depth}
    return node

'''
Try splitting [node.data] into two sets based on some predicates, and fitting 
two models respectively for the two sets of data. 
- If the averaged loss of the two models is better than the loss of the best 
    model fitted to [node.model], then "split" [node] by doing the following
    a. create [node]'s left and right children, 
    b. record the predicate that yields the lowest average loss. Specifically, 
       record the [threshold] and the [j_feature] we split data at in [node],
    c. update [node]'s [loss], delete the [data] at node
- Otherwise: do nothing

Given: 
    [node]: the node (which is currently a leaf) that we are going to "split" on 
            based on some predicates 
    [logspace], [model], [splittable], [max_depth], [min_samples_leaf] are the 
        same as in [_build_tree]

[Return]: None
'''

def _split_traverse_node(node, model, splittable, max_depth, min_samples_leaf, logspace):
    # Extract data
    X, y = node["data"]
    N, _ = X.shape
    # Find feature splits that might improve loss
    did_split = False
    loss_best = node["loss"]
    if loss_best <= 1e-4:
        return  
    # Perform threshold split search only if node has not hit max depth
    if (node["depth"] >= 0) and (node["depth"] < max_depth):
        for j_feature in splittable:
            threshold_search = []
            for i in range(N):
                threshold_search.append(X[i, j_feature])
            threshold_search = list(set(threshold_search))  # deduplicate
            # Perform threshold split search on j_feature
            for threshold in threshold_search:
                for sign in ["==", "<="]:
                    # Split data based on threshold
                    (X_left, y_left), (X_right, y_right) = _split_data(
                        j_feature, threshold, X, y, sign)
                    N_left, N_right = len(X_left), len(X_right)
                    # Make sure each leaf has samples size more than [min_samples_leaf]
                    split_conditions = [N_left >= min_samples_leaf,
                                        N_right >= min_samples_leaf]
                    # Do not attempt to split if split conditions not satisfied
                    if not all(split_conditions):
                        continue
                    # Compute weight loss function
                    loss_left, model_left = _fit_model(X_left, y_left, model, logspace)
                    loss_right, model_right = _fit_model(X_right, y_right, model, logspace)
                    loss_split = model_right.combine_loss(
                        loss_left, loss_right, len(X_left), len(X_right))
                    # Update best parameters if the loss is visibly lower
                    if loss_split < loss_best - THRESHOLD:
                        did_split = True
                        loss_best = loss_split
                        left_right_loss_best = [loss_left, loss_right]
                        models_best = [model_left, model_right]
                        data_best = [(X_left, y_left), (X_right, y_right)]
                        j_feature_best = j_feature
                        threshold_best = threshold
                        sign_best = sign
    # Return terminal node if split is not advised
    if not did_split:
        return 
    # Update node information based on splitting result
    node["j_feature"] = j_feature_best
    node["threshold"] = threshold_best
    node["sign"] = sign_best
    del node["data"]  # delete node stored data
    # Extract splitting results
    (X_left, y_left), (X_right, y_right) = data_best
    model_left, model_right = models_best
    loss_left, loss_right = left_right_loss_best
    node["children"]["left"] = _create_node_fitted(
        X_left, y_left, node["depth"]+1, loss_left, model_left, parent=node["name"], direction="l")
    node["children"]["right"] = _create_node_fitted(
        X_right, y_right, node["depth"]+1, loss_right, model_right, parent=node["name"], direction="r")
    # Split nodes
    _split_traverse_node(node["children"]["left"], model, splittable, max_depth, min_samples_leaf, logspace)
    _split_traverse_node(node["children"]["right"], model, splittable, max_depth, min_samples_leaf, logspace)
    
    return

'''
Given: 
    The data
        [X_init]: np.array of initial states
        [y_post]: np.array of expected post
    [model]: the type of model on leaves. currently, it is always the 
            [linear_model] class
    [splittable]: the list of features that can be used to form predicates
                  (see No.1 `Some choices` on the top of this file)
    [max_depth]: maximum depth of the model tree; currently set to 2
    [min_samples_leaf]: minimum number of samples required for a leaf; currently 
                        set to 1
    [logspace]: a boolean that indicates whether we are in `log mode`, 
                i.e., whether we let the leaf models to be the multiplicative models   
                
            **  When we are in the `log mode`, [X_init] and [y_post] would 
                contain the logarithmic values of the original data. We train 
                **linear models** on that logarithmic-valued data, and 
                exponentiate the linear model in [to_string]. The parameter 
                [logspace] will be passed down to helper functions that 
                [_build_tree] relies on, but eventually, it will only be used 
                in [to_string] **
Return: 
    a tree from [root] that fits the data given by ([X_init], [y_post]) 
    whose leaf model are as required by [logspace] and [model]
'''
def _build_tree(X_init, y_post, logspace, model, splittable, max_depth, min_samples_leaf):

    '''
    Recursively split node + traverse node until splitting stops helping or 
    other split_conditions are not satisfied 
    Currently, we grow the tree in the following order 
    (nodes are created in the numerical order of indices annotated), 
            1
          /   \
        2     3
        / \    / \
      4   5  10 11
      / \ / \
      6 7 8 9
    ''' 
    root = _create_root(X_init, y_post, model, logspace)  # depth 0 root node

    _split_traverse_node(root, model, splittable, max_depth, min_samples_leaf, logspace)
    return root  


'''
 Fit [model] such that [model].predict([X]) is approximately [y]
'''
def _fit_model(X, y, model, logspace):
    _, d = X.shape
    model_copy = deepcopy(model)  # must deepcopy the model!
    model_copy.fit(X, y)
    y_pred = model_copy.predict(X)
    if logspace:
        loss = mean_squared_error(exp_no_nan(y), exp_no_nan(y_pred))
    else:
        loss = mean_squared_error(y,y_pred)
    assert loss >= 0.0
    return loss, model_copy


'''
 Split entries in X based on whether the [j_feature] of X is [sign] [threshold], 
 and split y such that entries in X still correspond to same entries in y. 
'''
def _split_data(j_feature, threshold, X, y, sign):
    if len(X) == 0 and len(y) == 0:
        return (X, y), (X, y)
    else:
        if sign == "==":
            idx_left = np.where(X[:, j_feature] == threshold)[0]
        else:
            if sign == ">":
                idx_left = np.where(X[:, j_feature] > threshold)[0]
            else:
                if not (sign == "<="):
                    pdb.set_trace()
                try:
                    idx_left = np.where(X[:, j_feature] <= threshold)[0]
                except IndexError:
                    pdb.set_trace()
        idx_right = np.delete(np.arange(0, len(X)), idx_left)
        assert len(idx_left) + len(idx_right) == len(X)
        try:
            return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])
        except TypeError:
            pdb.set_trace()

def _prepare_data(data, features, logspace):
    X,y = load_expected_post(data, features)
    if logspace:
        X = log_no_nan(X)
        y = log_no_nan(y)
    return X, y

'''
 Return opsign such that for any a, b, (a sign b) iff not (a opsign b)
'''
def _op_sign(sign):
    if sign == "<=":
        return ">"
    elif sign == ">":
        return "<="
    elif sign == "==":
        return "!="

'''
 This function roughly corresponds to [extractInv] described in the Fig. 2 of 
 the paper. It returns a string representation of the model tree rooted at 
 [node]. 
''' 
def extract_invariant(node, features, logspace, depth, round_digit):
    # Empty node
    if depth ==0 or (node["children"]["left"] is None and node["children"]["right"] is None):
        return node["model"].to_string(features, logspace, round_digit)
    else:
        if logspace:
            threshold = exp_int_no_nan(node["threshold"])
        else:
            threshold = node["threshold"]
        threshold_str = "{} {} {:.3f} ".format(
            features[node['j_feature']], node["sign"], threshold)
        l = extract_invariant(node["children"]["left"], features, logspace, depth-1, round_digit)
        r = extract_invariant(node["children"]["right"], features, logspace, depth-1, round_digit)
        opsign = _op_sign(node["sign"])
        return "[{}]*({})+[{}]*({})".format(threshold_str, l, threshold_str.replace(node["sign"], opsign), r)

'''
 [extract_loss(node,depth)] calculates the loss of the model tree rooted at 
 [node] on the [node.data]. 
'''
def extract_loss(node, depth):
    if depth == 0 or (node["children"]["left"] is None and node["children"]["right"] is None):
        return node["loss"]
    else:
        lchild = node["children"]["left"]
        left_loss = extract_loss(lchild, depth-1)
        rchild = node["children"]["right"]
        right_loss = extract_loss(rchild, depth-1)
        return (left_loss * lchild["n_samples"] + right_loss * rchild["n_samples"]) / (lchild["n_samples"] + rchild["n_samples"])

'''
 [_predict_one(node, x, depth)] use the model tree rooted at [node] to make 
 prediction on [x], where [x] is one entry of the data
'''
def _predict_one(node, x, depth):
    no_children = node["children"]["left"] is None and \
        node["children"]["right"] is None
    if no_children or depth == 0:
        y_pred = node["model"].predict([x])
        return y_pred
    else:
        if x[node["j_feature"]] <= node["threshold"]:
            return _predict_one(node["children"]["left"], x, depth - 1)
        else:
            return _predict_one(node["children"]["right"], x, depth - 1)
        

'''
 [tree_depth(node)] returns the depth of the model tree rooted at [node].
'''
def tree_depth(node):
    no_children = node["children"]["left"] is None and \
        node["children"]["right"] is None
    if no_children:
        return 0
    else:
        return 1 + max(tree_depth(node["children"]["left"]), tree_depth(node["children"]["right"]))
  
class TreeLearner(Learner):
    def __init__(self, features, splittable, logspace, fit_intercept: bool, max_depth, min_samples_leaf):
        self.features = features
        self.splittable = splittable
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.logspace = logspace
        self.model = linear_model(fit_intercept= fit_intercept)
    
    def learn_inv(self, data):
        X_init, Y_post = _prepare_data(data, self.features, self.logspace)
        tree = _build_tree(X_init, Y_post, self.logspace, self.model, 
                           self.splittable, self.max_depth, self.min_samples_leaf)
        invlist = []
        # If we are in the log mode, then we only round learned coefficients to 
        # integers. After exponentiating linear models, we would get multiplicative
        # models \Prod (v_i)^{a_i}, where a_i is the coefficient. Non-integers 
        # coefficients would yield candidate invariants that the verifiers can't
        # handle
        if self.logspace:
            var = np.var(exp_no_nan(Y_post))
            digits = [0]
        else:
            var = np.var(Y_post)
            digits = [0,1,2,3]
        for depth in range(min(self.max_depth, tree_depth(tree)+1)):
            loss = extract_loss(tree, depth)
            if loss < var/5:
                for round_digit in digits:
                    inv = extract_invariant(tree, self.features, self.logspace, depth, round_digit) 
                    invlist.append((inv, loss * (1 + depth) * (1 + round_digit)))
                    #scale loss by the complexity of the invariant
        print("From model trees learned, we extract the following candidate\
            (sub)invariants: ")
        for inv in invlist:
            print(inv)
        return invlist