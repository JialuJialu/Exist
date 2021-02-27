"""
 ModelTree.py  (author: Anson Wong / git: ankonzoid)
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from graphviz import Digraph
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pdb
import math
import matplotlib
import matplotlib.pyplot as plt

THRESHOLD = 1e-4


class ModelTreeInv(object):

    def __init__(self, model, header, fit_used,
                 testing_known_model, known_model, max_depth, min_samples_leaf,
                 search_type, n_search_grid, j_feature_range, sign, loss_func):

        self.model = model
        self.header_init = header
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.search_type = search_type
        self.n_search_grid = n_search_grid
        self.j_feature_range = j_feature_range
        self.tree = None
        # not use a feature to fit if have already using it for splitting
        self.no_repeat = not(fit_used)
        self.bfs = False
        self.sign = sign
        assert self.sign == ">" or self.sign == "<=" or self.sign == "=="
        self.sort_reverse = True
        self.fitting_history = {}
        self.testing_known_model = testing_known_model
        self.known_model = known_model
        self.loss_func = loss_func
        self.norm_p = 2

    def get_params(self, deep=True):
        return {
            "model": self.model.get_params() if deep else self.model,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "search_type": self.search_type,
            "n_search_grid": self.n_search_grid,
            "j_feature_range": self.j_feature_range,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return "{}({})".format(class_name, ', '.join(["{}={}".format(k, v) for k, v in self.get_params(deep=False).items()]))

    # ======================
    # Fit
    # ======================
    def fit(self, X_init, y_post, verbose=False):

        # Settings
        model = self.model
        min_samples_leaf = self.min_samples_leaf
        max_depth = self.max_depth
        search_type = self.search_type
        n_search_grid = self.n_search_grid
        j_feature_range = self.j_feature_range
        header_init = self.header_init
        no_repeat = self.no_repeat

        if verbose:
            print(" max_depth={}, min_samples_leaf={}, search_type={}...".format(
                max_depth, min_samples_leaf, search_type))

        def _build_tree(X_init, y_post, j_feature_range):

            global index_node_global

            '''
            splitting [parent] based on its [j_feature] at [threshold] and
            create the child at [parent]'s [direction]
            [not_to_fit] tracks all features that ever used for splitting
            on the path from root to the node to be created
            '''
            def _create_node(X_init, y_post, depth, container, parent, direction, not_to_fit, verbose=True):
                # X_inv, y_inv = getXyfrominv(X_cur, X_next, parent,
                #                             direction, header_iter, root, parent_j_feature, parent_threshold, self.sign, model_sibling=None, verbose=False)
                # # we can assume that left has been created
                node_name = "root" if parent is None else "{}_{}".format(
                    parent, direction)
                # loss_node, model_node = _fit_model_both(
                #     X_inv, y_inv, X_init, y_post, X_cons, y_cons, not_to_fit, self.use_inv, model, verbose=True,
                #     node_name=node_name, fitting_hist=self.fitting_history, sort_reverse=self.sort_reverse)
                loss_node, model_node = _fit_model(X_init, y_post, model)
                node = {"name": node_name,
                        # "parent_name": parent,
                        "index": container["index_node_global"],
                        "loss": loss_node,
                        "model": model_node,
                        "data": (X_init, y_post),
                        "n_samples": len(X_init),
                        "not_to_fit": not_to_fit[:],
                        "j_feature": None,
                        "threshold": None,
                        "children": {"left": None, "right": None},
                        "direction": direction,
                        # "parent_j_feature": parent_j_feature,
                        # "parent_threshold": parent_threshold,
                        # "root": root,
                        "depth": depth}

                if verbose:
                    header = self.header_init
                    variables = [header[i] for i in range(
                        len(header)) if not (i in node["not_to_fit"])]
                    model_print, _ = node["model"].to_string(variables)
                    print("Creating node {} with model: \n {}".format(
                        node["name"], model_print))

                container["index_node_global"] += 1
                return node

                result = _splitter(node, model, header_init, self.sign,
                                   no_repeat, not_to_fit=not_to_fit,
                                   loss_func=self.loss_func,
                                   min_samples_leaf=min_samples_leaf,
                                   j_feature_range=j_feature_range,
                                   max_depth=max_depth,
                                   search_type=search_type,
                                   n_search_grid=n_search_grid)

            '''
            Recursively split node + traverse node until a terminal node is reached
            Currently, we grow the tree in a order that is neither DFS nor BFS.
            For instance, when the following tree is constructed,
            nodes are created in the numerical order of indices annotated.
                   1
                 /   \
                2     3
               / \    / \
              4   5  10 11
             / \ / \
             6 7 8 9
            '''
            def _split_traverse_node(node, container):
                if not no_repeat:
                    not_to_fit = []
                else:
                    not_to_fit = node["not_to_fit"]
                # Perform split and collect result
                result = _splitter(node, model, header_init, self.sign,
                                   no_repeat, not_to_fit=not_to_fit,
                                   loss_func=self.loss_func,
                                   min_samples_leaf=min_samples_leaf,
                                   j_feature_range=j_feature_range,
                                   max_depth=max_depth,
                                   search_type=search_type,
                                   n_search_grid=n_search_grid)

                # Return terminal node if split is not advised
                if not result["did_split"]:
                    if verbose:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(" {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(
                            depth_spacing_str, node["index"], node["depth"], node["loss"], result["N"]))
                    return

                # Update node information based on splitting result
                node["j_feature"] = result["j_feature"]
                node["threshold"] = result["threshold"]
                del node["data"]  # delete node stored data

                # Extract splitting results
                (X_left, y_left), (X_right, y_right) = result["data"]
                model_left, model_right = result["models"]

                # Report created node to user
                if verbose:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    print(" {}node {} @ depth {}: loss={:.6f}, j_feature={}, threshold={:.6f}, N=({},{})".format(
                        depth_spacing_str, node["index"], node["depth"], node["loss"], node["j_feature"], node["threshold"], len(X_left), len(X_right)))

                # Create children nodes
                if not no_repeat:
                    new_not_to_fit = []
                else:
                    new_not_to_fit = not_to_fit+[result["j_feature"]]
                node["children"]["left"] = _create_node(
                    X_left, y_left, node["depth"]+1, container, parent=node["name"], direction="l", not_to_fit=new_not_to_fit)
                node["children"]["right"] = _create_node(
                    X_right, y_right, node["depth"]+1, container, parent=node["name"], direction="r", not_to_fit=new_not_to_fit)
                node["children"]["left"]["model"] = model_left
                node["children"]["right"]["model"] = model_right

                # Split nodes
                _split_traverse_node(node["children"]["left"], container)
                _split_traverse_node(node["children"]["right"], container)

            container = {"index_node_global": 0}  # mutatable container
            root = _create_node(X_init, y_post, 0, container,
                                parent=None, direction=None, not_to_fit=[])  # depth 0 root node
            # split and traverse root node
            _split_traverse_node(root, container)

            return root

            def _split_known_model(node, container):
                pass  # TODO

            # split and traverse root node
            if self.testing_known_model:
                _split_known_model(root, container)
            else:
                if self.bfs:
                    _split_traverse_node(root, container)
            return root

        # Construct tree
        self.tree = _build_tree(X_init, y_post, j_feature_range)
        return self.sign, self.sort_reverse

    # ======================
    # Predict
    # ======================
    def predict(self, X):
        assert self.tree is not None

        def _predict(node, x):
            no_children = node["children"]["left"] is None and \
                node["children"]["right"] is None
            if no_children:
                x_use = [x[i]
                         for i in range(len(x)) if not (i in node["not_to_fit"])]
                try:
                    y_pred_x = node["model"].predict([x_use])[0]
                except ValueError:
                    pdb.set_trace()
                return y_pred_x
            else:
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    return _predict(node["children"]["left"], x)
                else:  # x[j] > threshold
                    return _predict(node["children"]["right"], x)

        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred

    # ======================
    # Explain
    # ======================
    def explain(self, X, header_init):
        assert self.tree is not None
        header = [w.replace("init_", "") for w in header_init]

        def _explain(node, x, explanation):
            no_children = node["children"]["left"] is None and \
                node["children"]["right"] is None
            if no_children:
                variables = [header[i] for i in range(
                    len(header)) if not (i in node["not_to_fit"])]
                explanation.append(node["model"].to_string(variables)[0])
                return explanation
            else:
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    explanation.append("{} = {:.3f} <= {:.3f}".format(
                        header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                    return _explain(node["children"]["left"], x, explanation)
                else:  # x[j] > threshold
                    explanation.append("{} = {:.3f} > {:.3f}".format(
                        header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                    return _explain(node["children"]["right"], x, explanation)

        explanations = [_explain(self.tree, x, []) for x in X]
        return explanations

    # ======================
    # Loss
    # ======================
    # TODO: to fix, but it is not used for plot or fitting
    def loss(self, X_init, y_post):
        y_pred = self.predict(X_init)
        return mean_squared_error(y_pred, y_post)

    # ======================
    # Tree diagram
    # ======================
    def export_graphviz(self, output_filename, feature_names, known_invariant,
                        export_png=True, export_pdf=False):

        graph_attr = {
            "label": "{} is a ground truth invariant".format(known_invariant)}
        g = Digraph('g', graph_attr=graph_attr, node_attr={
            'shape': 'record', 'height': '.1'})

        def build_graphviz_recurse(node, parent_node_index=0, parent_depth=0, edge_label=""):

            # Empty node
            if node is None:
                return

            # Create node
            node_index = node["index"]
            if self.no_repeat:
                used_features = [feature_names[i] for i in range(
                    len(feature_names)) if i not in node["not_to_fit"]]
            else:
                used_features = feature_names
            if node["children"]["left"] is None and node["children"]["right"] is None:
                threshold_str = ""
                explain_str, norm = node["model"].to_string(used_features)
            else:
                threshold_str = "{} {} {:.3f} ".format(
                    feature_names[node['j_feature']], self.sign, node["threshold"])
                explain_str, norm = node["model"].to_string(used_features)
            label_str = "{} \\n n_samples = {}\\n loss = {:.5f} \\n{} \\n {}".format(
                threshold_str, node["n_samples"], node["loss"], explain_str, norm)

            # Create node
            nodeshape = "rectangle"
            bordercolor = "black"
            fillcolor = "white"
            fontcolor = "black"
            g.attr('node', label=label_str, shape=nodeshape)
            g.node('node{}'.format(node_index),
                   color=bordercolor, style="filled",
                   fillcolor=fillcolor, fontcolor=fontcolor)

            # Create edge
            if parent_depth > 0:
                g.edge('node{}'.format(parent_node_index),
                       'node{}'.format(node_index), label=edge_label)

            # Traverse child or append leaf value
            l = build_graphviz_recurse(node["children"]["left"],
                                       parent_node_index=node_index,
                                       parent_depth=parent_depth + 1,
                                       edge_label="")
            r = build_graphviz_recurse(node["children"]["right"],
                                       parent_node_index=node_index,
                                       parent_depth=parent_depth + 1,
                                       edge_label="")

        # Build graph
        build_graphviz_recurse(self.tree,
                               parent_node_index=0,
                               parent_depth=0,
                               edge_label="")
        # Export pdf
        if export_pdf:
            print("Saving model tree diagram to '{}.pdf'...".format(output_filename))
            g.format = "pdf"
            g.render(filename=output_filename, view=False, cleanup=True)

        # Export png
        if export_png:
            print("Saving model tree diagram to '{}.png'...".format(output_filename))
            g.format = "png"
            g.render(filename=output_filename, view=False, cleanup=True)

    # ======================
    # Form inv_func diagram
    # ======================
    def export_func(self, feature_names):

        def generate_invariant(node):

            # Empty node
            if node is None:
                return ""

            # Create node
            if self.no_repeat:
                used_features = [feature_names[i] for i in range(
                    len(feature_names)) if i not in node["not_to_fit"]]
            else:
                used_features = feature_names
            if node["children"]["left"] is None and node["children"]["right"] is None:
                threshold_str = ""
                explain_str, norm = node["model"].to_string(used_features)
                return explain_str
            else:
                threshold_str = "{} {} {:.3f} ".format(
                    feature_names[node['j_feature']], self.sign, node["threshold"])
                explain_str, norm = node["model"].to_string(used_features)

                # Traverse child or append leaf value
                l = generate_invariant(node["children"]["left"])
                r = generate_invariant(node["children"]["right"])

                opsign = op_sign(self.sign)
                return "[{}]({})+[{}]({})".format(threshold_str, l, threshold_str.replace(self.sign, opsign), r)

        def generate_txt(node, path):

            # Empty node
            if node is None:
                return ""

            opsign = op_sign(self.sign)
            if node["children"]["left"] is None and node["children"]["right"] is None:
                path_str = ", ".join(path)
                if self.no_repeat or len(node["not_to_fit"]) > 0:
                    pdb.set_trace()
                coefs = node["model"].model
                feature_lst = ["f{}:{}".format(i, coefs[i])
                               for i in range(len(coefs))]
                feature_str = ", ".join(feature_lst)
                return "{}\n{}".format(path_str, feature_str)
            else:
                threshold_str_l = "[{} {} {:.3f}]".format(
                    feature_names[node['j_feature']], self.sign, node["threshold"])
                threshold_str_r = threshold_str_l.replace(self.sign, opsign)
                path_l = path + [threshold_str_l]
                path_r = path + [threshold_str_r]
                # Traverse child or append leaf value
                l = generate_txt(node["children"]["left"], path_l)
                r = generate_txt(node["children"]["right"], path_r)

                return "{}\n\n{}".format(l, r)

        # Build graph
        invariant = generate_invariant(self.tree)

        return invariant, generate_invariant, generate_txt, self.tree

    def plot_fitting_hist(self, filename, just_plot, feature_names):
        if just_plot:
            hist = np.load('{}.npy'.format(filename),
                           allow_pickle='TRUE').item()
            filename += "_{}_{}_fitting".format(self.sign,
                                                "reverse_sort" if self.sort_reverse else "sort")
        else:
            hist = self.fitting_history
            savefilename = "_".join(filename.split("_")[:-4])
            np.save('{}.npy'.format(savefilename), hist)
        nodes = list(hist.keys())
        n = len(nodes)
        max_x_num = len(hist["root"]["y"])
        plot_size = (max(min(int(0.2 * max_x_num), 2**9), 50), 30*n)
        if n == 1:  # only one node
            fig, axes = plt.subplots(2, 1, figsize=plot_size)
            # meaningless plot to fill the space for subplot 1
            axes[1].plot(np.arange(1), np.arange(1))
        else:
            fig, axes = plt.subplots(n, 1, figsize=plot_size)
        for i in range(n):
            node = nodes[i]
            X = hist[node]["X"]
            xheader = [feature_names[i] for i in range(
                len(feature_names)) if i not in hist[node]["not_to_fit"]]
            if self.sort_reverse:
                arg_idx = np.lexsort(tuple(X[:, X.shape[1] - 1 - i]
                                           for i in range(X.shape[1])))
            else:
                arg_idx = np.lexsort(tuple(X[:, i] for i in range(X.shape[1])))
            X = X[arg_idx]
            y = hist[node]["y"][arg_idx]
            ypred = hist[node]["y_pred"][arg_idx]
            xaxis = np.arange(len(y))
            axes[i].set_xticks(xaxis)
            xticklabels = np.array([[round(q, 2) for q in row] for row in X])
            axes[i].set_xticklabels(xticklabels, fontsize=7, rotation=90)
            # green if data point from invariant constraint blue ow
            cutoff = hist[node]["cutoff"]
            colors = np.array([[0, 1, 0] if arg_idx[i] < cutoff
                               else [0, 0, 1] for i in xaxis])
            axes[i].scatter(xaxis, y, c=colors, label="y", linewidth=7.0)
            axes[i].plot(xaxis, ypred, '-', label="y_pred", linewidth=6.0)
            axes[i].set_title('fitting node {}'.format(node), fontsize=40)
            axes[i].set_xlabel(
                'data/program points {}'.format(xheader), fontsize=40)
            axes[i].set_ylabel('expected post/next point', fontsize=40)
            axes[i].legend()
        if self.testing_known_model:
            fig.suptitle("Using ground inv to predict y", fontsize=40)
        else:
            fig.suptitle("Using learned inv to predict y", fontsize=40)
        fig.savefig("{}.png".format(filename))


# ***********************************
#
# Side functions
#
# ***********************************
        # result = _splitter(node, model, header_init, self.sign,
        #                    no_repeat, not_to_fit=not_to_fit,
        #                    loss_func=self.loss_func,
        #                    min_samples_leaf=min_samples_leaf,
        #                    j_feature_range=j_feature_range,
        #                    max_depth=max_depth,
        #                    search_type=search_type,
        #                    n_search_grid=n_search_grid)

def _splitter(node, model, header_init, sign,
              no_repeat, not_to_fit, loss_func, min_samples_leaf,
              j_feature_range=None,
              max_depth=5,
              search_type="greedy", n_search_grid=100):

    # Extract data
    X, y = node["data"]
    depth = node["depth"]
    N, d = X.shape

    # Find feature splits that might improve loss
    did_split = False
    loss_best = node["loss"]
    data_best = None
    models_best = None
    j_feature_best = None
    threshold_best = None

    split_range = [i for i in j_feature_range if i not in not_to_fit]

    # Perform threshold split search only if node has not hit max depth
    if (depth >= 0) and (depth < max_depth):

        for j_feature in split_range:

            # If using adaptive search type, decide on one to use
            search_type_use = search_type
            if search_type == "adaptive":
                if N > n_search_grid:
                    search_type_use = "grid"
                else:
                    search_type_use = "greedy"

            # Use decided search type and generate threshold search list (j_feature)
            threshold_search = []
            if search_type_use == "greedy":
                for i in range(N):
                    threshold_search.append(X[i, j_feature])
            elif search_type_use == "grid":
                x_min, x_max = np.min(X[:, j_feature]), np.max(X[:, j_feature])
                dx = (x_max - x_min) / n_search_grid
                for i in range(n_search_grid+1):
                    threshold_search.append(x_min + i*dx)
            else:
                exit("err: invalid search_type = {} given!".format(search_type))

            threshold_search = list(set(threshold_search))  # deduplicate
            # Perform threshold split search on j_feature
            for threshold in threshold_search:

                # Split data based on threshold
                (X_left, y_left), (X_right, y_right) = _split_data(
                    j_feature, threshold, X, y, sign)
                N_left, N_right = len(X_left), len(X_right)

                # Splitting conditions
                split_conditions = [N_left >= min_samples_leaf,
                                    N_right >= min_samples_leaf]

                # Do not attempt to split if split conditions not satisfied
                if not all(split_conditions):
                    continue

                # Compute weight loss function
                loss_left, model_left = _fit_model(X_left, y_left, model)
                loss_right, model_right = _fit_model(X_right, y_right, model)
                loss_split = math.sqrt(loss_left ** 2 + loss_right ** 2)

                # print("Trying on {} based on {:.2f} with left loss {:.6f}, right loss {:.6f}, averaged loss {:.6f}, left has {:.0f}, right has {:.0f}".format(
                #     j_feature, threshold, loss_left, loss_right, loss_split, N_left, N_right))

                # Update best parameters if loss is lower
                if loss_split < loss_best:
                    print("previous best: {:.6f}".format(loss_best))
                    print("Splitting on {} based on {:.2f} with left loss {:.2f}, right loss {:.2f}, averaged loss {:.2f}, left has {:.0f}, right has {:.0f}".format(
                        j_feature, threshold, loss_left, loss_right, loss_split, N_left, N_right))
                    did_split = True
                    loss_best = loss_split
                    models_best = [model_left, model_right]
                    data_best = [(X_left, y_left), (X_right, y_right)]
                    j_feature_best = j_feature
                    threshold_best = threshold

    # Return the best result
    result = {"did_split": did_split,
              "loss": loss_best,
              "models": models_best,
              "data": data_best,
              "j_feature": j_feature_best,
              "threshold": threshold_best,
              "N": N}

    return result


def _fit_model(X, y, model):
    model_copy = deepcopy(model)  # must deepcopy the model!
    model_copy.fit(X, y)
    y_pred = model_copy.predict(X)
    loss = model_copy.loss(X, y, y_pred)  # X_used to used in loss
    assert loss >= 0.0
    return loss, model_copy


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


def _predict0(model, not_to_fit, X):
    if X.shape[0] > 0:
        _, d = X.shape
        X = [[row[i] for i in range(d) if i not in not_to_fit] for row in X]
        try:
            return model.predict(X)
        except AttributeError:
            pdb.set_trace()
    return np.array([])


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


def _record_fitting(X, y, y_pred, cutoff, not_to_fit):
    return {"y": y,
            "y_pred": y_pred,
            "X": X,
            "cutoff": cutoff,
            "not_to_fit": not_to_fit
            }  # indices where X_init start}


def _expandby1(X):
    dummybias = np.ones((len(X), 1))
    newX = np.concatenate((np.array(X), dummybias), axis=1)
    return newX


def op_sign(sign):
    if sign == "<=":
        return ">"
    elif sign == ">":
        return "<="
    elif sign == "==":
        return "!="
