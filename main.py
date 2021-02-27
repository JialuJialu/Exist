from copy import deepcopy
# import cvxpy as cp
# from cvxpy.atoms.axis_atom import AxisAtom
# from cvxpy.atoms.norm import norm
from model_tree.models.linear_regr_constraint import linear_regr_constraint, linear_regr_constraint_L1, linear_regr_constraint_L2
from model_tree.models.linear_regr_lasso import linear_regr_lasso
from model_tree.models.linear_regr import linear_regr
from model_tree.run_model_tree import runModelTree
from src.data_utils import AggregateData
from src.ex_prog import geo_0, geo_0a, geo_0b,  geo_0c, ex1, ex2, ex3a, ex3b, ex3, ex3nest, ex3hard
from src.ex_prog import ex4, ex4a, ex5, ex5y, ex5yp, ex5p, ex7, ex8, ex8p, ex9, ex9p, ex10,  ex11, ex11a
from src.ex_prog import ex12, ex13, ex15, ex15a, ex17, ex18, ex19
from src.ex_prog import ex20, ex21, ex22
from src.ex_prog import exp0a, exp1a, exp1b, exp1c, exp3a, exp3b, exp12, exp19a, exp19b, exp21
# from src.ex_prog import ex21, ex22, ex22a, ex23
import os
import pdb
import itertools
import numpy as np
import pandas as pd
import random
import pickle
import cProfile
import timeit
from datetime import datetime
from collections import defaultdict
import sys
import csv
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%m:%d:%H:%M")

# parameters that do not need to change when running experiements
UPDATE_CSV = True  # False just want to fit learning and don't want collect data again
TESTING_KNOWN_MODEL = False
FIT_USED = True  # FIT_USED False if require a truly linear moel
FIT_intercept = False
Bootstrapping = False
# only meaningful when Bootstrapping is True, it is the ratio of subsample size: sample size
Bootstrapping_ratio = 1

# May change while running experiements
NUM_RUNS = int(sys.argv[1])
PLOT_fitting = False
PLOT_only = False
# When not Bootstrapping, we sample independently for nBAG times
# if Bootstrapping, then we only sample data once, and subsample to get bags;
# numBAG = [1, 2, 5, 10, 15]
numBAG = [5]
MAX_DEPTH = 2
MIN_SAMPLE_LEAF = 5

bool_space = np.linspace(0, 1, 2)
non_neg_int_space = np.linspace(0, 4, 5)
trivial_space = np.linspace(0, 0, 1)
prob_space = np.linspace(0, 1, 11)

# ---
INIT_GRID2bool1int = [{"bool": [bool1, bool2],
                       "non_neg_int": [int1]}
                      for bool1 in bool_space
                      for bool2 in bool_space
                      for int1 in non_neg_int_space]

INIT_GRID1bool1int = [{"bool": [bool1],
                       "non_neg_int": [int1]}
                      for bool1 in bool_space
                      for int1 in non_neg_int_space]

INIT_GRID1bool1float = [{"bool": [bool1],
                         "float": [float1]}
                        for bool1 in bool_space
                        for float1 in prob_space]

INIT_GRID1bool2int = [{"bool": [bool1],
                       "non_neg_int": [int1, int2]}
                      for bool1 in bool_space
                      for int1 in non_neg_int_space
                      for int2 in non_neg_int_space]

INIT_GRID3bool2int = [{"bool": [bool1, bool2, bool3],
                       "non_neg_int": [int1, int2]}
                      for bool1 in bool_space
                      for bool2 in bool_space
                      for bool3 in bool_space
                      for int1 in non_neg_int_space
                      for int2 in non_neg_int_space]

INIT_GRID0bool3int = [{"bool": [],
                       "non_neg_int": [int1, int2, int3]}
                      for int1 in non_neg_int_space
                      for int2 in non_neg_int_space
                      for int3 in non_neg_int_space]

INIT_GRID2bool0int = [{"bool": [bool1, bool2],
                       "non_neg_int": []}
                      for bool1 in bool_space
                      for bool2 in bool_space]

INIT_GRID0bool1int = [{"bool": [],
                       "non_neg_int": [int1]}
                      for int1 in non_neg_int_space]

INIT_GRID1bool0int = [{"bool": [bool1],
                       "non_neg_int": []}
                      for bool1 in bool_space]

INIT_GRID0bool2int = [{"bool": [],
                       "non_neg_int": [int1, int2]}
                      for int1 in non_neg_int_space
                      for int2 in non_neg_int_space]
# def generateGrid(int_num, bool_num):
#     m = len(int_space)**int_num * len(bool_space)**bool_num
#     grid = [{"bool": [], "int": []} for _ in range(m)]
#     for i in range(int_num):
#         for bool_i in bool_space:


normal = np.linspace(0.1, 0.9, 9)
fine_grid = np.linspace(0.05, 0.95, 19)
GAP = normal[1] - normal[0]
inptinpts0 = [(prob1, prob2, prob3)
              for prob1 in trivial_space for prob2 in trivial_space for prob3 in trivial_space]
inptinpts1 = [(prob1, prob2, prob3)
              for prob1 in normal for prob2 in trivial_space for prob3 in trivial_space]
inptinpts2 = [(prob1, prob2, prob3)
              for prob1 in normal for prob2 in normal for prob3 in trivial_space]


'''
"examplename": (example, inptinpts(number of probabilities it takes),
INIT_GRID(based on the number of each type needed to initialize the program state),
Uncon_GRID(based on the number of each type needed to initialize the program state)
)
'''
# TODO: supply sign for in the end of pair as in geo_0
progs = {
    # 100 * 2 * 5 * 20 * 2
    "geo_0": (geo_0, inptinpts1, INIT_GRID1bool1int, "<="),
    # "geo_0a": (geo_0a, inptinpts1, INIT_GRID1bool2int, "<="),
    # "geo_0b": (geo_0b, inptinpts1, INIT_GRID1bool2int), #TODO
    # "geo_0c": (geo_0c, inptinpts1, INIT_GRID1bool2int), #TODO
    # "ex2": (ex2, inptinpts2, INIT_GRID0bool3int,">"),
    # "ex3": (ex3, inptinpts2, INIT_GRID2bool1int), #TODO
    # "ex3nest": (ex3nest, inptinpts2, INIT_GRID2bool1int, ">"),
    # "exp3nest_a": (exp3nest_a, inptinpts2, INIT_GRID2bool1int, ">"),
    # "ex3a": (ex3a, inptinpts2, INIT_GRID1bool1int, ">"),
    # "ex3b": (ex3b, inptinpts2, INIT_GRID1bool1int,">"),
    # "ex4": (ex4, inptinpts0, INIT_GRID0bool3int, "<="),
    # "ex5": (ex5, inptinpts0, INIT_GRID1bool2int, "<="),
    # "ex5y": (ex5y, inptinpts0, INIT_GRID1bool2int, "<="),
    # "ex5yp": (ex5yp, inptinpts1, INIT_GRID1bool2int, "<="),
    # "ex5p": (ex5p, inptinpts1, INIT_GRID1bool2int, "<="),
    # "ex7": (ex7, inptinpts1, INIT_GRID0bool3int,"<="),
    # "ex8": (ex8, inptinpts1, INIT_GRID0bool3int, "<="),
    # "ex8a": (ex8a, inptinpts1, INIT_GRID0bool3int, "<="),
    # "ex9": (ex9, inptinpts0, INIT_GRID0bool2int, "<="),
    # "ex9p": (ex9p, inptinpts1, INIT_GRID0bool2int, "<="),
    # "ex10": (ex10, inptinpts0, INIT_GRID0bool3int,"<="),
    # "ex11": (ex11, inptinpts1, INIT_GRID2bool0int, "<="),
    # "ex11a": (ex11a, inptinpts1, INIT_GRID2bool0int,"<="),
    # "ex12": (ex12, inptinpts2, INIT_GRID0bool1int,"<="),
    # "ex13": (ex13, inptinpts2, INIT_GRID0bool3int),
    # "ex15": (ex15, inptinpts0, INIT_GRID0bool2int, "<="),
    # "ex15a": (ex15a, inptinpts0, INIT_GRID0bool2int, "<="),
    # "ex17": (ex17, inptinpts1, INIT_GRID1bool1float,"=="),
    # "ex18": (ex18, inptinpts1, INIT_GRID0bool3int,"<="),
    # "ex19": (ex19, inptinpts2, INIT_GRID2bool0int,">"),
    # "exp19a": (exp19a, inptinpts2, INIT_GRID2bool0int),
    # "exp19b": (exp19b, inptinpts2, INIT_GRID2bool0int),
    # "ex20": (ex20, inptinpts1, INIT_GRID3bool2int,">"),
    # "ex20a": (ex20a, inptinpts1, INIT_GRID3bool2int),
    # "ex21": (ex21, inptinpts1, INIT_GRID0bool2int, "<="),
    # "exp0a": (exp0a, inptinpts1, INIT_GRID1bool1int),
    # "ex3a": (exp3a, inptinpts2, INIT_GRID1bool1int),
    # "ex3b": (exp3b, inptinpts2, INIT_GRID1bool1int),
}  # programs to run


def get_stats(progname, proginfo, filename):
    prog, inpts, initgrid, _ = proginfo
    aggre_data = AggregateData()
    if UPDATE_CSV:
        for inpt in inpts:
            print(inpt)
            for init_states in initgrid:
                hists = None  # One hists for one set of input
                for _ in range(NUM_RUNS):  # repeatedly run
                    hists = prog(progname, inpt, hists, init_states)
                try:
                    aggre_data.end_sampling_runs(hists, inpt)
                except AttributeError:
                    pdb.set_trace()
        df = aggre_data.to_df()
        filename_csv = os.path.join("csv", "{}.csv".format(filename))
        df.to_csv(filename_csv, index=False)
    else:
        hists = None
        hists = prog(progname, inpts[0], hists,
                     initgrid[0], constraint=False)
        aggre_data.end_sampling_runs(hists, inpts[0])
    return aggre_data.qual_feature_indices, aggre_data.known_inv, aggre_data.known_model


'''
Recall that node has the following structure
            node = {"name": node_name,
                    "parent_name": parent,
                    "index": container["index_node_global"],
                    "loss": loss_node,
                    "model": model_node,
                    "data": (X_cur, X_next, X_init, y_post),
                    "constraint": (X_cons, y_cons),
                    "n_samples": len(X_inv) + len(X_init),
                    "not_to_fit": not_to_fit[:],
                    "j_feature": None,
                    "threshold": None,
                    "children": {"left": None, "right": None},
                    "direction": direction,
                    "parent_j_feature": parent_j_feature,
                    "parent_threshold": parent_threshold,
                    "root": root,
                    "depth": depth}
'''

'''
Return true if tree2 has the same structure as tree 1
'''


def check_same_structure(tree1, tree2):
    if tree1["name"] != tree2["name"]:
        return False
    else:
        if tree1["j_feature"] != tree2["j_feature"] or tree1["threshold"] != tree2["threshold"]:
            return False
        else:
            check_left = (tree1["children"]["left"]
                          is None and tree1["children"]["left"] is None) or (check_same_structure(
                              tree1["children"]["left"], tree2["children"]["left"]))
            check_right = (tree1["children"]["right"]
                           is None and tree1["children"]["right"] is None) or (check_same_structure(
                               tree1["children"]["right"], tree2["children"]["right"]))
            return check_left and check_right

# '''
# Return true if tree2 has the same structure as tree 1 except tree 2 making more splits
# '''


def check_tree_subtype(tree1, tree2):
    if tree1["name"] != tree2["name"] or tree1["parent_name"] != tree2["parent_name"]:
        return False
    else:
        if tree1["j_feature"] != tree2["j_feature"] or tree1["threshold"] != tree2["threshold"]:
            return False
        else:
            check_left = (tree1["children"]["left"] is None) or (check_same_structure(
                tree1["children"]["left"], tree2["children"]["left"]))
            check_right = (tree1["children"]["right"] is None) or (check_same_structure(
                tree1["children"]["right"], tree2["children"]["right"]))
            return check_left and check_right


def find_common_structure(treelist):
    treelist_ = np.array([deepcopy(tree) for tree in treelist])
    n = len(treelist_)
    if np.any([tree is None for tree in treelist]):
        return [None for _ in range(n)]
    namelist = [tree["name"] for tree in treelist_]
    same_name = np.all([namelist[i] == namelist[0] for i in range(n)])
    thresholdlist = [tree["threshold"] for tree in treelist_]
    same_threshold = np.all(
        [thresholdlist[i] == thresholdlist[0] for i in range(n)])
    j_featurelist = [tree["j_feature"] for tree in treelist_]
    same_j_feature = np.all(
        [j_featurelist[i] == j_featurelist[0] for i in range(n)])

    if not same_name:
        return [None for _ in range(n)]
    else:
        if not (same_j_feature and same_threshold):
            for tree in treelist_:
                tree["children"]["right"], tree["children"]["left"] = None, None
                tree["j_feature"], tree["threshold"] = None, None
            return treelist_
        else:
            leftlist = find_common_structure(
                [tree["children"]["left"] for tree in treelist_])
            rightlist = find_common_structure(
                [tree["children"]["right"] for tree in treelist_])
            for treeidx in range(n):
                treelist_[treeidx]["children"]["right"] = rightlist[treeidx]
                treelist_[treeidx]["children"]["left"] = leftlist[treeidx]
            return treelist_


def avg_models(models):
    modelmodel = np.mean(np.array([model.model for model in models]), axis=0)
    modellambda = np.mean(np.array([model.lambd for model in models]))
    return linear_regr_constraint(lambd=modellambda, model=modelmodel)


def median_models(models):
    modelmodel = np.median(np.array([model.model for model in models]), axis=0)
    modellambda = np.median(np.array([model.lambd for model in models]))
    return linear_regr_constraint(lambd=modellambda, model=modelmodel)


def aggregate_trees(treelist, aggre_method):
    if np.all(treelist == None):
        return None
    assert not np.any(treelist == None)
    avg_tree = treelist[0]
    try:
        avg_tree["loss"] = np.mean([tree["loss"] for tree in treelist])
    except TypeError:
        pdb.set_trace()
    avg_tree["model"] = aggre_method([tree["model"] for tree in treelist])
    avg_tree["data"] = ()
    avg_tree["constraint"] = ()
    avg_tree["n_samples"] = np.mean([tree["n_samples"] for tree in treelist])
    avg_tree["children"]["left"] = aggregate_trees(
        np.array([tree["children"]["left"] for tree in treelist]), aggre_method)
    avg_tree["children"]["right"] = aggregate_trees(
        np.array([tree["children"]["right"] for tree in treelist]), aggre_method)
    return avg_tree


def classify_tree_list(treelist):
    classified_tree_list = defaultdict(list)
    names_classified_tree_list = defaultdict(list)
    counter = 0
    for treeidx in range(len(treelist)):
        tree = treelist[treeidx]
        tree_added = False
        for i in classified_tree_list.keys():
            if check_same_structure(classified_tree_list[i][0], tree):
                classified_tree_list[i].append(tree)
                names_classified_tree_list[i].append(treeidx)
                tree_added = True
        if not tree_added:
            classified_tree_list[counter] = [tree]
            names_classified_tree_list[counter] = [treeidx]
            counter += 1
    return classified_tree_list, names_classified_tree_list


def find_most_common_structure(classified_tree_list):
    most_common_structure_treeclass = None
    maxlen = 0
    for _, treeclass in classified_tree_list.items():
        if len(treeclass) > maxlen:
            maxlen = len(treeclass)
            most_common_structure_treeclass = treeclass
    return most_common_structure_treeclass


def write_to_csv(learned_inv):
    learned_inv_dict = {}
    learnd_inv_array = np.array(learned_inv)
    learned_inv_dict["method"] = learnd_inv_array[:, 0]
    learned_inv_dict["prog_name"] = learnd_inv_array[:, 1]
    learned_inv_dict["known_inv"] = learnd_inv_array[:, 2]
    learned_inv_dict["invariant"] = learnd_inv_array[:, 3]
    learned_inv_dict["prob_grid_size"] = learnd_inv_array[:, 4]
    learned_inv_dict["Bootstrapping_subsampling"] = str(Bootstrapping)
    learned_inv_dict["Aggregate_size"] = learnd_inv_array[:, 5]
    learned_inv_dict["use_var_more_than_once"] = str(FIT_USED)
    learned_inv_dict["NUM_RUNS"] = str(NUM_RUNS)
    learned_inv_dict["Bootstrapping_ratio"] = str(Bootstrapping_ratio)
    df = pd.DataFrame.from_dict(learned_inv_dict)

    saving_filename = os.path.join(
        "invariants", "learned_invariants_{}.csv".format(dt_string))
    df.to_csv(saving_filename, index=False)


def save_to_text(norm, name, known_inv, invariant, probinputs, text, nBAG):
    filename = os.path.join(
        "txt", "{}_{}.txt".format(name, dt_string))
    with open(filename, "w") as f:
        f.write("benchmark: {}\n".format(name))
        f.write("\n")
        f.write("invariant: {}\n".format(invariant))
        f.write("\n")
        parameters = "method={}, prob_grid_size={}, Aggregate_size={}, Num_runs={}, Max_depth={}, Min_samples_leaf={}".format(
            norm, len(probinputs), nBAG, NUM_RUNS, MAX_DEPTH, MIN_SAMPLE_LEAF)
        f.write("paramenters: {}\n".format(parameters))
        f.write("\n")
        f.write("model_tree_begin\n")
        f.write("\n")
        f.write(text)
        f.write("\n")
        f.write("model_tree_end\n")
        f.write("\n")
    f.close()


learned_inv = []
for name, proginfo in progs.items():
    nBAG = max(numBAG)
    tot_time = timeit.default_timer()
    samp_time = 0
    tree_time = 0
    regressors = [
        # (linear_regr, "MSE"),
        (linear_regr_constraint, "MSEwithConstraint"),
        # (linear_regr_constraint_L1, "L1withConstraint"),
        # (linear_regr_constraint_L2, "L2withConstraint")
    ]
    for regresser, norm in regressors:
        plot_fitting = PLOT_fitting
        filename = name
        if Bootstrapping:
            time = timeit.default_timer()
            qual_features_indices, known_inv, known_model = get_stats(
                name, proginfo, filename)
            samp_time = samp_time + timeit.default_timer() - time
        treelist = []
        for T in range(nBAG):
            filename = "{}_{}".format(name, T)
            if not Bootstrapping:
                time = timeit.default_timer()
                qual_features_indices, known_inv, known_model = get_stats(
                    name, proginfo, filename)
                samp_time = samp_time + timeit.default_timer()-time
            time = timeit.default_timer()
            model = regresser(fit_intercept=FIT_intercept)
            m = runModelTree(model, filename, norm, qual_features_indices,
                             known_inv, known_model, TESTING_KNOWN_MODEL,
                             PLOT_only, plot_fitting,
                             Bootstrapping, Bootstrapping_ratio,
                             sign=list(proginfo)[-1],
                             fit_used=FIT_USED, max_depth=MAX_DEPTH,
                             min_samples_leaf=MIN_SAMPLE_LEAF)

            invariant, inv_func_recurse, generate_txt, learned_tree = m.run()
            picklename = os.path.join(
                "pickle", "{}.p".format(filename))
            pickle.dump(learned_tree, open(picklename, 'wb'))
            tree_time = timeit.default_timer() + tree_time - time
            treelist.append(learned_tree)
            learned_inv.append(
                [norm, name, known_inv, invariant, len(proginfo[1]), nBAG])
            # eagerly write to the file so the previous data won't be lost when we stuck in some examples
            write_to_csv(learned_inv)

            if T+1 in numBAG:

                classified_tree_list, names_classified_tree_list = classify_tree_list(
                    treelist)
                print(names_classified_tree_list)
                most_common_structure_treeclass = find_most_common_structure(
                    classified_tree_list)

                for (aggregate_method, method) in [(avg_models, "mean"), (median_models, "median")]:
                    average_tree = aggregate_trees(
                        most_common_structure_treeclass, aggregate_method)
                    invariant_string = inv_func_recurse(average_tree)

                    super_tree = find_common_structure(treelist)
                    average_tree2 = aggregate_trees(
                        super_tree, aggregate_method)
                    invariant_string2 = inv_func_recurse(average_tree2)

                    learned_inv.append(
                        [norm, "{}_{}".format(name, method), known_inv, invariant_string, len(proginfo[1]), T+1])
                    learned_inv.append(
                        [norm, "{}_supertree_{}".format(name, method), known_inv, invariant_string2, len(proginfo[1]), T+1])

                    write_to_csv(learned_inv)
                    save_to_text(norm, "{}_{}".format(name, method), known_inv,
                                 invariant_string, proginfo[1], generate_txt(average_tree, []), nBAG)
                    save_to_text(norm, "{}_supertree_{}".format(
                        name, method), known_inv, invariant_string2, proginfo[1], generate_txt(average_tree2, []), nBAG)
    tot_time = timeit.default_timer()-tot_time

    row = [name, tot_time, samp_time, tree_time, samp_time/nBAG, tree_time/nBAG, nBAG,
           NUM_RUNS, len(proginfo[1]), str(Bootstrapping), str(FIT_USED), proginfo[3]]
    with open('invariants/used_time', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)
