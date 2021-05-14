from src.ex_prog import BiasPri, BiasDir, Prinsys, Duel, Unif, Detm, RevBin
from src.ex_prog import LinExp, Seq0, Seq1, Nest, Sum0, Sum1, DepRV
from src.ex_prog import GeoAr0, GeoAr1, GeoAr2, GeoAr3, Bin0, Bin1, Bin2, Bin3
from src.ex_prog import Geo0, Geo1, Geo2, Fair, Mart, Gambler0
from copy import deepcopy
from model_tree.run_model_tree import runModelTree
from model_tree.models.linear_regr_pnorm import linear_regr_2norm
from model_tree.models.linear_regr import linear_regr
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

'''
A set of parameters that helps experimenting with the tool.
Default setup if you just want to run the algorithm we described in the paper:
UPDATE_CSV = True
TESTING_KNOWN_MODEL = False
PLOT_fitting = False
PLOT_only = False
Bootstrapping = False
Bootstrapping_ratio = 1
PURE_linear = False
FIT_intercept = True
Regressors = [(linear_regr, "scikit_regression")]
'''
# whether to collect the data again and update CSV or retriving data from
# existing CSV and learn the model. CSV files are under directory `csv`.
UPDATE_CSV = False
# whether to test how a known model fits data, v.s. learn a model from data without prior knowledge
TESTING_KNOWN_MODEL = False
# whether to plot how the model fits data. Plotting gives us insights into how the
# model fits the data but takes time. Plots are under directory `output`.
PLOT_fitting = False
# PLOT_only only makes sense when PLOT_fitting is True. It specifies whether to
# only plot with the historical data and model, or learn a model again before plotting.
# If PLOT_only is True, there must be existing data in the `output` directory in the format of `.npy`.
PLOT_only = False
# whether get multiple samples of data by bootstrapping, instead of by rerunning the program
Bootstrapping = False
# Bootstrapping_ratio is only meaningful when Bootstrapping is True.
# It is the ratio of subsample size to the sample size.
Bootstrapping_ratio = 1
# Given that we fit data with model trees that having linear models on leaves:
# PURE_linear specifies whether to make the restriction that the whole model tree (as a whole)
# should be a linear model;
PURE_linear = False
# Fit_intercept specifies whether we let the leave model,
# which is the linear function to fit with intercept.
FIT_intercept = True
# A list of choices for leaf models
# Regressors = [(linear_regr_2norm, "2norm")]
Regressors = [(linear_regr, "scikit_regression")]

'''
A set of parameters to determine how much data to collect for the tool.
These are the set of hyper-parameters of the tool.
'''
# The number of runs from each initialization
#NUM_RUNS = int(sys.argv[1])
NUM_RUNS = 500
# We learn max(numBAG) model trees in total
# When Bootstrapping is False, we recollect the data multiple times to train multiple models;
# When Bootstrapping is True, we only collect the data once,
# and subsample to get `bags` and learn multiple models.
numBAG = [1,3,5]
# Max depth of the model tree
MAX_DEPTH = 2
# Minimum number of samples
MIN_SAMPLE_LEAF = 5
# The space for boolean on which we perform grid search
bool_space = np.linspace(0, 1, 2)
# The space for integer on which we perform grid search
int_space = np.linspace(-2,2,5)
# The space for probability choices on which we perform grid search
prob_space = np.linspace(0, 1, 11)
# Grids of initial states for each program based on the number of variables
# (that exist in multiple iterations) of each type in them
INIT_GRID2bool1int = [{"bool": [bool1, bool2],
                       "int": [int1]}
                      for bool1 in bool_space
                      for bool2 in bool_space
                      for int1 in int_space]

INIT_GRID1bool1int = [{"bool": [bool1],
                       "int": [int1]}
                      for bool1 in bool_space
                      for int1 in int_space]

INIT_GRID1bool1float = [{"bool": [bool1],
                         "float": [float1]}
                        for bool1 in bool_space
                        for float1 in prob_space]

INIT_GRID1bool2int = [{"bool": [bool1],
                       "int": [int1, int2]}
                      for bool1 in bool_space
                      for int1 in int_space
                      for int2 in int_space]

INIT_GRID3bool2int = [{"bool": [bool1, bool2, bool3],
                       "int": [int1, int2]}
                      for bool1 in bool_space
                      for bool2 in bool_space
                      for bool3 in bool_space
                      for int1 in int_space
                      for int2 in int_space]

INIT_GRID0bool3int = [{"bool": [],
                       "int": [int1, int2, int3]}
                      for int1 in int_space
                      for int2 in int_space
                      for int3 in int_space]

INIT_GRID2bool0int = [{"bool": [bool1, bool2],
                       "int": []}
                      for bool1 in bool_space
                      for bool2 in bool_space]

INIT_GRID0bool1int = [{"bool": [],
                       "int": [int1]}
                      for int1 in int_space]

INIT_GRID1bool0int = [{"bool": [bool1],
                       "int": []}
                      for bool1 in bool_space]

INIT_GRID0bool2int = [{"bool": [],
                       "int": [int1, int2]}
                      for int1 in int_space
                      for int2 in int_space]

# The grids of choices for probabilities. We seperated them out from other
# variables when experimenting the idea, but later decided to treat them the
# same way as the other variables. They can be merged with the previous INIT_GRIDs
# but we kept the old implementation for convenience.
trivial_space = np.linspace(0, 0, 1)
normal = np.linspace(0.1, 0.9, 9)
fine_grid = np.linspace(0.05, 0.95, 19)
probinpts0 = [()]
probinpts1 = [(prob1, ) for prob1 in normal]
probinpts2 = [(prob1, prob2) for prob1 in normal for prob2 in normal]

# Rounding to [decimal] if [decimal_rounding] is true, otherwise round to [provided_value]


def rounding(number, decimal_rounding, decimal, provided_value=None):
    if decimal_rounding:
        return round(number, decimal)
    else:
        distance = np.abs(np.array(provided_value) - number)
        idx = list(distance).index(min(distance))
        return provided_value[idx]


def default_rounding(x): return rounding(x, True, 4)

'''
Info of example programs in the following format:
"name": (instrumented programs, probinpts(choices for probabilities), INIT_GRID,
sign of predicates in the model tree)

For the sign, "<=" and ">" would be symmetric because "a > b" is just "not a <= b";
and since the set of values we try in INIT_GRID and probinpts are discrete,
"<" and ">=" will have the same expressive power as ">" and "<=".
Thus, we can just input "<=" by default unless we want to try "==".
'''

progs = {
    # "Geo0": (Geo0, probinpts1, INIT_GRID1bool1int, "=="),
    # "Geo1": (Geo1, probinpts1, INIT_GRID1bool2int, "=="),
    # "Geo2": (Geo2, probinpts1, INIT_GRID1bool2int, "=="),
    # "Fair": (Fair, probinpts2, INIT_GRID2bool1int, "=="),
    # "Mart": (Mart, probinpts1, INIT_GRID0bool3int, "<="),
    # "Gambler0": (Gambler0, probinpts0, INIT_GRID0bool3int, "<="),
    "GeoAr0": (GeoAr0, probinpts1, INIT_GRID1bool2int, "<="),
    # "GeoAr1": (GeoAr1, probinpts0, INIT_GRID1bool1int, "<="),
    # "GeoAr2": (GeoAr2, probinpts0, INIT_GRID1bool2int, "<="),
    "GeoAr3": (GeoAr3, probinpts1, INIT_GRID1bool1int, "<="),
    # "Bin0": (Bin0, probinpts1, INIT_GRID0bool3int, "<="),
    # "Bin1": (Bin1, probinpts1, INIT_GRID0bool3int, "<="),
    # "Bin2": (Bin2, probinpts1, INIT_GRID0bool3int, "<="),
    # "Bin3": (Bin3, probinpts0, INIT_GRID0bool3int, "<="),
    # "LinExp": (LinExp, probinpts0, INIT_GRID3bool2int, "<="),
    # "Seq0": (Seq0, probinpts2, INIT_GRID2bool1int, "<="),
    # "Seq1": (Seq1, probinpts2, INIT_GRID2bool1int, "<="),
    # "Nest": (Nest, probinpts2, INIT_GRID2bool1int, "<="),
    # "Sum0": (Sum0, probinpts1, INIT_GRID0bool2int, "<="),
    # "Sum1": (Sum1, probinpts0, INIT_GRID0bool2int, "<="),
    # "DepRV": (DepRV, probinpts0, INIT_GRID0bool3int, "<="),
    # "BiasPri": (BiasPri, probinpts1, INIT_GRID2bool0int, "=="),
    # "BiasDir": (BiasDir, probinpts1, INIT_GRID2bool0int, "<="),
    # "Prinsys": (Prinsys, probinpts2, INIT_GRID0bool1int, "<="),
    # "Duel": (Duel, probinpts2, INIT_GRID2bool0int, "<="),
    # "Unif": (Unif, probinpts0, INIT_GRID0bool2int, "<="),
    # "Detm": (Detm, probinpts0, INIT_GRID0bool2int, "<="),
    # "RevBin": (RevBin, probinpts1, INIT_GRID0bool2int, "<="),
}

# example_types = {}
'''
Run example programs in ex_prog.py to collect data
'''


def get_stats(progname, proginfo, filename):
    prog, inpts, initgrid, _ = proginfo
    hists = None  #
    if UPDATE_CSV:
        for inpt in inpts:
            print(inpt)
            for init_states in initgrid:
                for _ in range(NUM_RUNS):  # repeatedly run
                    hists = prog(progname, inpt, hists, init_states)
                try:
                    hists.end_sampling_runs(inpt)
                except AttributeError:
                    pdb.set_trace()
        # example_types[progname] = hists.types
        df = hists.to_df()
        filename_csv = os.path.join("csv", "{}.csv".format(filename))
        df.to_csv(filename_csv, index=False)
    else:
        hists = prog(progname, inpts[0], hists, initgrid[0])
    return hists.variable_indices, hists.known_inv, hists.known_model


'''
The main routine of the tool
'''


def main():
    learned_inv = []
    for name, proginfo in progs.items():
        nBAG = max(numBAG)
        # for timing
        tot_time = timeit.default_timer()
        samp_time = 0
        tree_time = 0
        # iterating through choices of leaf models
        for regresser, leafmodelname in Regressors:
            filename = name
            if Bootstrapping:
                time = timeit.default_timer()
                variable_indices, known_inv, known_model = get_stats(
                    name, proginfo, filename)
                samp_time = samp_time + timeit.default_timer() - time
            treelist = []
            for T in range(nBAG):
                if not Bootstrapping:
                    filename = "{}_{}".format(name, T)
                    time = timeit.default_timer()
                    variable_indices, known_inv, known_model = get_stats(
                        name, proginfo, filename)
                    samp_time = samp_time + timeit.default_timer()-time
                time = timeit.default_timer()
                model = regresser(fit_intercept=FIT_intercept)
                m = runModelTree(model, filename, leafmodelname, variable_indices,
                                 known_inv, known_model, TESTING_KNOWN_MODEL,
                                 PLOT_only, PLOT_fitting,
                                 Bootstrapping, Bootstrapping_ratio,
                                 sign=list(proginfo)[-1],
                                 pure_linear=PURE_linear, max_depth=MAX_DEPTH,
                                 min_samples_leaf=MIN_SAMPLE_LEAF)

                invariant, inv_func_recurse, generate_txt, learned_tree = m.run()
                if TESTING_KNOWN_MODEL or PLOT_only:
                    continue
                # picklename = os.path.join(
                #     "pickle", "{}.p".format(filename))
                # pickle.dump(learned_tree, open(picklename, 'wb'))
                tree_time = timeit.default_timer() + tree_time - time
                treelist.append(learned_tree)
                learned_inv.append(
                    [leafmodelname, name, known_inv, invariant, len(proginfo[1]), nBAG])
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
                            [leafmodelname, "{}_{}".format(name, method), known_inv, invariant_string, len(proginfo[1]), T+1])
                        learned_inv.append(
                            [leafmodelname, "{}_supertree_{}".format(name, method), known_inv, invariant_string2, len(proginfo[1]), T+1])

                        write_to_csv(learned_inv)
                        save_to_text(leafmodelname, "{}_{}".format(name, method), known_inv,
                                     invariant_string, proginfo[1], generate_txt(average_tree, []), T+1)
                        save_to_text(leafmodelname, "{}_supertree_{}".format(
                            name, method), known_inv, invariant_string2, proginfo[1], generate_txt(average_tree2, []), T+1)

        tot_time = timeit.default_timer()-tot_time
        row = [name, tot_time, samp_time, tree_time, samp_time/nBAG, tree_time/nBAG, nBAG,
               NUM_RUNS, len(proginfo[1]), str(Bootstrapping), str(PURE_linear), proginfo[3]]
        with open('invariants/used_time', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(row)


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


'''
Define two trees to have the same structures if they are the same except having 
different leaf models. 
'''
'''
 Return: true if tree2 can be pruned into a tree that has the same structure as tree1, 
 return false otherwise
'''


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


'''
 Return: the deepest tree T such that each T' in treelist can be pruned into 
 a tree that has the same structure as T. 
'''


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


'''
 Average a list of linear functions by taking the mean of each coeefficients
'''


def avg_models(models):
    modelmodel = np.mean(np.array([model.model for model in models]), axis=0)
    roundedmodel = np.array(list(map(default_rounding, modelmodel)))
    return linear_regr_2norm(fit_intercept=FIT_intercept, model=roundedmodel)


'''
 Average a list of linear functions by taking the median of each coeefficient
'''


def median_models(models):
    modelmodel = np.median(np.array([model.model for model in models]), axis=0)
    roundedmodel = np.array(list(map(default_rounding, modelmodel)))
    return linear_regr_2norm(fit_intercept=FIT_intercept, model=roundedmodel)


'''
 Assume: all trees in treelist has the same structure 
 Return: a tree whose structure is the same as any tree in treelist, 
 and every leaf model of which is the average of leaf models of trees in treelist 
 at the same position. 
'''


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


'''
 Return: 
 classified_tree_list: a dictionary with keys to be some number indexing 
 different structures, and values to be lists of trees -- trees are in the same 
 list iff they have the same structure. 
 names_classified_tree_list: similar to classified_tree_list, except its values, 
 it contains the trees' indices in the input treelist, instead of the trees themselves. 
'''


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


'''
 Assume:  classified_tree_list is a dictionary with keys to be some number indexing 
 different structures, and values to be lists of trees -- trees are in the same 
 list iff they have the same structure. 
 Return: the list of same structure trees in classified_tree_list.values() 
 that has the most members. We just pick the one last seen when there is a tie. 
'''


def find_most_common_structure(classified_tree_list):
    most_common_structure_treeclass = None
    maxlen = 0
    for _, treeclass in classified_tree_list.items():
        if len(treeclass) >= maxlen:
            maxlen = len(treeclass)
            most_common_structure_treeclass = treeclass
    return most_common_structure_treeclass


'''
 Assume: learned_inv is a list of list. 
 Each entry of learned_inv is in the form of 
 [leafmodelname, name, known_inv, learned invariants, len(proginfo[1]), 
  the number of models it aggregates]
  
 The function writes info in learned_inv and other global parameters into a csv. 
'''


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
    learned_inv_dict["purely linear"] = str(PURE_linear)
    learned_inv_dict["NUM_RUNS"] = str(NUM_RUNS)
    learned_inv_dict["Bootstrapping_ratio"] = str(Bootstrapping_ratio)
    df = pd.DataFrame.from_dict(learned_inv_dict)

    saving_filename = os.path.join(
        "invariants", "learned_invariants_{}.csv".format(dt_string))
    df.to_csv(saving_filename, index=False)


'''
 Save the parameters and a text representation of the learned model tree that 
 can be passed to perform different rounding schemes under the directory `txt`
'''


def save_to_text(leafmodelname, name, known_inv, invariant, probinputs, text, nBAG):
    filename = os.path.join(
        "txt", "{}_{}.txt".format(name, dt_string))
    with open(filename, "w") as f:
        f.write("benchmark: {}\n".format(name))
        f.write("\n")
        f.write("invariant: {}\n".format(invariant))
        f.write("\n")
        parameters = "leaf model={}, prob_grid_size={}, Aggregate_size={}, Num_runs={}, Max_depth={}, Min_samples_leaf={}".format(
            leafmodelname, len(probinputs), nBAG, NUM_RUNS, MAX_DEPTH, MIN_SAMPLE_LEAF)
        f.write("paramenters: {}\n".format(parameters))
        f.write("\n")
        f.write("model_tree_begin\n")
        f.write("\n")
        f.write(text)
        f.write("\n")
        f.write("model_tree_end\n")
        f.write("\n")
    f.close()


main()
