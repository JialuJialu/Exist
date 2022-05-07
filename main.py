from wolframclient.evaluation import WolframLanguageSession
from learners.NN import NNLearner
from learners.Tree import TreeLearner
import pandas as pd
import os
import time
from datetime import datetime
from cegis import cegis_one_prog, get_var_types
import sys
import json
import argparse
import copy

parser = argparse.ArgumentParser()

parser.add_argument("-sub", "--subinv", help="put 'yes' (or 'y') if \
    you want to learn subinvariant; otherwise, the program learns exact invariant")
parser.add_argument("-cached", "--cached", help="put 'yes' (or 'y') \
    if you want to use cached program traces as the data; otherwise, \
        the program would sample new data")
parser.add_argument("-nruns", "--nruns", help="number of runs from each initial\
    state", type = int)
parser.add_argument("-nstates", "--nstates", help="number of initial states \
    we sample for each benchmark", type = int)

args = parser.parse_args()

# By default, the program will learn exact invariants 
if str(args.subinv).lower() in ["yes", "y"]: 
    exact = False
else: 
    exact = True

# By default, the program will sample new data
if str(args.cached).lower() in ["yes", "y"]: 
    sample_new_data = False
else: 
    sample_new_data = True
    
if args.nruns is None:
    NUM_runs = 500
else:
    NUM_runs = args.nruns
    
if args.nstates is None:
    NUM_init_states = 500
else:
    NUM_init_states = args.nruns

PATH = os.path.realpath("")
assumed_shape = "post + [G] * model"

"""
Given: 
    [features_log]: a list of strings denoting 
                    expressions generated for the multiplicative models 
                    (we call it `log mode` because it's implemented through 
                    taking the logarithms of the data)
    [features_linear]: a list of strings denoting 
                    expressions generated for the linear models (`linear mode`)
    [log_not_split]: a list of strings in [features_log] that denote
                    expressions involving probability variables
                    (in general, we don't want the predicates in the model trees 
                    to split on probabilities)
    [linear_not_split]: a list of strings in [features_linear] that denote
                        expressions involving probability variables
    [config]: a dictionary loaded from a .json file under `\configuration_files`
                
                [config] is not used in [prepare_tree_learner]
                but we have it so [prepare_tree_learner], [prepare_NN_sub], 
                [prepare_NN_exact] have the same type

Returns:
    [Learners]: a list of objects that take in data and output models (see class 
                [Learner] of `learners/abstract_learner.py`; we call them 
                learners). More specifically, [prepare_tree_learner(...)] 
                returns two learners of class [TreeLearner] (see 
                `learners/Tree.py`. ) They both train model trees in the 
                conventional way (using divde and conquer), with the objective 
                of finding a model tree that translates to an exact invariant. 
                One learner assumes linear models on the leaves, and 
                the another learner assumes multiplicative models on the leaves. 
"""


def prepare_tree_learner(
    features_log, features_linear, log_not_split, linear_not_split, config
) -> list:
    splittables_log = [
        i for i in range(len(features_log)) if features_log[i] not in log_not_split
    ]
    splittables_linear = [
        i
        for i in range(len(features_linear))
        if features_linear[i] not in linear_not_split
    ]
    learners = []
    learners.append(
        TreeLearner(
            features_linear,
            splittables_linear,
            False,
            True,
            max_depth=2,
            min_samples_leaf=1,
        )
    )
    learners.append(
        TreeLearner(
            features_log, splittables_log, True, True, max_depth=2, min_samples_leaf=1
        )
    )
    return learners

"""
Given: 
    Same as for [prepare_tree_learner] 
Returns:
    [Learners]: Same type as for [prepare_tree_learner], but [prepare_NN_sub(...)] 
            returns two learners that train the **neural model trees** by 
            gradient descent, with the objective of finding a model tree that 
            translates to an subinvariant. One learner assumes linear models on 
            the leaves, and the another learner assumes multiplicative models 
            on the leaf. 
"""

def prepare_NN_sub(
    features_log, features_linear, log_not_split, linear_not_split, config
) -> list:
    splittables_log = []
    splittables_linear = []
    # Uncomment if you want the neural model trees to make split 
    # (See documentations `Some choices: ... ` in `learners/NN.py`)
    # vars = [var for types in get_var_types(config).values() for var in types]
    # splittables_log = [
    #     i for i in range(len(features_log)) if features_log[i] not in log_not_split
    # ]
    # splittables_linear = [
    #     i
    #     for i in range(len(features_linear))
    #     if features_linear[i] not in linear_not_split
    # ]
    learners = []
    # train model trees as neural networks
    learners.append(
        NNLearner(
            features_log,
            splittables_log,
            assumed_shape,
            subinv=True,
            fit_logspace=False,
            fit_intercept=True,
        )
    )
    learners.append(
        NNLearner(
            features_linear,
            splittables_linear,
            assumed_shape,
            subinv=True,
            fit_logspace=True,
            fit_intercept=True,
        )
    )
    return learners

"""
Given: 
    Same as for [prepare_tree_learner] 
Returns:
[Learners]: Same type as for [prepare_NN_sub], except that the returned learners 
            have the objective to return a model tree that translates to an 
            exact invariant. 
            
[prepare_NN_exact] is not used in the paper. 
"""
# def prepare_NN_exact(
#     features_log, features_linear, log_not_split, linear_not_split, config
# ) -> list:
#     splittables_log = []
#     splittables_linear = []
#     # Uncomment if you want the neural model trees to make split 
#     # (See documentations `Some choices: ... ` in `learners/NN.py`)
#     # vars = [var for types in get_var_types(config).values() for var in types]
#     # splittables_log = [
#     #     i for i in range(len(features_log)) if features_log[i] not in log_not_split
#     # ]
#     # splittables_linear = [
#     #     i
#     #     for i in range(len(features_linear))
#     #     if features_linear[i] not in linear_not_split
#     # ]
#     learners = []
#     # train model trees as neural networks
#     learners.append(
#         NNLearner(
#             features_log,
#             splittables_log,
#             assumed_shape,
#             subinv=False,
#             fit_logspace=False,
#             fit_intercept=True,
#         )
#     )
#     learners.append(
#         NNLearner(
#             features_linear,
#             splittables_linear,
#             assumed_shape,
#             subinv=False,
#             fit_logspace=True,
#             fit_intercept=True,
#         )
#     )
#     return learners


"""
[get_config] loads the json object in [progname]'s configuration file
"""
def get_config(progname):
    with open(os.path.join(PATH, "configuration_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config

def config_select_pre(config, idx):
    config_idx = copy.deepcopy(config)
    config_idx["wp"]["pre"] = config_idx["wp"]["pre"][idx]
    return config_idx


if exact:
    print("Mode: EXACT INVARIANTS")
    prepare_learners = prepare_tree_learner
    header = ["generated_inv", "post", "sampling_time", "learning_time", "verifying_time", \
    "total_time"]
else:
    print("Mode: SUBINVARIANTS")
    prepare_learners = prepare_NN_sub
    header = ["generated_subinv", "pre", "post", "sampling_time", "learning_time", "verifying_time", \
    "total_time"] 
    
session = WolframLanguageSession()
with open(os.path.join(PATH, "program_list.txt"), "r") as f:
    prognames = f.read().split("\n")
    
results = {}
for progname in prognames:
    config = get_config(progname)
    post = config["wp"]["post"]
    guard = config["wp"]["guard"]
    pre_lst = config["wp"]["pre"]
    nsubtasks = 1 if exact else len(pre_lst)
    for idx in range(nsubtasks): # iterate through each preexpectation in the list
        config_idx = config_select_pre(config, idx)
        pre = pre_lst[idx]
        if exact:
            print("     Benchmark name: {}; Post-exp: {}".format(progname, post))
        else: 
            print("     Benchmark name: {}; Post-exp: {}; Pre-exp: {}".format(progname, post, pre))
        start_time = time.time()
        inv, st, lt, vt = cegis_one_prog(
            progname,
            config_idx,
            prepare_learners,
            NUM_runs,
            NUM_init_states,
            exact,
            sample_new_data,
            session,
            assumed_shape,
        )
        total = time.time() - start_time
        complete_inv = f"{post} + {guard} * {inv}"
        if exact: 
            print("For {}: we get {}\n".format(progname, complete_inv))
            results[progname] = [complete_inv, post,st,lt,vt,total]
        else: 
            print("For {} with preexpetation {}: we get {}\n".format\
                (progname, pre_lst[idx],complete_inv))
            results["{}_{}".format(progname, str(idx))] = [complete_inv, pre, post,st,lt,vt,total]
        # we save eagerly so even if the program crashes at a benchmark
        # we have previous results all saved
        df = pd.DataFrame.from_dict(results, orient="index", columns=header)
        df.to_csv(os.path.join(PATH, "results", "{}-{}.csv".format\
            (prognames[0], "exact" if exact else "sub")))
session.terminate()
