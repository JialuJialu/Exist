import os
import sys
import json
import pdb
from sympy.utilities.iterables import flatten
import pandas as pd
import numpy as np
import time
from feature_generation import generate_features_linear, generate_features_log
from sampler import sample, sample_counterex

# from verifier_copy import Verifier
from verifier import Verifier
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE

PATH = os.path.realpath("")

"""
This function roughly correspond to the Algorithm Exist in Fig. 2. 
We comment the correspondence to the pseudocode below. 
Given:
    [progname]: names of the benchmark to run. It is used to load the program 
                from ex_prog.py or ex_prog_sub.py. 
    [config]: the json object in [progname]'s configuration file
    [get_learners]: a function that returns a list of objects that implements 
                    "Learner" object in learners/abstract_learner. Currently, 
                    [get_learners] is one of the following: 
                    [prepare_tree_learner], [prepare_NN_sub], and [prepare_NN_exact]
    [NUM_runs]: number of runs from each initial states
    [NUM_init]: number of initial states sampled
    [exact]: a boolean that is [True] if we want to learn exact invariants and 
             [False] if we are trying to learn subinvariants
    [sample_new]: a boolean indicating whether we load data from csv files 
                    instead of sampling new data
    [supply_feature]: a boolean indicating whether we use user-supplied features
    [session]: a wolfram alpha session
    [assumed_shape]: the assumed shape of the (sub)invariants. 
                    Currently, it is always "post + [G] * model", and we explain
                    in the paper why the assumption is without loss of generality. 
Returns: 
    a tuple (inv, sampling_time, learning_time, verifying_time) where
    [inv]: str. The string is a verified invariant when our algorithm succeeds. 
                Otherwise, the string indicating not being able to find one, 
                and include a list of candidate invariants we can find
    [sampling_time]: number of seconds spent on sampling and saving data
    [learning_time]: number of seconds spent on training the model and 
                    translating the model to invariants
    [verifying_time]: number of seconds spent on verifying the candidate 
                      invariants. 
"""


def cegis_one_prog(
    progname: str,
    config,
    get_learners,
    NUM_runs: int,
    NUM_init: int,
    exact: bool, 
    sample_new: bool,
    session,
    assumed_shape,
):
    var_types = get_var_types(config)
    task = get_task(config)
    # The following block corresponds to `feat ← getFeatures(prog, pexp)` in Fig. 2
    add_features_dic = get_user_supplied_features(config, exact)
    add_features_dic = defaultdict(list, add_features_dic)
    features_log, log_not_split = generate_features_log(
        var_types, add_features_dic, exact
    )
    features_linear, linear_non_split = generate_features_linear(
        var_types, add_features_dic, exact
    )
    features = list(set(features_log + features_linear))
    print("Exist generates the following set of features:\n     {}".format(features))
    # This block roughly corresponds to `states ← sampleStates(feat, nstates)
    #                      data ← sampleTraces(prog, pexp, feat, nruns, states)`
    # Both [sampleStates] and [sampleTraces] are implemented in [sample]. 
    filename = get_filename(progname, exact)
    before_sampling = time.time()
    print("     Start sampling {}".format(progname))
    data = sample(
        progname,
        exact,
        assumed_shape,
        filename,
        var_types,
        features,
        task,
        NUM_runs,
        NUM_init,
        sample_new,
    )
    after_sampling = time.time()
    print(
        "     It takes {} time to sample {}".format(
            after_sampling - before_sampling, progname
        )
    )
    sampling_time = after_sampling - before_sampling
    learning_time = 0
    verifying_time = 0
    learned_inv_dict = {}
    loop_count = 1
    # The following corresponds to the `while not timed out:` loop in Fig. 2
    while learning_time + verifying_time <= 600:
        # The following block roughly corresponds to `models ← learnInv(feat, data)
        #                                       candidates ← extractInv(models)`
        # [learn_cand_dic] iterates through [learner] in [learners], and 
        # [learner] performs both [learnInv] and [extractInv] on [data]
        # [cand_dic] is a dictionary that maps candidates for invariants to their 
        # loss on the data
        learners = get_learners(
            features_log, features_linear, log_not_split, linear_non_split, config
        )
        before_learning = time.time()
        cand_dic = learn_cand_dic(learners, data)
        after_learning = time.time()
        learning_time += after_learning - before_learning
        if len(cand_dic.keys()) == 0:
            s = "We cannot find a model that fit well; try supply more features"
            break
        # The for-loop [for inv in candidates: verified, cex ← verifyInv(inv, prog)] 
        # is baked in [verify_cand] here. 
        verified, inv_info = verify_cand(
            cand_dic, exact, task, session, var_types, assumed_shape
        )
        verifying_time += time.time() - after_learning
        # If any candidate in cand_dic is verified, then returns it
        if verified:
            return inv_info[0], sampling_time, learning_time, verifying_time
        else:
            counter_ex = inv_info
            learned_inv_dict.update(cand_dic)
            before_add_data = time.time()
            print("     Sampling more data for {}".format(progname))
            # The following block roughly corresponds to 
            # `states ← states ∪ cex
            #  states ← states ∪ sampleStates(feat, nstates)
            #  data ← data ∪ sampleTraces(prog, pexp, feat, nruns, states)`
            loop_count += 1
            add_data = sample_counterex(
                progname,
                exact,
                assumed_shape,
                filename,
                features,
                task,
                counter_ex,
                NUM_runs * loop_count,
            )
            more_sample = sample(
                progname,
                exact,
                assumed_shape,
                filename,
                var_types,
                features,
                task,
                NUM_runs * loop_count,
                int(NUM_init / 10),
                True,
                mode="a",
                includeheader=False,
            )
            sampling_time += time.time() - before_add_data
            data = combine_data(exact, features, data, add_data, more_sample)
    # Messages when timeout
    s = "We are not able to verify an (sub)invariant. \n We have learned the \
        following candidates for (sub)invariant: "
    for key in learned_inv_dict.keys():
        s = "{}\n{}".format(s, key)
    return s, sampling_time, learning_time, verifying_time


# --------------------Learning-related helper functions-------------------------------


"""
Learn a list of candidate invariants using each learner and returns [cand_dic], 
a dictionary mapping candidates invariants and their loss
"""


def learn_cand_dic(learners, data):
    cand_dic = {}
    for learner in learners:
        invlist = learner.learn_inv(data)
        for inv in invlist:
            cand_dic[inv[0]] = inv[1]
    return cand_dic


# --------------------Verifying-related helper functions-------------------------------
"""
The function finds top-5 least loss candidate invariants in [cand_dic] 
and checks if any of them is a correct invariant. 

Given: 
    [exact], [session], [assumed_shape] are just as in [cegis_one_prog]
    [cand_dic]: a dictionary that maps candidates for invariants to their 
                loss on the data
    [exact]: a boolean indicating whether we are learning exact invariants
    [task]: a dictionary {"guard": -- the loop guard --,
                             "loopbody": -- the loop body --,
                             "post": -- the post-expectation given -- ,
                             "pre": -- the pre-expectation given }
    [var_types]: a dictionary that maps fields "Reals", "Integers", "Booleans", 
                "Probs" to a list of variables of that type
Returns: 
    a tuple (verified, extra_info) where 
    [verified]: bool. It is [True] if one of the top-5 least loss candidate 
                invariants in [cand_dic] is correct. 
    [extra_info]: If [verified] is [True], then [extra_info] is the tuple 
                (- the correct invariants -, - its loss -); 
                Otherwise, [extra_info] is a list of counterexamples
"""


def verify_cand(cand_dic, exact: bool, task, session, var_types, assumed_shape):
    counter_examples = []
    sorted_keys = sorted(cand_dic, key=cand_dic.get, reverse=False)[:5]
    for inv in sorted_keys:
        print("Trying to verify {}".format(inv))
        inv_verifier = Verifier(inv, exact, assumed_shape, task, session)
        res = inv_verifier.compute_conditions(var_types)
        if len(res) == 0:
            return True, (inv, cand_dic[inv])
        else:
            counter_examples += res
    return False, counter_examples


# ---------------------Basic helper functions -----------------------------------


"""
[get_var_types] returns a dictionary that maps fields "Reals", "Integers", 
"Booleans", "Prob" to a list of variables of that type
"""


def get_var_types(config):
    var_config = config["Sample_Points"]
    var_config = defaultdict(list, var_config)
    return var_config

"""
[get_task] returns [task], which given as input to [verify_cand]. 
"""
def get_task(config):
    return config["wp"]

"""
[get_user_supplied_features] finds user-supplied features if there exists any. 
It returns a dictionary that associates types ("Reals", "Probs", "Integers" and 
"Booleans") to user-supplied features of these types. 
"""
def get_user_supplied_features(config, exact):
    if exact:
        try:
            add_features_dic = config["additional features for exact"]
        except KeyError:
            add_features_dic = {}
    else: 
        try:
            add_features_dic = config["additional features for sub"]
        except KeyError:
            add_features_dic = {}
    return add_features_dic

"""
[get_filename] returns filename(s) where we store sampled data into
"""
def get_filename(progname, exact: bool):
    if exact:
        filename = os.path.join(PATH, "generated_data", progname + "_expected_post.csv")
        return filename
    else:
        filename_weight = os.path.join(PATH, "generated_data", progname + "_weight.npy")
        filename_cur = os.path.join(PATH, "generated_data", progname + "_G_init.csv")
        filename_next = os.path.join(PATH, "generated_data", progname + "_G_next.csv")
        return (filename_weight, filename_cur, filename_next)

"""
Roughly, [combine_data] unions [data], [add_data] and [more_sample]. 
The for-loops of the second branch of [if-then-else] are there to make sure 
that [data], [add_data] and [more_sample] have the same columns
"""
def combine_data(exact, features, data, add_data, more_sample):
    if exact:
        return pd.concat([data, add_data, more_sample], axis=0)
    else:
        exist_add_data = add_data[0].shape[0] > 0
        exist_more_sample = more_sample[0].shape[0] > 0
        for feature in list(data[0].columns):
            if feature in features or feature == "1":
                if feature not in list(add_data[0].columns) and exist_add_data:
                    add_data[0][feature] = add_data[0].eval(feature, engine="python")
                if feature not in list(more_sample[0].columns) and exist_more_sample:
                    more_sample[0][feature] = more_sample[0].eval(
                        feature, engine="python"
                    )
        for feature in list(data[1].columns):
            if feature in features or feature == "1":
                if feature not in list(add_data[1].columns) and exist_add_data:
                    add_data[1][feature] = add_data[1].eval(feature, engine="python")
                if feature not in list(more_sample[1].columns) and exist_more_sample:
                    more_sample[1][feature] = more_sample[1].eval(
                        feature, engine="python"
                    )
        data0 = pd.concat([data[0], add_data[0], more_sample[0]], axis=0)
        data1 = pd.concat([data[1], add_data[1], more_sample[1]], axis=0)
        weight = np.concatenate((data[2], add_data[2], more_sample[2]), axis=0)
        return (data0, data1, weight)
