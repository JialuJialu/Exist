from itertools import permutations
from itertools import combinations
import pdb



"""
Given:
    [var_list]: a list of integers or a list of real variables,
Returns
    [lst]: a list of expressions made up by integer or real variables 
    such that 
    1. for any x, y in [var_list], x-y is in [lst].
    2. for any ordered pair x, y in [prob], x+y is in [lst].
"""
def add_var_features(var_list):
    lst = []
    for ele in permutations(var_list, 2):
        a1, b1 = ele
        lst.append(f"({a1}-{b1})")
    for ele in combinations(var_list, 2):
        a1, b1 = ele
        lst.append(f"({a1}+{b1})")
    lst = lst + var_list
    return lst



"""
Given
    [prob]: a list of names for probability variables,
Returns
    [lst] a list of expressions made up by probability variables 
such that 
1. for any p in [prob], 1+p and 1-p are in [lst]
2. for any ordered pair p, q in [prob], p+q, and (p+q-p*q) are in [lst].
"""
def add_prob_features(prob):
    lst = []
    for ele in prob:
        lst.append(f"(1+{ele})")
        lst.append(f"(1-{ele})")
    for ele in combinations(prob, 2):
        a1, b1 = ele
        lst.append(f"({a1}+{b1})")
        lst.append(f"(({a1}+{b1})-{a1}*{b1})")
    lst = lst + prob
    return lst

"""
Given:
    [var_list]: a list of integers or a list of real variables,
Returns
    [lst]: a list of expressions made up by integer or real variables 
    such that 
    1. for any x in [var_lst], x*x in [lst]
    2. for any ordered pair x, y in [var_lst], (x*y) are in [lst].
"""
def multiply_var_features(var_list):
    lst = []
    for ele in var_list:
        lst.append(f"({ele}*{ele})")
    for ele in combinations(var_list, 2):
        a1, b1 = ele
        lst.append(f"({a1}*{b1})")

    lst = lst + var_list
    return lst

"""
Given
    [prob]: a list of names for probability variables [prob],
Returns:
    [lst]: a list of expressions made up by probability variables
such that for any ordered pair p, q in [prob], p*q is in [lst].
"""
def multiply_prob_features(prob):
    lst = []
    for ele in combinations(prob, 2):
        a1, b1 = ele
        lst.append(f"({a1}*{b1})")
    lst = lst + prob
    return lst

# """
# Given
#     [real_list]: a list of real variables,
#     [int_list]: a list of integer variables,
# Returns:
#     [lst]: a list of expressions including r * i for any r in [real_list] 
#     and i in [int_list]
# """
# def multiply_var_features_2(real_list, int_list):
#     real_list_expanded = []
#     for a1 in int_list:
#         for b1 in real_list:
#             real_list_expanded.append(f"({a1}*{b1})")
#     return real_list_expanded


"""
Given:
    [var_types] and [add_features_dic] are both dictionaries that map keys 
        "Real", "Integers", "Booleans", "Probs" to lists of variable names
    [exact]: a boolean indicating whether we learn exact invariant or subinvariant
    
Returns:
    [features]: a list of generated features. Features include variables 
    themselves, sum of variables, and subtraction of variables
"""

def generate_features_log(var_types, add_features_dic, exact):
    probs = var_types["Probs"] + add_features_dic["Probs"]
    reals = var_types["Reals"] + add_features_dic["Reals"]
    ints = var_types["Integers"] + add_features_dic["Integers"]
    bools = var_types["Booleans"] + add_features_dic["Booleans"]
    if exact:
        expanded_probs = add_prob_features(probs)
        expanded_reals_ints = add_var_features(reals + ints)
        expanded_bools = add_var_features(bools)
        return expanded_probs + expanded_reals_ints + expanded_bools, probs
    else:
        return probs + reals + ints + bools, probs


"""
Given:
    same as in [generate_features_log].
    
Returns:
    [features]: a list of generated features. Features include variables 
    themselves and products of variables. 
"""
def generate_features_linear(var_types, add_features_dic, exact):
    probs = var_types["Probs"] + add_features_dic["Probs"]
    reals = var_types["Reals"] + add_features_dic["Reals"]
    ints = var_types["Integers"] + add_features_dic["Integers"]
    bools = var_types["Booleans"] + add_features_dic["Booleans"]
    if exact:
        expanded_probs = multiply_prob_features(probs) 
        expanded_reals_ints = multiply_var_features(reals + ints)
        expanded_bools = multiply_var_features(bools)
        # more_reals = multiply_var_features_2(probs + reals, ints)
        return expanded_probs + expanded_reals_ints + expanded_bools, probs
    else:
        return probs + reals + ints + bools, probs

