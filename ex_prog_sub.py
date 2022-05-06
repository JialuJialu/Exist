import random
from scipy.stats import bernoulli
from scipy.stats.morestats import WilcoxonResult


'''
Given:
    [scope]: a python environment obtained by calling local()
    [task]: a dictionary {"guard": -- the loop guard --,
                             "loopbody": -- the loop body --,
                             "post": -- the post-expectation given -- ,
                             "pre": -- the pre-expectation given }
Return: 
    a boolean specifying whether the loop guard specified in [task] is True in 
    [scope]
'''
def G(scope, task):
    guard = task["guard"].replace("[", "(").replace("]", ")").replace("&", "*")
    guard_bool = eval(guard, scope)
    assert (type(guard_bool) == bool) or (guard_bool == 0) or (guard_bool) == 1
    return bool(guard_bool)

'''
Given:
    [state]: a dictionary that maps features to values
    [scope], [task] are the same as in [G(scope, task)]
Return: 
    an updated state with additional keys ["pre"], ["post"], and ["G"] and map 
    them respectively to the values of preexpectation, postexpectation and the 
    loop guard in the [scope]
'''
def get_post_and_pre(state, scope, task):
    state["post"] = eval(task["post"].replace("[", "(").replace("]", ")"), scope)
    state["pre"] = eval(task["pre"].replace("[", "(").replace("]", ")"), scope)
    state["G"] = G(scope, task)
    return state

'''
Given:
    [state], [scope], [task] are the same as in [G(scope, task)]
Return: 
    update [state]'s binding to keys ["pre"], ["post"], and ["G"] to the values 
    of preexpectation, postexpectation and the loop guard in the [scope], 
    then return the updated dictionary. 
'''
def update_pre_post_state(scope, inpt, task):
    keys = [
        key
        for key in inpt.keys()
        if not ((key == "pre") or (key == "post") or (key == "G"))
    ]
    post_state = {key: eval(key, scope) for key in keys}
    post_state = get_post_and_pre(post_state, scope, task)
    return post_state


"""
Given:
    [inpt]: dictionary that maps program variables to their initial values
            we use that to represent one initial states
    [task]: the same as in [get_post_and_pre]
    [terms]: a list of features
    [NUM_runs]: an integer indicating the number of runs
    [assumption]: a string indicating the assumed shape of the invariants. 
                    Currently, the parameter is unused because we always assume 
                    the shape to be "post + [G] * model"
Return ([G], [data])
    [G]: A boolean that indicates whether the loop guard is satisfied on the 
        initial state [inpt]
    [data]: 
        If [G] is false, then [data] is an empty list []
        If [G] true, then [data] is [(inpt', post_states_lst, 1) # weight = 1]
        [inpt'] is the dictionary [inpt] with ["pre"], ["post"], and ["G"] 
        associated with the preexpectation, postexpectation and the loop guard 
        evaluated on the [inpt] state. [post_states_lst] is the set of post 
        states sampled from running the loop body from [inpt]
"""

def template_for_new_benchmarks(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    # if loop guard is false, then we just return 
    if not G(inpt, task):
        return False, inpt
    # sampling
    post_states_lst = []
    for _ in range(NUM_runs):
        # TODO 1: initialize variables to their values in inpt
        # TODO 2: code the probabilistic loop in python
        # TODO 3: add the postexpectation to the variable [post]
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Geo0(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    # if loop guard is false, then we just return 
    if not G(inpt, task):
        return False, inpt
    # sampling
    post_states_lst = []
    for _ in range(NUM_runs):
        p1 = inpt["p1"]
        z = inpt["z"]
        flip = inpt["flip"]
        # start of the loop body
        d = bernoulli.rvs(size=1, p=p1)[0]
        if d:
            flip = 1
        else:
            z = z + 1
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Geo1(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    # sampling
    post_states_lst = []
    for _ in range(NUM_runs):
        p1 = inpt["p1"]
        z, x = inpt["z"], inpt["x"]
        flip = inpt["flip"]
        # start of the loop body
        d = bernoulli.rvs(size=1, p=p1)[0]
        if d:
            flip = 1
        else:
            x = x * 2
            z = z + 1
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Geo2(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    # sampling
    post_states_lst = []
    for _ in range(NUM_runs):
        p1 = inpt["p1"]
        z, x = inpt["z"], inpt["x"]
        flip = inpt["flip"]
        # start of the loop body
        d = bernoulli.rvs(size=1, p=p1)[0]
        if d:
            flip = 1
        else:
            x = x + 1
            z = z + 1
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1



def Fair(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    # sampling
    post_states_lst = []
    for _ in range(NUM_runs):
        p1, p2, count = inpt["p1"], inpt["p2"], inpt["count"]
        c1, c2 = inpt["c1"], inpt["c2"]
        # start the loop body
        c1 = bernoulli.rvs(size=1, p=p1)[0]
        if c1:
            count = count + 1
        c2 = bernoulli.rvs(size=1, p=p2)[0]
        if c2:
            count = count + 1
        # end the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1



def Mart(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    # sampling
    post_states_lst = []
    for _ in range(NUM_runs):
        p, rounds, b = inpt["p"], inpt["rounds"], inpt["b"]
        c = inpt["c"]
        # start the loop body
        d = bernoulli.rvs(size=1, p=p)
        if d:
            c = c + b
            b = 0
        else:
            c = c - b
            b = 2 * b
        rounds += 1
        # end the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Gambler0(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        z, x, y = inpt["z"], inpt["x"], inpt["y"]
        # start the loop body
        d = bernoulli.rvs(size=1, p=0.5)[0]
        if d:
            x = x + 1
        else:
            x = x - 1
        z = z + 1
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1



def GeoAr0(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        p, z, x, y = inpt["p"], inpt["z"], inpt["x"], inpt["y"]
        d = bernoulli.rvs(size=1, p=p)[0]
        y = y + 1
        if d:
            z = 0
        else:
            x = x + y
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Bin0(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        p, n, x, y = inpt["p"], inpt["n"], inpt["x"], inpt["y"]
        # start of the loop body
        d = bernoulli.rvs(p, size=1)[0]
        if d:
            x = x + y
        n = n - 1
        # end of the loop body
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Bin2(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, n, y, p = inpt["x"], inpt["n"], inpt["y"], inpt["p"]
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            x = x + n
        else:
            x = x + y
        n = n - 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Sum0(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, n, p = inpt["x"], inpt["n"], inpt["p"]
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            x = x + n
        n = n - 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def DepRV(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, n, y = inpt["x"], inpt["n"], inpt["y"]
        d = bernoulli.rvs(size=1, p=0.5)[0]
        if d:
            x = x + 1
        else:
            y = y + 1
        n = n - 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1



def BiasDir(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, y, p = inpt["x"], inpt["y"], inpt["p"]
        d1 = bernoulli.rvs(size=1, p=p)[0]
        if d1:
            x = 1
        else:
            x = 0
        d2 = bernoulli.rvs(size=1, p=p)[0]
        if d2:
            y = 1
        else:
            y = 0
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1



def Prinsys(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, p1, p2 = inpt["x"], inpt["p1"], inpt["p2"]
        d1 = bernoulli.rvs(size=1, p=p1)[0]
        if d1:
            x = 0
        else:
            d2 = bernoulli.rvs(size=1, p=p2)[0]
            if d2:
                x = -1
            else:
                x = 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Bin1(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, n, M, p = inpt["x"], inpt["n"], inpt["M"], inpt["p"]
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            x = x + 1
        n = n + 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Duel(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        t, c, p1, p2 = (
            inpt["t"],
            inpt["c"],
            inpt["p1"],
            inpt["p2"],
        )
        if t == 1:
            d1 = bernoulli.rvs(size=1, p=p1)[0]
            if d1:
                c = 0
            else:
                t = 0
        else:
            d2 = bernoulli.rvs(size=1, p=p2)[0]
            if d2:
                c = 0
            else:
                t = 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def LinExp(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        n, z = inpt["n"], inpt["z"]
        x1 = bernoulli.rvs(size=1, p=0.5)[0]
        x2 = bernoulli.rvs(size=1, p=0.5)[0]
        x3 = bernoulli.rvs(size=1, p=0.5)[0]
        c1 = x1 + x2 + x3  # c1 = "x1 or x2 or x3"
        if c1 >= 1:
            c1 = 1
        c2 = x1 + (1 - x2) + x3  # c2 = "not x1 or x2 or x3"
        if c2 >= 1:
            c2 = 1
        c3 = (1 - x1) + x2 + x3  # c3 = "not x2 or x1 or x3"
        if c3 >= 1:
            c3 = 1
        n = n - 1
        z += c1 + c2 + c3
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def RevBin(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        x, z, p = inpt["x"], inpt["z"], inpt["p"]
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            x = x - 1
        z = z + 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Geo0c(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    # sampling
    for _ in range(NUM_runs):
        p, z, flip = inpt["p"], inpt["z"], inpt["flip"]
        d = bernoulli.rvs(size=1, p=p)[0]
        if d:
            flip = 1
        else:
            z = z + 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1



def Unif(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    
    # sampling
    for _ in range(NUM_runs):
        x, count = inpt["x"], inpt["count"]
        x = x + random.uniform(0, 2)
        count = count + 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1


def Detm(inpt, task, NUM_runs, assumption):
    inpt = get_post_and_pre(inpt, inpt, task)
    if not G(inpt, task):
        return False, inpt
    post_states_lst = []
    
    # sampling
    for _ in range(NUM_runs):
        x, count = inpt["x"], inpt["count"]
        x = x + 1
        count = count + 1
        scope = locals()
        post_state = update_pre_post_state(scope, inpt, task)
        post_states_lst.append(post_state)
    return True, (inpt, post_states_lst, 1) # weight = 1