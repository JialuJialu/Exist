import random
from scipy.stats import bernoulli

"""
All the benchmark programs in the following have the same types and are 
structured in the same way. 
Given:
    [inpt]: dictionary that maps program variables to their initial values
            we use that to represent one initial states
    [terms]: a list of features
    [NUM_runs]: an integer indicating the number of runs
Return:
    [inpt]: a dictionary that 
        - maps features to their initial values
        - maps "expected_post" to the expected value of the terminating value of 
        the benchmark's postexpectation
        - maps "post-init" to the expected value of the terminating value of 
        the benchmark's postexpectations minus the initial value of that 
        postexpectation. (Note that postexpectation is just an expression)
"""


def template_for_new_benchmarks(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    # record the initial value of features
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    # When the loop guard is false, the loop would just exit on the initial 
    # state. So we don't have to sample. 
    if not (inpt["string representat of the loop guard"] == 0):
        return None
    # sampling
    for _ in range(NUM_runs):
        # TODO 1, initialize variables to their values in inpt
        # TODO 2, code the probabilistic loop in python
        # TODO 3, add the postexpectation to the variable [post]
        pass
    # calculating the expected value of the post
    expected_post = post / NUM_runs
    inpt["expected_post"] = expected_post
    # we substract the initial value of the postexpectation (in this case, "z")
    # because invariants are in the shape of "post + [G] * model", and we are 
    # fitting the model here.
    inpt["post-init"] = expected_post - inpt["string representation of the postexpectation"]
    return inpt

def Geo0(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    # record the initial value of features
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    # when the loop guard is false, the loop would just exit on the initial 
    # state. So we don't have to sample. 
    if not (inpt["flip"] == 0):
        return None
    # sampling
    for _ in range(NUM_runs):
        p = inpt["p1"]
        z = inpt["z"]
        flip = inpt["flip"]
        while flip == 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                flip = 1
            else:
                z = z + 1
        post += z
    # calculating the expected value of the post
    expected_post = post / NUM_runs
    inpt["expected_post"] = expected_post
    # we substract the initial value of the postexpectation (in this case, "z")
    # because invariants are in the shape of "post + [G] * model", and we are 
    # fitting the model here.
    inpt["post-init"] = expected_post - inpt["z"]
    return inpt


def Geo1(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    # record the initial value of features
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    # when the loop guard is false, the loop would just exit on the initial 
    # state. So we don't have to sample.  
    if not (inpt["flip"] == 0):
        return None
    # sampling
    for _ in range(NUM_runs):
        p1 = inpt["p1"]
        z, x = inpt["z"], inpt["x"]
        flip = inpt["flip"]
        while flip == 0:
            d = bernoulli.rvs(size=1, p=p1)[0]
            if d:
                flip = 1
            else:
                x = x * 2
                z = z + 1
        post += z
    #calculating the expected value of the post
    expected_post = post / NUM_runs
    inpt["expected_post"] = expected_post
    # we substract the initial value of the postexpectation (in this case, "z")
    # because invariants are in the shape of "post + [G] * model", and we are 
    # fitting the model here.
    inpt["post-init"] = expected_post - inpt["z"]
    return inpt


def Geo2(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["flip"] == 0):
        return None
    for _ in range(NUM_runs):
        p1 = inpt["p1"]
        z, x = inpt["z"], inpt["x"]
        flip = inpt["flip"]
        while flip == 0:
            d = bernoulli.rvs(size=1, p=p1)[0]
            if d:
                flip = 1
            else:
                x = x + 1
                z = z + 1
        post += z
    expected_post = post / NUM_runs
    inpt["expected_post"] = expected_post
    inpt["post-init"] = expected_post - inpt["z"]
    return inpt


def Geo0a(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["flip"] == 0):
        return None
    for _ in range(NUM_runs):
        p, z, x, flip = inpt["p1"], inpt["z"], inpt["x"], inpt["flip"]
        while flip == 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            x += 1
            if (x % 2) == 0:
                if d:
                    flip = 1
                else:
                    z = z + 2
        post += z
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["z"]
    return inpt


def Geo0b(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["flip"] == 0):
        return None
    for _ in range(NUM_runs):
        p, z, x, flip = inpt["p1"], inpt["z"], inpt["x"], inpt["flip"]
        while flip == 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            x += 1
            if d:
                flip = 1
            else:
                if (x % 2) == 0:
                    z = z + 2
        post += z
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["z"]
    return inpt


def Fair(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if inpt["c1"] or inpt["c2"]:
        return None
    for _ in range(NUM_runs):
        p1, p2, count = inpt["p1"], inpt["p2"], inpt["count"]
        c1, c2 = inpt["c1"], inpt["c2"]
        while not (c1 or c2):
            c1 = bernoulli.rvs(size=1, p=p1)[0]
            if c1:
                count = count + 1
            c2 = bernoulli.rvs(size=1, p=p2)[0]
            if c2:
                count = count + 1
        post += count
    post /= NUM_runs
    inpt["expected_post"] = post
    inpt["post-init"] = post - inpt["count"]
    return inpt


def Mart(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["b"] > 0):
        return None
    for _ in range(NUM_runs):
        p, rounds, b = inpt["p"], inpt["rounds"], inpt["b"]
        c = inpt["c"]
        while b > 0:
            d = bernoulli.rvs(size=1, p=p)
            if d:
                c = c + b
                b = 0
            else:
                c = c - b
                b = 2 * b
            rounds += 1
        post += rounds
    post /= NUM_runs
    inpt["expected_post"] = post
    inpt["post-init"] = post - inpt["rounds"]
    return inpt


def Gambler0(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["x"] > 0 and inpt["y"] > inpt["x"]):
        return None
    for _ in range(NUM_runs):
        z, x, y = inpt["z"], inpt["x"], inpt["y"]
        while x > 0 and y > x:
            d = bernoulli.rvs(size=1, p=0.5)[0]
            if d:
                x = x + 1
            else:
                x = x - 1
            z = z + 1
        post += z
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["z"]
    return inpt

"""
We do not know the ground truth wpe for [Gambler_general] and it's not tested. 
"""

def Gambler_general(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["x"] > 0 and inpt["y"] > inpt["x"]):
        return None
    for _ in range(NUM_runs):
        p, z, x, y = inpt["p"], inpt["z"], inpt["x"], inpt["y"]
        while x > 0 and y > x:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                x = x + 1
            else:
                x = x - 1
            z = z + 1
        post += z
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["z"]
    return inpt


def GeoAr0(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if inpt["z"] == 0:
        return None
    for _ in range(NUM_runs):
        p, z, x, y = inpt["p"], inpt["z"], inpt["x"], inpt["y"]
        while not (z == 0):
            d = bernoulli.rvs(size=1, p=p)[0]
            y = y + 1
            if d:
                z = 0
            else:
                x = x + y
        post += x
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt


def Bin(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        p, n, x = inpt["p"], inpt["n"], inpt["x"]
        while n > 0:
            d = bernoulli.rvs(p, size=1)[0]
            if d:
                x = x + 1
            n = n - 1
        post += x
    post /= NUM_runs
    post = round(post, 2)
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt


def Bin0(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        p, n, x, y = inpt["p"], inpt["n"], inpt["x"], inpt["y"]
        while n > 0:
            d = bernoulli.rvs(p, size=1)[0]
            if d:
                x = x + y
            n = n - 1
        post += x
    post /= NUM_runs
    post = round(post, 2)
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt

def Bin1(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] < inpt["M"]):
        return None
    for _ in range(NUM_runs):
        x, n, M, p = inpt["x"], inpt["n"], inpt["M"], inpt["p"]
        while n < M:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                x = x + 1
            n = n + 1
        post += x
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt


def Bin2(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        x, n, y, p = inpt["x"], inpt["n"], inpt["y"], inpt["p"]
        while n > 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                x = x + n
            else:
                x = x + y
            n = n - 1
        post += x
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt


def Sum0(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        x, n, p = inpt["x"], inpt["n"], inpt["p"]
        while n > 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                x = x + n
            n = n - 1
        post += x
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt


def DepRV(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        x, n, y = inpt["x"], inpt["n"], inpt["y"]
        while n > 0:
            d = bernoulli.rvs(size=1, p=0.5)[0]
            if d:
                x = x + 1
            else:
                y = y + 1
            n = n - 1
        post += x * y
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"] * inpt["y"]
    return inpt


"""
We do not know the ground truth wpe for [DepRV_general] when p is not 
initialized to 0.5 for this one and it's not tested. 
"""


def DepRV_general(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        x, n, y, p = inpt["x"], inpt["n"], inpt["y"], inpt["p"]
        while n > 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                x = x + 1
            else:
                y = y + 1
            n = n - 1
        post += x * y
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"] * inpt["y"]
    return inpt


def BiasDir(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["x"] - inpt["y"] == 0):
        return None
    for _ in range(NUM_runs):
        x, y, p = inpt["x"], inpt["y"], inpt["p"]
        while x - y == 0:
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
        post += x
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["x"]
    return inpt


def Prinsys(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    inpt["[x==1]"] = int(inpt["x"] == 1)
    if not (inpt["x"] == 0):
        return None
    for _ in range(NUM_runs):
        x, p1, p2 = inpt["x"], inpt["p1"], inpt["p2"]
        while x == 0:
            d1 = bernoulli.rvs(size=1, p=p1)[0]
            if d1:
                x = 0
            else:
                d2 = bernoulli.rvs(size=1, p=p2)[0]
                if d2:
                    x = -1
                else:
                    x = 1
        post += eval("x==1")
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["[x==1]"]
    return inpt




def Duel(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["c"] == 1):
        return None
    for _ in range(NUM_runs):
        turn, continuing, p1, p2 = (
            inpt["t"],
            inpt["c"],
            inpt["p1"],
            inpt["p2"],
        )
        while continuing == 1:
            if turn == 1:
                d1 = bernoulli.rvs(size=1, p=p1)[0]
                if d1:
                    continuing = 0
                else:
                    turn = 0
            else:
                d2 = bernoulli.rvs(size=1, p=p2)[0]
                if d2:
                    continuing = 0
                else:
                    turn = 1
        post += turn
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["t"]
    return inpt


def LinExp(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["n"] > 0):
        return None
    for _ in range(NUM_runs):
        n, z = inpt["n"], inpt["z"]
        while n > 0:
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
        post += z
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["z"]
    return inpt


def RevBin(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["x"] > 0):
        return None
    for _ in range(NUM_runs):
        x, z, p = inpt["x"], inpt["z"], inpt["p"]
        while x > 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                x = x - 1
            z = z + 1
        post += z
    post /= NUM_runs
    inpt["expected_post"], inpt["post-init"] = post, post - inpt["z"]
    return inpt


def Geo0c(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["flip"] == 0):
        return None
    for _ in range(NUM_runs):
        p, z, flip = inpt["p1"], inpt["z"], inpt["flip"]
        if p < 0.5:
            p = 1 - p
        while flip == 0:
            d = bernoulli.rvs(size=1, p=p)[0]
            if d:
                flip = 1
            else:
                z = z + 1
        post += z
    post /= NUM_runs
    inpt["expected_post"] = post
    inpt["post-init"] = post - inpt["z"]
    return inpt



"""
This benchmark is currently not tested because the verifier can't handle 
checking its ground truth weakest preexpectation
"""

def Nest(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["flip1"] == 0):
        return None
    for _ in range(NUM_runs):
        p1, p2, x, flip1, flip2 = (
            inpt["p1"],
            inpt["p2"],
            inpt["x"],
            inpt["flip1"],
            inpt["flip2"],
        )
        while flip1 == 0:
            d1 = bernoulli.rvs(size=1, p=p1)[0]
            if d1:
                while flip2 == 0:
                    d2 = bernoulli.rvs(size=1, p=p2)[0]
                    if d2:
                        x = x + 1
                    else:
                        flip2 = 1
                flip2 = 0
            else:
                flip1 = 1
        post += x
    post /= NUM_runs
    inpt["expected_post"] = post
    inpt["post-init"] = post - inpt["x"]
    return inpt



def Unif(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["x"] <= 10):
        return None
    for _ in range(NUM_runs):
        x, count = inpt["x"], inpt["count"]
        while x <= 10:
            x = x + random.uniform(0, 2)
            count = count + 1
        post += count
    post /= NUM_runs
    inpt["expected_post"] = post
    inpt["post-init"] = post - inpt["count"]
    return inpt


def Detm(inpt, terms, NUM_runs):
    post = 0
    terms = [ele for ele in terms if ele not in inpt.keys()]
    for ele in terms:
        inpt[ele] = round(eval(ele, {"__builtins__": None}, inpt), 2)
    if not (inpt["x"] <= 10):
        return None
    for _ in range(NUM_runs):
        x, count = inpt["x"], inpt["count"]
        while x <= 10:
            x = x + 1
            count = count + 1
        post += count
    post /= NUM_runs
    inpt["expected_post"] = post
    inpt["post-init"] = post - inpt["count"]
    return inpt

