from wolframclient.language.expression import WLSymbol, WLFunction
from wolframclient.language import wlexpr, Global, wl
from wolframclient.deserializers import WXFConsumer, binary_deserialize
import itertools
from sampler import sample_by_type
import pdb
import re
import numpy as np
from wolframclient.evaluation import SecuredAuthenticationKey


class MathConsumer(WXFConsumer):
    """Implements a consumer that consumes output returned by wolfram's 
    NMaximize module, and transforms it to python equivalent"""

    def build_function(self, head, args, **kwargs):
        if head == wl.Complex and len(args) == 2:
            return complex(*args)
        elif head == wl.Alternatives:
            return args
        elif head == wl.DirectedInfinity:
            if args[0] == -1:
                return -float("inf")
            else:
                return float("inf")
        else:
            return super().build_function(head, args, **kwargs)

    def consume_symbol(self, current_token, tokens, **kwargs):
        # Extract the data present in the current_token
        if "Global" in str(current_token.data):
            return current_token.data.strip("Gloabl`")
        else:
            return super().consume_symbol(current_token, tokens, **kwargs)


class Verifier:
    """
    Given:
        [invariant]: a string representation of a candidate invariant
        [exact], [assumed_shape], [session], [task] are the same as in 
        [cegis_one_prog]
    """

    def __init__(self, invariant, exact, assumed_shape, task, session) -> None:
        invariant = invariant.replace("**", "^")
        self.inv = invariant
        self.task = task
        self.session = session
        self.exact = exact
        self.assumed_shape = assumed_shape
        self.task["guard"] = self.task["guard"].replace("(", "").replace(")", "")

    '''
    If [self.exact] is True, [compute_conditions] checks if [self.inv] is an 
    exact invariant for [self.task] and returns a list of counterexamples if 
    it is not; 
    If [self.exact] is False, [compute_conditions] checks if [self.inv] is a 
    subinvariant for [self.task] and returns a list of counterexamples if 
    it is not.  
    '''
    def compute_conditions(self, var_types):
        if self.exact:
            return self.compute_conditions_exact(var_types)
        else:
            return self.compute_conditions_sub(var_types, self.assumed_shape)

    """
    Given
    [var_types]: a dictionary that map keys "Real", "Integers", "Booleans", 
                "Probs" to lists of variable names
    Return: 
    [output_list]: a list of dictionaries that represent program states on which 
                    [self.invariant]  = [G] · wpe(P, [self.invariant] ) + [¬G] · post
                    is violated
    
    1. Since we assume that [inv] has the shape [G] · [inv'] + post, we only 
    need to check [G] · ([inv'] + post) =  [G] · wpe(P, [G] · [inv'] + post)  
              
    2. To check lhs = rhs, we ask the verifier to maximize(rhs - lhs) and 
    maximize(lhs -rhs), and they are both 0 iff lhs = rhs
    
    3. Sometimes the verifier makes error and return program states that are not 
    counter-examples, so we double check that by asserting 
    abs(lhs(s) - rhs(s)) < 1e-4 for counter-example program states [s] found. 
    We check abs(lhs(s) - rhs(s)) < 1e-4 instead of lhs = rhs to be more 
    tolerant of numerical errors.
    """

    def compute_conditions_exact(self, var_types):
        ib_list = var_types["Integers"] + var_types["Booleans"]
        # [G] · ([inv'] + post) =  [G] · wpe(P, [G] · [inv'] + post)
        whole_inv = "{}*({}) + {}".format(self.task["guard"], self.inv, self.task["post"])
        lhs = self.task["guard"] + "*({} + {})".format(self.inv, self.task["post"])
        rhs = self.task["guard"] + "*({})".format(
            wp_expression(whole_inv, self.task["loopbody"])
        )
        maxlmr, inslgr = maximize(evaluate(lhs, rhs, "-"), var_types)
        maxrml, insrgl = maximize(evaluate(rhs, lhs, "-"), var_types)
        final = maxlmr + inslgr + maxrml + insrgl
        output = dump_output(self.session, self.inv, final)
        output_list = extract_wl_output(self.session, output, ib_list)
        for i in range(len(output)):
            if output[i] == "false":
                output_list.insert(i, sample_by_type(var_types))
        print("          Wolfram alpha finds {} counterexamples\
            ".format(len(output_list)))
        not_valid = [] # a list of invalid counterexamples
        # check the counterexamples
        for cex in output_list:
            try:
                lhs = (
                    lhs.replace("[", "(")
                    .replace("]", ")")
                    .replace("^", "**")
                    .replace("&", ")*(")
                )
                rhs = (
                    rhs.replace("[", "(")
                    .replace("]", ")")
                    .replace("^", "**")
                    .replace("&", ")*(")
                )
                lhs_val = eval(lhs, {"__builtins__": None}, cex)
                rhs_val = eval(rhs, {"__builtins__": None}, cex)
                if abs(lhs_val - rhs_val) < 1e-4:
                    not_valid.append(cex)
            except (OverflowError, ZeroDivisionError) as e:
                continue
        res = [cex for cex in output_list if cex not in not_valid]
        print("          {} of those counterexamples are indeed counterexamples\:".format(len(res)))
        for cex in res:
            print("          {}".format(cex))
        return res
    """
    Given
    [var_types]: a dictionary that map keys "Real", "Integers", "Booleans", 
                "Probs" to lists of variable names
    [assumed_shape]: it is currently not used
    Return: 
    [output_list]: a list of dictionaries that represent program states on which 
                   one of the following is violated: 
                    (a) [G] * model + post <= wp(body, [G] * model + post)
                    (b) pre <= [G] * model + post
                    
    1. Since we assume that [inv] has the shape [G] · [inv'] + post, for 
    requirement (a) we only need to check 
    [G] * (model + post) <= wp(body, [G] * model + post)  
              
    2. To check lhs <= rhs, we ask the verifier maximize(lhs - rhs), and the 
    max value is sub-zero iff lhs = rhs
    
    3. The verifier makes error and return program states that are not 
    counter-examples sometimes, so we double check that by asserting 
    lhs(s) - rhs(s) < 1e-4 for counter-example program states [s] found. 
    We check lhs(s) - rhs(s) < 1e-4 instead of lhs <= rhs to be more tolerant 
    of numerical errors. 
    """
    def compute_conditions_sub(self, var_types, assumed_shape):
        whole_inv = "{}*({}) + {}".format(
            self.task["guard"], self.inv, self.task["post"]
        )
        GIplusPost = self.task["guard"] + "*({} + {})".format(
            self.inv, self.task["post"]
        )
        wpbodyI = wp_expression(whole_inv, self.task["loopbody"])
        # [G] * (model + post) <= wp(body, [G] * model + post)
        max1, ins1 = maximize(evaluate(GIplusPost, wpbodyI, "-"), var_types)
        pre = self.task["pre"]
        # pre <= [G] * model + post
        max2, ins2 = maximize(evaluate(pre, whole_inv, "-"), var_types)
        final = max1 + max2 + ins1 + ins2
        ib_list = var_types["Integers"] + var_types["Booleans"]
        output = dump_output(self.session, self.inv, final)
        output_list = extract_wl_output(self.session, output, ib_list)
        print("Wolfram alpha finds {} counterexamples: ".format(len(output_list)))
        print(output_list)
        not_valid = []
        # double check the counterexamples
        for cex in output_list:
            try:
                for (lhs, rhs) in [(GIplusPost, wpbodyI), (pre, whole_inv)]:
                    lhs = lhs.replace("[", "(").replace("]", ")").replace("^", "**")
                    rhs = rhs.replace("[", "(").replace("]", ")").replace("^", "**")
                    lhs_val = eval(lhs, {"__builtins__": None}, cex)
                    rhs_val = eval(rhs, {"__builtins__": None}, cex)
                    if lhs_val - rhs_val < 1e-4:
                        not_valid.append(cex)
            except (OverflowError, ZeroDivisionError) as e:
                continue
        res = [cex for cex in output_list if cex not in not_valid]
        print("{} of those counterexamples are indeed counterexamples:".format(len(res)))
        for cex in res:
            print("{}".format(cex))
        return res


"""
Given: 
    [inv]: a string representation of a candidate invariant
    [loopbody]: a string representation of the loop body where 
        - we use ";" to sequence commands on the top level
        - we represent `if b then c else d` by {c}[b]{d}
        - we represent any probabilistic choice as {statements}[p]{statements}
        - if corresponding to each choice there is just a single statement then 
        we represent loopbody as c1[p]c2, else for multiple statements we use 
        curly braces i.e. {c1,c2}[p]{c3,c4}
Return: 
    [inv]: a string representation of weakp(inv, loopbody)
"""


def wp_expression(inv, loopbody):
    loopbody = loopbody.split(";")[::-1]  
    for ele in loopbody:
        parts = extract_parts_exp(ele)
        # this is the case for handling assignments
        if len(parts) == 1:
            if parts[0] == "skip":
                inv = inv
            else:
                ele = ele.split("=")
                inv = inv.replace(ele[0], f"({ele[1]})")
        else:
            for i in range(len(parts)):
                parts[i] = parts[i].replace(",", ";")
            if is_equalities(parts[1]):  # this is the case for handling conditionals `if then else`
                inv1 = f"[{parts[1]}]*({wp_expression(inv,parts[0])})"
                inv2 = f"[{negate(parts[1])}]*({wp_expression(inv,parts[2])})"
                inv = f"{inv1}+{inv2}"
            else:  # this is the case for handling probabilistic branching
                inv1 = f"{parts[1]}*({wp_expression(inv,parts[0])})"
                inv2 = f"(1-{parts[1]})*({wp_expression(inv,parts[2])})"
                inv = f"{inv1}+{inv2}"
    return inv


"""
[evaluate] performs the following steps:
    1. calculate [inv] = [lhs][ope][rhs]. 
    2. Let [result] be the list of predicates appeared in [inv]. 
    [evaluate] then returns a predicate obtained by (partial) quantifier 
    elimination of [inv]:
    [req] = \BigOr_{[ele] is a true/false setting of [result]} 
                                ([inv] with [ele] substitute for its [result]),

    Example:
    Let [inv] = [lhs][ope][rhs] 
    Say [inv] is of type P1*w1-P2*w2, where P1 & P2 are predictes and w1 & w2 are 
    expressions in terms of program variables.
    Then [evaluate(lhs, rhs, ope)] returns:
    (0*w1-0*w2)&&!P1&&!P2 || (0*w1-1*w2)&&!P1&&P2 || (0*w1-1*w2)&&P1&&!P2 || (1*w1-1*w2)&&P1&&P2
"""


def evaluate(lhs: str, rhs: str, ope: str):
    req = ""
    inv = f"{lhs}{ope}({rhs})"
    inv = inv.replace("&", "]*[")
    inv = inv.replace(" ", "")
    inv = inv.replace("or", " or ")
    result = extract_predicates(inv)
    result = list(set(result))
    result = [f"({ele[1:-1]})" for ele in result]
    nums = extract_num_predicates(result)
    inv = inv.replace("[", "(")
    inv = inv.replace("]", ")")
    for ele in nums:
        inv = inv.replace(ele, str(int(eval(ele))))
    result = [ele for ele in result if ele not in nums]
    lst = list(itertools.product([0, 1], repeat=len(result)))
    for ele in lst:
        st = ""
        inv1 = inv
        for i in range(len(result)):
            if ele[i] == 0:
                neg = negate(result[i])
                st += str(neg) + "&&"
            else:
                st += result[i] + "&&"
            inv1 = inv1.replace(result[i], str(ele[i]))
        st += "(" + inv1 + ")"
        st = "(" + st + ")"
        st += "||"
        req += st
    req = req.strip("||")
    return req


"""
Given
    [out]: the piecewise expression returned by the function [evaluate]
    [var_dict]: dictionary that stores the variables of each type
Return: 
    [finalstr]: a list of string formed to input to the Wolfram Alpha solver. 
Each string encodes one clause in [out] as a maximization problem. 
So we not only want to know whether that clause in [out] is unsatisfiable, 
but also the biggest unsatisfaction of that clause. 

1. Because Wolfram Alpha's [Maximize] can fail sometimes, we double up with 
[Findinstance(the desired inequality is not satisfied)]
2. When formulating the string [svarl] to restrict the domain of each variable, 
    we have two choices. 
    a. We search the value of Boolean variables and Integer variables 
    among integers. 
    b. We only require them to be Reals and later round them into integers. 
    There is a trade-off: 
    While in principle we should require Boolean variables and Integer veriables 
    as Integers, Wolfram Alpha's NMaximize works better with reals than integers; 
    on the other hand, if we naively adopting the option (b), we may want 
    integer x > 0, and if we only require x to be reals, then the solver may 
    return x = 0 + Infinitesimal, which would get rounded to 0 instead of 1. 
    
    We adopt obtion (b) and handle the above-mentioned problem in 
    [process_boundary] function.
"""


def maximize(out: str, var_types):

    ext_st = []
    ext_st += [f"(0<={i}<=1)" for i in var_types["Booleans"]]
    # not exactly 0 < i < 1 so it does not take counterexamples that take forever to run
    ext_st += [f"(0.01<={i}<=0.99)" for i in var_types["Probs"]]
    # avoid huge integers 
    ext_st += [f"(0<={i}<=25)" for i in var_types["Reals"] + var_types["Integers"]]
    out = out.split("||")
    out = [i.replace("or", "||") for i in out]
    out = [i.replace("not", "!") for i in out]
    out = [i[1:-1] for i in out]
    out = [
        (i.split("&&")[-1], ("&&").join(i.split("&&")[:-1] + ext_st))
        if len(i.split("&&")) >= 1
        else (i.split("&&")[-1], ("&&").join(ext_st))
        for i in out
    ]
    # hack: treat all variables as reals to make it easier to optimize
    # TODO: explain why real_dom contains booleans but not integers?
    real_dom = var_types["Booleans"] + var_types["Probs"] + var_types["Reals"]
    s_varl = "Element[" + ("|").join(real_dom + var_types["Integers"]) + ",Reals]"
    s_varl = "{" + s_varl + "}"
    maximizestr = [
        f"Check[N[NMaximize[{{Simplify[{out[i][0]}],Reduce[Rationalize[{process_boundary(out[i][1], real_dom)}]]}},{s_varl}]],false,Power::infy]"
        for i in range(len(out))
    ]
    instancestr = [
        f"Check[FindInstance[{out[i][0]}>0 && {out[i][1]}, {s_varl}],false,Power::infy]"
        for i in range(len(out))
    ]
    return maximizestr, instancestr


"""
Given: 
    [session]: wolfram alpha session
    [output]: a list of result returned by the wolframalpha engine
    [ib_list]: a list of integer and boolean variables
Return:
    [counter_list]: A list of counter-examples that maximize the violation of 
clause in the disjunction [out]
"""


def extract_wl_output(session, output, ib_list):
    temp_output = []
    output = [i for i in output if i != "false"]
    session.evaluate(wlexpr("asso[r : List[__Rule]] := Association[r]"))
    for ele in output:
        if len(ele) == 0:  # filter out empty tuple found by FindInstance
            continue
        if len(ele) == 1:  # first part of tuple found by FindInstance
            temp_output.append(ele[0])
        elif (round(ele[0], 2) > 0  ):  
            # rounding is important since NMaximize maximizes objective 
            # numerically with respect to variables.
            temp_output.append(ele[1])
    temp_output = [session.evaluate(Global.asso(i)) for i in temp_output]
    for i in range(len(temp_output)):
        temp_output[i] = {
            (k_i if type(k) == tuple else k): v
            for k, v in temp_output[i].items()
            for k_i in k
        }
    final_output = []
    for ele in temp_output:
        ele = dict(ele)
        for key, value in ele.items():
            if key in ib_list:
                ele[key] = int(round(value))
        final_output.append(ele)
    return final_output






"""
Store the each input clause and its corressponding output in a file(useful for debugging).
"""

def dump_output(session, inv, final):
    output = [session.evaluate_wxf(wl.N(wlexpr(i))) for i in final]
    output = [binary_deserialize(i, consumer=MathConsumer()) for i in output]
    with open("Wolframalpha.txt", "a") as f:
        f.write(f"For invariant:  {inv}\n\n")
        for i in range(len(final)):
            f.write(f"for case {i}\n")
            f.write(f"{final[i]}\n")
            f.write(f"{output[i]}\n")
    return output


'''[negate] returns negation of a predicate'''

def negate(predicate: str):

    d = {
        "not": "",
        "==": "!=",
        "!=": "==",
        "<=": ">",
        ">=": "<",
        ">": "<=",
        "<": ">=",
    }
    for operator in d:
        if operator in predicate:
            return predicate.replace(operator, d[operator])


"""
Helper function of [evaluate]
Given:
    [inv]: a string representing a piecewise expression
Return
    [ans]: a list of all predicates present in a string
"""


def extract_predicates(inv: str):
    ans = []
    s = ""
    idx = 0
    while idx < len(inv):
        if inv[idx] == "[":
            while idx < len(inv) and inv[idx] != "]":
                s += inv[idx]
                idx += 1
            ans.append(f"{s}]")
            s = ""
        idx += 1
    return ans


"""
Helper function of [evaluate]
Given
    [res]: a list of predicates
Returns
    [l]: a list of predicates that do not contain any variable(ex: [1==0])
"""


def extract_num_predicates(res):
    l = []
    for ele in res:
        ele_copy = ele.replace("or", "")
        ele_copy = ele_copy.replace("not", "")
        c = 0
        for char in ele_copy:
            if char.isalpha():  # True if all the characters are alphabet letters
                c = 1
        if c == 0:  # c == 0 if none of the characters is a alphabetical letter
            l.append(ele)
    return l

'''Helper function for wp_expression'''

def is_equalities(check):
    equalities = ["==", "!=", ">=", "<=", ">", "<"]
    for i in equalities:
        if i in check:
            return True
    return False


"""
Helper function for wp_expression
Given a pobabilistic choice statement as exp. This function breaks it into three
parts: {left_part}[center_part]{right_part}
Given: 
    [exp]: a string denoting a pobabilistic choice statement as exp. 
Return: 
    [parts]: a list of components in [exp] as [left_part, center_part, right_part]
"""


def extract_parts_exp(exp):
    parts = []
    exp1 = ""
    i = 0
    while i < len(exp):
        if exp[i] == "{":
            exp1 = ""
            i = i + 1
            while i < len(exp) and exp[i] != "}":
                exp1 += exp[i]
                i += 1
            i += 1
            parts.append(f"{exp1}")
        elif exp[i] == "[":
            exp1 = ""
            i = i + 1
            while i < len(exp) and exp[i] != "]":
                exp1 += exp[i]
                i += 1
            i += 1
            parts.append(f"{exp1}")
        else:
            exp1 = ""
            while i < len(exp) and (exp[i] != "{" and exp[i] != "["):
                exp1 += exp[i]
                i += 1
            parts.append(exp1)
    return parts


"""
Wolfram Alpha's NMaximize always tries to find the global maximum of the 
objective function subject to the constraints given. However, if the maximum 
is achieved only infinitesimally outside the region banned by the constraints, 
rounding the maximizing assignments will give us assignments that violate 
the constraints.
Example: NMaximize{-x, x > 0} will give us {infinitesimal at x = infinitesimal}, 
and it will get rounded to {0 at x = 0}, which violate the contraint. 

[process_boundary] will handle this by processing all constraints involving 
integer variables.  
Example: It will transform the constraint {x is an integer and x > 0} into 
{x is an integer and x >= 1}. Now, we would query the verifier 
NMaximize{x, x >= 1}, and the verifier will return us {1 at x=1}. 
After rounding to integers, it stays at {1 at x = 1}, which is correct.
"""


def process_boundary(constr, real_dom):
    real_dom = sorted(real_dom, key=len, reverse=True)
    predicates = constr.split("&&")
    int_predicates = constr.split("&&")
    for var in real_dom:
        for ele in predicates:
            if var in ele and ele in int_predicates:
                int_predicates.remove(ele)
    int_predicates_processed = []
    for ele in int_predicates:
        if "<" in ele and "<=" not in ele:
            ele = ele.replace("<", "<=")
            ele = ele[:-1] + "-1)"
        elif ">" in ele and ">=" not in ele:
            ele = ele.replace(">", ">=")
            ele = ele[:-1] + "+1)"
        int_predicates_processed.append(ele)
    for ele, ele1 in zip(int_predicates, int_predicates_processed):
        constr = constr.replace(ele, ele1)
    return constr


