# Description of what each benchmark is doing

Geometric distribution

| Program id| Post | Loop Invariant | Type of Invariant| Program Behavior | Challenges| What we learn | 
| ----------| -----------    | ----------       | ------   | --- | --- | ---|
| geo_0       | z | (z >= 0) * z + (z >= 0 and flip == 0) * ((1 - p)/p)| Linear | made flip to 1 probabilistically | | Almost exactly |
| geo_0a      | z | same as ex0 | Linear | ex0 with extra variable x | x introduces noise| noise in the (flip == 0) branch  between prob1 and 1| 
| geo_0b      | z | (z >= 0) * z + (z >= 0 and flip == 0) * 2((1 - p)/p) | Linear |  ex0  but no-op in odd cycles | i introduces noise | almost exactly| 
| geo_0c      | z | unclear| Linear | geo_0 but only update z in even cycles | the current method of generating inv generates trivial data sets| unclear| 
| ex1       | count | [c1 or c2] * count + [not (c1 or c2)]* (count + (p_1 + p_2)/ (p_1 + p_2 - p_1 * p_2))  | Linear | tossing 2 fair coins in one while loop | | almost the invariant | 
|ex2       | rounds |  rounds + [b > 0] * ((1 - p)/ p)      | Linear | Martingale betting strategy| | almost the invariant |
|ex4   | z | [x − 1 >= 0 and x − y + 1 <= 0] * (z + x * y − x^2) |Non-linear | Gambler's ruin problem (fixed p = 0.5) | | |
|ex5   | x | [x >= 0 and z − 1 <= 0 and z >= 0 and y >= 0] * (x + 3 * z * y) |Non-Linear | a variant of Geometric Distribution: increase y in each iteration and add y to x in each successful iteration | | |

Binomial Distributions

| Program id| Post | Loop Invariant | Type of Invariant| Program Behavior | Challenges| What we learn | 
| ----------| -----------    | ----------       | ------   | --- | --- | ---|
|ex7 | x |[y >= 0 and n − 1 >= 0] * (x + p * n * y) | Non-linear | Standard Binomial Distribution
|ex8 | x | [y >= 0 and n − 1 >= 0] * (x + (0.5) * p * n * n + 0.5 * p * n − 3 * p * n * y) | Non-linear | Binomial Distribution that adds n if not successed and n decreases in each iteration
| ex18 | x | (x ≥ 0 and x − n ≤ 0)([n − M ≤ 0] * (x - prob*n + prob*M) + [n − M > 0] * x) | Linear | Algorithm which generates a sample x distributed binomially with parameters p and M | | the invariant except some extra split |
|ex20 | count | [n<=0] * count + [n>0] * (count + n(21/8))| Linear| the likelihood of 3-SAT under uniformly random assignment |  | Almost exactly|  
|ex20a | count | [n<=0] * count + [n>0] * (count + n(21/8))| Linear| ex20 but not recording x1, x2, x3 |  | Off| 
|ex21 | z | [x - 1 >= 0] + [x-1 < 0] * z, my guess: [x >= 1] * (z + x/(2-prob)) + (x <= 0) (z) | Linear | a program that always terminate | | it learns [x > 1] * (z + x/(2-prob)) + (x = 0)(z) + [x = 1] (z + 1), which i now think make more sense | 

Others Distributions

| Program id| Post | Loop Invariant | Type of Invariant| Program Behavior | Challenges| What we learn | 
| ----------| -----------    | ----------       | ------   | --- | --- | ---|
|ex3a | x | [flip != 0] * x +[flip = 0] * (x + prob1 / (1-prob1))  |Linear |first loop of a program with two (non-nested) loops | | almost exactly |
|ex3b | x | [flip != 0] * x + [flip = 0] * (x - prob2/(1-prob2))   |Linear |second loop of a program with two (non-nested) loops | negative coefficient, x performs badly | almost exactly|
|ex9 | x |[n > 0] * (x + p * (0.5n(n+1))) + [n <= 0] * x | Non-linear | Sum of Random Series| 
|ex10| x * y |-(1/4) * n + x * y + (1/2) * x * n + (1/2) * y * n + (1/4) * n^2 | Non-linear| The product of dependent variables
|ex5   | TODO | [x >= 0 and z − 1 <= 0 and z >= 0 and y >= 0] * (x + 3 * z * y) |Non-Linear |
|ex6 | TODO| [z >= 0 and z − 1 <= 0 and x >= 0 and y >= 0] * 10 * (5 * p * z − 2 * p * z * z + x + p * n * x) |Non- Linear |
|ex11 | x | [x == 0 and y − 1 == 0] − [x − 1 == 0 and y==0] | Linear | Simulation of fair coin with biased coin| | [y=1 and x =1] * 0.5 + [y=0][0.5x + 0.5] not sure| 
|ex11a| x | [x != y] * x + [x == y] * 1/2 | Linear | ex11 with an equivalent invariant | | (x=1)([y!=x] + [y=x]*0.5) + [x=0][x=y]*0.5, equivalent to the invariant|  
|ex12 | [x == 1] | [x = 0] * (1-prob2) + [x = 1] | Linear | probability prob1 of exiting the loop, and probability 1-prob2 of having x == 1 when exiting | negative number need to split x twice to differentiate x==0, x==1, x==-1, may be -1 doesn't matter| (x=1) + (x=0 or x = -1)(1.1 - prob) 
|ex12a | [x == 1] | [x = 0] * (1-prob2) + [x = 1] | Linear | ex12 but dummy variable x' to facilitate split on x twice | x' introduces non-linearity|  (x=1)(0.5x' + 0.5) + (x=0 )(1 - prob), equivalent to the invariant   
|ex13 |  [h >= 0 and t >= 0 and count >= 0] | [h >= 0 and t >= 0 and count >= 0] * (1) | Linear| a stochastic verstion of  hare and tortoise | it was presented in Winkens with the trivial invariant | it learns 1 
|ex14 | same as ex 22 | Omitted |
|ex15 | count | y - x, but my guess is [x + y > 10] *1 + [x + y <= 10] (10 - x - y)/(3prob) | Non-linear? | exit the loop if x + y > 10, otherwise, for prob1, y increase by 2 and x increase by a quantity sampled from uniform(0,2) | drawing from a uniform distribution besides a bernoulli distribution; it's using (x+y) in both the predicate and the linear function; how much cheating is okay? We are currently hardcoding (10-x-y)/(3prob) | off: Not resemble the invariant, lots of floating number coefficient | 
|ex16 | z(the counter for running time) | 1 + \sum_{l=0}^{w} ([x>l] * (3 + 2 * \sum_{k=0}^{w} ((#col+l)/N)^k) - (2 * [cp[i] == 0] * [x>0] * \sum_{k=0} ^{w} (#col/N)^k)| Non-linear | the runtime of a coupon collecting progroam | hard to even hardcode | SKIP for now
|ex17 | x | [x >= 0 and x − 1 <= 0 and (b − 1 == 0 or x == 0 or x − 1 == 0)] * (x) | Linear| generate biased coin with probability x from a fair coin  | the predicate involves a conjunction TO DECIDE RECORD WHAT | it learns x |
|ex17a | x | [x >= 0 and x − 1 <= 0 and (b − 1 == 0 or x == 0 or x − 1 == 0)] * (x) | Linear| ex17 but record  (b − 1 == 0 or x == 0 or x − 1 == 0) as a variable|  | same as in ex17 
|ex17b | x | unclear | Linear| ex17a but randomized the "fair" coin's probability| | (b = 1) (x=1) ((b − 1 == 0 or x == 0 or x − 1 == 0) (0.5) + 0.5) + (b=0)x  , which seems equivalent to the invariant of ex17|
|ex19 | n (number of turns it take to end the game)| [t == A and c == 0] + [t == A and C == 1] * (p1 / (p1 + p2 - p1 * p2))+  [t == B and c == 1] * ((1 - p2) / (p1 + p2 - p1 * p2))|Linear | "Duelling cowboy" A and B take turns and at their turn, they win and end the game with some probability associated to them |  TODO: check the asymmetric invariant as we changed the program to be symmetric to both players; | correct at 2 branches and wrong at 1 branch |


