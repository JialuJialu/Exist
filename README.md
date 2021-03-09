# Exist
=======
# Credits
The directory `model_tree` is adapted from [this repo](https://github.com/ankonzoid/LearningX/tree/master/advanced_ML/model_tree), which implements a model tree learning algorithm. 

# What to install
* numpy
* sklearn
* cvxpy

# How to Run
* To run: uncomment example programs you want to run in `main.py` and run `python3 main.py` in the `preexp` directory. 
* To add example programs: 
add programs with instrumented code to `src/ex_prog.py` and add them to the dictionary `progs` in 
`main.py`

## Useful parameters
* NUM_RUNS: number of runs that we sample before averaging for each initial setting
* UPDATE_CSV = True: False if just want to fit learning and don't want collect data again
* TESTING_KNOWN_MODEL: if described the known invariant in the `ex_prog`, this would tell how that model fits.
* FIT_intercept: used for the linear regression in model tree leaf
* USE_INV: whether we use the subcondition `I <= wp(body, I)` to generate data
* BAGGING: If we do bagging 
* Sample_ratio: the size of each bag/subsample compared to original sample size
* nBAG: number of bags

## How to tweak existing examples and make new example: 
If you want to record some extra expressions to test how robust the tool is, 
you add the extra expressions into `VarInfo`
(include constant expressions in the first input list to `VarInfo`, and include the variable expressions in the second input), 
and add them into each `record_predicate_in_loop`, which takes a list of expressions in the order of constant expressions, 
variable expressions evaluated at the initial state, variable expressions evaluated at the current program point. 
See more documentations on the top of the `ex_prog.py`. 

Please make a new example while tweaking, and add the new example to the `progs` in `main.py`. 

## How to run profiling 
First,  uncomment example programs you want to run in `main.py` and run `python3 -m cProfile -o profiling/FILENAME.pstats main.py` 
to execute the program while profiling. 

Then, run `../gprof2dot/gprof2dot.py -f pstats profiling/FILENAME.pstats | dot -Tsvg -o profiling/FILENAME.svg` to generate a diagram representing the profiling results. 

# TODO 
- test the code under different settings
- examine and run all experiments again
- update the chart in Benchmarks, maybe using an excel file instead if that is easier

