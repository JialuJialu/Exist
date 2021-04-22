# Exist
This is a research prototype for the data-driven approach to synthesize invariants 
for probabilistic programs, as described in more details in TODO. 

# Repo structure
- Directory `model_tree`: it is adapted from code in [this repo](https://github.com/ankonzoid/LearningX/tree/master/advanced_ML/model_tree), which implements a model tree learning algorithm. 
- Directory `src` includes instrumented example programs (`ex_prog.py`), the class for managing the data from program traces (`data_utils.py`), and manual verifications of candidate invariants we synthesize. 
- `main.py` loads example programs from `src/ex_prog.py`, gets their traces (stored directory `csv`), synthesizes candidates invariants (stored in `invariants` as text and in `pickle` as trees). 
- `rounding.py` is for rounding the candidates invariants we get. 

# What to install
* numpy
* sklearn
* cvxpy

# How to Run
* To run: uncomment example programs you want to run in `main.py` and run `python3 main.py` in the command line. 
* To add example programs: 
add programs to `src/ex_prog.py` with code instrumentation following guideline there; 
then add them to the dictionary `progs` in `main.py`
 
