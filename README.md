# Data-Driven Invariant Learning for Probabilistic Programs

This repository hosts the artifact of our CAV 2022 paper [Data-Driven Invariant Learning for Probabilistic Programs](https://baojia.lu/assets/preprints/EXIST.pdf).

## What does Exist do?
Exist is a method for learning invariants for probabilistic programs (Section 3 of the paper). Exist executes the program multiple times on a set of input states, and then uses machine learning algorithms to learn models encoding possible invariants. A CEGIS-like loop is used to iteratively expand the set of input states given counterexamples to the invariant conditions. 
Here we present a concrete implementation of Exist tailored for handling two problems: learning exact invariants (Section 4), and learning sub-invariants (Section 5). Our method for exact invariants learns a model tree [1], a generalization of binary decision trees to regression. The constraints for sub-invariants are more difficult to encode as a regression problem, and our method learns a neural model tree [2] with a custom loss function. While the models differ, both algorithms leverage off-the-shelf learning algorithms. 


## Structure of this Artifact

Our implementation roughly follows the pseudocode presented in Fig. 2 of our [paper](https://arxiv.org/abs/2106.05421). 
We document the correspondence between our python code and
individual lines of the pseudocode in the following places: 
- `cegis.py` (above and inline function `cegis_one_prog`); 
- `sampler.py` (above function `sampler_by_type` and `sample`);
- `learners/NN.py` (above function `makeModelTree`) and `learners/Tree.py` (above function `extract_invariant`) 

## Getting Started
We present three methods to set up the environment to run our code. The most recommended method is through Anaconda, but we also provide two options through Docker. The second method pulls a publicly released Docker image that takes about 7.2 GB space, while the third uses a Dockerfile to build a docker image and would take less space. 
Currently, our Docker images only support machines with Intel chips.

Our tool uses Wolfram Engine, so no matter which method you choose, the first step is to set up the wolfram engine. 
0. Create a Wolfram login by visiting: https://account.wolfram.com/login/create

### First method: Install dependencies through Anaconda
1.  Set up and activate Wolfram Engine on your machine as instructed in this [link](https://support.wolfram.com/45743). 
2. Make sure that you have anaconda installed. 
3. At the root directory of artifact, create a conda environment from the exist.yml file by running `conda env create -f exist_conda.yml`
4. Activate the new environment by running  `conda activate exist`
5. Verify that the new environment was installed correctly by running  `conda env list` and checking that `exist` is in the list

### Second Method: Pull publicly released Docker Image 

1. Make sure that you have Docker Engine and Docker Desktop installed.
Launch the Docker Desktop to activate the docker daemon. 
2. Open terminal and enter the directory `Exist` through  `cd Exist`
3. Pull the docker image using the command:  `docker pull nitesh2008/inv2022:login`
(Note: “nitesh2008/inv2022:login” is publicly released image on docker hub)
4. Execute the following command to run a docker container with the pulled image: 
`docker run --name exist_artifact -it -v ${PWD}:/project nitesh2008/inv2022:login /bin/bash`. 
This command would create a container named `exist_artifact` and mount the base directory `Exist` to the directory `project` in the container. 
5. Type command “wolframscript” on terminal. It will ask for your Wolfram login credentials. Type in your Wolfram login credentials. 
6. Enter the the directory `project`  by typing `cd project` 

### Third Method: Build Docker Image through dockerfile
1. Download the Wolfram Engine suitable to your operating system from [here](https://www.wolfram.com/engine/). Move the downloaded installer file inside the directory `Exist`. 
2. Open terminal in the directory `Exist` and type command `docker build -t inv2022:01 .`. This will build an image `inv2022:01` locally with all dependencies required to execute the code. 
3. Type `. local_env/bin/activate` to activate the environment.
4. Execute the following command to run a docker container with the build image: `docker run --name pyinv -it -v ${PWD}:/project inv2022:01`
5. Type `cd project/` and execute the wolfram engine installer, during installation follow the instructions carefully.
6. Finally, login to wolframscript using the login credentials you registered in step 0. 

## Evaluation Instructions

### Test the installed environment:
Run the command `python main.py -test` and check if it exits normally. 

### Test the tool on benchmarks:
To learn exact invariants: 
0. Run `python main.py`
1. If the tool exits unexpectedly and you want to restart
			the experiment without repeating benchmarks you have already finished, you
			can remove those benchmarks from `program_list.txt`.
2. After the script finishes, check the generated invariants and the running time 
in the most recent `*-exact.csv` file under the folder `results`. 

To learn sub-invariants: 
0. If you removed benchmarks from 
`program_list.txt` in the previous step, now you can copy the list of benchmarks 
from `program_list_all.txt` and paste them in `program_list.txt`. 
1. Run `python main.py -sub yes`
2. After the script finishes, check the generated invariants and the running time 
in the most recent `*-sub.csv` file under the folder `results`. 

### Remarks:
#### Other arguments to the command
Besides running `python main.py -sub yes`, you can also run the program with the following
arguments
- `python main.py -cached yes` would use saved data in `\generated_data` instead of sample new data for training new models. 
- `python main.py -nruns 1000` would run the benchmark from each initial state for 1000 time instead of the default 500 times. You can also input other integers in place of 1000. 
- `python main.py -nstates 1000` would sample 1000 (instead of 500 by default) initial states for each benchmark. You can also input other integers in place of 1000. 

#### Comparison to results in our paper
- For exact invariants, the learned invariants recorded in the `*-exact.csv` file 
should be the equivalent to (though might not be exactly the same as) the
learned invariants in Table 1 of the paper. Due to the probabilistic nature of
our algorithm, the running time differ in different trials, and `Exist` may fail to find
invariants for some benchmarks that we reported successful. If you are curious,
you can try rerun failed invariants  with `python main.py -nstates 1000 -nruns
1000` to see if more data helps. 
- Subinvariants of a given task are not unique, so the subinvariants recorded
in the `*-sub.csv` file may be different from the learned subinvariants in
Table 2 of the paper. When they differ, you can check whether the learned
subinvariant is bigger than the given preexpectation and smaller than the exact
invariant we generated above. An expression is a valid subinvariant if the
answers to both questions are yes. While the returned invariants are
''verified'' by the verifier, they may still be wrong occasionally because
limitations of Wolfram Alpha.


#### Wolfram alpha disconnected
We use Python’s wolframclient module to create a connected session with the wolfram engine kernel, thereafter the session interacts with the Wolfram Engine executable. Sometimes wolframclient fails to create a session even after multiple tries or the session gets disconnected in the middle of an execution, and you may get errors that look like the following. (Check [this google doc](https://docs.google.com/document/d/1raf8veEzBY87vRD16tjRLFlni13tpx-5y2bI1YQwdDM/edit#heading=h.jh6zo3yov47z). ) 
The best way we found to address this problem is to reboot the computer. Although rebooting is a bit annoying, this problem does not happen frequently. 


## How to extend Exist

### Add benchmarks
- Encode the program in `ex_prog.py` and `ex_prog_sub.py` following the structure laid out in 
`template_for_new_benchmarks`. 
- Add configurations of the program to `{progname}.json` following `template.json`. 
  - The field `Sample_Points` should include all program variables in `progname`. 
			 Some of its subfields (`Probs`, `Integers`, `Booleans`, `Reals`) may be omitted 
if there are no program variables of that type. 
  - The field `wp` and all its subfields are required. 
  - Optional: provide user-supplied features `additional features for sub` and 
		`additional features for sub` 

**Our code follows modular design, so you can implement alternative sampler, learner,
verifier and cegis procedure easily.**

#### Change the sampler
- Make sure that you implement `sample` and `sample_counterex` to have the same type as 
ours in `sample.py`; or, implement a sampler in any way you like, and update the usage of sampler in
`cegis.py` and the postprocessing of sampled data in `/learners/utils.py`.

#### Add learners
- Add a python class `New_learner` that inherits the `Learner` class in `learners/abstract_learner.py`. 
- Define a function `prepare_new_learner` in `main.py` to instantiate instances of `New_learner`. 
- Specify when `prepare_learners` should be `prepare_new_learners` in `main.py`. 

#### Change the verifier 
- Make sure that you implement a class `Verifier` with functions `__init__` and `compute_conditions` having the same type as ours in `verifier.py`. 
; or, implement a verifier in any way you like, and update the usage of verifier in
the function `verify_cand` in `cegis.py`.


## References

Quinlan, J.R.: Learning with continuous classes. In: Australian Joint Conference on Artificial Intelligence (AI), Hobart, Tasmania, vol. 92, pp. 343– 348 (1992)

Yang, Y., Morillo, I.G., Hospedales, T.M.: Deep neural decision trees. CoRR abs/1806.06988 (2018), URL http://arxiv.org/abs/1806.06988


