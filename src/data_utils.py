import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from collections import defaultdict
import pdb
import sys

'''
RecordStat is an object that records useful info of an example program and its traces, 
and also supports tranforming them into a pandas.dataframe
'''


class RecordStat(object):
    '''
    - preds_str: a string describing the names of features recorded
    - known_inv: a known invariant in text, just for printing
    - known_model: a known invariant in the form of model tree, used when TESTING_KNOWN_MODEL
    - ninput: the number of probabilities used by the program
    - variable_indices: indices of features that can vary in different iterations.
    When fitting the model tree, we only split on the variables.
    '''

    def __init__(self, preds_str, known_inv, known_model, ninput, variable_indices, types):
        self.pred_all_runs = []
        self.predvec = []
        self.preds_str = [preds_str.strip()
                          for preds_str in preds_str.split(',')]
        # self.post_lst = []
        self.ninput = ninput
        self.variable_indices = [i+self.ninput for i in variable_indices]
        self.known_inv = known_inv
        self.known_model = known_model
        self.data_dict = defaultdict(list)
        self.types = types

    def record_predicate(self, predvec):
        self.predvec = predvec

    def record_predicate_end(self, post):
        # self.post_lst.append(post)
        self.pred_all_runs.append(np.array(self.predvec + [post]))

    def end_sampling_runs(self, prob):
        self.data_dict[prob] += self.pred_all_runs
        # clear up pred_all_runs for the next prob
        self.pred_all_runs = []
        self.post_lst = []

    def to_df(self):
        ninput = self.ninput

        data = self.data_dict
        dataarray = [list(inpt)[:ninput]+list(run)
                     for inpt in data.keys() for run in data[inpt]]
        datadict = {}
        fields = ["prob{}".format(i+1)
                  for i in range(ninput)] + self.preds_str + ["post"]
        for i in range(len(fields)):
            datadict[fields[i]] = [row[i] for row in dataarray]

        df = pd.DataFrame.from_dict(datadict)
        df = df.groupby(fields[: -1]).agg({fields[-1]: ['mean']})
        df.columns = ['post']
        df = df.reset_index()
        df = df.drop(
            columns=[cur for cur in fields if cur.startswith("cur")])  # ad lib

        return df
