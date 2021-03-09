import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from collections import defaultdict
import pdb
import sys

# maxint = sys.maxsize
maxint = 10000


def compareLst(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False
    return True


class RecordStat(object):
    '''
    - NUM_ITR: the program can iterate different number of iterations in different runs,
    NUM_ITR limits the maximum number of iteration that we recorded into
    self.pred_all runs
    - preds_str: a string describing names of fields recorded
    - wp: weakest pre-expectation transformer on string symbolic expressions (not used)
    ninput: the number of probabilities input to the program
    - qual_feature_indices: indices of qualitative features, we use it to guide model
    tree to split on these features (when we set KNOW_QUAL = True in main)
    - self.post_lst: a list tracking the evaluated value of post after each run of
    the program
    '''

    def __init__(self, preds_str, known_inv, known_model, ninput, qual_feature_indices):
        self.pred_all_runs = []
        self.predvec = []
        self.preds_str = [preds_str.strip()
                          for preds_str in preds_str.split(',')]
        self.post_lst = []
        self.ninput = ninput
        self.qual_feature_indices = qual_feature_indices
        self.known_inv = known_inv
        self.known_model = known_model

    def record_predicate(self, predvec):
        self.predvec = predvec

    def record_predicate_end(self, post):
        self.post_lst.append(post)
        self.pred_all_runs.append(np.array(self.predvec))


'''
AggregateData is one level higher than RecordState.
We create one Aggregate for one program
After running one program a desired number of times with one probability initialization,
information in RecordState pours into AggregateDate.
'''


class AggregateData(object):
    '''
    - self.raw_dict has the format
        {probability_initialization:
            [
                [ entry i j for iteration j ]
                for run i
            ]
        }
    where entry i j = [field 1' value before iteration j of run i, ...,
                      field n' value before iteration j of run i,
                      post after all all iterations of run i]
    -- self.qual_feature_indices, self.preds_str, self.ninput are taken from
    RecordState
    '''

    def __init__(self):
        self.raw_dict = defaultdict(list)
        self.qual_feature_indices = None
        self.preds_str = None
        self.ninput = None
        self.known_inv = None
        self.known_model = None

    def end_sampling_runs(self, hists, prob):  # post = z_expt in our example
        # create raw_dict
        preds = np.array(hists.pred_all_runs)
        try:
            nruns, nfields = preds.shape
        except ValueError:
            pdb.set_trace()
        post_array = np.array(
            [[hists.post_lst[k]] for k in range(nruns)])
        runs_and_post = np.concatenate((preds, post_array), axis=1)
        self.raw_dict[prob] += list(runs_and_post)

        if self.ninput is None:
            self.ninput = hists.ninput
            self.preds_str = hists.preds_str
            self.qual_feature_indices = [
                i+self.ninput for i in hists.qual_feature_indices]
            self.known_model = hists.known_model
            self.known_inv = hists.known_inv
        else:
            assert (self.ninput == hists.ninput)
            assert (self.preds_str == hists.preds_str)

    def to_df(self):
        ninput = self.ninput

        data = self.raw_dict
        dataarray = [list(inpt)[:ninput]+list(run)
                     for inpt in data.keys() for run in data[inpt]]

        datadict = {}
        fields = ["prob{}".format(i+1)
                  for i in range(ninput)] + self.preds_str + ["post"]
        for i in range(len(fields)):
            datadict[fields[i]] = [row[i] for row in dataarray]

        df = pd.DataFrame.from_dict(datadict)
        # post-process df3: average post
        df = df.groupby(fields[: -1]).agg({fields[-1]: ['mean']})
        df.columns = ['post']
        df = df.reset_index()
        df = df.drop(
            columns=[cur for cur in fields if cur.startswith("cur")])  # ad lib

        return df
