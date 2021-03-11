"""
 Adapted from model_tree.py by Anson Wong (git: ankonzoid)

  With hyperparameters specified during initialization, 
 [runModelTree] fits a model tree to the data in the given input file. 
 
 While keeping the main functionalities of Anson's code, we make adjustment to 
 faciliate performing the experiments that we were interested in. 
 
 Updates highlights:
 - We changed it to a subroutine called by `../main.py`
 - Get rid of original text outputs that try to express the quality of model 
 on each data point, and enable plotting how model fits the data in a scatterplot. 
 - Support subsampling when doing Bootstrapping
"""
import os
import csv
import numpy as np
from model_tree.src.ModelTree import ModelTree
from model_tree.src.utils import load_csv_data, cross_validate


class runModelTree(object):
    def __init__(self, model, filename, leafmodelname, variable_indices, known_inv, known_model, testing_known_model, plot_only, plot_fitting, bootstrapping, sample_ratio, sign, pure_linear, max_depth, min_samples_leaf):
        self.model = model  # the leaf model
        self.leafmodelname = leafmodelname
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.variable_indices = variable_indices
        self.sign = sign
        self.pure_linear = pure_linear
        self.known_inv = known_inv
        self.known_model = known_model
        self.testing_known_model = testing_known_model
        assert not (not plot_fitting and plot_only)
        self.plot_fitting = plot_fitting
        self.plot_only = plot_only
        self.bootstrapping = bootstrapping
        self.sample_ratio = sample_ratio
        self.folder = "csv"
        self.filename = filename

    def run(self):
        data_csv_data_filename = os.path.join(
            self.folder, "{}.csv".format(self.filename))
        X_init, y_post, header = load_csv_data(
            data_csv_data_filename, mode="regr", verbose=True)
        assert len(X_init) == len(y_post)
        # Subsampling from data if self.bootstrapping is True
        if self.bootstrapping:
            b = len(X_init)
            sample_init = np.random.choice(
                np.array(b), int(b * self.sample_ratio), replace=True)
            X_init, y_post = X_init[sample_init], y_post[sample_init]

        # Build model tree
        model_tree = ModelTree(self.model, header,
                               self.pure_linear,
                               testing_known_model=self.testing_known_model,
                               known_model=self.known_model,
                               max_depth=self.max_depth,
                               min_samples_leaf=self.min_samples_leaf,
                               search_type="greedy", n_search_grid=100,
                               variable_indices=self.variable_indices,
                               sign=self.sign,
                               leafmodelname=self.leafmodelname)

        filename = "{}_{}".format(self.filename, self.leafmodelname)
        output_filename = os.path.join("output", filename)
        header = [w.replace("init_", "") for w in header]

        if self.plot_only:
            if self.testing_known_model:
                output_filename += "_known_inv"
            else:
                output_filename += "_learned"
            model_tree.plot_fitting_hist(
                output_filename, just_plot=True, feature_names=header)
            return None, None, None, None
        else:
            print("Training model tree with '{}'...".format(
                self.model.__class__.__name__))
            # Train model tree
            sign, sort = model_tree.fit(X_init, y_post, verbose=True)
            y_pred = model_tree.predict(X_init)
            sort_str = "reversed_sort" if sort else "sort"
            if self.testing_known_model:
                output_filename += "_known_inv_{}_{}".format(sign, sort_str)
            else:
                output_filename += "_learned_{}_{}".format(sign, sort_str)
            invariant_string, generate_invariant, generate_txt, learned_tree = model_tree.export_func(
                header)
            # Plot how the model fits data
            if self.plot_fitting:
                model_tree.plot_fitting_hist(
                    output_filename+"_fitting", just_plot=False, feature_names=header)

        return invariant_string, generate_invariant, generate_txt, learned_tree
