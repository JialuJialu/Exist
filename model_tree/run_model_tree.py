"""

 model_tree.py  (author: Anson Wong / git: ankonzoid)

 Given a classification/regression model, this code builds its model tree.

"""
import os
import pickle
import csv
import numpy as np
from model_tree.src.ModelTree_inv import ModelTreeInv
from model_tree.src.utils import load_csv_data, cross_validate, load_csv_data_no_post, load_csv_data_with_pre


class runModelTree(object):
    def __init__(self, model, filename, norm, qual_feature_indices, known_inv, known_model, testing_known_model, plot_only, plot_fitting, bag, sample_ratio, sign, fit_used, max_depth, min_samples_leaf, mode="regr"):

        self.model = model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.qual_feature_indices = qual_feature_indices
        self.sign = sign
        self.fit_used = fit_used
        self.mode = mode
        self.known_inv = known_inv
        self.known_model = known_model
        assert not (plot_fitting and plot_only)
        self.plot_fitting = plot_fitting
        self.plot_only = plot_only
        self.testing_known_model = testing_known_model
        self.folder = "csv"
        self.norm = norm
        self.bag = bag
        self.sample_ratio = sample_ratio
        self.filename = filename

    def run(self):
        # ====================
        # Settings
        # ====================
        mode = self.mode  # "clf" / "regr"
        save_model_tree = False  # save model tree?
        cross_validation = True  # cross-validate model tree?

        data_csv_data_filename = os.path.join(
            self.folder, "{}.csv".format(self.filename))
        X_init, y_post, header_init = load_csv_data(
            data_csv_data_filename, mode=mode, verbose=True)
        assert len(X_init) == len(y_post)
        if self.bag:
            b = len(X_init)
            sample_init = np.random.choice(
                np.array(b), int(b * self.sample_ratio), replace=True)
            X_init, y_post = X_init[sample_init], y_post[sample_init]

        leaf_model = self.model
        # Build model tree
        model_tree = ModelTreeInv(leaf_model, header_init,
                                  self.fit_used,
                                  testing_known_model=self.testing_known_model,
                                  known_model=self.known_model,
                                  max_depth=self.max_depth,
                                  min_samples_leaf=self.min_samples_leaf,
                                  search_type="greedy", n_search_grid=100,
                                  j_feature_range=self.qual_feature_indices,
                                  sign=self.sign,
                                  loss_func=self.norm)
        # set qual range here

        # ====================
        # Train model tree
        # ====================
        filename = "{}_{}".format(self.filename, self.norm)
        output_filename = os.path.join("output", filename)
        header = [w.replace("init_", "") for w in header_init]

        if self.plot_only:
            if self.testing_known_model:
                output_filename += "_known_inv"
            else:
                output_filename += "_learned"
            model_tree.plot_fitting_hist(output_filename, True, header)
        else:
            print("Training model tree with '{}'...".format(
                leaf_model.__class__.__name__))
            # sort only matteers to the plot
            sign, sort = model_tree.fit(X_init, y_post, verbose=True)
            y_pred = model_tree.predict(X_init)
            sort_str = "reversed_sort" if sort else "sort"
            if self.testing_known_model:
                output_filename += "_known_inv_{}_{}".format(sign, sort_str)
            else:
                output_filename += "_learned_{}_{}".format(sign, sort_str)
            invariant_string, generate_invariant, generate_txt, learned_tree = model_tree.export_func(
                header)
            if self.plot_fitting:
                model_tree.plot_fitting_hist(
                    output_filename+"_fitting", self.plot_only, header)

        # ====================
        # Save model tree results
        # ====================
        if save_model_tree:
            model_tree_filename = os.path.join(
                "output", "{}.p".format(filename))
            print("Saving model tree to '{}'...".format(model_tree_filename))
            pickle.dump(leaf_model, open(model_tree_filename, 'wb'))

        # # ====================
        # # Cross-validate model tree
        # # ====================
        # if cross_validation and (not self.use_inv):
        #     cross_validate(model_tree, X, y, kfold=5, seed=1)
        # TODO

        return invariant_string, generate_invariant, generate_txt, learned_tree
