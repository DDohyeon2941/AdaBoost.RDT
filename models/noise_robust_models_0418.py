# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:57:12 2021

@author: dohyeon
"""

import time
import numpy as np

from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils.validation import _num_samples

from sklearn.ensemble import AdaBoostRegressor


class AdaBoost_RT(AdaBoostRegressor):
    def __init__(self,
                 n_estimators=25,
                 base_estimator=None,
                 pie=0.05):
        super(AdaBoost_RT, self).__init__(n_estimators=n_estimators,
                                        base_estimator=base_estimator)
        self.pie = pie
    def fit(self, X, y, sample_weight=None, showtimes=False):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X.astype(np.float32), y)
        self.X = X
        self.y = y


        if not sample_weight is None:
            sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        else:
            sample_weight = np.ones(y.shape[0]) / y.shape[0]
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()


        # Check loss
        """
        if self.loss not in ('linear', 'square', 'exponential','absolute_relative'):
            raise ValueError(
                "loss must be 'linear', 'square', 'exponential', or 'absolute_relative'")
        """

        # Clear any previous fit results
        self.raw_sample_weights_ = np.zeros((y.shape[0],self.n_estimators))
        self.nor_sample_weights_ = np.zeros((y.shape[0],self.n_estimators))

        self.estimators_ = []
        self.error_rates_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.betas_ = np.zeros(self.n_estimators, dtype=np.float64)

        self.times_ = np.zeros(self.n_estimators)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            start = time.time()
            self.nor_sample_weights_[:,iboost] = sample_weight

            # Get beta and error_rate instead of estimator_weight, estimator_error
            sample_weight, beta, error_rate = self._boost(
                iboost, self.X, self.y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break


            self.error_rates_[iboost] = error_rate
            self.betas_[iboost] = beta
            self.raw_sample_weights_[:,iboost] = sample_weight

            itime = time.time() - start
            self.times_[iboost] = itime
            if showtimes:
                print("Round: %s, time: %.2f" % (iboost, itime))
            # Stop if error is zero

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self


    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.
            Controls also the bootstrap of the weights used to train the weak
            learner.
            replacement.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)),
            size=_num_samples(X),
            replace=True,
            p=sample_weight,
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)

        y_copy = y.copy()
        # Add small value for zero demands to calculate error rate of each sample
        y_copy[y_copy==float(0)] = 1e-20
        error_vect = np.abs(y_predict - y_copy)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_rate_vector = (masked_error_vector / np.abs(y_copy[sample_mask]))
        error_rate = np.mean(np.average(error_rate_vector > self.pie, weights=masked_sample_weight, axis=0))

        """
        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()
        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None
        """
        #beta = estimator_error / (1.0 - estimator_error)
        beta = error_rate ** 2

        # Boost weight using AdaBoost.R2 alg
        #estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:

            sample_weight[sample_mask] *= np.power(beta, (error_rate_vector <= self.pie).astype(int) * self.learning_rate)

            """

            sample_weight[sample_mask] *= self.learning_rate
            sample_weight[error_rate_vector <= self.pie] *= beta

            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )
            """

        return sample_weight, beta, error_rate



    def _predict(self, X):

        # y_preds.shape is (max_iter, len(X))
        y_preds = np.array([reg.predict(X) for reg in self.estimators_])
        
        # weighted majority vote
        y_pred = np.sum(np.log(1.0/self.betas_)[:,None] * y_preds, axis=0) / np.log(1.0/self.betas_).sum()

        return y_pred






