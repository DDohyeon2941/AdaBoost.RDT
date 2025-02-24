# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:14:50 2022

@author: dohyeon
"""

import user_utils as uu
from copy import deepcopy

import numpy as np

import time
import warnings


from sklearn.base import clone, BaseEstimator, is_regressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import stable_cumsum
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, Lasso

from sklearn.tree import (
    DecisionTreeRegressor,
    ExtraTreeRegressor,
    BaseDecisionTree,
    DecisionTreeClassifier,
)

from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils.validation import _check_sample_weight, _num_samples, check_is_fitted

from statsmodels.stats.outliers_influence import variance_inflation_factor

"""Modify the Residual_DT"""

np.seterr(divide='ignore', invalid='ignore')
OUTLIER_THR = 3
USEALL = True

class AvgExtractor:
    def fit(self, X, y):
        self.average_target = np.mean(y)
        return self
    def predict(self, X):
        return np.ones(X.shape[0])*self.average_target

def _filter_cols(X_subset):
    """Extract candidate columns using VIF Score"""
    global USEALL
    if USEALL: return np.arange(X_subset.shape[1])



class hierarchical_Estimator(BaseEstimator):
    RANDOM_NUM = np.random.choice(range(10**6),1)
    def __init__(
            self,
            *,
            criterion="squared_error",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            ccp_alpha=0.0,
            alpha=1.0,
            avg_threshold=1,
            avg_ratio=0.6):

        max_depth = np.iinfo(np.int32).max if max_depth is None else max_depth
        self.dt_obj = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha)
        #self.lr_obj = LinearRegression()
        #self.lr_obj = Ridge(alpha=alpha)
        self.lr_obj = Lasso(alpha=alpha)
        self.avg_obj = AvgExtractor()


        self.criterion = criterion
        self.splitter = splitter
        self.max_depth  = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha= alpha
        self.ccp_alpha = ccp_alpha
        self.avg_threshold = avg_threshold
        self.avg_ratio = avg_ratio

        self.bootstrap_1d_ = None
        self.leaf_node_labels_1d_ = None



    def fit(self, X, y, sample_weight=None):
        self.dt_obj.max_depth = self.max_depth
        self.dt_obj.max_features = self.max_features
        self.dt_obj.random_state = self.random_state

        if sample_weight is None:
            sample_weight = np.ones(len(y))/len(y)
        temp_X =  deepcopy(X)

        self.dt_obj.fit(temp_X, y)
        self.leaf_node_unique = np.where(self.dt_obj.tree_.children_left==-1)[0]

        ##
        split_features = np.unique(self.dt_obj.tree_.feature)[1:]


        #print(self.dt_obj.max_depth)
        leaf_node_sample_idxs =  self.dt_obj.apply(temp_X)
        self.pred_leaf_node_labels = leaf_node_sample_idxs

        unique_leaf_labels_pred = np.unique(leaf_node_sample_idxs)
        assert np.all(np.isin(unique_leaf_labels_pred, self.leaf_node_unique))
        lr_dict = {}
        #ipdb.set_trace()

        for node_label in unique_leaf_labels_pred:
            # Divide the datasets, and extract candidate columns based on VIF scores
            node_idx_1d = np.where(leaf_node_sample_idxs==node_label)[0]
            node_x = _safe_indexing(temp_X, node_idx_1d)
            node_y = _safe_indexing(y, node_idx_1d)

            ##
            node_std_zero = np.where(np.std(node_x, axis=0)==0)[0]

            node_sel_cols = np.setdiff1d(np.arange(node_x.shape[1]),
                                         np.union1d(split_features, node_std_zero))
            if node_sel_cols.shape[0] == 0:
                node_sel_cols = np.setdiff1d(np.arange(node_x.shape[1]), node_std_zero)

            node_error = node_y - np.mean(node_y)
            low_error_ratio = (np.abs(node_error) < self.avg_threshold).sum() / node_error.shape[0]

            if low_error_ratio < self.avg_ratio:
                if not node_sel_cols.shape[0] == 0:
                    lr_dict[node_label] = [node_sel_cols, ]
                    leaf_node_lr = deepcopy(self.lr_obj)
                else:
                    node_sel_cols = np.arange(node_x.shape[1])
                    lr_dict[node_label] = [node_sel_cols, ]
                    leaf_node_lr = deepcopy(self.avg_obj)
            else:
                leaf_node_lr = deepcopy(self.avg_obj)
                lr_dict[node_label] = [node_sel_cols, ]

            leaf_node_lr.fit(node_x[:, node_sel_cols], node_y)
            lr_dict[node_label].append(leaf_node_lr)

            leaf_node_lr = None

        self.lr_dict = lr_dict
        temp_X = None
        return self

    def predict(self, X):
        temp_x =  deepcopy(X)
        leaf_node_sample_idxs = self.dt_obj.apply(temp_x)
        unique_leaf_labels_pred = np.unique(leaf_node_sample_idxs)
        assert np.all(np.isin(unique_leaf_labels_pred, self.leaf_node_unique))
        pred_1d = np.zeros(X.shape[0])
        for node_label in unique_leaf_labels_pred:
            node_idx_1d = np.where(leaf_node_sample_idxs==node_label)[0]
            node_lr = self.lr_dict[node_label][1]
            node_col_idx = self.lr_dict[node_label][0]
            node_x = _safe_indexing(temp_x, node_idx_1d)
            node_pred = node_lr.predict(node_x[:,node_col_idx])
            #negative to zero
            node_pred[node_pred<0.0] = 0.0
            if node_pred.ndim>1:
                node_pred = node_pred.squeeze()
            pred_1d[node_idx_1d] = node_pred
        return pred_1d



class Noise_corrector:
    def __init__(self,
                 is_corrected=True,
                 outlier_thr=3):
        self.is_corrected = is_corrected
        self.outlier_thr = outlier_thr

    def __call__(self, n_data, n_iter, init_sample_weight):

        self.min_value = init_sample_weight[0]
        self.outliers_mask_2d_ = np.zeros((n_data, n_iter), dtype=np.float64)
        self.residual_2d_ = np.zeros((n_data, n_iter), dtype=np.float64)
        self.min_sample_weights_ = np.zeros(n_iter, dtype=np.float64)

    def _correct_noise(self, iboost, real_y, pred_y, sample_weight):
        temp_sample_weight = sample_weight.copy()
        if self.min_value is None:
            min_val = np.min(temp_sample_weight)
        else:
            min_val = self.min_value
    
        outliers_mask_base = np.zeros(real_y.shape[0])
        temp_residual_1d = real_y - pred_y
        outlier_idx = np.where(temp_residual_1d > self.outlier_thr)[0]
        temp_sample_weight[outlier_idx] = min_val
        outliers_mask_base[outlier_idx] = 1.0

        self.outliers_mask_2d_[:, iboost] = outliers_mask_base
        self.residual_2d_[:, iboost] = temp_residual_1d
        self.min_sample_weights_[iboost] = min_val

        return temp_sample_weight

def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.
    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.
    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.
    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:
        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


def _positive_mask(predicted_residual_1d):
    pred_tree = (predicted_residual_1d > float(0))*float(1)
    return pred_tree

class Residual_DT:
    def __init__(self,
                 with_residual_dt=True,
                 max_depth=4,
                 weighted_fit=True):
        self.with_residual_dt = with_residual_dt
        self.resid_dt_obj = DecisionTreeRegressor(max_depth=max_depth)
        self.weighted_fit = weighted_fit
        self.conv_func = _positive_mask

    def __call__(self, n_data, n_iter):
        self.pred_residual_2d_ =  np.zeros((n_data, n_iter), dtype=np.float64)
        self.worst_mask_2d_ =  np.zeros((n_data, n_iter), dtype=np.float64)
        self.worst_preds_ =  np.zeros(n_iter, dtype=np.float64)
        self.resid_dt_objs_ = []

    def _get_residual_dt(self, iboost, X, residual_1d, sample_weight):
        temp_dt_obj = deepcopy(self.resid_dt_obj)
        temp_x = deepcopy(X)
        if self.weighted_fit:
            temp_dt_obj.fit(X, residual_1d, sample_weight=sample_weight)
        else:
            temp_dt_obj.fit(X, residual_1d)
        pred_residual_1d = temp_dt_obj.predict(temp_x)

        self.pred_residual_2d_[:, iboost] = pred_residual_1d
        self.resid_dt_objs_.append(temp_dt_obj)

        self._make_variable(iboost, pred_residual_1d)

        temp_dt_obj = None
        temp_x = None

    def _make_variable(self, iboost, predicted_residual):
        worst_pred = np.max(np.abs(predicted_residual))
        pred_tree = self.conv_func(predicted_residual)

        self.worst_mask_2d_[:, iboost] = pred_tree
        self.worst_preds_[iboost] = worst_pred


    def concat_X(self, iboost, raw_X):
        return np.concatenate([raw_X,
                               self.worst_mask_2d_[:,iboost-1].reshape(-1,1)], axis=1)


class AdaBoostRegressor_ModelTree_base(AdaBoostRegressor):
    def __init__(self,
                 n_estimators=25,
                 base_estimator=None,
                 loss='linear',
                 noise_obj=None):
        super(AdaBoostRegressor_ModelTree_base,self).__init__(n_estimators=n_estimators,
                                                         base_estimator=base_estimator,
                                                         loss=loss)
        self.noise_obj = noise_obj

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
    
        # TODO: Remove in v1.2
        # criterion "mse" and "mae" would cause warnings in every call to
        # DecisionTreeRegressor.fit(..)
        if isinstance(estimator, (DecisionTreeRegressor, ExtraTreeRegressor, hierarchical_Estimator)):
            if getattr(estimator, "criterion", None) == "mse":
                estimator.set_params(criterion="squared_error")
            elif getattr(estimator, "criterion", None) == "mae":
                estimator.set_params(criterion="absolute_error")
    
        # TODO(1.3): Remove
        # max_features = 'auto' would cause warnings in every call to
        # Tree.fit(..)
        if isinstance(estimator, BaseDecisionTree):
            if getattr(estimator, "max_features", None) == "auto":
                if isinstance(estimator, DecisionTreeClassifier):
                    estimator.set_params(max_features="sqrt")
                elif isinstance(estimator, DecisionTreeRegressor):
                    estimator.set_params(max_features=1.0)


        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

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
        
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            y_numeric=is_regressor(self),
            )

        sample_weight = _check_sample_weight(sample_weight, X, np.float64, copy=True)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")
        
        # Check parameters
        self._validate_estimator()
    
        # Clear any previous fit results

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        self.times_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.raw_sample_weights_2d_ = np.zeros((len(y), self.n_estimators), dtype=np.float64)
        self.nor_sample_weights_2d_ = np.zeros((len(y), self.n_estimators), dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration

        self.noise_obj(n_data=len(y),
                       n_iter=self.n_estimators,
                       init_sample_weight=sample_weight)


        random_state = check_random_state(self.random_state)
        self.nor_sample_weights_2d_[:, 0] = sample_weight
        time_base = 0

        for iboost in range(self.n_estimators):
            start = time.time()
            # Boosting step

            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )
            self.raw_sample_weights_2d_[:, iboost] = sample_weight
  
            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            itime = time.time() - start
            time_base += itime
            self.times_[iboost] = itime

            if showtimes:
                print("Round: %s, est_time: %.2f, total_time: %.2f" % (iboost, itime, time_base))
  
            # Stop if error is zero
            if estimator_error == 0:
                break
  
            sample_weight_sum = np.sum(sample_weight)
  
            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                break
  
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break
  
            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
                self.nor_sample_weights_2d_[:, iboost+1] = sample_weight

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

        self.estimators_[iboost].bootstrap_1d_ = bootstrap_idx
        self.estimators_[iboost].leaf_node_labels_1d_ = estimator.dt_obj.apply(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == "square":
            masked_error_vector **= 2
        elif self.loss == "exponential":
            masked_error_vector = 1.0 - np.exp(-masked_error_vector)

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

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )

        if self.noise_obj.is_corrected:
            sample_weight  = self.noise_obj._correct_noise(iboost, y,
                                                           y_predict,
                                                           sample_weight)

        return sample_weight, estimator_weight, estimator_error


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

class AdaBoostRegressor_ModelTree(AdaBoostRegressor_ModelTree_base):
    def __init__(self,
                 n_estimators=25,
                 base_estimator=None,
                 loss='linear',
                 noise_obj=None,
                 resid_dt_obj=None):
        super(AdaBoostRegressor_ModelTree,self).__init__(
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            loss=loss,
            noise_obj=noise_obj)

        self.resid_dt_obj = resid_dt_obj
        self.cdf_weight = 0
        self.median_pred = 0

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
        init_time = time.time()

        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            y_numeric=is_regressor(self),
            )

        sample_weight = _check_sample_weight(sample_weight, X, np.float64, copy=True)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")
        
        # Check parameters
        self._validate_estimator()
    
        # Clear any previous fit results

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        self.times_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.raw_sample_weights_2d_ = np.zeros((len(y), self.n_estimators), dtype=np.float64)
        self.nor_sample_weights_2d_ = np.zeros((len(y), self.n_estimators+1), dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration

        self.noise_obj(n_data=len(y),
                       n_iter=self.n_estimators,
                       init_sample_weight=sample_weight)


        random_state = check_random_state(self.random_state)
        self.nor_sample_weights_2d_[:, 0] = sample_weight
        time_base = 0

        for iboost in range(self.n_estimators):
            start = time.time()
            # Boosting step
            if iboost == 0:
                self.resid_dt_obj(n_data=len(y), n_iter=self.n_estimators)
            else:
                X = self.resid_dt_obj.concat_X(iboost, X)
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )
            self.raw_sample_weights_2d_[:, iboost] = sample_weight
  
            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            itime = time.time() - start
            time_base += itime
            self.times_[iboost] = itime

            if showtimes:
                print("Round: %s, est_time: %.2f, total_time: %.2f" % (iboost, itime, time_base))
  
            # Stop if error is zero
            if estimator_error == 0:
                self.whole_time_ = time.time() - init_time
                break
  
            sample_weight_sum = np.sum(sample_weight)
  
            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                self.whole_time_ = time.time() - init_time
                break
  
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                self.whole_time_ = time.time() - init_time
                break

            # Normalize
            sample_weight /= sample_weight_sum
            self.nor_sample_weights_2d_[:, iboost+1] = sample_weight
            self.resid_dt_obj._get_residual_dt(iboost=iboost,
                                               X=X,
                                               residual_1d=self.noise_obj.residual_2d_[:, iboost],
                                               sample_weight=sample_weight)

        self.whole_time_ = time.time() - init_time
        return self


    @logging_time
    def obj_predict_elements1(self, XX, limit, returnX=False):
        temp_x = deepcopy(XX)
        predictions = np.zeros((temp_x.shape[0], limit), dtype=np.float64)
        resid_predictions = np.zeros((temp_x.shape[0], limit), dtype=np.float64)
    
        for est_idx in range(limit):
            iestimator = self.estimators_[est_idx]
            predictions[:, est_idx] = iestimator.predict(temp_x)
            idt = self.resid_dt_obj.resid_dt_objs_[est_idx]
            pred_resid = idt.predict(temp_x)
            resid_predictions[:, est_idx] = pred_resid
            temp_mask = self.resid_dt_obj.conv_func(pred_resid)
            temp_x = np.concatenate([temp_x, temp_mask.reshape(-1,1)], axis=1)
            pred_resid=None
        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)
    
        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
    
        if returnX:
        # Return median predictions
            return predictions, sorted_idx, weight_cdf, resid_predictions, temp_x
        else:
            return predictions, sorted_idx, weight_cdf, resid_predictions
    
    def predict_log(self, XX):
    
        limit = len(self.estimators_)
        pred_elements = self.obj_predict_elements1(XX, limit, returnX=False)
    
        cdf_weight = get_cdf_weight_avg(self.resid_dt_obj.pred_residual_2d_, pred_elements[3], limit)
        self.cdf_weight = cdf_weight
        pred_y = extract_optimal(pred_elements[0],
                                 pred_elements[1],
                                 pred_elements[2],
                                 cdf_weight)
        median_pred_y = extract_optimal(pred_elements[0],
                                        pred_elements[1],
                                        pred_elements[2],
                                        np.ones(cdf_weight.shape[0])*0.5)

        pred_y[pred_y<0] = float(0)
        median_pred_y[median_pred_y<0] = float(0)

        return pred_y, median_pred_y
    
    def predict(self, XX):
        pred_y, median_pred_y = self.predict_log(XX)
        self.median_pred = np.exp(median_pred_y)-1
        return np.exp(pred_y)-1


    def predict_log_median(self, XX):
        limit = len(self.estimators_)
        pred_elements = self.obj_predict_elements1(XX, limit, returnX=False)
        cdf_weight = np.ones(XX.shape[0]) * 0.5
        pred_y = extract_optimal(pred_elements[0], pred_elements[1], pred_elements[2], cdf_weight)
        pred_y[pred_y<0] = float(0)
        return pred_y

    def predict_median(self, XX):
        return np.exp(self.predict_log_median(XX))-1


def extract_optimal(pred_2d, s_idx, w_cdf, opt_loc):
    opt_or_above = w_cdf >= (opt_loc * w_cdf[:, -1])[:, np.newaxis]
    opt_idx = opt_or_above.argmax(axis=1)
    opt_estimators = s_idx[np.arange(pred_2d.shape[0]), opt_idx]
    return pred_2d[np.arange(pred_2d.shape[0]), opt_estimators]

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def get_cdf_weight_avg(trn_pred_resid, tst_pred_resid, limit):
    trn_avg_pred = np.mean(trn_pred_resid[:, :limit], axis=1)
    tst_avg_pred = np.mean(tst_pred_resid[:, :limit], axis=1)
    #scaler = MinMaxScaler(feature_range=(0.05,0.95))
    scaler = StandardScaler()
    scaler.fit(trn_avg_pred.reshape(-1,1))
    output = scaler.transform(tst_avg_pred.reshape(-1,1)).squeeze()

    return clipping_cdf_weight(sigmoid(output))


def clipping_cdf_weight(raw_cdf_weight, min_val=0.10, max_val=0.90):
    raw_cdf_weight[raw_cdf_weight<min_val] = min_val
    raw_cdf_weight[raw_cdf_weight>max_val] = max_val
    return raw_cdf_weight






