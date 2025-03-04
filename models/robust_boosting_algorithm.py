# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:03:21 2022

@author: dohyeon
"""

"""
Stage1


"""
import time
from abc import ABCMeta
from abc import abstractmethod

import numbers
import warnings

from sklearn.exceptions import NotFittedError
from functools import partial


import numpy as np


from scipy.optimize import minimize

from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier

from sklearn.dummy import DummyRegressor


from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import deprecated
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.stats import _weighted_percentile


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE, DOUBLE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaseEnsemble

from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.ensemble._gradient_boosting import predict_stage
from sklearn.ensemble._gradient_boosting import _random_sample_mask

from sklearn.ensemble._gb_losses import MultinomialDeviance,BinomialDeviance
from sklearn.ensemble._gb_losses import LossFunction, LeastSquaresError, LeastAbsoluteError, HuberLossFunction, QuantileLossFunction, ExponentialLoss, LeastAbsoluteError, RegressionLossFunction
from sklearn.ensemble._gb import VerboseReporter

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

C=1.547
theta = 0.0001
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

class mloss(RegressionLossFunction):
    def __init__(self):
        global C
        global theta
        self.c = C
        self.theta = theta
        self.a1s = []

        super(mloss,self).__init__()

    def init_estimator(self):
        #DummyRegressor(strategy="mean")
        return DecisionTreeRegressor(criterion="absolute_error", max_depth=2,  min_samples_leaf=20)
    def __call__(self, y_real, y_pred, sample_weight=None):
        if sample_weight is None:
            return np.mean((y_real - y_pred.ravel()) ** 2)
        else:
            return (
                1
                / sample_weight.sum()
                * np.sum(sample_weight * ((y_real - y_pred.ravel()) ** 2))
            )

    def tukey_bisquare(self, u):
        if np.abs(u)<=self.c:
            output = 1-(1-(u/self.c)**2)**3
        else:
            output=1
        return output

    def tukey_weight(self, u):
        if np.abs(u)<=self.c:
            output = (6*(u**5)-12**self.c**2*u**3+6*self.c**4*u) /self.c**6
            #output = 1-3*((1-(u/c)**2)**2)*(-2*(u/c))*(1/c)
        else:
            output = 0
        return output

    def obj_func(self, cc, y_real, y_pred):
        output = np.sum([self.tukey_bisquare(u=(y_i-y_hat_i)/cc) for y_i, y_hat_i in zip(y_real, y_pred)])
        output = (output/len(y_real))-0.5
        return output
    
    def cal_const(self, cc, y_real, y_pred):
        output = np.sum([(self.tukey_bisquare(u=(y_i-y_hat_i)/cc))*((y_i-y_hat_i)/cc) for y_i, y_hat_i in zip(y_real, y_pred)])
        return output**-1
    
    def negative_gradient_func(self, cc, x_idx, y_real, y_pred):
        const = self.cal_const(cc=cc, y_real=y_real, y_pred=y_pred)
        return -const*self.tukey_weight(u=(y_real[x_idx]-y_pred[x_idx])/cc)
    @logging_time
    def negative_gradient(self, real_y, pred_y_copy,  k=None, sample_weight=None,tol=None):
        func2 = partial(self.obj_func, y_real=real_y, y_pred=pred_y_copy)
        a1 = minimize(func2, 1, method='BFGS',options={'disp': True}, tol=tol, maxiter=50)
        print(a1.x)
        grd = np.array([self.negative_gradient_func(cc=a1.x, x_idx=xx, y_real=real_y, y_pred=pred_y_copy) for xx in range(real_y.shape[0])])
        #ipdb.set_trace()
        self.a1s.append(a1)
        return -grd.ravel()


    def get_init_raw_predictions(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1, 1).astype(np.float64)


    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        """Least squares does not need to update terminal regions.
        But it has to update the predictions.
        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n,)
            The weight of each sample.
        sample_mask : ndarray of shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        pass

class HuberLossFunction1(RegressionLossFunction):
    """Huber loss function for robust regression.

    M-Regression proposed in Friedman 2001.

    Parameters
    ----------
    alpha : float, default=0.9
        Percentile at which to extract score.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.gamma = None

    def init_estimator(self):
        return DummyRegressor(strategy="quantile", quantile=0.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Huber loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        gamma = self.gamma
        if gamma is None:
            if sample_weight is None:
                gamma = np.percentile(np.abs(diff), self.alpha * 100)
            else:
                gamma = _weighted_percentile(
                    np.abs(diff), sample_weight, self.alpha * 100
                )
        print(gamma)
        gamma_mask = np.abs(diff) <= gamma
        if sample_weight is None:
            sq_loss = np.sum(0.5 * diff[gamma_mask] ** 2)
            lin_loss = np.sum(gamma * (np.abs(diff[~gamma_mask]) - gamma / 2))
            loss = (sq_loss + lin_loss) / y.shape[0]
        else:
            sq_loss = np.sum(0.5 * sample_weight[gamma_mask] * diff[gamma_mask] ** 2)
            lin_loss = np.sum(
                gamma
                * sample_weight[~gamma_mask]
                * (np.abs(diff[~gamma_mask]) - gamma / 2)
            )
            loss = (sq_loss + lin_loss) / sample_weight.sum()
        return loss

    def negative_gradient(self, y, raw_predictions, sample_weight=None, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        if sample_weight is None:
            gamma = np.percentile(np.abs(diff), self.alpha * 100)
        else:
            gamma = _weighted_percentile(np.abs(diff), sample_weight, self.alpha * 100)
        gamma_mask = np.abs(diff) <= gamma
        residual = np.zeros((y.shape[0],), dtype=np.float64)
        residual[gamma_mask] = diff[gamma_mask]
        residual[~gamma_mask] = gamma * np.sign(diff[~gamma_mask])
        self.gamma = gamma
        return residual

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        gamma = self.gamma
        diff = y.take(terminal_region, axis=0) - raw_predictions.take(
            terminal_region, axis=0
        )
        median = _weighted_percentile(diff, sample_weight, percentile=50)
        diff_minus_median = diff - median
        tree.value[leaf, 0] = median + np.mean(
            np.sign(diff_minus_median) * np.minimum(np.abs(diff_minus_median), gamma)
        )

def _get_mad(data):
    return np.median(np.absolute(data - np.median(data)))

class Robloss(HuberLossFunction):
    def __init__(self, alpha):
        super(Robloss, self).__init__(alpha=alpha)

    def init_estimator(self):
        return DummyRegressor(strategy="quantile", quantile=0.5)


    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Huber loss.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        #print(sample_weight)
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        gamma = self.gamma
        if gamma is None:
            if sample_weight is None:
                gamma = _get_mad(diff)
            else:
                gamma =  _get_mad(diff)
        #print(np.vstack(np.unique(diff, return_counts=True)).T)
        print(gamma)
        gamma_mask = np.abs(diff) <= gamma
        if sample_weight is None:
            sq_loss = np.sum(0.5 * diff[gamma_mask] ** 2)
            lin_loss = np.sum(gamma * (np.abs(diff[~gamma_mask]) - gamma / 2))
            loss = (sq_loss + lin_loss) / y.shape[0]
        else:
            sq_loss = np.sum(0.5 * sample_weight[gamma_mask] * diff[gamma_mask] ** 2)
            lin_loss = np.sum(
                gamma
                * sample_weight[~gamma_mask]
                * (np.abs(diff[~gamma_mask]) - gamma / 2)
            )
            loss = (sq_loss + lin_loss) / sample_weight.sum()
        return loss

    def negative_gradient(self, real_y, pred_y_copy, sample_weight=None, **kargs):
        """Compute the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = pred_y_copy.ravel()
        diff = real_y - raw_predictions
        if sample_weight is None:
            gamma =  _get_mad(diff)
        else:
            gamma =  _get_mad(diff)
        gamma_mask = np.abs(diff) <= gamma
        residual = np.zeros((real_y.shape[0],), dtype=np.float64)
        residual[gamma_mask] = diff[gamma_mask]
        residual[~gamma_mask] = gamma * np.sign(diff[~gamma_mask])
        self.gamma = gamma
        return residual

class BaseGradientBoosting1(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting."""

    @abstractmethod
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_estimators,
        criterion,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_depth,
        min_impurity_decrease,
        init,
        subsample,
        max_features,
        ccp_alpha,
        random_state,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    @abstractmethod
    def _validate_y(self, y, sample_weight=None):
        """Called by fit to validate y."""

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self.loss_
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(
                y, raw_predictions_copy, k=k, sample_weight=sample_weight
            )

            # induce regression tree on residuals
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha,
            )

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight, check_input=False)

            # update tree leaves
            loss.update_terminal_regions(
                tree.tree_,
                X,
                y,
                residual,
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""
        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than 0 but was %r" % self.n_estimators
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be greater than 0 but was %r" % self.learning_rate
            )

        if (
            self.loss not in self._SUPPORTED_LOSS
            or self.loss not in LOSS_FUNCTIONS
        ):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        # TODO: Remove in v1.2
        if self.loss == "ls":
            warnings.warn(
                "The loss 'ls' was deprecated in v1.0 and "
                "will be removed in version 1.2. Use 'squared_error'"
                " which is equivalent.",
                FutureWarning,
            )
        elif self.loss == "lad":
            warnings.warn(
                "The loss 'lad' was deprecated in v1.0 and "
                "will be removed in version 1.2. Use "
                "'absolute_error' which is equivalent.",
                FutureWarning,
            )

        if self.loss == "deviance":
            loss_class = (
                MultinomialDeviance
                if len(self.classes_) > 2
                else BinomialDeviance
            )
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        if is_classifier(self):
            self.loss_ = loss_class(self.n_classes_)
        elif self.loss in ("huber", "quantile", "huber1", "rob"):
            self.loss_ = loss_class(self.alpha)
        else:
            self.loss_ = loss_class()

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == "zero"):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    "Got init={}".format(self.init)
                )

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but was %r" % self.alpha)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classifier(self):
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    "Invalid value for max_features: %r. "
                    "Allowed string values are 'auto', 'sqrt' "
                    "or 'log2'."
                    % self.max_features
                )
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if 0.0 < self.max_features <= 1.0:
                max_features = max(int(self.max_features * self.n_features_in_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change, (numbers.Integral, type(None))):
            raise ValueError(
                "n_iter_no_change should either be None or an integer. %r was passed"
                % self.n_iter_no_change
            )

    def _init_state(self):
        """Initialize model state and allocate model state data structures."""

        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K), dtype=object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators), dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, "train_score_"):
            del self.train_score_
        if hasattr(self, "oob_improvement_"):
            del self.oob_improvement_
        if hasattr(self, "init_"):
            del self.init_
        if hasattr(self, "_rng"):
            del self._rng

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes."""
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError(
                "resize with smaller n_estimators %d < %d"
                % (total_n_estimators, self.estimators_[0])
            )

        self.estimators_ = np.resize(
            self.estimators_, (total_n_estimators, self.loss_.K)
        )
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        if self.subsample < 1 or hasattr(self, "oob_improvement_"):
            # if do oob resize arrays or create new if not available
            if hasattr(self, "oob_improvement_"):
                self.oob_improvement_ = np.resize(
                    self.oob_improvement_, total_n_estimators
                )
            else:
                self.oob_improvement_ = np.zeros(
                    (total_n_estimators,), dtype=np.float64
                )

    def _is_initialized(self):
        return len(getattr(self, "estimators_", [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    @abstractmethod
    def _warn_mae_for_criterion(self):
        pass

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, default=None
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.criterion in ("absolute_error", "mae"):
            # TODO: This should raise an error from 1.1
            self._warn_mae_for_criterion()

        if self.criterion == "mse":
            # TODO: Remove in v1.2. By then it should raise an error.
            warnings.warn(
                "Criterion 'mse' was deprecated in v1.0 and will be "
                "removed in version 1.2. Use `criterion='squared_error'` "
                "which is equivalent.",
                FutureWarning,
            )

        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        # Since check_array converts both X and y to the same dtype, but the
        # trees use different types for X and y, checking them separately.

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
        )

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_sample_weight(sample_weight, X)

        y = column_or_1d(y, warn=True)

        if is_classifier(self):
            y = self._validate_y(y, sample_weight)
        else:
            y = self._validate_y(y)

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            X, X_val, y, y_val, sample_weight, sample_weight_val = train_test_split(
                X,
                y,
                sample_weight,
                random_state=self.random_state,
                test_size=self.validation_fraction,
                stratify=stratify,
            )
            if is_classifier(self):
                if self._n_classes != np.unique(y).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        "The training data after the early stopping split "
                        "is missing some classes. Try using another random "
                        "seed."
                    )
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == "zero":
                raw_predictions = np.zeros(
                    shape=(X.shape[0], self.loss_.K), dtype=np.float64
                )
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = (
                        "The initial estimator {} does not support sample "
                        "weights.".format(self.init_.__class__.__name__)
                    )
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError as e:
                        # regular estimator without SW support
                        raise ValueError(msg) from e
                    except ValueError as e:
                        if (
                            "pass parameters to specific steps of "
                            "your pipeline using the "
                            "stepname__parameter"
                            in str(e)
                        ):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = self.loss_.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "estimators_.shape[0]=%d when "
                    "warm_start==True" % (self.n_estimators, self.estimators_.shape[0])
                )
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse="csr")
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X,
            y,
            raw_predictions,
            sample_weight,
            self._rng,
            X_val,
            y_val,
            sample_weight_val,
            begin_at_stage,
            monitor,
        )

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, "oob_improvement_"):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        sample_weight,
        random_state,
        X_val,
        y_val,
        sample_weight_val,
        begin_at_stage=0,
        monitor=None,
    ):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(
                    y[~sample_mask],
                    raw_predictions[~sample_mask],
                    sample_weight[~sample_mask],
                )

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i,
                X,
                y,
                raw_predictions,
                sample_weight,
                sample_mask,
                random_state,
                X_csc,
                X_csr,
            )

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(
                    y[sample_mask],
                    raw_predictions[sample_mask],
                    sample_weight[sample_mask],
                )
                self.oob_improvement_[i] = old_oob_score - loss_(
                    y[~sample_mask],
                    raw_predictions[~sample_mask],
                    sample_weight[~sample_mask],
                )
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter), sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], self.loss_.K), dtype=np.float64
            )
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(X, self.init_).astype(
                np.float64
            )
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
            yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        self._check_initialized()

        relevant_trees = [
            tree
            for stage in self.estimators_
            for tree in stage
            if tree.tree_.node_count > 1
        ]
        if not relevant_trees:
            # degenerate case where all trees have only one node
            return np.zeros(shape=self.n_features_in_, dtype=np.float64)

        relevant_feature_importances = [
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in relevant_trees
        ]
        avg_feature_importances = np.mean(
            relevant_feature_importances, axis=0, dtype=np.float64
        )
        return avg_feature_importances / np.sum(avg_feature_importances)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features,)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape \
                (n_trees_per_iteration, n_samples)
            The value of the partial dependence function on each grid point.
        """
        if self.init is not None:
            warnings.warn(
                "Using recursion method with a non-constant init predictor "
                "will lead to incorrect partial dependence values. "
                "Got init=%s."
                % self.init,
                UserWarning,
            )
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        n_estimators, n_trees_per_stage = self.estimators_.shape
        averaged_predictions = np.zeros(
            (n_trees_per_stage, grid.shape[0]), dtype=np.float64, order="C"
        )
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                tree = self.estimators_[stage, k].tree_
                tree.compute_partial_dependence(
                    grid, target_features, averaged_predictions[k]
                )
        averaged_predictions *= self.learning_rate

        return averaged_predictions

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `n_features_` was deprecated in version 1.0 and will be "
        "removed in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_

class GradientBoostingRegressor1(RegressorMixin, BaseGradientBoosting1):

    # TODO: remove "ls" in version 1.2
    _SUPPORTED_LOSS = (
        "squared_error",
        "ls",
        "absolute_error",
        "lad",
        "huber",
        "quantile",
        "huber1",
        "rob",
        "mloss"
    )

    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
    ):

        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )

    def _validate_y(self, y, sample_weight=None):
        if y.dtype.kind == "O":
            y = y.astype(DOUBLE)
        return y

    def _warn_mae_for_criterion(self):
        # TODO: This should raise an error from 1.1
        warnings.warn(
            "criterion='mae' was deprecated in version 0.24 and "
            "will be removed in version 1.1 (renaming of 0.26). The "
            "correct way of minimizing the absolute error is to use "
            " loss='absolute_error' instead.",
            FutureWarning,
        )

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

    # FIXME: to be removed in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `n_classes_` was deprecated "
        "in version 0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def n_classes_(self):
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_classes_ attribute.".format(self.__class__.__name__)
            ) from nfe
        return 1

LOSS_FUNCTIONS = {
    "squared_error": LeastSquaresError,
    "ls": LeastSquaresError,
    "absolute_error": LeastAbsoluteError,
    "lad": LeastAbsoluteError,
    "huber": HuberLossFunction,
    "quantile": QuantileLossFunction,
    "deviance": None,  # for both, multinomial and binomial
    "exponential": ExponentialLoss,
    "huber1": HuberLossFunction1,
    "rob":Robloss,
    "mloss":mloss
}