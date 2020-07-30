"""This file contains implementation of Offset and Imbalance Aware random forest Classifier. The most of
the logic is copied from Scikit-Learn ``RandomForestClassifier`` implementation.

The sampling process during training procedure is customized, see the `_parallel_build_trees` and
`_sample_positive_indices` methods.
"""

import numbers
import threading
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

from ._utils import check_random_state, _joblib_parallel_args, _check_sample_offsets, _partition_estimators

# from `sklearn/ensemble/_forest.py`
MAX_INT = np.iinfo(np.int32).max


class OiasRandomForestClassifier(BaseEnsemble, ClassifierMixin):  # can't use anything else but BaseEnsemble
    """
    Offset and Imbalance Aware random forest Classifier.

    Variant of the Random Forest algorithm able to take into account the imbalance of both
    benign and pathogenic variants, and imbalance of variants with respect to donor/acceptor
    site offset.
    """

    def __init__(self,
                 # OIAS parameters
                 class_ratio=10,
                 bins=(np.iinfo(np.int32).min, np.iinfo(np.int32).max),

                 # ensemble parameters
                 n_estimators=100,
                 n_positives=None,
                 bootstrap=True,
                 n_jobs=None,
                 verbose=0,

                 # tree parameters
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 random_state=None,
                 ccp_alpha=0.0):
        """
        Instantiate the classifier.

        Parameters
        ----------
        class_ratio : positive float or int
            ratio of negative/positive instances (``n_negatives`` = ``class_ratio`` * ``n_positives``)
        bins : array-like of shape (1,), default (np.int32.min, np.int32.max)
            a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform bin
            widths. None by default
        n_estimators : positive int
            number of random forest trees
        n_positives : float | int | None, default=None
            the maximum number of positive examples to draw from the total available when training a single estimator:
            - if ``float``, this indicates a fraction of the total and should be within the interval `(0, 1)`;
            - if ``int``, this indicates the exact number of samples;
            - if ``None``, this indicates the total number of samples.
        bootstrap : bool
            when training estimator, elements are sampled w/ replacement if True. Otherwise, the sampling is
            performed w/o replacement
        n_jobs : int, default=None
            The number of jobs to run in parallel. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors.

        See Scikit-learn's documentation for ``RandomForestClassifier`` for explanation of the remaining hyperparameters.
        """
        super().__init__(
            DecisionTreeClassifier(),
            n_estimators=n_estimators,
            # DecisionTreeClassifier's parameters copied from `sklearn.ensemble.RandomForestClassifier`
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
        )
        self.class_ratio = class_ratio
        self.bins = bins
        self.n_positives = n_positives

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha

        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.verbose = verbose

        # required by sklearn code, but not supported by OIAS RF
        self.warm_start = False
        self.class_weight = None

    def fit(self, X, y, offsets=None, pos_label=1):
        """
        Build a forest of trees from the training set (X, y) using offset information.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        offsets : array-like of shape (n_samples,), default=None
            Variant offsets. If None, then offset is not used during the sampling
            process.
        pos_label : int, default=1
            Positive class label value.

        Returns
        -------
        self : object
        """
        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=np.floating)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=np.int)

        if offsets is None:
            raise ValueError("offsets must not be None")

        offsets = _check_sample_offsets(offsets, X)

        # Validate bins
        if self.bins is None:
            raise ValueError("bins must not be None")

        self.bins = check_array(self.bins, ensure_2d=False, dtype=np.int)
        if np.any(self.bins[:-1] >= self.bins[1:]):
            raise ValueError("bin boundaries must be sorted")

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples,), for example using ravel().",
                          DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != np.float or not y.flags.contiguous:
            y = np.ascontiguousarray(y)

        # Get number of positive examples to sample for each tree
        n_pos_samples = _get_n_samples_bootstrap(n_samples=X.shape[0], max_samples=self.n_positives)

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warnings.warn("Warm-start fitting without increasing n_estimators does not "
                          "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False, random_state=random_state) for _ in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(t, self, X, y,
                                               offsets=offsets,
                                               n_pos_samples=n_pos_samples,
                                               pos_label=pos_label)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k], axis=1), axis=0)

            return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = \
                np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        return y, expanded_class_weight


def _parallel_build_trees(tree: DecisionTreeClassifier, forest: OiasRandomForestClassifier,
                          X, y, offsets=None, n_pos_samples=None, pos_label=1):
    """
    Private function used to fit a single tree in parallel.
    """
    # 0 - prepare masks and other info
    random_instance = check_random_state(tree.random_state)

    assert X.shape[0] == y.shape[0], "`X` and `y` must have the same number of rows"

    pos_mask = y == pos_label
    pos_idxs, _ = np.where(pos_mask)
    neg_idxs, _ = np.where(~pos_mask)

    # 1 - sample given number of positive indices
    pos_idxs_sample = _sample_positive_indices(offsets[pos_idxs], forest.bins, size=n_pos_samples,
                                               random_instance=random_instance, replace=forest.bootstrap)

    # 2 - sample negative indices without replacement, while respecting the class ratio
    # at most `n_neg` indices are sampled
    n_neg = min(len(neg_idxs), round(forest.class_ratio * pos_idxs_sample.shape[0]))
    neg_idxs_sample = random_instance.choice(a=len(neg_idxs), size=n_neg, replace=False)

    # 3 - create the dataset - use the sampled indices to select elements from positive and negative indices,
    #   then use the positive and negative we select the instances
    X_current = np.concatenate((X[pos_idxs[pos_idxs_sample]],  # positive instances
                                X[neg_idxs[neg_idxs_sample]]),  # negative instances
                               axis=0)
    y_current = np.concatenate((y[pos_idxs[pos_idxs_sample]],  # positive instances
                                y[neg_idxs[neg_idxs_sample]]),  # negative instances
                               axis=0)
    # sanity check
    assert X_current.shape[0] == y_current.shape[0], "`X_current` and `y_current` must have the same number of rows"

    # 4 - make indices to shuffle the X/y arrays
    perms = random_instance.permutation(X_current.shape[0])

    # 5 - fit the tree
    return tree.fit(X_current[perms], y_current[perms], check_input=False)


def _sample_positive_indices(a, bins, size, random_instance=None, replace=True):
    """
    Draw selected number of elements from array, the elements are drawn uniformly from provided bins.

    Parameters
    ----------
    a : array-like
        array with elements, the elements must be within bin boundaries
    bins : array-like of ints
        array with bin boundaries, at least one bin is required
    size : int
        number of elements to draw
    random_instance: None | int | numpy.random.RandomState
        Random number generator. If None, then a new random seed is initialized . If int, then the value is used as random seed.
    replace : bool
        Elements are sampled w/ replacement if True, otherwise the sampling is performed w/o replacement. Note that if
        ``replace == False``, then at most ``min(size, n_samples)`` indices are sampled
    Returns
    -------
    indices : array with shape (size,)
        array with indices to sample elements from the array ``a``.
    """
    # 0 - initial checks
    n_bins = bins.shape[0] - 1
    assert n_bins > 0, "Expected at least one bin, got {}".format(n_bins)

    # validate that the elements are within the bounds
    if np.any(a[(bins.min() >= a) | (a > bins.max())]):
        raise ValueError('`a` must be within `bins` boundaries [{},{}]'.format(bins.min(), bins.max()))

    n_samples = a.shape[0]
    assert n_samples > 0, "Expected at least one sample, got {}".format(n_samples)

    if not replace:
        # we cannot draw more elements than `n_samples`, since sample without replacement
        size = min(size, n_samples)

    random_instance = check_random_state(random_instance)

    # 1 - figure out the bin for each variant, creating an array with mapping from `offset` -> `bins`
    variant_bin_idx = np.digitize(x=a, bins=bins, right=True)
    n_used_bins = len(np.unique(variant_bin_idx))

    # 2 - count number of variants present in individual bins
    bin_count = np.bincount(variant_bin_idx)[1:]

    # 3 - count number of variants that are located in variant's bin
    #   and use the number to calculate the probability of the variant being sampled
    n_variants_in_bin = bin_count[variant_bin_idx - 1]
    variant_proba = 1 / (n_used_bins * n_variants_in_bin)  # TODO - how about underflow?

    # 4 - sample the required number of variant indices using variant probabilities
    return random_instance.choice(n_samples, size=size, replace=replace, p=variant_proba)


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]
