"""Utility methods from scikit-learn."""

import numbers
from distutils.version import LooseVersion

import numpy as np
from joblib import effective_n_jobs
from sklearn.utils import check_array


def _check_sample_offsets(offsets, X):
    n_samples = _num_samples(X)

    offsets = check_array(offsets, accept_sparse=False, ensure_2d=False, dtype=np.int64, order="C")

    if offsets.ndim != 1:
        raise ValueError("Offsets must be 1D array")

    if offsets.shape != (n_samples,):
        raise ValueError("offsets.shape == {}, expected {}!"
                         .format(offsets.shape, (n_samples,)))

    return offsets


# sklearn/utils/validation.py
def _check_sample_weight(sample_weight, X, dtype=None):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
       Input sample weights.

    X : nd-array, list or sparse matrix
        Input data.

    dtype: dtype
       dtype of the validated `sample_weight`.
       If None, and the input `sample_weight` is an array, the dtype of the
       input is preserved; otherwise an array with the default numpy dtype
       is be allocated.  If `dtype` is not one of `float32`, `float64`,
       `None`, the output will be of dtype `float64`.

    Returns
    -------
    sample_weight : ndarray, shape (n_samples,)
       Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None or isinstance(sample_weight, numbers.Number):
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=dtype)
        else:
            sample_weight = np.full(n_samples, sample_weight,
                                    dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
            order="C"
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight.shape == {}, expected {}!"
                             .format(sample_weight.shape, (n_samples,)))
    return sample_weight


# sklearn/utils/validation.py
def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)


# sklearn/utils/validation.py
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def compute_sample_weight(class_weight, y, indices=None):
    """Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, list of dicts, "balanced", or None, optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data:
        ``n_samples / (n_classes * np.bincount(y))``.

        For multi-output, the weights of each column of y will be multiplied.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Array of original class labels per sample.

    indices : array-like, shape (n_subsample,), or None
        Array of indices to be used in a subsample. Can be of length less than
        n_samples in the case of a subsample, or equal to n_samples in the
        case of a bootstrap subsample with repeated indices. If None, the
        sample weight will be calculated over the full sample. Only "balanced"
        is supported for class_weight if this is provided.

    Returns
    -------
    sample_weight_vect : ndarray, shape (n_samples,)
        Array with sample weights as applied to the original y
    """

    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))
    n_outputs = y.shape[1]

    if isinstance(class_weight, str):
        if class_weight not in ['balanced']:
            raise ValueError('The only valid preset for class_weight is '
                             '"balanced". Given "%s".' % class_weight)
    elif (indices is not None and
          not isinstance(class_weight, str)):
        raise ValueError('The only valid class_weight for subsampling is '
                         '"balanced". Given "%s".' % class_weight)
    elif n_outputs > 1:
        if (not hasattr(class_weight, "__iter__") or
                isinstance(class_weight, dict)):
            raise ValueError("For multi-output, class_weight should be a "
                             "list of dicts, or a valid string.")
        if len(class_weight) != n_outputs:
            raise ValueError("For multi-output, number of elements in "
                             "class_weight should match number of outputs.")

    expanded_class_weight = []
    for k in range(n_outputs):

        y_full = y[:, k]
        classes_full = np.unique(y_full)
        classes_missing = None

        if class_weight == 'balanced' or n_outputs == 1:
            class_weight_k = class_weight
        else:
            class_weight_k = class_weight[k]

        if indices is not None:
            # Get class weights for the subsample, covering all classes in
            # case some labels that were present in the original data are
            # missing from the sample.
            y_subsample = y[indices, k]
            classes_subsample = np.unique(y_subsample)

            weight_k = np.take(compute_class_weight(class_weight_k,
                                                    classes_subsample,
                                                    y_subsample),
                               np.searchsorted(classes_subsample,
                                               classes_full),
                               mode='clip')

            classes_missing = set(classes_full) - set(classes_subsample)
        else:
            weight_k = compute_class_weight(class_weight_k,
                                            classes_full,
                                            y_full)

        weight_k = weight_k[np.searchsorted(classes_full, y_full)]

        if classes_missing:
            # Make missing classes' weight zero
            weight_k[np.in1d(y_full, list(classes_missing))] = 0.

        expanded_class_weight.append(weight_k)

    expanded_class_weight = np.prod(expanded_class_weight,
                                    axis=0,
                                    dtype=np.float64)

    return expanded_class_weight


def compute_class_weight(class_weight, classes, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    """
    # Import error caused by circular imports.
    from sklearn.preprocessing import LabelEncoder

    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can "
                         "be in y")
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y) / (len(le.classes_) *
                               np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            else:
                weight[i] = class_weight[c]

    return weight


# sklearn/utils/fixes.py
def _joblib_parallel_args(**kwargs):
    """Set joblib.Parallel arguments in a compatible way for 0.11 and 0.12+

    For joblib 0.11 this maps both ``prefer`` and ``require`` parameters to
    a specific ``backend``.

    Parameters
    ----------

    prefer : str in {'processes', 'threads'} or None
        Soft hint to choose the default backend if no specific backend
        was selected with the parallel_backend context manager.

    require : 'sharedmem' or None
        Hard condstraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based even
        if the user asked for a non-thread based backend with
        parallel_backend.

    See joblib.Parallel documentation for more details
    """
    import joblib

    if joblib.__version__ >= LooseVersion('0.12'):
        return kwargs

    extra_args = set(kwargs.keys()).difference({'prefer', 'require'})
    if extra_args:
        raise NotImplementedError('unhandled arguments %s with joblib %s'
                                  % (list(extra_args), joblib.__version__))
    args = {}
    if 'prefer' in kwargs:
        prefer = kwargs['prefer']
        if prefer not in ['threads', 'processes', None]:
            raise ValueError('prefer=%s is not supported' % prefer)
        args['backend'] = {'threads': 'threading',
                           'processes': 'multiprocessing',
                           None: None}[prefer]

    if 'require' in kwargs:
        require = kwargs['require']
        if require not in [None, 'sharedmem']:
            raise ValueError('require=%s is not supported' % require)
        if require == 'sharedmem':
            args['backend'] = 'threading'
    return args

# sklearn
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
