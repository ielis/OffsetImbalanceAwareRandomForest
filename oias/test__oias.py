import unittest
from collections import namedtuple

import numpy as np
import numpy.testing as nt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from ._oias import _sample_positive_indices, _parallel_build_trees, OiasRandomForestClassifier

X, y = load_iris(return_X_y=True)

MockOiasRandomForestClassifier = namedtuple('MockOiasRandomForestClassifier',
                                            ('bins', 'bootstrap', 'class_ratio'))


class TestOiasRandomForest(unittest.TestCase):

    def setUp(self) -> None:
        bins = np.array([-10, 0, 2, 10])
        self.params = {
            'class_ratio': 2, 'bins': bins,
            'criterion': 'gini',
            'max_depth': 3, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.,
            'max_features': 'auto', 'max_leaf_nodes': 2,
            'min_impurity_decrease': 0., 'min_impurity_split': None,
            'random_state': 123, 'ccp_alpha': 0.
        }

        train_indices = np.array([51, 52, 53, 54, 55,  # class 1
                                  101, 102, 103, 104, 105])  # class 2

        self.X = X[train_indices]
        self.y = y[train_indices]

    def test_fit(self):
        clf = OiasRandomForestClassifier(**self.params)
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9])
        clf.fit(self.X, self.y, offsets=offsets, pos_label=1)

        self.assertEqual(100, len(clf.estimators_))
        self.assertEqual(4, clf.n_features_)
        self.assertEqual(1, clf.n_outputs_)

        proba = clf[0].predict_proba(X[[0, 53, 149]])
        nt.assert_almost_equal(proba, np.array([[.0625, .9375], [.64285714, .35714286], [.0625, .9375]]), decimal=7)


class TestTreeBuilding(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.RandomState(123)
        train_indices = np.array([1, 2, 3, 4, 5,  # class 0
                                  101, 102, 103, 104, 105])  # class 2
        self.X = X[train_indices]
        self.y = y[train_indices].reshape(-1, 1)

    def test__parallel_build_trees(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9], dtype=np.int)

        forest = MockOiasRandomForestClassifier(bins=bins, bootstrap=True, class_ratio=2)

        n_pos_samples = 2
        pos_label = 2
        tree = _parallel_build_trees(DecisionTreeClassifier(random_state=self.rng), forest,
                                     self.X, self.y, offsets, n_pos_samples, pos_label)
        print(tree)


class TestSampling(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.RandomState(seed=10)

    def test__sample_positive_indices_w_replacement(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9], dtype=np.int)

        indices = _sample_positive_indices(a=offsets, bins=bins, size=10, random_instance=self.rng, replace=True)

        nt.assert_array_equal(np.array([7, 0, 5, 6, 3, 1, 1, 7, 1, 0]), indices)

    def test__sample_positive_indices_wo_replacement(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9], dtype=np.int)

        indices = _sample_positive_indices(a=offsets, bins=bins, size=5, random_instance=self.rng, replace=False)

        nt.assert_array_equal(np.array([7, 0, 5, 6, 3]), indices)

    def test__sample_positive_indices_wo_replacement__size_too_big(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9], dtype=np.int)

        with self.assertRaises(ValueError):
            _sample_positive_indices(a=offsets, bins=bins, size=11, random_instance=self.rng, replace=False)
