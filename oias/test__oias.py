import pickle
import unittest
from collections import namedtuple

import numpy as np
import numpy.testing as nt
from pkg_resources import resource_filename
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from oias import load_imbalanced_iris, OiasRandomForestClassifier
from ._oias import _sample_positive_indices, _parallel_build_trees

X, y = load_iris(return_X_y=True)

MockOiasRandomForestClassifier = namedtuple('MockOiasRandomForestClassifier',
                                            ('bins', 'bootstrap', 'class_ratio'))


class TestOiasRandomForestFitting(unittest.TestCase):

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

        y_proba = clf[0].predict_proba(X[[0, 53, 149]])

        nt.assert_almost_equal(y_proba, np.array([[0.1666667, 0.8333333],
                                                  [1., 0.],
                                                  [0.1666667, 0.8333333]]),
                               decimal=7)


class TestOiasRandomForestPrediction(unittest.TestCase):

    def setUp(self) -> None:
        self.X, self.y, self.offsets = load_imbalanced_iris()
        self.X_train, self.X_test, self.y_train, self.y_test, self.offsets_train, self.offsets_test = train_test_split(
            self.X, self.y, self.offsets, train_size=.2, stratify=self.y, shuffle=True, random_state=123)

        self.bins = np.array([-10, 0, 2, 10])

        res_path = resource_filename(__name__, 'test_data/model.v0.1.ser')
        with open(res_path, 'rb') as fh:
            self.clf = pickle.load(fh)

    def test_predict_proba(self):
        """Test the `predict_proba` method."""
        y_proba = self.clf.predict_proba(self.X_test)
        nt.assert_almost_equal(y_proba[[1, 3, 4]], np.array([[.975063551, .0249364486],
                                                             [.896291270, .103708730],
                                                             [.975063551, .0249364486]]),
                               decimal=7)

    def test_predict(self):
        """Test the `predict` method."""
        y_pred = self.clf.predict(self.X_test)
        nt.assert_array_equal(y_pred[[1, 3, 8]], np.array([0, 0, 1]))

    @unittest.skip
    def test_train_model(self):
        """This test serves to train & serialize the model that is used in tests above."""
        params = {
            'class_ratio': 2., 'bins': self.bins,
            'criterion': 'gini',
            'max_depth': 3, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.,
            'max_features': 'auto', 'max_leaf_nodes': 2,
            'min_impurity_decrease': 0., 'min_impurity_split': None,
            'random_state': 123, 'ccp_alpha': 0.
        }

        clf = OiasRandomForestClassifier(**params).fit(self.X_train, self.y_train, offsets=self.offsets_train)
        with open(resource_filename(__name__, 'test_data/model.v0.1.ser'), 'wb') as fh:
            pickle.dump(clf, fh)


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
        tree_ = tree.tree_
        nt.assert_array_equal(tree_.feature, np.array([2, -2, -2]))
        nt.assert_array_equal(tree_.value, np.array([[[4, 2]], [[4, 0]], [[0, 2]]]))
        nt.assert_array_equal(tree_.value, np.array([[[4, 2]], [[4, 0]], [[0, 2]]]))
        nt.assert_array_equal(tree_.value, np.array([[[4, 2]], [[4, 0]], [[0, 2]]]))
        nt.assert_array_equal(tree_.children_left, np.array([1, -1, -1]))
        nt.assert_array_equal(tree_.children_right, np.array([2, -1, -1]))


class TestSampling(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.RandomState(seed=10)

    def test__sample_positive_indices_w_replacement(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-9, -1, 2, 5, 2, 1, 2, 1, 6, 10], dtype=np.int)

        indices = _sample_positive_indices(a=offsets, bins=bins, size=10, random_instance=self.rng, replace=True)

        nt.assert_array_equal(np.array([7, 0, 5, 7, 3, 1, 1, 7, 1, 0]), indices)

    def test__sample_positive_indices_wo_replacement(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9], dtype=np.int)

        indices = _sample_positive_indices(a=offsets, bins=bins, size=5, random_instance=self.rng, replace=False)

        nt.assert_array_equal(np.array([8, 0, 7, 4, 1]), indices)

    def test__sample_positive_indices_wo_replacement__size_too_big(self):
        bins = np.array([-10, 0, 2, 10])
        offsets = np.array([-5, -1, 2, 1, 2, 1, 2, 1, 4, 9], dtype=np.int)

        indices = _sample_positive_indices(a=offsets, bins=bins, size=11, random_instance=self.rng, replace=False)

        # we expect to receive 10 samples only
        nt.assert_array_equal(np.array([8, 0, 7, 4, 1, 9, 2, 6, 5, 3]), indices)

        # we expect to see all possible indices
        nt.assert_array_equal(np.arange(10), np.sort(indices))

    def test__sample_positive_indices_wo_replacement__a_outside_of_boundaries(self):
        bins = np.array([-10, 0, 2, 10])
        with self.assertRaises(ValueError):  # below
            _sample_positive_indices(a=np.array([-10, -1]), bins=bins, size=10, random_instance=self.rng, replace=True)

        with self.assertRaises(ValueError):  # above
            _sample_positive_indices(a=np.array([-9, 11]), bins=bins, size=10, random_instance=self.rng, replace=True)
