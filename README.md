# OffsetImbalanceAwareRandomForest

Offset and imbalance-aware random forest classifier.

This classifier is an extension of Scikit-learn's `RandomForestClassifier`, with adjusted sampling process of the 
positive examples. The sampling is aware of the imbalance in distribution of examples with respect to an `offset` (feature).
> In our situation, this represents distance of a genomic variant from the start/end of an exon  

During training, positive examples are sampled uniformly from the pre-defined `bins` (hyperparameter, offset regions),
ensuring that an equal number of examples is sampled from each bin. Negative examples are then randomly extracted 
according to the `class_ratio` (hyperparameter). 

Predictions are performed in the same way as in a regular random forest. 

## How to install

In order to install the code into the current Python environment, run:

```bash
cd OffsetImbalanceAwareRandomForest
pip install .
```

## How to use

Run the code below in order to train the `OiasRandomForestClassifier` on the Iris dataset.

```bash
from sklearn.model_selection import train_test_split
from oias import OiasRandomForestClassifier, load_imbalanced_iris

# prepare toy dataset
X, y, offset = load_imbalanced_iris()
X_train, X_test, y_train, y_test, offset_train, offset_test = train_test_split(X, y, offset, 
                                                                               test_size=.25, shuffle=True, 
                                                                               stratify=y, random_state=123)


# train the OIAS random forest
bins = [-10, 0, 2, 10]
clf = OiasRandomForestClassifier(class_ratio=2, bins=bins)
y_pred = clf.fit(X_train, y_train, offsets=offset_train).predict(X_test)


# evaluate
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred, adjusted=True)
```

