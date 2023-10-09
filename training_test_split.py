"""
a script to split the data downloaded from automl challenge website
https://automl.chalearn.org/data
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


dataset_name = ["christine", "jasmine", "madeline", "philippine", "sylvine"] [4]
dataset_dir = Path.cwd() / "datasets" / dataset_name

feature_dir = dataset_dir / (dataset_name + "_train.data")
label_dir = dataset_dir / (dataset_name + "_train.solution")

features = pd.read_csv(str(feature_dir), header=None, delim_whitespace=True).to_numpy()
labels = pd.read_csv(str(label_dir), header=None, delim_whitespace=True).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.33,
                                                    stratify=labels,
                                                    random_state=42)

pd.DataFrame(X_train).to_csv(str(dataset_dir / "features_train.data"), header=None, index=None, sep=" ")
pd.DataFrame(y_train).to_csv(str(dataset_dir / "labels_train.data"), header=None, index=None, sep=" ")

pd.DataFrame(X_test).to_csv(str(dataset_dir / "features_test.data"), header=None, index=None, sep=" ")
pd.DataFrame(y_test).to_csv(str(dataset_dir / "labels_test.data"), header=None, index=None, sep=" ")


