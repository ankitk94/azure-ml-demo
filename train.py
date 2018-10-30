# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from sklearn.externals import joblib
import os
import numpy as np
import pickle

os.makedirs('./outputs', exist_ok=True)

run = Run.get_submitted_run()

X, y = load_diabetes(return_X_y=True)

run = Run.get_submitted_run()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

reg = Ridge(alpha=0.65)
reg.fit(data["train"]["X"], data["train"]["y"])
model_path = 'model.pkl'
f = open(model_path, 'wb')
pickle.dump(reg, f)



with open(model_path, "wb") as file:
    from sklearn.externals import joblib
    joblib.dump(reg, file)
run.upload_file(model_path,  model_path)
os.remove(model_path)
