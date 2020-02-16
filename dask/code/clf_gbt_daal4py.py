# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
from cloudpickle import dump
from pathlib import Path
import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client
import daal4py as d4p

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact

from typing import IO, AnyStr, Union, List, Optional, Tuple

def clf_gbt_daal4py(
    context: MLClientCtx,
    target_path: str,
    name: str,
    key: str,
    n_classes: int,
    categories: List[str],
    test_size: float = 0.1,
    gbt_params = {    
        'fptype':                       'float',
        'maxTreeDepth':                 3,
        'minSplitLoss':                 0.1,
        'shrinkage':                    0.1,
        'observationsPerTreeFraction':  1,
        'lambda_':                      1,
        'maxBins':                      256,
        'featuresPerNode':              0,
        'minBinSize':                   5,
        'minObservationsInLeafNode':    1,
        'nClasses':                     2
    }
) -> None:
    """Train Logistic Regression classifier
    
    :param context:         the function context
    :param dask_client:     dask client scheduler json file
    :param train_set:       training (features, labels) tuple
    :param valid_set:       validation (features, labels) tuple
    :param target_path:     destimation folder for training artifacts
    :param name:            model name
    :param key:             key of model in artifact store
    :param params:          glm parameters
    """
#     scheduler_file = os.path.join(target_path, str(dask_client))
#     dask_client = Client(scheduler_file=scheduler_file)
    
#     xtrain = dask_client.datasets[train_set[0]]
#     ytrain = dask_client.datasets[train_set[1]]
    
#     xvalid = dask_client.datasets[valid_set[0]]
#     yvalid = dask_client.datasets[valid_set[1]]

    # temporary hack
    X = pd.read_parquet('/User/repos/demos/dask/dataset/partitions')
    y = X.pop('ArrDelay')
    X['CRSDepTime'] = X['CRSDepTime'].clip(upper=2399)
    y = (y.fillna(16) > 15)

    for cat in categories:
        le = LabelEncoder()
        X[cat] = le.fit_transform(X[cat])
        # loses this in the transform?
        X[cat] = X[cat].astype('category')
        fp = os.path.join(target_path, cat+'-encoding.pkl')
        # save this encoder
        dump(le, open(fp, 'wb'))
    
    xtrain, xtest, ytrain, ytest = \
        train_test_split(X,
                         y,
                         test_size=test_size,
                         random_state=1)
    
    # One-hot encoding - todo
 
    clf = d4p.gbt_classification_training(**gbt_params)
    clf_result = clf.compute(xtrain, ytrain[:, np.newaxis])

#     filepath = os.path.join(target_path, name)
#     dump(clf, open(filepath, 'wb'))
#     context.log_artifact(key, target_path=filepath)
    
#     filepath = os.path.join(target_path, 'results-' + name)
#     dump(clf_result, open(filepath, 'wb'))
#     context.log_artifact(key, target_path=filepath)