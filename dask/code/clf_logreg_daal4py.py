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

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact

from typing import IO, AnyStr, Union, List, Optional, Tuple

def clf_logreg_daal4py(
    context: MLClientCtx,
    dask_client: Union[DataItem, str],
    train_set: Tuple[str, str],
    valid_set: Tuple[str, str],
    target_path: str,
    name: str,
    key: str,
    params = {'nClasses':2, 'interceptFlag' : True},
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
    scheduler_file = os.path.join(target_path, str(dask_client))
    dask_client = Client(scheduler_file=scheduler_file)
    
    xtrain = dask_client.datasets[train_set[0]]
    ytrain = dask_client.datasets[train_set[1]]
    xtrain = dd.concat([xtrain, ytrain], axis=1).dropna().compute()
    ytrain = xtrain.pop(ytrain.name)
    
    xvalid = dask_client.datasets[valid_set[0]]
    yvalid = dask_client.datasets[valid_set[1]]
    
    try:
        clf = d4p.logistic_regression_training(**params)

        clf_result = clf.compute(xtrain.values, ytrain.values)

        filepath = os.path.join(target_path, name)
        dump(clf, open(filepath, 'wb'))
        context.log_artifact(key, target_path=filepath)

        filepath = os.path.join(target_path, 'results-' + name)
        dump(clf_result, open(filepath, 'wb'))
        context.log_artifact(key, target_path=filepath)
    except Exception as e:
        print(f'FAILED {e}')
        