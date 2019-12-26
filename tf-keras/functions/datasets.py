# Copyright 2019 Iguazio
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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import Optional, List
from sklearn.datasets import make_classification

from mlrun.execution import MLClientCtx

from functions.tables import log_context_table

__all__ = ["create_binary_classification"]

def create_binary_classification(
    context: MLClientCtx = None,
    n_samples: int = 1_000_000, 
    m_features: int = 20,
    features_hdr: Optional[List[str]] = None,
    weight: float = 0.50, 
    target_path: str = '',
    key: str = ''
):
    """Create a binary classification sample dataset and save.

    File name will be 'iguazio-binary-{n_samples-m_features}'.

    test:
    >>> n, m = 100_000, 20
    >>> fn = create_binary_classification(None, n, m, 0.5, name='test.csv')
    >>> assert fn = 'test.csv'
    >>> testdf = pd.read_csv('test.csv')
    >>> assert testdf.shape == (n, m)
    
    TODO: 
        - refactor so that all sklearn params can be modified
        - multiclass
    
    :param context:       function context
    :param n_samples:     number of rows/samples
    :param m_features:    number of cols/features
    :param features_hdr:  header for features array
    :param weights:       fraction of sample (neg)
    :param target_path:   destimation for file
    :param key:           key of data in artifact store
    
    Returns filename of created data (includes path).
    """
    # create file name and check directories
    name = f'simdata-{n_samples:0.0e}X{m_features}.parquet'.replace('+','')
    
    filename = os.path.join(target_path, name)

    features, labels = make_classification(n_samples=n_samples, 
                                           n_features=m_features, 
                                           n_informative=5, 
                                           n_classes=2, 
                                           n_clusters_per_class=1, 
                                           weights=[weight], # False
                                           shuffle=True,
                                           random_state=1)
    
    # make dataframes, add column names, concatenate (X, y)
    X = pd.DataFrame(features)
    if not features_hdr:
        X.columns=['feat_'+str(x) for x in range(m_features)]
    else:
        X.columns=features_hdr
    
    y = pd.DataFrame(labels, columns=['labels'])
    data = pd.concat([X, y], axis=1)
    
    log_context_table(context, data, filename, 'features', overwrite=True)
    
    return filename