import os
import json
import numpy as np
import pandas as pd

import pyarrow.parquet as pq
import pyarrow as pa
from cloudpickle import dump

import pyarrow.parquet as pq
import pyarrow as pa

import dask
import dask.dataframe as dd
from dask.distributed import Client

import lightgbm
import dask_lightgbm.core as dlgbm

from dask_ml.preprocessing import LabelEncoder
from dask_ml.model_selection import train_test_split

from typing import IO, AnyStr, Union, List, Optional, Tuple 

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_valid_test_splitter(
    context: Optional[MLClientCtx] = None,
    dask_client: Union[DataItem, str] = '',
    dask_key: str = '',
    label_column: str = '',
    categories: list = [],
    test_size: float = 0.1,
    train_val_split: float = 0.75,
    target_path: str = '',
    name: str = '',
    key: str = '',
    random_state = 1,
    lgb_params = {}
) -> None:
    """Split raw data input into train, validation and test sets.
    
    The following transformations are applied here:
    * transform the label column to binary
    * LabelEncode all categorical features and pickle encoders
    * split data three ways: train, vaslidation and test sets
    * the test set is set aside
    
    
    Note: this will replace any published cluster keys xtrain, ytrain, 
          xvalid, yvalid, xtest, ytest

    :param context:         the function context
    :param dask_client:     ('scheduler.json')
    :param dask_key:        key of source table on cluster
    :param label_column:    ground-truth (y) labels
    :param categories:      categorical variable columns
    :param test_size:       (0.1) test set size
    :param train_val_split: (0.75) Once the test set has been removed the 
                            training set gets this proportion.
    :param target_path:     folder location of files
    :param name:            model name
    :param key:              model key in artifact store
    :param random_state:    rng seed
    :param params:          lightgbm parameters
    """
    scheduler_file = os.path.join(target_path, str(dask_client))
    dask_client = Client(scheduler_file=scheduler_file)
    
    features = dask_client.datasets[dask_key]
    
    # generate labels
    labels = features.pop(label_column)
    labels = (labels.fillna(16) > 15)
    
    # features engineering
    features['CRSDepTime'] = features['CRSDepTime'].clip(upper=2399)
    features = features.categorize(columns=categories)
    feature_classes = dict()
    
    # Category encoding
    #before_encoding_shape = features.shape[1]
    
    for cat in categories:
        le = LabelEncoder()
        features[cat] = le.fit_transform(features[cat])
        fp = os.path.join(target_path, cat+'-encoding.pkl')
        # save this encoder
        dump(le, open(fp, 'wb'))
    
    #after_encoding_shape = features.shape[1]
    #context.logger.info(f'N FEATURES:\nbefore {before_encoding_shape}\nafter  {after_encoding_shape}')
    
    # One-hot encoding - todo
    
    
    
    
    # splits
    x, xtest, y, ytest = train_test_split(
        features,
        labels,
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True, 
        blockwise=None)
   
    xtrain, xvalid, ytrain, yvalid = train_test_split(
        x, y, 
        train_size=train_val_split, 
        random_state=random_state,
        shuffle=True, 
        blockwise=None)

    # save header
    f = os.path.join(target_path, 'header.pkl')
    dump(features.columns.values, open(f, 'wb'))
    context.log_artifact('header', target_path=f)
    
    # unpublish current dataset items
    current_ds = dask_client.list_datasets()
    
    for ds in current_ds:
        if ds in ['xtrain', 'ytrain', 'xvalid', 'yvalid']:
            print(f'dataset {ds} exists, unpublishing')
            dask_client.unpublish_dataset(ds)
    
    # publish new dataset items
    dask_client.datasets['xtrain'] = xtrain
    dask_client.datasets['ytrain'] = ytrain
    dask_client.datasets['xvalid'] = xvalid
    dask_client.datasets['yvalid'] = yvalid
    
    # set these aside, make them available as artifacts
    test_set = dd.multi.concat([xtest, ytest], axis=1)
    test_set = test_set.reset_index()
    
    fp = os.path.join(target_path, 'test_set')
    dd.to_parquet(test_set, fp, append=False, compute=True, write_index=False)
    context.log_artifact('test_set', target_path=fp)
    