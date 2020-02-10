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
from pathlib import Path
import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, wait

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from typing import IO, AnyStr, Union, List, Optional

def parquet_to_dask(
    context: MLClientCtx,
    parquet_url: Union[DataItem, str, Path, IO[AnyStr]],
    sample = 0.3,
    shards: int = 4,
    threads_per: int = 4,
    memory_limit: str = '2GB',
    dask_key: str = '',
    target_path: str = '',
    scheduler_filename: str = 'scheduler.json',
    scheduler_key: str = 'scheduler'
) -> None:
    """Load parquet dataset into dask cluster
    
    If no cluster is found loads a new one and persist the data to it. It
    should not be necessary to create a new cluster when the function
    is run as a 'dask' job.
    
    Note that only `sample` fraction of the data will be managed by the cluster
    
    :param context:         the function context
    :param parquet_url:     url of the parquet file or partitioned dataset as either
                            artifact DataItem, string, or path object (see pandas read_csv)
    :param sample:           sample size as fraction 0.0-1.0
    :param shards:          number of workers to launch
    :param threads_per:     number of threads per worker
    """
    if hasattr(context, 'dask_client'):
        context.logger.info('found cluster...')
        dask_client = context.dask_client
    else:
        context.logger.info('starting new cluster...')
        cluster = LocalCluster(n_workers=shards,
                               threads_per_worker=threads_per,
                               processes=True,
                               memory_limit=memory_limit)
        dask_client = Client(cluster)

    context.logger.info(dask_client)
 
    df = dd.read_parquet(parquet_url).sample(frac=sample)
    df = df.sample(frac=sample)

    context.logger.info(f'column header {df.columns.values}')
    
    #df = dask_client.persist(df)
    dask_client.datasets[dask_key] = df
        
     # share the scheduler
    filepath = os.path.join(target_path, scheduler_filename)
    dask_client.write_scheduler_file(filepath)
    context.log_artifact(scheduler_key, target_path=filepath)
  