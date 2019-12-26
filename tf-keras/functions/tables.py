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
import io
from os import path, makedirs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mlrun.execution import MLClientCtx
from mlrun.artifacts import TableArtifact
from mlrun.datastore import DataItem


__all__ = ["get_context_table", "log_context_table"]


def get_context_table(ctxtable: DataItem) -> pd.DataFrame:
    """Get table from context.
    
    Convenience function to retrieve a table via a blob.
    
    :param ctxtable: table saved in the context, which needs
                     to be deserialized.
        
    In this demonstration tables are stored in parquet format and passed
    between steps as blobs.  We could also pass folder or file names
    in the context, which may be faster.
    
    Returns a pands DataFrame
    """
    blob = io.BytesIO(ctxtable.get())
    return pd.read_parquet(blob, engine="pyarrow")


def log_context_table(
    context: MLClientCtx,
    table: pd.DataFrame,
    target_path: str = "",
    key: str = "",
    overwrite: bool = False
) -> None:
    """Log a table through the context.
    
    The table is written as a parquet file, and its target
    path is saved in the context.
    
    :param context:     function context
    :param table:       the object we wish to store
    :param target_path: location (folder) of our DataItem
    :param key:         key of the object in the artifact store
    :param, overwrite:  overwrite if object exists at target_path/name
    """
#    makedirs(target_path, exist_ok=True)
    
    print(f"writing {target_path}")
    if overwrite:
        pq.write_table(pa.Table.from_pandas(table), target_path)

    context.log_artifact(key, target_path=target_path)
