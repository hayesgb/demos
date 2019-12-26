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
from urllib.request import urlretrieve

from mlrun.execution import MLClientCtx
from typing import (IO, 
                    AnyStr, 
                    TypeVar, 
                    Union, 
                    List, 
                    Tuple, Any)
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

from functions.tables import log_context_table

__all__ = ["arc_to_parquet"]


def arc_to_parquet(
    context: MLClientCtx,
    archive_url: Union[str, Path, IO[AnyStr]],
    header: Union[None, List[str]] = None,
    name: str = "",
    target_path: str = "",
    chunksize: int = 10_000,
    log_data: bool = True,
) -> None:
    """Open a file/object archive and save as a parquet file.
    
    Args:
    :param context:     function context
    :param archive_url: any valid string path consistent with the path variable
                        of pandas.read_csv. ncluding strings as file paths, as urls, 
                        pathlib.Path objects, etc...
    :param header:      column names
    :param target_path: destination folder of table
    :param chunksize:   (0) row size retrieved per iteration
    :param log_data:    (True) if True, log the data so that it is available
                        at the next step
    """
    os.makedirs(target_path, exist_ok=True)

    if not name.endswith(".parquet"):
        name += ".parquet"

    dest_path = os.path.join(target_path, name)

    if not os.path.isfile(dest_path):
        context.logger.info("destination file does not exist, downloading")
        pqwriter = None
        for i, df in enumerate(
            pd.read_csv(archive_url, chunksize=chunksize, names=header)
        ):
            table = pa.Table.from_pandas(df)
            if i == 0:
                pqwriter = pq.ParquetWriter(dest_path, table.schema)
            pqwriter.write_table(table)

        if pqwriter:
            pqwriter.close()

        context.logger.info(f"saved table to {dest_path}")
    else:
        context.logger.info("destination file exists")

    if log_data:
        context.logger.info("logging data to context")
        df = pq.read_table(dest_path).to_pandas()
        log_context_table(context, df, target_path, name)

