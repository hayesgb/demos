# Dask

## acquire data and load dask cluster
**[1. remote archive to local parquet](#1.%20remote%20archive%20to%20local%20parquet.ipynb)**<br>
**[2. parquet to dask cluster](#2.%20parquet%20to%20dask%20cluster.ipynb)**<br>
* **[describe.py](#describe.py)** a fire-and-forget utility that simply summarizes a table, could be used as template for any quick-and-dirty EDA or other experiment.<br>

## feature engineering and train & test sets
**[3. generate train and test sets](#3.%20generate%20train%20and%20test%20sets.ipynb)**<br>

## Training models
**[4. lightgbm on a dask cluster](#4.%20lightgbm%20on%20dask%20cluster.ipynb)**<br>
**[4. gradient boosting using intel's daal4py](#4.%20lightgbm%20on%20dask%20cluster.ipynb)**<br>
**[4. xgboost on a dask cluster](#4.%20lightgbm%20on%20dask%20cluster.ipynb)**<br>
**[4. glm-logreg using dask-ml, xgboost or daal4py on a dask cluster](#4.%20lightgbm%20on%20dask%20cluster.ipynb)**<br>

Once loaded on the cluster, the dask scheduler's details are stored
in the `scheduler` artifact and can be accessed either natively (by copying the scheduler's json config and instantiating a client), or by querying the mlrun database.  The latter may be quite convenient when there are multiple clusters running multiple pipleines and user experiments

(wip - these are being refactored)
5. [evaluate]()
6. [deploy]()