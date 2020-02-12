# Dask

In this demo, we load a relatively _small_ big-data file (~120 mio rows, ~10 numerical features, 3 categories) and train a sample on a number of models on a dask cluster.  Although the entire dataset will be saved in step 1., the loaded sample in step 2. can be smaller or larger (set `with_replacement=True` and `frac > 1.0`) than the original dataset.  With the airlines data, setting a sample fraction of 0.01 returns > 1 million rows.

2 kinds of categorical variable encodings are available, `LabelEncoder` and `OneHotEncoder`.  The latter generates a large sparse features matrix.


## Install

* create a folder `'/User/repos'`
* in that folder clone https://github.com/yjb-ds/demos.git
* in Jupyter enter the  folder and run the sheets in sequence

## pipeline components
**[1. remote archive to local parquet](#1.%20remote%20archive%20to%20local%20parquet.ipynb)**<br>
a long-running routine that downloads the airlines archive, stores it as a parquet partitioned dataset and creates an artifact called **`airlines`** that points to the parquet folder<br>

**[2. parquet to dask cluster](#2.%20parquet%20to%20dask%20cluster.ipynb)**<br>
loads a sample from a parquet dataset into an existing dask cluster

* **[describe.py](#describe.py)**<br>
fire-and-forget utility that summarizes a table in the background and when completed stores the result as an artifact `table-summary`

could be used as template for any quick-and-dirty EDA or other experiment<br>

**3. generate train and test sets**<br>

* **[LabelEncoder](#3.%20generate%20train%20and%20test%20sets.ipynb)**<br>
convert categorical variables into numerical types
* **[OneHotEncoder]((#3.%20generate%20train%20and%20test%20sets-hotencode.ipynb)**)<br>
one-hot encode categorical variables

## Training models
**[4. lightgbm on a dask cluster](#4.%20lightgbm%20on%20dask%20cluster.ipynb)**<br>

**[4. gradient boosting using intel's daal4py](#4.%20gbt%20batch%20daal4py.ipynb)**<br>
**[4. xgboost on a dask cluster](#4.%20xgboost%20on%20dask%20cluster.ipynb)**<br>
**4. glm-logreg using dask-ml, xgboost or daal4py on a dask cluster]()**<br>
* #4.%20lightgbm%20on%20dask%20cluster.ipynb
Once loaded on the cluster, the dask scheduler's details are stored in the `scheduler` artifact and can be accessed either natively reading the scheduler's json config file, or by querying the mlrun database.  The latter may be quite convenient when there are multiple clusters running multiple pipleines and user experiments.

(wip - these are being refactored)
5. [evaluate]()
6. [deploy]()