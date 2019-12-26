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
import joblib
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from typing import Any
from os import path, makedirs
import importlib

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from functions.tables import log_context_table

__all__ = [
    #"log_context_model",
    #"get_context_model",
    "classifier_gen",
    "METRICS",
    "FeaturesEngineer",
    "Classifier",
    "class_instance"
]

def log_context_model(
    context: MLClientCtx,
    model: Any = None,
    artifact_key: str = "",
    target_path: str = "",
    name: str = "",
    labels: dict = {
        "framework": "tf-keras",
        "version": "2.0.0b1",
        "gpu": False,
        "model_type": "classifier",
    },
) -> None:
    """Pickle the model.
    
    DEPRECIATE
    
    Uses joblib to pickle a model at the `target_path` using `name`.
    This has only been tested on scikit-learn, XGBoost and
    LightGBM models.

    :param context:         context
    :param artifact_key:    name for the model as an arifact
    :param target_path:     destination path of the model
    :param name:            file name of the model
    :param model:           the estimated model
    :param labels:          artifact labels
    """
    makedirs(target_path, exist_ok=True)
    file_path = path.join(target_path, name)
    joblib.dump(model, open(file_path, "wb"))
    context.log_artifact(artifact_key, target_path=target_path, labels=labels)


def get_context_model(
    context: MLClientCtx, model: DataItem = "", name: str = ""
) -> Any:
    """Get a model from the context.
    
    DEPRECIATE
    
    :param context: context
    :param model:   model path
    :param name:    file name of the model

    Returns a model, typed here as `Any` since we can't really know
    in this simple example.
    """
    modelpath = os.path.join(str(model), name)
    model = joblib.load(open(modelpath, "rb"))
    return model

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# generate a simple classifier for this dataset
def classifier_gen(
    metrics = METRICS, 
    output_bias=None,
    m_features: int = 20,
    dropout: float = 0.5
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(m_features,)),
      keras.layers.Dropout(dropout),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

Classifier = KerasClassifier(build_fn=classifier_gen)    

class FeaturesEngineer(BaseEstimator, TransformerMixin):
    """Engineer features from raw input.

    A standard transformer mixin that can be inserted into a scikit learn Pipeline.
    
    To use, 
    >>> ffg = FeaturesEngineer()
    >>> ffg.fit(X)
    >>> x_transformed = ffg.transform(X)
    or
    >>> ffg = FeaturesEngineer()
    >>> x_transformed = ffg.fit_transform(X)
    
    In a pipeline:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> transformers = [('feature_gen', FeaturesEngineerFeature()), 
                        ('scaler', StandardScaler())]
    >>> transformer_pipe = Pipeline(transformers)
    """
    def __init__(
        self,
        headers: str,
        target_path: str,
        key: str
    ):
        """initialize method.

        :param header:       in case the header gets lost (put in context)
        :param target_path:  destination folder for features
        :param name:         file name for features array
        :param key:          key name for features in context
        """
        self.target_path = target_path
        self.key = key
    
    def fit(self, X, y=None):
        """fit is unused here
        """
        return self

    def transform(self, X, y=None):
        """Transform raw input features a preprocessing step.
        
        :param X: Raw input features, as a pandas Dataframe 
        
        Returns a cleaned DataFrame of features.
        """
        x = X.copy()
        
        # do some cool feature engineering:here we replace by a N(2,2) series
        m = 2.0
        s = 2.0
        n, f = x.shape
        print(n, f)
        
        x.values[:, f-1] = np.random.normal(m, s, n)
        
        # could also save copy of features for further exploration
        #log_context_table(context, pd.DataFrame(x), self.target_path+'/features.parquet', self.key, overwrite=True)
        
        x = x.astype('float')
        
        return x
    

def class_instance(module_class: str):
    """Instantiate a class from strings
    
    For example, to instantiate a sklearn StandardScaler,
    use:
    >>> ss = class_instance('sklearn.preprocessing.data.StandardScaler')
     
    :param module_class:   module and class name
    
    """
    splits = module_class.split('.')
    module = '.'.join(splits[:-1])
    aclass = splits[-1]

    module = importlib.import_module(module)
    class_ = getattr(module, aclass)
    return class_    