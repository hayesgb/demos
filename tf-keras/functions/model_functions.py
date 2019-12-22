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

__all__ = [
    "log_context_model",
    "get_context_model",
    "log_context_model_json",
    "get_context_model_json",
]


def log_context_model(
    context: MLClientCtx,
    artifact_key: str = "",
    target_path: str = "",
    name: str = "",
    model: Any = None,
    labels: dict = {
        "framework": "tf-keras",
        "version": "2.0.0b1",
        "gpu": False,
        "model_type": "classifier",
    },
) -> None:
    """Pickle the model.
    
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

    :param context: context
    :param model:   model path
    :param name:    file name of the model

    Returns a model, typed here as `Any` since we can't really know
    in this simple example.
    """
    modelpath = os.path.join(str(model), name)
    model = joblib.load(open(modelpath, "rb"))
    return model


def log_context_model_json(model, target_path: str = "", name: str = "") -> None:
    """Save a keras model.

    :param model:       model to store
    :param target_path: destination of model file
    :param name:        model name
    """
    filepath = path.join(target_path, name)
    # architecture
    json_model = model.model.to_json()
    with open(filepath + ".json", "w") as fh:
        fh.write(json_model)
    # weights
    model.model.save_weights(filepath + "-weights.h5", overwrite=True)


def get_context_model_json(target_path: str = "", name: str = "model") -> None:
    """Get a saved keras model.
    """
    filepath = path.join(target_path, name)
    # architecture
    json_model = model.model.to_json()
    with open(path.join(target_path, name) + ".json", "w") as fh:
        fh.write(json_model)
    # weights
    model.model.save_weights(
        path.join(target_path, name) + "-weights.h5", overwrite=True
    )

