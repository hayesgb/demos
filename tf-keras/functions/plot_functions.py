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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact

matplotlib.rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

__all__ = ["plot_validation", "plot_roc"]


def plot_validation(
    context: MLClientCtx,
    train_loss: np.ndarray,
    valid_loss: np.ndarray,
    artifact_key: str = "",
    title: str = "training validation results",
    xlabel: str = "epoch",
    ylabel: str = "logloss",
    fmt: str = "png",
) -> None:
    """Plot train and validation loss curves.
    
    These curves represent the training round losses from the training
    and validation sets. The actual type of loss curve depends on the
    algorithm and selcted metrics.

    :param context:         The context.
    :param artifact_key:    The plot"s key in the context.
    :param train_loss:      Vector of loss metric estimates for training set.
    :param valid_loss:      Predictions given a test sample and an
                            estimated model.
    :param title:           Plot title.
    :param xlabel:          X-axis label.
    :param ylabel:          Y-axis label.
    :param fmt:             The file image format (png, jpg, ...), and the
                            saved file extension.
    """
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title("")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(["train", "valid"])
    context.log_artifact(PlotArtifact(artifact_key, body=plt.gcf()))


def plot_roc(
    context: MLClientCtx,
    artifact_key: str,
    ytest: np.ndarray,
    ypred: np.ndarray,
    title: str = "roc curve",
    xlabel: str = "false positive rate",
    ylabel: str = "true positive rate",
    fmt: str = "png",
) -> matplotlib.figure.Figure:
    """Plot an ROC curve.
    
    :param context:         The context.
    :param artifact_key:    The plot"s key in the context.
    :param ytest:           Ground-truth labels.
    :param ypred:           Predictions given a test sample and 
                            an estimated model.
    :param title:           Plot title.
    :param xlabel:          X-axis label (not tick labels).
    :param ylabel:          Y-axis label (not tick labels).
    :param fmt:             The file image format (png, jpg, ...),
                            and the saved file extension.
    """
    fpr_xg, tpr_xg, _ = roc_curve(ytest, ypred)

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr_xg, tpr_xg, label="tf-keras")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    context.log_artifact(PlotArtifact(artifact_key, body=plt.gcf()))

