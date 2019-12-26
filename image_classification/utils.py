import os
import zipfile
import json
from tempfile import mktemp
import pandas as pd
from mlrun.artifacts import TableArtifact
from mlrun.execution import MLClientCtx


def open_archive(
    context: MLClientCtx, target_dir: str = "content", archive_url: str = ""
):
    """Open a file/object archive into a target directory
    
    :param context:     function context
    :param target_dir:  destination of file(s)
    :param archive_url: source location as url.
    """
    os.makedirs(target_dir, exist_ok=True)
    context.logger.info("Verified directories")

    context.logger.info("Extracting zip")
    with zipfile.ZipFile(archive_url, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    context.logger.info(f"extracted archive to {target_dir}")
    context.log_artifact("content", target_path=target_dir)


def categories_map_builder(
    context: MLClientCtx,
    source_dir: str = "",
    df_filename: str = "file_categories_df.csv",
    map_filename: str = "categories_map.json",
):
    """Create DataFrame and label images from folder
    
    :param context:      MLClientXtx,
    :param source_dir:   location of folder
    :param map_filename: filename format <category>.NN.jpg
    """

    filenames = [file for file in os.listdir(source_dir) if file.endswith(".jpg")]
    categories = []

    for filename in filenames:
        category = filename.split(".")[0]
        categories.append(category)

    df = pd.DataFrame({
        "filename": filenames, 
        "category": categories})
    df["category"] = df["category"].astype("str")

    categories = df.category.unique()
    categories = {i: category for i, category in enumerate(categories)}
    with open(os.path.join(context.out_path, map_filename), "w") as f:
        f.write(json.dumps(categories))

    context.logger.info(categories)
    context.log_artifact("categories_map", src_path=map_filename)
    context.log_artifact(TableArtifact("file_categories", df=df, src_path=df_filename))
