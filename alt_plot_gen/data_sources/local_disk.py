
import pandas as pd

import os


def import_dataset():
    """
    return the raw dataset from local disk or cloud storage
    """

    path = os.path.join(
        os.environ.get("LOCAL_DATA_PATH"),
        "wiki_movie_plots_deduped.csv")

    try:

        df = pd.read_csv(path)  # read all rows

    except pd.errors.EmptyDataError:

        return None  # end of data

    return df
