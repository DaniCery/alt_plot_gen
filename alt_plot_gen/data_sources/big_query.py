#from google.cloud import bigquery

import pandas as pd
import os

#from alt_plot_gen.ml_logic.params import PROJECT, DATASET


def import_dataset() -> pd.DataFrame:
    """
    return a big query dataset table
    format the output dataframe according to the provided data types
    """

    '''
    use this code to upload dataset if you have it on a bucket
    '''
    bucket_path = os.path.join('gs://',
        os.environ.get("BUCKET_NAME"),
        "wiki_movie_plots_deduped.csv")

    big_query_df = pd.read_csv(bucket_path)  # read all rows

    '''
    use this code to upload dataset if you have it on tables (connection with bq client)
    '''
    #table = f"{PROJECT}.{DATASET}.{table}"

    #client = bigquery.Client()

    #rows = client.list_rows(table)

    # convert to expected data types
    #big_query_df = rows.to_dataframe()

    #if big_query_df.shape[0] == 0:
    #    return None  # end of data


    return big_query_df
