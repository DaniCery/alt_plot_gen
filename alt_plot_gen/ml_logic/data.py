from alt_plot_gen.ml_logic.params import (COLUMN_NAMES_RAW,
                                            DTYPES_RAW_OPTIMIZED,
                                            DTYPES_RAW_OPTIMIZED_HEADLESS,
                                            DTYPES_PROCESSED_OPTIMIZED
                                            )

from alt_plot_gen.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)
from alt_plot_gen.data_sources.local_disk import import_dataset

from alt_plot_gen.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant plots
    or columns for the training set
    """
    '''
    # remove useless/redundant columns
    df = df.drop(columns=['key'])

    # remove buggy transactions
    df = df.drop_duplicates()  # TODO: handle in the data source if the data is consumed by chunks
    df = df.dropna(how='any', axis=0)
    df = df[df.passenger_count > 0]
    df = df[df.fare_amount > 0]

    # remove irrelevant/non-representative transactions (rows) for a training set
    print("\n✅ data cleaned")
    '''
    ### Prepare
    df = import_dataset()
    df['Plot'] = df['Plot'].map(lambda x: " ".join(x.split()[:350]))  #cut all plots until the 350th word

    #Create a very small test set to compare generated text with the reality
    test_set = df.sample(n = 200)
    df = df.loc[~df.index.isin(test_set.index)]

    #Reset the indexes
    test_set = test_set.reset_index()
    df = df.reset_index()
    #For the test set only, keep last 50 words in a new column, then remove them from original column
    test_set['True_end_plot'] = test_set['Plot'].str.split().str[-50:].apply(' '.join)
    test_set['Plot'] = test_set['Plot'].str.split().str[:-50].apply(' '.join)
    return df, test_set

'''
def get_chunk(source_name: str,
              index: int = 0,
              chunk_size: int = None,
              verbose=False) -> pd.DataFrame:
    """
    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
    Always assumes `source_name` (CSV or Big Query table) have headers,
    and do not consider them as part of the data `index` count.
    """

    if "processed" in source_name:
        columns = None
        dtypes = DTYPES_PROCESSED_OPTIMIZED
    else:
        columns = COLUMN_NAMES_RAW
        if os.environ.get("DATA_SOURCE") == "big query":
            dtypes = DTYPES_RAW_OPTIMIZED
        else:
            dtypes = DTYPES_RAW_OPTIMIZED_HEADLESS

    if os.environ.get("DATA_SOURCE") == "big query":

        chunk_df = get_bq_chunk(table=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                verbose=verbose)

        return chunk_df

    chunk_df = get_pandas_chunk(path=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                columns=columns,
                                verbose=verbose)

    return chunk_df


def save_chunk(destination_name: str,
               is_first: bool,
               data: pd.DataFrame) -> None:
    """
    save chunk
    """

    if os.environ.get("DATA_SOURCE") == "big query":

        save_bq_chunk(table=destination_name,
                      data=data,
                      is_first=is_first)

        return

    save_local_chunk(path=destination_name,
                     data=data,
                     is_first=is_first)
'''
