from datetime import datetime
from sys import modules
# $WIPE_BEGIN
import pytz
import pandas as pd
import os
import torch
from alt_plot_gen.ml_logic.data import clean_data, clean_plot
from alt_plot_gen.ml_logic.registry import load_model
from alt_plot_gen.interface.main import text_generation
from alt_plot_gen.interface.main import build_train_model
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH
from alt_plot_gen.ml_logic.model import get_pretrained
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset
from tqdm import trange
import torch.nn.functional as F
from alt_plot_gen.ml_logic.generation import generate

# $WIPE_END'''

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# # $WIPE_BEGIN
# # 💡 Preload the model to accelerate the predictions
# # We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# # The trick is to load the model in memory when the uvicorn server starts
# # Then to store the model in an `app.state.model` global variable accessible across all routes!
# # This will prove very useful for demo days

app.state.model = load_model()
# # $WIPE_END

# # get the tokenizer needed for the generate function
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# ##define a new clean_data function
def clean_data(dataseries):

    # Clean Plot column
    dataseries['Plot'] = dataseries['Plot'].apply(clean_plot)


    # Cut plot wuth more than 1024 tokens to adapt to gpt-2 medium limitations
    dataseries['Plot'] = dataseries['Plot'].map(lambda x: " ".join(x.split()[:350]))  #cut all plots until the 350th word


    #For the test set only, keep last 50 words in a new column, then remove them from original column
    dataseries['True_end_plot'] = dataseries['Plot'].str.split().str[-50:].apply(' '.join)
    dataseries['Plot'] = dataseries['Plot'].str.split().str[:-50].apply(' '.join)

    return dataseries

def tokenize_plots(dataseries):
    class Token_plot(Dataset):
        def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
            self.plots = []

            for row in dataseries['Plot']:
                self.plots.append(torch.tensor(
                        self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
                    ))
            if truncate:
                self.plots = self.plots[:20000]
            self.plots_count = len(self.plots)

        def __len__(self):
            return self.plots_count

        def __getitem__(self, item):
            return self.plots[item]

    dataset = Token_plot(dataseries['Plot'].values[0], truncate=True, gpt2_type="gpt2")   #list of tensors (tokenized plots)

    return dataset
# #define a new preprocess function
def preprocess(dataseries):
    print("\n⭐️ use case: preprocess")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    ds_cleaned = clean_data(dataseries)
    plot = ds_cleaned['Plot'].values[0]
    plot_tokenized = tokenizer.encode(plot)

    print(f"\n✅ data tokenized")

    return plot_tokenized
# # http://127.0.0.1:8000/generate?
# @app.get("/generate")
# def generate(title: str,
#             release_year: int):
#     """
#     we use type hinting to indicate the data types expected
#     for the parameters of the function
#     FastAPI uses this information in order to hand errors
#     to the developpers providing incompatible parameters
#     FastAPI also provides variables of the expected data type to use
#     without type hinting we need to manually convert
#     the parameters of the functions which are all received as strings
#     """
#     # $CHA_BEGIN
#     data = pd.read_csv(f'{LOCAL_DATA_PATH}/wiki_movie_plots_deduped.csv')

#     #if release_year is not None:
#     for movie in data:
#         locate_plot = data.loc[(data['Title'] == title) & (data['Release Year'] == release_year)]
#         plot = locate_plot['Plot'].values[0]

#     plot_preprocessed=preprocess(plot)

#     # else:
#     #     for movie in data:
#     #         plot_list = data.loc[(data['Title'] == title)]
#     #         latest_plot = plot_list.sort_values('Release Year', ascending=True)
#     #         plot = latest_plot.loc[latest_plot['Release Year'] == latest_plot['Release Year'].max()]
#     #         plot = plot['Plot'].values[0]


#     model = app.state.model


#     #Run the functions to generate the alternative endings
#     full_test_generated_plot = text_generation(model, tokenizer, plot_preprocessed)

#     return full_test_generated_plot

#     #⚠️ fastapi only accetpts simple python data types as a return value
#     #among which dict, list, str, int, float, bool
#     #in order to be able to convert the api response to json

#     # $CHA_END'''

# @app.get("/")
# def root():
#     # $CHA_BEGIN
#     return dict(greeting="Hello")
#     # $CHA_END


app.state.model = load_model()

@app.get("/generated")
def generated(title: str, release_year: int):

    dataset = pd.read_csv(f'{LOCAL_DATA_PATH}/wiki_movie_plots_deduped.csv')

    for movie in dataset:
        locate_plot = dataset.loc[(dataset['Title'] == title) & (dataset['Release Year'] == release_year)]
        selected_plot = locate_plot['Plot'].values[0]

    plot_preproc = preprocess(locate_plot)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #plot_tokenized = tokenizer.encode(selected_plot)
    model = app.state.model

    #select plot you want an alternative ending of
    #index (from 0 to 200) of the movie you want to test (set input_raw you want to ask the user to insert from console)


    #Run the functions to generate the alternative endings
    #alternative_end, full_test_generated_plot = text_generation(model, tokenizer, plot_preproc)
    generated_plot = generate(model, tokenizer, plot_preproc, entry_count=10, entry_length=200, #maximum number of words
    top_p=0.8, temperature=1.)
    # return alternative_end, full_test_generated_plot
    # alternative_end, full_test_generated_plot = text_generation(model, tokenizer, locate_plot)

    return generated_plot[-1]
