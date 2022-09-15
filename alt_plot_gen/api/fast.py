from datetime import datetime
from sys import modules
# $WIPE_BEGIN
import pytz
import pandas as pd
import os
import torch
from alt_plot_gen.ml_logic.data import clean_data, clean_plot
from alt_plot_gen.ml_logic.registry import load_model
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH
from transformers import GPT2Tokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset
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

#define a new clean_data function
def clean_data_2(dataseries):

    # Clean Plot column
    dataseries['Plot'] = dataseries['Plot'].apply(clean_plot)


    # Cut plot wuth more than 1024 tokens to adapt to gpt-2 medium limitations
    dataseries['Plot'] = dataseries['Plot'].map(lambda x: " ".join(x.split()[:350]))  #cut all plots until the 350th word


    #For the test set only, keep last 50 words in a new column, then remove them from original column
    dataseries['True_end_plot'] = dataseries['Plot'].str.split().str[-50:].apply(' '.join)
    dataseries['Plot'] = dataseries['Plot'].str.split().str[:-50].apply(' '.join)

    return dataseries
#define a new tokenizer function
def tokenize_plots_2(dataseries):
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
#define a new preprocess function
def preprocess_2(dataseries):
    print("\n⭐️ use case: preprocess")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    ds_cleaned = clean_data(dataseries)
    plot = ds_cleaned['Plot'].values[0]
    plot_tokenized = tokenizer.encode(plot)

    print(f"\n✅ data tokenized")

    return plot_tokenized

app.state.model = load_model()

@app.get("/generated")
def generated(title: str, release_year: int):

    dataset = pd.read_csv(f'{LOCAL_DATA_PATH}/wiki_movie_plots_deduped.csv')

    for movie in dataset:
        locate_plot = dataset.loc[(dataset['Title'] == title) & (dataset['Release Year'] == release_year)]
        selected_plot = locate_plot['Plot'].values[0]

    plot_preproc = preprocess_2(locate_plot)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = app.state.model

    #Run the functions to generate the alternative endings
    generated_plot = generate(model, tokenizer, plot_preproc, entry_count=10, entry_length=200, #maximum number of words
    top_p=0.8, temperature=1.)

    return generated_plot[-1]
