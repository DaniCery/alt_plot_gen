from datetime import datetime
import pytz
import pandas as pd


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alt_plot_gen.ml_logic.data import clean_data
from alt_plot_gen.ml_logic.generation import generate
from alt_plot_gen.ml_logic.encoders import tokenize_plots
from alt_plot_gen.ml_logic.model import get_pretrained, train

def preprocess():
    """
    Preprocess the dataset by
    1) cleaning
    2) preprocessing
    3) tokenizing
    """
    print("\n⭐️ Start...")

    df, test_set = clean_data()

    cleaned_row_count = len(df)

    print(f"\n✅ data processed: ({cleaned_row_count} cleaned)")

    #df = preprocess_features(df)

    dataset = tokenize_plots(df)  #list of tensors (tokenized plots) #all genres

    return dataset, test_set


def build_train_model(dataset):

    # initialize model: get_pretrained from gpt-2
    tokenizer, model = get_pretrained()

    print(f"\n✅ got gpt-2 pretrained model")

    # model params
    batch_size=16
    epochs=5
    lr=2e-5
    max_seq_len=400
    warmup_steps=200

    # pack tensor and train the model incrementally
    model = train(dataset, model, tokenizer,
                batch_size=batch_size, epochs=epochs, lr=lr,
                max_seq_len=max_seq_len, warmup_steps=warmup_steps,
                gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
                test_mode=False,save_model_on_epoch=False)

    return model, tokenizer

#Function to generate multiple sentences
def text_generation(model, tokenizer, test_data):
    #generated_plots = []
    #for i in range(len(test_data)):

    x = generate(model, tokenizer, test_data, entry_count=1)  #top_p=0.8, temperature=1.

    #generated_plots.append(x)

    print(f"\n✅ generation created")

    #show only generated text
    a = test_data.split()[-200:] #Get the matching string we want (200 words)
    b = ' '.join(a)
    c = ' '.join(x) #Get all that comes after the matching string
    my_generation = c.split(b)[-1]

    #Finish the sentences when there is a point, remove after that
    to_remove = my_generation.split('.')[-1]
    final = my_generation.replace(to_remove,'')
    return final



app = FastAPI()

# Define a root `/` endpoint
# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get('/predict')
def predict(plot_number):
    # select plot based on number
    dataset, test_set = preprocess()
    #model, tokenizer = build_train_model(dataset)

    #select plot you want an alternative ending of
    #index (from 0 to 200) of the movie you want to test (set input_raw you want to ask the user to insert from console)
    i = plot_number
    selected_plot = test_set['Plot'][i]  #take the 100th of the test set as example

    #Run the functions to generate the alternative endings
    #test_generated_plot = text_generation(model, tokenizer, selected_plot)


    '''
    #print results
    print('\n ✅ Base Plot: ')
    print(selected_plot)
    print('\n ✅ True end: ')
    print(test_set['True_end_plot'][i])
    print('\n ✅ Alternative ending: ')
    print(test_generated_plot)
    print('\n ✅ Full generated Plot: ')
    print(selected_plot + test_generated_plot)
    '''


    return {'plot_number': plot_number}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
