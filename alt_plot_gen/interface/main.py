#from alt_plot_gen.ml_logic.params import (CHUNK_SIZE,
#                                      DATASET_SIZE,
#                                      VALIDATION_DATASET_SIZE)

from alt_plot_gen.ml_logic.data import clean_data
#from alt_plot_gen.ml_logic.preprocessor import preprocess_features
from alt_plot_gen.ml_logic.generation import generate
from alt_plot_gen.ml_logic.encoders import tokenize_plots
from alt_plot_gen.ml_logic.model import get_pretrained, train
#from alt_plot_gen.ml_logic.registry import load_model, save_model

def preprocess():
    """
    Preprocess the dataset by
    1) cleaning
    2) preprocessing
    3) tokenizing
    """
    print("\n⭐️ use case: preprocess")

    df, test_set = clean_data()

    cleaned_row_count = len(df)

    print(f"\n✅ data processed: ({cleaned_row_count} cleaned)")

    #df = preprocess_features(df)

    dataset = tokenize_plots(df)  #list of tensors (tokenized plots) #all genres

    print(f"\n✅ data tokenized")

    return dataset, test_set


def build_train_model(dataset):

    # initialize model: get_pretrained from gpt-2
    tokenizer, model = get_pretrained()

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


# save model
'''
params = dict(
    # model parameters
    batch_size=16,
    epochs=5
    lr=2e-5
    max_seq_len=400
    warmup_steps=200

    # data source
    #training_set_size=DATASET_SIZE,
    #val_set_size=VALIDATION_DATASET_SIZE,
    #row_count=row_count
    )
#save_model(model=model, params=params)  #, metrics=dict(mae=val_mae)
'''

#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
  generated_plots = []
  #for i in range(len(test_data)):
  x = generate(model, tokenizer, test_data['Plot'][100], entry_count=1)  #top_p=0.8, temperature=1.
  generated_plots.append(x)
  return generated_plots



if __name__ == '__main__':
    dataset, test_set = preprocess()
    model, tokenizer = build_train_model(dataset)

    #select plot you want an alternative ending of
    selected_plot = test_set['Plot'][50]  #take the 100th of the test set as example
    test_generated_plot = text_generation(selected_plot)

    #Run the functions to generate the alternative endings
    print(test_generated_plot)
