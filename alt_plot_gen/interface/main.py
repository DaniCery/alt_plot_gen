#from alt_plot_gen.ml_logic.params import (CHUNK_SIZE,
#                                      DATASET_SIZE,
#                                      VALIDATION_DATASET_SIZE)

from alt_plot_gen.ml_logic.data import clean_data
#from alt_plot_gen.ml_logic.preprocessor import preprocess_features
from alt_plot_gen.ml_logic.generation import generate
from alt_plot_gen.ml_logic.encoders import tokenize_plots
from alt_plot_gen.ml_logic.model import get_pretrained, train
#from alt_plot_gen.ml_logic.registry import load_model, save_model

import os
import torch
from transformers import GPT2Tokenizer
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH

def preprocess():
    """
    Preprocess the dataset by
    1) cleaning
    2) preprocessing
    3) tokenizing
    """
    print("\n⭐️ use case: preprocess")

    df, test_set = clean_data()
    # save test set to be used in website
    test_set_file = os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                                            "test_set.csv")
    test_set.to_csv(test_set_file, index = False)

    cleaned_row_count = len(df)

    print(f"\n✅ data processed: ({cleaned_row_count} cleaned)")

    #df = preprocess_features(df)

    dataset = tokenize_plots(df)  #list of tensors (tokenized plots) #all genres

    print(f"\n✅ data tokenized")

    return dataset, test_set


def build_train_model(dataset):

    # initialize model: get_pretrained from gpt-2
    tokenizer, model = get_pretrained()

    print(f"\n✅ got pretrained model")

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

    print(f"\n✅ data trained")

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

#Function to generate multiple sentences
def text_generation(model, tokenizer, test_data):
    #generated_plots = []
    #for i in range(len(test_data)):

    x = generate(model, tokenizer, test_data, entry_count=1, entry_length=60) #, top_p=0.8, temperature=1.

    #generated_plots.append(x)

    print(f"\n✅ generation created")

    #show only generated text
    a = test_data.split()[-60:] #Get the matching string we want (200 words)
    b = ' '.join(a)
    c = ' '.join(x) #Get all that comes after the matching string
    my_generation = c.split(b)[-1]

    #Finish the sentences when there is a point, remove after that
    just_alternative =[]
    to_remove = my_generation.split('.')[-1]
    just_alternative = my_generation.replace(to_remove,'')
    return just_alternative, x



if __name__ == '__main__':

    #dataset, test_set = preprocess()
    #model, tokenizer = build_train_model(dataset)

    model_path = os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                                            "trained_model.pt")

    model = torch.load(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #select plot you want an alternative ending of
    #index (from 0 to 200) of the movie you want to test (set input_raw you want to ask the user to insert from console)
    i = 101
    selected_plot = test_set['Plot'][i]  #take the 100th of the test set as example
    '''
        selected_plot = "Willard Isenbaum, a lonely insurance man with wild sexual fantasies, decides to propose to the new secretary, Susie, whom he has only known for a day and to whom he has never spoken. He spends the entire morning before work fantasizing about having sex with her, but his attempts to approach her fail. His female boss sends him to investigate a claim filed by Painless Martha, an aging tattoo artist, who works in a prison. Martha believes in a Ouija board message saying that she will be killed by a wizard on a Tuesday. \
    When Willard tells her that the insurance company won't pay until her death, she dies of a heart attack. Her will stipulates that her killer must take care of her duck. After the duo spend a night in jail, the duck takes Willard to a brothel. After a wild night of partying, they wind up in the desert, where the duck dresses Willard in women's clothing in an attempt to get a ride. After several encounters with an old prospector dying of thirst, a racist police officer, two lesbians, and a short Mexican man, they are finally picked up by a trucker. \
    Back at his apartment, Willard creates a makeshift sex object, which the duck eats. Shortly after, Willard discovers that Duck is a she, and has sex with her. The following morning, Willard and the duck go to Willard's job, where Willard has sex with his female boss and quits his job shortly after. Willard and the duck leave, and the movie ends with Willard saying that Duck was a good duck after all."
    '''
    #Run the functions to generate the alternative endings
    alternative_end, full_test_generated_plot = text_generation(model, tokenizer, selected_plot)

    #print results
    print('\n ✅ Base Plot: ')
    print(selected_plot)
    print('\n ✅ True end: ')
    #print(test_set['True_end_plot'][i])
    print('\n ✅ Alternative end: ')
    print(alternative_end)
    print('\n ✅ Full plot with alternative ending: ')
    print(full_test_generated_plot)

    del model
