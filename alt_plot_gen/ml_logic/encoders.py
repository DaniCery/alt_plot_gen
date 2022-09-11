import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

#from alt_plot_gen.ml_logic.utils import simple_time_and_memory_tracker


def tokenize_plots(df: pd.DataFrame) -> np.ndarray:
    class Token_plot(Dataset):
        def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
            self.plots = []

            for row in df['Plot']:
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

    dataset = Token_plot(df['Plot'], truncate=True, gpt2_type="gpt2")   #list of tensors (tokenized plots)

    return dataset
