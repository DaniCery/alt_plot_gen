import time
start = time.perf_counter()
#end = time.perf_counter()

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup



#Get the tokenizer and model
def get_pretrained():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return tokenizer, model

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

import os
def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=5, lr=2e-5,          #PARAMETERS TUNING
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):
    acc_steps = 100
    #device=torch.device("cuda")
    #model = model.cuda()

    try:
        '''
        #importing model from bucket
        bucket_path_model = os.path.join('gs://',
        os.environ.get("BUCKET_NAME"),
        "wreckgar-49.pt")
        '''
        #importing model from local path
        path_model = os.path.join(
        os.environ.get("LOCAL_DATA_PATH"),
        "wreckgar-49.pt")
        model = torch.load(path_model)

        print('\n ✅ wreckgar-49.pt model imported',)

    except:
        print('\n ✅ training model... ')
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
        )

        train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss=0
        accumulating_batch_count = 0
        input_tensor = None

        for epoch in range(epochs):

            print(f"Training epoch {epoch}")
            print(loss)
            for idx, entry in tqdm(enumerate(train_dataloader)):
                (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

                if carry_on and idx != len(train_dataloader) - 1:
                    continue

                #input_tensor = input_tensor.to(device)
                #outputs = model(input_tensor, labels=input_tensor)
                #loss = outputs[0]
                #loss.backward()

                if (accumulating_batch_count % batch_size) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                accumulating_batch_count += 1
                input_tensor = None
            if save_model_on_epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
                )
        print(f"\n✅ data trained")
        
    return model
