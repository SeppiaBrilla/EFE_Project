import torch
from transformers import BertTokenizer, RobertaTokenizer, LongformerTokenizer, AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import random
from tqdm import tqdm
import os
from helper import dict_lists_to_list_of_dicts

class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def pretrain_bert(data_loader, model, optimizer, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
            x = batch.copy()
            del x["output_ids"]
            y = batch["output_ids"].to(device)
            x = {key: val.to(device) for key, val in x.items()}
            outputs = model(**x, labels=y)
            loss = outputs.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x
            del y
            torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}")

def get_tokenizer(bert_type:str):
    if "FacebookAI/roberta-base" == bert_type:
        return RobertaTokenizer.from_pretrained(bert_type)
    elif "bert-base-uncased" == bert_type:
        return BertTokenizer.from_pretrained(bert_type)
    elif "allenai/longformer-base-4096" == bert_type:
        return LongformerTokenizer.from_pretrained(bert_type)
    elif "microsoft/codebert-base" == bert_type:
        return AutoTokenizer.from_pretrained(bert_type)
    elif "../weights/tokenizer_weights" == bert_type:
        return AutoTokenizer.from_pretrained(bert_type)
    else:
        raise Exception("bert type unrecognised")

params_files = [f for f in os.listdir("../bert")]
params = []
for file in params_files:
    with open(os.path.join("../bert", file)) as f:
        params.append(f.read())

bert_type = "microsoft/codebert-base"
tokenizer = get_tokenizer("../weights/tokenizer_weights")
model = AutoModelForMaskedLM.from_pretrained(bert_type)

tokenize_params = dict_lists_to_list_of_dicts(tokenizer(params, padding=True, truncation=True, return_tensors='pt'))
for param in tokenize_params:
    param["output_ids"] = param["input_ids"]
    for i in range(param["input_ids"].shape[0]):
        if param["attention_mask"][i] == 0:
            continue
        p = random.random()
        if p > .15:
            p = random.random()
            if p < .8:
                param["input_ids"][i] = tokenizer.mask_token_id
            elif p < .9:
                param["input_ids"][i] = random.randint(0,782)


dataset = CustomDataset(tokenize_params)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

pretrain_bert(data_loader, model, optimizer)
torch.save(model.state_dict(), "../weights/bert/encoder")