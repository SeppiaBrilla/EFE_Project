from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, LongformerModel, LongformerTokenizer, AutoTokenizer, AutoModel
from random import randint
import torch.nn as nn
import torch
from json import loads
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from neuralNetwork import NeuralNetwork, In_between_epochs
from helper import Dataset, dict_lists_to_list_of_dicts, one_hot_encoding, predict_dataloader
from typing import Tuple, Callable
from sys import argv


class Model(NeuralNetwork):
    def __init__(self, base_model_name:str, num_classes:int, dropout:float = .5) -> None:
        super().__init__()
        if "FacebookAI/roberta-base" == base_model_name:
            self.bert = RobertaModel.from_pretrained(base_model_name)
        elif "bert-base-uncased" == base_model_name:
            self.bert = BertModel.from_pretrained(base_model_name)
        elif "allenai/longformer-base-4096" == base_model_name:
            self.bert = LongformerModel.from_pretrained(pretrained_model_name_or_path = base_model_name)
        elif "microsoft/codebert-base" == base_model_name:
            self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        else:
            self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout)

        self.middle_layer = nn.Linear(self.bert.config.hidden_size, 100)

        self.output_layer_class = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        out = self.middle_layer(encoded_input)
        out = self.relu(out)
        return self.output_layer_class(out)
    
class Evaluate_time(In_between_epochs):

    def __init__(self, len_train, len_val, len_test, min_train, min_validation, min_test, maj_train, maj_validation, maj_test, sb_train, sb_validation, sb_test, test_dataloader, train_dataloader, times_matrix) -> None:
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test
        self.vb_train = min_train
        self.vb_validation = min_validation
        self.vb_test = min_test
        self.maj_train = maj_train
        self.maj_validation = maj_validation
        self.maj_test = maj_test
        self.sb_train = sb_train
        self.sb_validation = sb_validation
        self.sb_test = sb_test
        self.test_dataloader = test_dataloader
        self.times_matrix = times_matrix
        self.train_dataloader = train_dataloader 
        
    def __call__(self, model: torch.nn.Module, loaders: dict[str, torch.utils.data.DataLoader], device: 'torch.device|str', output_extraction_function:Callable) -> bool:
        train_prediction = predict_dataloader(model, self.train_dataloader, device, output_extraction_function)
        validation_prediction = predict_dataloader(model, loaders["validation"], device, output_extraction_function)
        test_prediction = predict_dataloader(model, self.test_dataloader, device, output_extraction_function)
        pred_train = [self.times_matrix[i, train_prediction[i]] for i in range(self.len_train)]
        pred_val = [self.times_matrix[self.len_train + i, validation_prediction[i]] for i in range(self.len_val)]
        pred_test = [self.times_matrix[self.len_train + self.len_val + i, test_prediction[i]] for i in range(self.len_test)]

        print(f"train set:\nvb:{self.vb_train}     pred:{sum(pred_train)}      sb:{self.maj_train}        smallest:{self.sb_train}. Unique values: {np.unique(train_prediction)}")
        print(f"validation set:\nvb:{self.vb_validation}     pred:{sum(pred_val)}     sb:{self.maj_validation}        smallest:{self.sb_validation}. Unique values: {np.unique(validation_prediction)}")
        print(f"test set:\nvb:{self.vb_test}     pred:{sum(pred_test)}     sb:{self.maj_test}        smallest:{self.sb_test}. Unique values: {np.unique(test_prediction)}")
        return False
      

def get_tokenizer(bert_type:str):
    if "FacebookAI/roberta-base" == bert_type:
        return RobertaTokenizer.from_pretrained(bert_type)
    elif "bert-base-uncased" == bert_type:
        return BertTokenizer.from_pretrained(bert_type)
    elif "allenai/longformer-base-4096" == bert_type:
        return LongformerTokenizer.from_pretrained(bert_type)
    else:
        return AutoTokenizer.from_pretrained(bert_type)
    
def get_time_matrix(shape:tuple, times:list[list[dict]]):
    time_matrix = np.zeros(shape)
    for i in range(len(times)):
        times_i = times[i]
        for j in range(len(times_i)):
            time_matrix[i,j] = times_i[j]["time"]
    return time_matrix

def get_dataloader(x, y, batch_size, test_buckets = []) -> Tuple[DataLoader, DataLoader, DataLoader]:
    BUCKETS = 10

    N_ELEMENTS = len(x)

    BUCKET_SIZE = N_ELEMENTS // BUCKETS

    TEST_BUCKETS = 1


    x_local = x.copy()
    y_local = y.copy()
    x_test, y_test = [], []
    
    if len(test_buckets) == 0: 
        for _ in range(TEST_BUCKETS):
            idx = randint(0, BUCKETS)
            while idx in test_buckets:
                idx = randint(0, BUCKETS)
            test_buckets.append(idx)

    for bucket in test_buckets:
        idx = bucket * BUCKET_SIZE
        for _ in range(BUCKET_SIZE):
            x_test.append(x_local.pop(idx))
            y_test.append(y_local.pop(idx))

    train_elements = (len(y_local) // 10) * 9
    x_train = x_local[:train_elements]
    y_train = y_local[:train_elements]

    x_validation = x_local[train_elements:]
    y_validation = y_local[train_elements:]

    train_dataset, val_dataset, test_dataset = Dataset(x_train, y_train), Dataset(x_validation, y_validation), Dataset(x_test, y_test)
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 
            DataLoader(val_dataset, batch_size=batch_size), 
            DataLoader(test_dataset, batch_size=batch_size)) 

def get_weights(unique_combinations:list, all_combinations:list) -> torch.Tensor:

    frequencies = [all_combinations.count(c) for c in unique_combinations]
    max_freq = max(frequencies)
    min_freq = min(frequencies)
    norm_factor = max_freq - min_freq 
    weights = [1 + ((max_freq - freq)/norm_factor) for freq in frequencies]
    weights = torch.Tensor(weights)
    return weights

        
def main():
    if argv[1] == '--help':
        print("network.py dataset batch_size bert_type tokenizer_type epochs")
        print("bert types: \n[1]FacebookAI/roberta-base \n[2]bert-base-uncased \n[3]allenai/longformer-base-4096 \n[4]microsoft/codebert-base")
        return
    dataset, batch_size, bert_type, tokenizer_type, epochs = argv[1], int(argv[2]), argv[3], argv[4], int(argv[5])

    if bert_type == "1":
        bert_type = "FacebookAI/roberta-base"
    elif bert_type == "2":
        bert_type = "bert-base-uncased"
    elif bert_type == "3":
        bert_type = "allenai/longformer-base-4096"
    elif bert_type == "4":
        bert_type = "microsoft/codebert-base"

    ### import the data
    f = open(dataset)
    data = loads(f.read())
    f.close()

    ### import the model specification

    all_times = [datapoint["all_times"] for datapoint in data]
    combinations = [d["combination"] for d in data[0]["all_times"]]
    instances_and_model = [d["instance_value"] for d in data]

    tokenizer = get_tokenizer(tokenizer_type)

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y_times = get_time_matrix((len(all_times), len(combinations)), all_times)
    for i in range(y_times.shape[0]):
        t_max = np.max(y_times[i,:])
        for j in range(y_times.shape[1]):
            y_times[i,j] = y_times[i,j] / t_max
    y_times = [ torch.from_numpy(y_times[i, :]).type(torch.float32) for i in range(y_times.shape[0])]
    y_class = [datapoint["combination"] for datapoint in data]
    y_class = one_hot_encoding(y_class, combinations)

    y_times = get_time_matrix((len(all_times), len(combinations)), all_times)
    for i in range(y_times.shape[0]):
        t_max = np.max(y_times[i,:])
        t_min = np.min(y_times[i,:])
        for j in range(y_times.shape[1]):
            y_times[i,j] = t_max - y_times[i,j] / t_min
    y_times = [ torch.from_numpy(y_times[i, :]).type(torch.float32) for i in range(y_times.shape[0])]
    softmax = nn.Softmax(dim=0)
    y_times = [ softmax(y_t) for y_t in y_times]

    y = y_class
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, batch_size)
    
    ### training 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    running_combinations = [d["combination"] for d in data]
    weights = get_weights(combinations, running_combinations)
    weights = weights.to(device)

    len_train = len(train_dataloader.dataset)
    len_val = len(validation_dataloader.dataset)
    len_test = len(test_dataloader.dataset)

    times_matrix = get_time_matrix((len(all_times), len(combinations)), all_times)
    min_train = [min(times_matrix[i, :]) for i in range(len_train)]
    min_val = [min(times_matrix[i, :]) for i in range(len_train, len_train + len_val)]
    min_test = [min(times_matrix[i, :]) for i in range(len_train + len_val, len_train + len_val + len_test)]

    new_y = [datapoint["combination"] for datapoint in data]
    new_y = one_hot_encoding(new_y, combinations)
    new_y = np.array([yt.tolist() for yt in new_y])
    majority_index = np.argmax([np.sum(new_y[:, i]) for i in range(len(combinations))])

    majority_train = [times_matrix[i, majority_index] for i in range(len_train)]
    majority_val = [times_matrix[len_train + i, majority_index] for i in range(len_val)]
    majority_test = [times_matrix[len_train + len_val + i, majority_index] for i in range(len_test)]

    sb_t = [sum(times_matrix[:len_train, i]) for i in range(len(combinations))]
    sb_v = [sum(times_matrix[len_train:len_train + len_val, i]) for i in range(len(combinations))]
    sb_te = [sum(times_matrix[len_train + len_val:len_train + len_val + len_test, i]) for i in range(len(combinations))]

    sb_train = min(sb_t)
    sb_validation = min(sb_v)
    sb_test = min(sb_te)
    in_between = Evaluate_time(
        len_train, len_val, len_test, 
        sum(min_train), sum(min_val), sum(min_test), 
        sum(majority_train), sum(majority_val), sum(majority_test), 
        sb_train, sb_validation, sb_test, 
        test_dataloader, DataLoader(train_dataloader.dataset, train_dataloader.batch_size, shuffle=False), times_matrix)

    length = len(combinations)
    model = Model(bert_type, length, dropout=.3)
    extraction_function = lambda x: torch.max(x, -1)[1].cpu()

    train_score, val_score = model.train_network(train_dataloader, 
                    validation_dataloader, 
                    torch.optim.SGD, 
                    loss_function=nn.CrossEntropyLoss(weights),
                    device=device, 
                    verbose=True, 
                    output_extraction_function= extraction_function, 
                    metrics={
                     # "accuracy_times": lambda y_true, y_pred: accuracy_score(y_true["times"], y_pred["times"]), 
                     "accuracy_class": accuracy_score, #lambda y_true, y_pred: accuracy_score(y_true["class"], y_pred["class"]), 
                     # "f1_score_times": lambda y_true, y_pred: f1_score(y_true["times"], y_pred["times"], average="macro"),
                     "f1_score_class": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")},
                    in_between_epochs={"validate_times":in_between},
                    learning_rate=1e-04,
                    epochs=epochs)

    torch.save(model.state_dict(), "code_bert_weights")
main()
