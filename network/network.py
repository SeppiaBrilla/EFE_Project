from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, LongformerModel, LongformerTokenizer
import torch.nn as nn
import torch
from json import loads, dumps
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from neuralNetwork import NeuralNetwork
from helper import Dataset, dict_lists_to_list_of_dicts, one_hot_encoding, predict_dataloader
from typing import Tuple
from sys import argv

class Model(NeuralNetwork):
    def __init__(self, base_model_name:str, num_classes:int, dropout:float = .5) -> None:
        super().__init__()
        if "FacebookAI/roberta-base" == base_model_name:
            self.bert = RobertaModel.from_pretrained(base_model_name)
        elif "bert-base-uncased" == base_model_name:
            self.bert = BertModel.from_pretrained(base_model_name)
        elif "allenai/longformer-base-4096":
            self.bert = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        else:
            raise Exception("bert type unrecognised")
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(self.bert.config.hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        
        return self.output_layer(encoded_input)
    

def get_tokenizer(bert_type:str):
    if "FacebookAI/roberta-base" == bert_type:
        return RobertaTokenizer.from_pretrained(bert_type)
    elif "bert-base-uncased" == bert_type:
        return BertTokenizer.from_pretrained(bert_type)
    elif "allenai/longformer-base-4096" == bert_type:
        return LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    else:
        raise Exception("bert type unrecognised")
    
def get_time_matrix(shape:tuple, times:list[list[dict]]):
    time_matrix = np.zeros(shape)
    for i in range(len(times)):
        times_i = times[i]
        for j in range(len(times_i)):
            time_matrix[i,j] = times_i[j]["time"]
    return time_matrix

def get_dataloader(x, y, batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    BUCKETS = 10

    N_ELEMENTS = len(x)

    BUCKET_SIZE = N_ELEMENTS // BUCKETS

    TRAIN_VALIDATION_BUCKETS = 9
    TEST_BUCKETS = 1

    TRAIN_SIZE = int((TRAIN_VALIDATION_BUCKETS / 10) * 9)
    VALIDATION_SIZE = TRAIN_VALIDATION_BUCKETS - TRAIN_SIZE

    buckets_x = [x[bucket * BUCKET_SIZE: (bucket + 1) * BUCKET_SIZE] for bucket in range(BUCKETS)]
    buckets_y = [y[bucket * BUCKET_SIZE: (bucket + 1) * BUCKET_SIZE] for bucket in range(BUCKETS)]

    x_train_validation = buckets_x[:TRAIN_VALIDATION_BUCKETS]
    y_train_validation = buckets_y[:TRAIN_VALIDATION_BUCKETS]


    x_train = np.ravel(x_train_validation[:TRAIN_SIZE]) 
    y_train = y_train_validation[0]
    for i in range(1, TRAIN_SIZE):
        y_train += y_train_validation[i]
    x_validation  = np.ravel(x_train_validation[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE])
    y_validation = y_train_validation[TRAIN_SIZE]
    for i in range(TRAIN_SIZE + 1, TRAIN_SIZE + VALIDATION_SIZE):
        y_validation += y_train_validation[i]

    x_test = np.ravel(buckets_x[-TEST_BUCKETS:])
    y_test = buckets_y[-TEST_BUCKETS]
    for i in range((BUCKETS - TEST_BUCKETS) + 1, BUCKETS):
        y_test += buckets_y[i]

    train_dataset = Dataset(x_train, y_train)
    validation_dataset = Dataset(x_validation, y_validation)
    test_dataset = Dataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, validation_dataloader, test_dataloader

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
        print("network.py dataset model_specifications batch_size bert_type epochs")
        print("bert types: \n[1]FacebookAI/roberta-base \n[2]bert-base-uncased \n[3]allenai/longformer-base-4096")
        return
    dataset, model_specifications, batch_size, bert_type, epochs = argv[1], argv[2], int(argv[3]), argv[4], int(argv[5])

    if bert_type == "1":
        bert_type = "FacebookAI/roberta-base"
    elif bert_type == "2":
        bert_type = "bert-base-uncased"
    elif bert_type == "3":
        bert_type = "allenai/longformer-base-4096"

    ### import the data
    f = open(dataset)
    data = loads(f.read())
    f.close()

    ### import the model specification

    f = open(model_specifications)
    problem_specification = f.read()
    f.close()

    ### data manipulation

    all_times = [datapoint["all_times"] for datapoint in data]
    combinations = [d["combination"] for d in data[0]["all_times"]]
    

    ### dataset creation
    
    tokenizer = get_tokenizer(bert_type)

    instances_and_model = [f"{problem_specification}\n\n{datapoint['instance']}" for datapoint in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = [datapoint["combination"] for datapoint in data]
    y = one_hot_encoding(y, combinations)
    
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(x, y, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    running_combinations = [d["combination"] for d in data]
    weights = get_weights(combinations, running_combinations)
    weights = weights.to(device)

    length = len(combinations)
    model = Model(bert_type, length, dropout=.1)
    extraction_function = lambda x: torch.max(x, -1)[1].cpu()
    train_score, val_score = model.train_network(train_dataloader, 
                    validation_dataloader, 
                    torch.optim.SGD, 
                    loss_function=nn.CrossEntropyLoss(weights),
                    device=device, 
                    verbose=True, 
                    output_extraction_function= extraction_function, 
                    metrics={"accuracy": accuracy_score, "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")},
                    learning_rate=1e-03,
                    epochs=epochs)

    for key in train_score:
                train_score[key] = [float(v) for v in train_score[key]]
                val_score[key] = [float(v) for v in val_score[key]]
    f = open("stats.json", 'w')
    f.write(dumps({"train":train_score, "validation":val_score}))
    print("evaluating on the training, validation and test set")

    model = model.to(device)
    model.eval()
    train_prediction = predict_dataloader(model, train_dataloader, device, extraction_function)
    validation_prediction = predict_dataloader(model, validation_dataloader, device, extraction_function)
    test_prediction = predict_dataloader(model, test_dataloader, device, extraction_function)

    len_train = len(train_dataloader.dataset)
    len_val = len(validation_dataloader.dataset)
    len_test = len(test_dataloader.dataset)

    times_matrix = get_time_matrix(np.array(y).shape, all_times)
    min_train = [min(times_matrix[i, :]) for i in range(len_train)]
    min_val = [min(times_matrix[i, :]) for i in range(len_train, len_train + len_val)]
    min_test = [min(times_matrix[i, :]) for i in range(len_train + len_val, len_train + len_val + len_test)]
    
    pred_train = [times_matrix[i, train_prediction[i]] for i in range(len_train)]
    pred_val = [times_matrix[len_train + i, validation_prediction[i]] for i in range(len_val)]
    pred_test = [times_matrix[len_train + len_val + i, test_prediction[i]] for i in range(len_test)]

    print(f"train set:\nvb:{sum(min_train)}     pred:{sum(pred_train)}")
    print(f"validation set:\nvb:{sum(min_val)}     pred:{sum(pred_val)}")
    print(f"test set:\nvb:{sum(min_test)}     pred:{sum(pred_test)}")
main()
