"""
A file where every function used by the network.py file is saved
"""
from torch import zeros
from torch.utils.data import Dataset
import torch

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def one_hot_encoding(data, values):
    length = len(values)

    out = []

    for datapoint in data:
        tensor = zeros(size=(length, ))

        idx = values.index(datapoint)
        tensor[idx] = 1.
        out.append(tensor)

    return out


def dict_lists_to_list_of_dicts(input_dict:dict) -> list[dict]:
    """
    A function that convert a dictionary of lists into a list of dictionaries
    Parameters
    ----------
    input_dict:dict
        The dictionary to convert
    
    Outputs
    -------
    The list of dictionaries
    """
    keys = input_dict.keys()
    list_lengths = [len(input_dict[key]) for key in keys]

    if len(set(list_lengths)) > 1:
        raise ValueError("All lists in the input dictionary must have the same length.")

    list_of_dicts = [{key: input_dict[key][i] for key in keys} for i in range(list_lengths[0])]

    return list_of_dicts


def to(data, device):
  if isinstance(data, dict):
    return {key: to(data[key], device) for key in data.keys()}
  elif isinstance(data, list):
    return [to(d, device) for d in data]
  elif isinstance(data, tuple):
    return tuple([to(d, device) for d in data])
  else:
    return data.to(device)
  
def remove(data):
  if isinstance(data, dict):
    for key in data.keys():
      remove(data[key])
  elif isinstance(data, list) or isinstance(data, tuple):
    for d in data:
      remove(d)
  else:
    del data


def predict_dataloader(model, loader, device, extraction_function):
    all_predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader):
          inputs = data[0]
          if True:
            inputs = to(data[0], device)
          outputs = model(inputs)
          predicted_classes = extraction_function(outputs)
          if isinstance(predicted_classes, dict):
                if not isinstance(all_predictions, dict) :
                    all_predictions = {}
                for key in predicted_classes.keys():
                    if not key in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key] += predicted_classes[key]
          else:
              all_predictions += predicted_classes
          if True:
            remove(inputs)
            torch.cuda.empty_cache()
    return all_predictions

