from gen_features import gen_features
from json import load
import numpy as np

f = open("data/datasets/dataset.json")
data = load(f)
f.close()

models = [d["combination"].split("_")[:-1] for d in data[0]["all_times"]]
models = ["_".join(d) for d in models]
models = np.unique(models).tolist()

for datapoint in data:
    instance = datapoint["instance"]

    for model in models:
        gen_features(f"../EssenceCatalog-runs/problems/csplib-prob001-CarSequencing/conjure-mode/portfolio4/{model}", f"{instance}", save=True, verbose=False, file_name=f"features/{model.split('/')[-1]}_{instance.split('/')[-1]}.json")
