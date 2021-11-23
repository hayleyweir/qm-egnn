import torch
from torch import nn
from torch_geometric.loader import DataLoader
from src import utils

import os
import numpy as np


def evaluate(
        model,
        device,
        test_dataset,
        target_index,
):
    loader_test = DataLoader(test_dataset, batch_size=len(test_dataset))
    loss_fn = nn.MSELoss()
    batch_to_device = utils.BatchToDevice(device)

    with torch.no_grad():
        for batch in loader_test:
            batch = batch_to_device(batch)
            pred = model(batch)
            loss = loss_fn(pred.flatten(), batch.y[:, target_index].flatten())
            test_loss = loss.item()

    print("------------------------------------------------")
    print(f'Test set loss: {test_loss:.2f}')
    print("------------------------------------------------")
    return test_loss


def dump_eval_dict(base_path, eval_dict):
    keys = ["index", "prediction", "target", "error"]
    header = ", ".join(keys)
    for dataset_key, data_dict in eval_dict.items():
        filename = os.path.join(base_path, f"{dataset_key}.txt")
        with open(filename, "w") as fp:
            fp.write(header + "\n")
            data = tuple([data_dict[key] for key in keys])
            for idx, pred, targ, err in zip(*data):
                fp.write(f"{idx:>7d} {pred:.8e} {targ:.8e} {err:.8e}\n")
        print(f"Saved: {filename}")


def get_eval_dict(model, device, target_index, **datasets):
    loaders = {}
    for dataset_key, dataset in datasets.items():
        loaders[dataset_key] = DataLoader(dataset, batch_size=1000)
    batch_to_device = utils.BatchToDevice(device)

    eval_dict = {}
    for dataset_key, loader in loaders.items():
        print(f"Processing: {dataset_key}")
        data_dict = {"index": [], "prediction": [], "target": []}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch_to_device(batch)
                pred = model(batch).cpu().numpy().flatten()
                targ = batch.y[:, target_index].cpu().numpy().flatten()

                data_dict["prediction"].append(pred)
                data_dict["target"].append(targ)
                data_dict["index"].append(batch.idx.numpy().flatten())
                print(f"{i + 1}/{len(loader)}", end="\r")
            print(f"{i + 1}/{len(loader)}", end="\n")

        eval_dict[dataset_key] = data_dict

    for dataset_key, data_dict in eval_dict.items():
        for key, value in data_dict.items():
            data_dict[key] = np.concatenate(value)
        data_dict["error"] = data_dict["prediction"] - data_dict["target"]

    return eval_dict
