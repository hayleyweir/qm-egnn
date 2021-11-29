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
    print(f"Test set loss: {test_loss:.2f}")
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


def load_eval_dict(base_path):
    filenames = ["training.txt", "validation.txt", "test.txt"]
    keys = ["index", "prediction", "target", "error"]
    eval_dict = {}
    for filename in filenames:
        dataset_key = filename.split(".")[0]
        filename = os.path.join(base_path, filename)
        data = np.loadtxt(filename, skiprows=1)
        data_dict = {}
        for key, column in zip(keys, data.T):
            data_dict[key] = column.flatten()
        data_dict["index"] = data_dict["index"].astype(int)
        eval_dict[dataset_key] = data_dict

    return eval_dict


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


def compute_metrics(eval_dict):
    for dataset_key, data_dict in eval_dict.items():
        prediction = data_dict["prediction"]
        target = data_dict["target"]
        error = data_dict["error"]
        data_dict["MAE"] = np.mean(np.abs(error))
        data_dict["RMSE"] = np.sqrt(np.mean(np.square(error)))
        data_dict["Mean Error"] = np.mean(error)
        data_dict["STD"] = np.std(error)
        data_dict["Mean Absolute Percent Error"] = np.mean(
            np.abs(error / target * 100)
        )
        data_dict["Max Absolute Error"] = np.max(np.abs(error))
        data_dict["Max Absolute Percent Error"] = np.max(
            np.abs(error / target * 100)
        )
        data_dict["Mean prediction"] = np.mean(prediction)
        data_dict["Mean target"] = np.mean(target)



def print_metrics(save_filename, eval_dict):
    header = ["Metric"] + list(eval_dict.keys())
    header = ", ".join(header)
    print("\n" + header)
    with open(save_filename, "w") as fp:
        fp.write(header + "\n")

    print_keys = [
        "MAE",
        "RMSE",
        "Mean Error",
        "STD",
        "Mean Absolute Percent Error",
        "Max Absolute Error",
        "Max Absolute Percent Error",
        "Mean prediction",
        "Mean target"
    ]
    # max_key_length = max([len(key) for key in print_keys])
    print_line = "{:<30}  " + "  ".join(["{:.4e}"] * 3)

    for key in print_keys:
        print_data = tuple(
            [data_dict[key] for data_dict in eval_dict.values()]
        )
        line = print_line.format(key, *print_data)
        print(line)
        with open(save_filename, "a") as fp:
            fp.write(line + "\n")

    print(f"\nSaved {save_filename}")
