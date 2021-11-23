#!/usr/bin/env python

import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
import os
import sys

from src.model import EGNN_network
from src.data_loader import load_dataset
from src import utils
from src import eval

np.random.seed(6)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EGNN model")
    parser.add_argument(
        "weights_path", type=str, help="Path to saved model weights"
    )
    parser.add_argument(
        "dataset_dir", type=str, help="Path to dataset (Only QM9 for now)"
    )
    parser.add_argument(
        "target_index", type=int, help="0=dipole, 4=homo-lumo gap"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save model predictions in",
    )
    parser.add_argument(
        "-u", "--hidden_units", type=int, default=128, help="Default: 128"
    )
    parser.add_argument(
        "-d", "--n_layers", type=int, default=7, help="Default: 7"
    )
    parser.add_argument(
        "-s",
        "--dataset_size_lim",
        type=int,
        help="Default: None (Entire dataset is used)",
    )
    parser.add_argument(
        "-i",
        "--GPU_index",
        type=int,
        help="Train model using GPU with given index. Default: CPU",
    )

    args = parser.parse_args().__dict__
    args["weights_path"] = os.path.abspath(args["weights_path"])
    args["dataset_dir"] = os.path.abspath(args["dataset_dir"])
    args["output_dir"] = os.path.abspath(args["output_dir"])
    return args


def main():
    args = parse_args()
    train_dataset, val_dataset, test_dataset = load_dataset(
        args["dataset_dir"], args["dataset_size_lim"]
    )
    model = EGNN_network(
        hidden_units=args["hidden_units"], n_layers=args["n_layers"]
    )
    n_parameters = sum(
        [tensor.flatten().size()[0] for tensor in model.parameters()]
    )
    print(f"Number of parameters: {n_parameters}")

    # 3. Set up CUDA environment
    device = torch.device("cpu")
    if args["GPU_index"] is not None:
        device = utils.setup_cuda(args["GPU_index"])
        model.cuda(device)

    # 4. Get eval_dict containing predictions, targets
    # and errors for all partitions
    eval_dict = eval.get_eval_dict(
        model,
        device,
        args["target_index"],
        training=train_dataset,
        validation=val_dataset,
        test=test_dataset,
    )

    # 5. Print the eval_dict to text files
    eval.dump_eval_dict(args["output_dir"], eval_dict)

    # 6. Plot the predictions vs target
    colors = ["blue", "green", "red"]
    fontsize = 20
    plt.figure(figsize=(20, 10))
    for i, (dataset_key, data_dict) in enumerate(eval_dict.items()):
        color = colors[i]
        plt.subplot(1, 3, i + 1)
        plt.scatter(
            data_dict["target"],
            data_dict["prediction"],
            c=color,
            label=dataset_key,
            s=2,
        )
        plt.xlabel("Target", fontsize=fontsize)
        plt.ylabel("Prediction", fontsize=fontsize)
        plt.axis("square")
        plt.title(dataset_key, fontsize=fontsize)


    # # 7. Print some metrics
    # with open(os.path.join(args["output_dir"], "metrics.txt"), "w") as fp:
    #     fp.write("Metric, " + ", ".join(list(eval_dict.keys())))

    #     # MAE




if __name__ == "__main__":
    sys.exit(main())
