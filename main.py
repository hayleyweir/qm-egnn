import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
import os

from src.model import EGNN_network
from src.train import train_model
from src.eval import evaluate
from src.data_loader import load_dataset
from src import utils


np.random.seed(6)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EGNN model")

    parser.add_argument(
        "dataset_dir", type=str, help="Path to dataset (Only QM9 for now)"
    )
    parser.add_argument(
        "target_index", type=int, help="0=dipole, 4=homo-lumo gap"
    )
    parser.add_argument(
        "num_epochs", type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=128, help="Default: 128"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=1e-5, help="Default: 1e-5"
    )
    parser.add_argument(
        "-g", "--gamma", type=float, default=0.999, help="Default: 0.999"
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
    args["dataset_dir"] = os.path.abspath(args["dataset_dir"])
    return args


def plot_loss(train_loss, val_loss, start_epoch=0):
    plt.plot(train_loss[start_epoch:], c="k", label="train")
    plt.plot(val_loss[start_epoch:], c="b", label="val")
    plt.legend()
    plt.savefig("Training_Val_Loss.png")
    plt.show()


def main():
    # 0. Set values
    args = parse_args()

    # 1. Load training, validation and test datasets
    train_dataset, val_dataset, test_dataset = load_dataset(
        args["dataset_dir"], args["dataset_size_lim"]
    )

    # 2. Initiate EGNN model
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

    # 4. Train model
    train_loss, val_loss = train_model(
        model,
        device,
        train_dataset,
        val_dataset,
        num_epochs=args["num_epochs"],
        batch_size=args["batch_size"],
        target_index=args["target_index"],
        learning_rate=args["learning_rate"],
        gamma=args["gamma"],
    )

    # 5. Plot loss
    plot_loss(train_loss, val_loss)

    # 6. Test model on test set
    # model = EGNN_network(hidden_units=hidden_units, n_layers=n_layers)
    model.load_state_dict(torch.load("saved_weights/BEST.pt"))
    model.eval()
    evaluate(model, device, test_dataset, args["target_index"])


if __name__ == "__main__":
    main()
