#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import sys

from src import eval

np.random.seed(6)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot predictions vs target")
    parser.add_argument(
        "base_path",
        type=str,
        nargs="+",
        help="Path to folder with predictions saved",
    )
    args = parser.parse_args().__dict__
    args["base_path"] = [os.path.abspath(path) for path in args["base_path"]]
    return args


def main():
    args = parse_args()
    for base_path in args["base_path"]:
        try:
            eval_dict = eval.load_eval_dict(base_path)
        except OSError:
            continue
        save_filename = os.path.join(base_path, "prediction_plot.png")

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
        plt.savefig(save_filename)
        print(f"\nSaved {save_filename}")


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
