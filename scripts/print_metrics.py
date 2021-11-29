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
        "base_path", type=str, nargs="+", help="Path to folder with predictions saved"
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
        eval.compute_metrics(eval_dict)
        save_filename = os.path.join(base_path, "metrics.txt")
        eval.print_metrics(save_filename, eval_dict)


if __name__ == "__main__":
    sys.exit(main())
