import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Plot loss")
parser.add_argument("run_name", type=str, help="name of run")
args = parser.parse_args().__dict__ 

with open("history.txt", "r") as f:
    ls = f.readlines()

train_losses = [float(l.split()[1]) for l in ls[1:]]
val_losses = [float(l.split()[2]) for l in ls[1:]]

plt.plot(train_losses, c="r", label="Train")
plt.plot(val_losses, c="b", label="Val")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(args["run_name"])
plt.savefig(f'{args["run_name"]}.png')
