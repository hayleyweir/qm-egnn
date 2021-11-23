import matplotlib.pyplot as plt

with open("history.txt", "r") as f:
    ls = f.readlines()

train_losses = [float(l.split()[1]) for l in ls[1:]]
val_losses = [float(l.split()[2]) for l in ls[1:]]

plt.plot(train_losses, c="r", label="Train")
plt.plot(val_losses, c="b", label="Val")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss.png")
