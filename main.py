import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import EGNN_network
from src.train import train_model
from src.eval import evaluate
from src.data_loader import load_dataset

np.random.seed(6)


def setup_cuda():
    print(f"GPU is available: {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    device = torch.device('cuda:7')
    print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"GPU index: {device.index}")


def plot_loss(train_loss, val_loss, start_epoch=0):
    plt.plot(train_loss[start_epoch:], c="k", label="train")
    plt.plot(val_loss[start_epoch:], c="b", label="val")
    plt.legend()
    plt.savefig("Training_Val_Loss.png")
    plt.show()


def main():
    # 0. Set values
    dataset_size = 10000
    batch_size = 10
    num_epochs = 10
    target_index = 0

    lr = 1e-5
    gamma = 0.999

    hidden_units = 128
    n_layers = 7

    # 1. Load training, validation and test datasets
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_size)

    # 2. Set up CUDA environment
    # setup_cuda()

    # 3. Initiate EGNN model
    model = EGNN_network(hidden_units=hidden_units, n_layers=n_layers)
    print(f"Number of parameters: {sum([tensor.flatten().size()[0] for tensor in model.parameters()])}")

    # 4. Train model
    train_loss, val_loss = train_model(model, train_dataset, val_dataset, num_epochs=num_epochs,
                                       batch_size=batch_size, target_index=target_index, learning_rate=lr, gamma=gamma)

    # 5. Plot loss
    plot_loss(train_loss, val_loss)

    # 6. Test model on test set
    # model = EGNN_network(hidden_units=hidden_units, n_layers=n_layers)
    model.load_state_dict(torch.load("saved_weights/BEST.pt"))
    model.eval()
    evaluate(model, test_dataset, target_index)


if __name__ == "__main__":
    main()
