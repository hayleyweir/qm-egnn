import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9

from src.model import EGNN_network
from src.train import train_model
from src.eval import evaluate

np.random.seed(6)


def load_dataset(data_size):
    dataset = QM9(root='./QM9')
    dataset.shuffle()
    dataset = dataset[:data_size]
    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)
    test_size = int(data_size * 0.1)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size: train_size + val_size]
    test_dataset = dataset[train_size + val_size: train_size + val_size + test_size]

    print(f"Size of training set: {len(train_dataset)}")
    print(f"Size of val set: {len(val_dataset)}")
    print(f"Size of test set: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def setup_cuda():
    print(f"GPU is available: {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    device = torch.device('cuda:7')
    print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"GPU index: {device.index}")


def plot_loss(start_epoch=0):
    plt.plot(train_loss[start_epoch:], c="k", label="train")
    plt.plot(test_loss[start_epoch:], c="b", label="test")
    plt.legend()
    plt.savefig("Training_Val_Loss.png")
    plt.show()


if __name__ == "__main__":
    # 0. Set values
    data_size = 100
    batch_size = 10
    num_epochs = 5
    target_index = 0
    lr = 1e-5

    hidden_units = 128
    n_layers = 7

    # 1. Load training, validation and test datasets
    train_dataset, val_dataset, test_dataset = load_dataset(data_size)

    # 2. Set up CUDA environment
    # setup_cuda()

    # 3. Initiate EGNN model
    model = EGNN_network(hidden_units=hidden_units, n_layers=n_layers)
    print(f"Number of parameters: {sum([tensor.flatten().size()[0] for tensor in model.parameters()])}")

    # 4. Train model
    train_loss, test_loss = train_model(model, train_dataset, val_dataset, num_epochs=num_epochs,
                                        batch_size=batch_size, target_index=target_index, learning_rate=lr)

    # 5. Plot loss
    plot_loss()

    # 6. Test model on test set
    # model = EGNN_network(hidden_units=hidden_units, n_layers=n_layers)
    model.load_state_dict(torch.load("saved_weights/BEST.pt"))
    model.eval()

    evaluate(model, test_dataset, target_index)
