import torch
from torch import nn
from torch_geometric.loader import DataLoader


def evaluate(
        model,
        device,
        test_dataset,
        target_index,
):
    loader_test = DataLoader(test_dataset, batch_size=len(test_dataset))
    loss_fn = nn.MSELoss()
    batch_to_device = ToDevice(device, ["edge_index", "pos", "z", "batch", "y"])

    with torch.no_grad():
        for batch in loader_test:
            batch = batch_to_device(batch)
            pred = model(batch)
            loss = loss_fn(pred.flatten(), batch.y[:, target_index].flatten())
            test_loss = loss.item()

    print ("------------------------------------------------")
    print(f'Test set loss: {test_loss:.2f}')
    print("------------------------------------------------")
    return test_loss
