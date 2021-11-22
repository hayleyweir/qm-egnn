import torch
from torch import nn
from torch_geometric.loader import DataLoader
import os
# from torch_geometric.transforms import ToDevice
import time

def text_progress_bar(fraction):
    nchars = 50
    nfilled = int(nchars * fraction)
    return (
        "["
        + "=" * nfilled
        + " " * (nchars - nfilled)
        + "] "
        + f"{int(100*fraction):3d}%"
    )

def make_weights_dir():
    if not os.path.isdir("saved_weights"):
        os.mkdir("saved_weights")

def summary(epoch_train_loss, epoch_val_loss, epoch, epoch_time):
    print (f'Epoch {epoch}: \t Training loss {epoch_train_loss:.3f} \t Val loss {epoch_val_loss:.3f} \t Time for epoch: {epoch_time/60.:.3f} min')


def train_model(
    model,
    # device,
    train_dataset,
    val_dataset,
    batch_size=64,
    num_epochs=30,
    learning_rate=1e-2,
    target_index=0,  # 0=dipole, 4=homo-lumo gap
    gamma=0.999
):
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    loader_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    loader_val = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)  # Just a single batch...
    # batch_to_device = ToDevice(device)
    # batch_to_device = ToDevice(device, ["edge_index", "pos", "z", "batch", "y"])

    make_weights_dir()
    tic = time.time()

    loss_fn = nn.MSELoss()
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        tic = time.time()
        epoch_train_loss = 0
        for batch in loader_train:
            # batch = batch_to_device(batch)
            pred = model(batch)
            #             print(pred)
            loss = loss_fn(pred.flatten(), batch.y[:, target_index].flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            epoch_train_loss += loss.item()

        epoch_train_loss_mean = epoch_train_loss / len(loader_train)
        train_loss.append(epoch_train_loss_mean)

        with torch.no_grad():
            for batch in loader_val:
                # batch = batch_to_device(batch)
                pred = model(batch)
                loss = loss_fn(pred.flatten(), batch.y[:, target_index].flatten())
                epoch_val_loss = loss.item()

        val_loss.append(epoch_val_loss)

        # Save every epoch
        # torch.save(model.state_dict(), f'saved_weights/{epoch}_weights.pt')

        # Check if best validation loss and if so save to `BEST.pt`
        if epoch_val_loss <= min(val_loss):
            print (f' - BEST SCORE, SAVING WEIGHTS to `BEST.pt`')
            torch.save(model.state_dict(), f'saved_weights/BEST.pt')


        print(text_progress_bar(float((epoch + 1) / num_epochs)), end="\r", flush=True)
        toc = time.time()

        summary(epoch_train_loss_mean, epoch_val_loss, epoch, toc-tic)


    print("\n")

    return train_loss, val_loss
