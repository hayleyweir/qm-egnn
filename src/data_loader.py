from torch_geometric.datasets import QM9


def load_dataset(dataset_size):
    dataset = QM9(root='./QM9')
    dataset.shuffle()
    dataset = dataset[:dataset_size]
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = int(dataset_size * 0.1)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size: train_size + val_size]
    test_dataset = dataset[train_size + val_size: train_size + val_size + test_size]

    print(f"Size of training set: {len(train_dataset)}")
    print(f"Size of val set: {len(val_dataset)}")
    print(f"Size of test set: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset