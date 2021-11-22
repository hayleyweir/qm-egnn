import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch import nn
from torch_scatter import segment_coo

from torch_geometric.transforms import ToDevice


class EGNN(nn.Module):
    def __init__(self, hidden_units=128, activation=nn.SiLU()):
        super(EGNN, self).__init__()

        self.activation = activation
        self.node_function = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            self.activation,
            nn.Linear(hidden_units, hidden_units),
        )

        self.edge_function = nn.Sequential(
            nn.Linear(hidden_units * 2 + 1, hidden_units),
            self.activation,
            nn.Linear(hidden_units, hidden_units),
            self.activation,
        )

        self.edge_inf = nn.Sequential(nn.Linear(hidden_units, 1), nn.Sigmoid())

    def forward(self, mol_input):

        receivers = mol_input.edge_index[0]
        senders = mol_input.edge_index[1]
        x = mol_input.x
        distances = mol_input.distances

        # Construction of messages
        messages = torch.cat(
            [x[receivers], x[senders], distances], axis=1
        )  # Need to include distance
        messages = self.edge_function(messages)
        message_weights = self.edge_inf(messages)  # Soft edges
        messages = messages * message_weights
        messages = segment_coo(messages, receivers)  # Aggregation of messages

        # Update the node representations
        x_new = torch.cat([x, messages], axis=1)
        x_new = self.node_function(x_new)
        return x_new


class EGNN_network(nn.Module):
    def __init__(self, hidden_units=128, activation=nn.SiLU(), n_layers=2):
        super(EGNN_network, self).__init__()

        self.activation = activation
        self.n_layers = n_layers

        self.embedding = nn.Embedding(20, hidden_units)
        self.egnn_list = nn.ModuleList(
            [
                EGNN(hidden_units=hidden_units, activation=activation)
                for i in range(n_layers)
            ]
        )

        self.output_layer_1 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            self.activation,
            nn.Linear(hidden_units, hidden_units),
        )

        self.output_layer_2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            self.activation,
            nn.Linear(hidden_units, 1),
        )

    def forward(self, mol_input):

        # Get the (squared) distances between all the atoms
        mol_input.x = self.embedding(mol_input.z)

        receivers = mol_input.edge_index[0]
        senders = mol_input.edge_index[1]
        coordinates = mol_input.pos
        distances = torch.square(
            coordinates[receivers] - coordinates[senders]
        ).sum(axis=1)
        distances = torch.unsqueeze(distances, 1)
        mol_input.distances = distances

        # EGNN layers
        for egnn_layer in self.egnn_list:
            mol_input.x = egnn_layer(mol_input)

        # Output layer
        y = self.output_layer_1(mol_input.x)
        y = segment_coo(y, mol_input.batch)
        y = self.output_layer_2(y)

        return y
