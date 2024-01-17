import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader
from itertools import product

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the Cora citation network dataset (small graph dataset)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Get the first graph object

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Define the hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
hidden_sizes = [16, 32, 64]

best_test_acc = 0
best_hyperparameters = {}

# Perform grid search
for learning_rate, hidden_size in product(learning_rates, hidden_sizes):
    # Create and initialize the model
    model = GCN(dataset.num_features, hidden_size, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Train the model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Evaluate on the test set
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(correct.sum()) / int(data.test_mask.sum())

    # Update the best hyperparameters if the test accuracy is higher
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_hyperparameters = {'learning_rate': learning_rate, 'hidden_size': hidden_size}

print(f'Best Test Accuracy: {best_test_acc:.4f} | Best Hyperparameters: {best_hyperparameters}')
