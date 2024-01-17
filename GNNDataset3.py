import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import csv

# Define the network used, layer 3 GCN
class GCN_NET3(torch.nn.Module):
    def __init__(self, num_features, hidden_size1, hidden_size2, classes):
        super(GCN_NET3, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv2 = GCNConv(hidden_size1, hidden_size2)
        self.conv3 = GCNConv(hidden_size2, classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, Graph):
        x, edge_index = Graph.x, Graph.edge_index
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, edge_index)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out, edge_index)
        out = self.softmax(out)
        return out

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

dataset = Planetoid(root='./', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_NET3(dataset.num_node_features, 128, 64, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

eval_T = 5
P = 3
i = 0
max_epoch = 300
setup_seed(seed=20)
temp_val_loss = 99999
L = []
L_val = []

model.train()
for epoch in range(max_epoch):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    _, val_pred = model(data).max(dim=1)
    loss_val = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])

    if (epoch % eval_T) == 0:
        if (temp_val_loss > loss_val):
            temp_val_loss = loss_val
            torch.save(model.state_dict(), "GCN_NET3.pth")
            i = 0
        else:
            i = i + 1
    if i > P:
        print("Early Stopping! Epoch: ", epoch)
        break

    L_val.append(loss_val.detach().item())
    val_correct = val_pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
    val_acc = val_correct / data.val_mask.sum()
    print('Epoch: {}  loss: {:.4f}  val_loss: {:.4f}  val_acc: {:.4f}'.format(epoch, loss.item(),
                                                                               loss_val.item(), val_acc.item()))
    L.append(loss.detach().item())
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load("GCN_NET3.pth"))
model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum()
print("Test accuracy is {:.4f}".format(acc.item()))

n = [i for i in range(len(L))]
plt.plot(n, L, label='train')
plt.plot(n, L_val, label='val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

training_accuracy = acc

"""
# Open the CSV file in append mode
with open("gnnCSV.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([training_accuracy])
"""
