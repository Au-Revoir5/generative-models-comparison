import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=784, h1=256, h2=64, n_classes=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_classes)
        self.drop = nn.Dropout(dropout)

    def hidden1(self, x):
        return F.relu(self.fc1(x.flatten(1)))

    def features(self, x):
        """z_MLP in R^64: the representation right before the final classifier."""
        return F.relu(self.fc2(self.drop(self.hidden1(x))))

    def forward(self, x):
        return self.fc3(self.drop(self.features(x)))

    def weights(self):
        """(W1, W2, W3) with shapes (256,784), (64,256), (10,64)."""
        return self.fc1.weight.detach(), self.fc2.weight.detach(), self.fc3.weight.detach()

    def biases(self):
        return self.fc1.bias.detach(), self.fc2.bias.detach(), self.fc3.bias.detach()
