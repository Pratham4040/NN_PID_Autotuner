import torch
import torch.nn as nn
import torch.optim as optim
class PlantNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
        print("initialised parameters")
    def forward(self, x):
        print("forward")
        return self.net(x)
class NeuralPlantModel:
    def __init__(self):
        self.model = PlantNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.loss_fn = nn.MSELoss()
        self.data = []
        print("initialised optimizer and stuff")
    def add_sample(self, t1, t2, u1, u2, target):
        self.data.append(([t1, t2, u1, u2], target))
        print("added a data")
        if len(self.data) > 1000:
            self.data.pop(0)
            print("popped data cause it execed 1000")
    def train_step(self):

        if len(self.data) < 20:
            return

        x = torch.tensor([d[0] for d in self.data], dtype=torch.float32)
        y = torch.tensor([[d[1]] for d in self.data], dtype=torch.float32)

        pred = self.model(x)

        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("train step completed")
        return loss.item()

    def predict(self, t1, t2, u1, u2):
        x = torch.tensor([[t1, t2, u1, u2]], dtype=torch.float32)
        print("predicted a thing")
        print(x)
        with torch.no_grad():
            return self.model(x).item()