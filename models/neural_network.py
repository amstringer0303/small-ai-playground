import numpy as np
import torch
from torch import nn

from models.base import ModelAdapter

torch.set_num_threads(1)


class TinyNetwork(nn.Module):
    def __init__(self, inputs, hidden, classes):
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden)
        self.output = nn.Linear(hidden, classes)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


class NeuralAdapter(ModelAdapter):
    def __init__(self, inputs, hidden, classes, seed):
        torch.manual_seed(seed)
        self.network = TinyNetwork(inputs, hidden, classes)
        self.reset_state = self.parameters()

    def parameters(self):
        return {name: value.detach().cpu().numpy().tolist() for name, value in self.network.named_parameters()}

    def fit(self, x, y, sample_weight, config, validation=None):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        sample_weight = torch.as_tensor(sample_weight, dtype=torch.float32)
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if config.epochs > 0 and not parameters:
            raise ValueError("All layers are frozen. Use evaluate-only or leave a trainable layer.")
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate) if parameters else None
        history = []
        if validation is not None:
            vx, vy = validation
            vx = torch.as_tensor(vx, dtype=torch.float32)
            vy = torch.as_tensor(vy, dtype=torch.long)
        self.network.train()
        histogram_bound = max(4.0, max(p.detach().abs().max().item() for p in self.network.parameters()) + 1)
        for epoch in range(config.epochs):
            optimizer.zero_grad()
            losses = nn.functional.cross_entropy(self.network(x), y, reduction="none")
            loss = (losses * sample_weight).sum() / sample_weight.sum()
            loss.backward()
            gradients = [p.grad.detach().abs().mean().item() for p in parameters if p.grad is not None]
            optimizer.step()
            with torch.no_grad():
                weights = torch.cat([p.flatten() for p in self.network.parameters()])
                histogram, edges = np.histogram(weights.numpy(), bins=20, range=(-histogram_bound, histogram_bound))
                validation_loss = nn.functional.cross_entropy(self.network(vx), vy).item() if validation is not None else None
                history.append({"epoch": epoch + 1, "loss": float(loss.item()),
                                "validation_loss": validation_loss,
                                "gradient_mean": float(np.mean(gradients)) if gradients else 0.0,
                                "weight_mean": weights.mean().item(), "weight_std": weights.std().item(),
                                "weight_histogram": histogram.tolist(), "histogram_edges": edges.tolist(),
                                "weight_min": weights.min().item(), "weight_max": weights.max().item()})
        self.network.eval()
        return history

    def predict_proba(self, x):
        self.network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return torch.softmax(self.network(tensor), dim=1).numpy()
