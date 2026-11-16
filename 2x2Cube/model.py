import torch
from torch import nn
from pathlib import Path
from cube import encode_onehot

# Value network: one-hot cube encoding (B,88) -> estimated moves to solved (B,1)

class DistanceNet(nn.Module):
    def __init__(self, input_size=88, hidden_sizes=(256,256)):
        super(DistanceNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        layers = []
        prev = input_size
        for size in self.hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        # No activation on output so it can be any real number
        layers.append(nn.Linear(prev, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def save_model(path, model, **info):
    # Save weights and what's needed to rebuild same architecture; the info holds
    # extra details (encoding, max_depth) and need to be plain python values (numbers, strings, lists)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "input_size": model.input_size,
        "hidden_sizes": list(model.hidden_sizes),
        "state_dict": model.state_dict(),
        "info": info,
    }, path)

def load_model(path):
    # Returns (model in eval mode, info). weights_only=True refuses to run code stored in the file
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DistanceNet(checkpoint["input_size"], checkpoint["hidden_sizes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["info"]

def network_heuristic(model):
    # Wrap trained DistanceNet as A* heuristic: cube -> estimated moves to solved
    model.eval()
    def h(cube):
        if cube.is_solved():
            return 0.0
        x = torch.from_numpy(encode_onehot(cube.cp[None], cube.co[None]))
        with torch.no_grad():
            # True distances aren't ever negative, so clamp at 0 only makes estimates better
            return max(0.0, model(x).item())
    return h