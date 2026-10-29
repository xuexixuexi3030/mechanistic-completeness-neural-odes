import math
import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    activations = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'swish': nn.SiLU, 'sigmoid': nn.Sigmoid, 'gelu': nn.GELU}
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}")
    return activations[name]()

def init_weights(module: nn.Module, method: str='glorot_normal'):
    if isinstance(module, nn.Linear):
        if method == 'glorot_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'glorot_normal':
            nn.init.xavier_normal_(module.weight)
        elif method == 'zeros':
            nn.init.zeros_(module.weight)
        else:
            raise ValueError(f"Unknown init method '{method}'")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class FNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int=4, hidden_units: int=50, activation: str='tanh', init_method: str='glorot_normal'):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(get_activation(activation))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(lambda m: init_weights(m, init_method))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SmallFNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int=3, hidden_units: int=32, activation: str='tanh', init_strategy: str='all_small'):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(get_activation(activation))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_small_weights(init_strategy)

    def _init_small_weights(self, strategy: str):
        linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        if strategy not in {'all_small', 'small_output'}:
            raise ValueError(f'Unknown SmallFNN init strategy: {strategy}')
        for index, module in enumerate(linear_layers):
            nn.init.xavier_normal_(module.weight)
            if strategy == 'all_small' or index == len(linear_layers) - 1:
                module.weight.data *= 0.01
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class OutputTransform(nn.Module):

    def __init__(self, transform: str='abs'):
        super().__init__()
        self.transform = transform

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.transform == 'abs':
            return torch.abs(y)
        elif self.transform == 'softplus':
            return torch.nn.functional.softplus(y)
        elif self.transform == 'none':
            return y
        raise ValueError(f"Unknown transform '{self.transform}'")
