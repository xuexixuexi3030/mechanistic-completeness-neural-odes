from .pinn import PINNModel
from .neural_ode import NeuralODESolver
from .mode_a import ModeAModel
from .mode_b import ModeBModel
from .mode_c import ModeCModel
MODEL_REGISTRY = {'pinn': PINNModel, 'node': NeuralODESolver, 'mode_a': ModeAModel, 'mode_b': ModeBModel, 'mode_c': ModeCModel}
