from dataclasses import dataclass


@dataclass
class GameSettings:
    speed: int = 20
    state_size: int = 11
    hidden_layer_size: int = 256
    action_size: int = 3
    max_memory: int = 100
    epsilon_decay: int = 80
    learning_rate: float = 0.001
    batch_size: int = 1000
    gamma: float = 0.9