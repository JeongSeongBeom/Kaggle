import torch
import torch.nn

class TitanicModel(torch.nn):
    def __init__(self, emdedding_size, num_numeric_features):
        super().__init__()

