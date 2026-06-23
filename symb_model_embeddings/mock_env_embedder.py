import numpy as np
import torch

def mock_embedding(size: int):
    #np.random.default_rng(42)
    torch_generator = torch.Generator().manual_seed(42)
    embedding_vector = torch.rand(size, generator=torch_generator)
    return embedding_vector

SIZE = 32
SYMB_EMBEDDINGS = {key: mock_embedding(SIZE) for key in ["pickup", "putdown", "pushdown", "pushleft", "pushright", "pushup"]}
