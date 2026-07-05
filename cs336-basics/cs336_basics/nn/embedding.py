import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        :param num_embeddings: int, Size of the vocabulary
        :param embedding_dim: int, Dimension of the embedding vectors, i.e., d_model
        :param device: torch.device | None = None, Device to store the parameters on
        :param dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        # 初始化索引表权重
        weight = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        [batch_size, max_len] -> [batch_size, max_len, emb_dim]
        """
        return self.weight[token_ids]
