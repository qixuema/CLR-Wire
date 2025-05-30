import torch
import numpy as np
import torch.nn.functional as F


def point_seq_tangent(point_seq: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    compute the tangent vectors of the point sequence
    """
    # use the difference of the points to compute the tangent vectors
    tangent = point_seq[..., 1:, :] - point_seq[..., :-1, :]

    # use the last tangent vector to complete the last point
    last_tangent = tangent[..., -1:, :]
    # if the tangent is a numpy array, use np.concatenate
    if isinstance(tangent, np.ndarray):
        tangent = np.concatenate([tangent, last_tangent], axis=-2)
        norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
        tangent = tangent / (norm + eps)
    elif isinstance(tangent, torch.Tensor):
        tangent = torch.cat([tangent, last_tangent], dim=-2)
        tangent = F.normalize(tangent, dim=-1, eps=eps)

    return tangent