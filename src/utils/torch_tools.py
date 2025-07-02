import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torchtyping import TensorType
from torch.nn import Module

def fmt(v):
    if torch.is_tensor(v) and v.dim() == 0:
        v = v.item()
    if isinstance(v, (float, int)):
        return f"{v:.3f}"
    else:
        return str(v)


def interpolate_1d(
    t: TensorType["bs", "n"],
    data: TensorType["bs", "c", "n"],
):
    """
    Perform 1D linear interpolation on the given data.

    Args:
    t (Tensor): Interpolation coordinates, with values in the range [0, 1], 
                of shape (batch_size, n).
    data (Tensor): Original data to be interpolated, 
                   of shape (batch_size, channels, num_points).

    Returns:
    Tensor: Interpolated data of shape (batch_size, channels, n).
    """
    # Check if input tensors have the expected dimensions
    assert t.dim() == 2, "t should be a 2D tensor with shape (batch_size, n)"
    assert data.dim() == 3, "data should be a 3D tensor with shape (batch_size, channels, num_points)"
    assert (0 <= t).all() and (t <= 1).all(), "t must be within [0, 1]"

    # Map interpolation coordinates from [0, 1] to [0, num_reso - 1]
    num_reso = data.shape[-1]
    t = t * (num_reso - 1)

    left = torch.floor(t).long()
    right = torch.ceil(t).long()
    alpha = t - left

    left = torch.clamp(left, max=num_reso - 1)
    right = torch.clamp(right, max=num_reso - 1)

    c = data.shape[-2]

    left = repeat(left, 'bs n -> bs c n', c=c)
    left_values = torch.gather(data, -1, left)

    right = repeat(right, 'bs n -> bs c n', c=c)
    right_values = torch.gather(data, -1, right)

    alpha = repeat(alpha, 'bs n -> bs c n', c=c)

    interpolated = (1 - alpha) * left_values + alpha * right_values
    
    return interpolated


def calculate_polyline_lengths(points: TensorType['b', 'n', 'c', float]) -> TensorType['b', float]:
    """
    Calculate the lengths of a batch of polylines.

    Args:
    points (torch.Tensor): Tensor of shape (batch_size, num_points, c),
                            where batch_size is the number of polylines,
                            num_points is the number of points per polyline,
                            and c corresponds to the 2D/3D coordinates of each point.

    Returns:
    torch.Tensor: Tensor of shape (batch_size,) representing the total length of each polyline.
    """

    if points.dim() != 3:
        raise ValueError("Input tensor must have shape (batch_size, num_points, c)")

    diffs = points[:, 1:, :] - points[:, :-1, :]
    distances = torch.norm(diffs, dim=2)
    polyline_lengths = distances.sum(dim=1)

    return polyline_lengths

def sample_edge_points(batch_edge_points, num_points=32):
    # example: (batch_size, 256, 3) -> (batch_size, 32, 3)

    t = torch.linspace(0, 1, num_points).to(batch_edge_points.device)
    bs = batch_edge_points.shape[0]
    t = repeat(t, 'n -> b n', b=bs)
    
    batch_edge_points = rearrange(batch_edge_points, 'b n c -> b c n')
    batch_edge_points = interpolate_1d(t, batch_edge_points)
    batch_edge_points = rearrange(batch_edge_points, 'b c n -> b n c')
    
    return batch_edge_points

def point_seq_tangent(
    point_seq: torch.Tensor, 
    channel_dim: int = -1,
    seq_dim: int = -2,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    compute the tangent vectors of the point sequence
    """
    # use the difference of the points to compute the tangent vectors
    tangent = point_seq.diff(dim=seq_dim)

    # use the last tangent vector to complete the last point
    last_tangent = tangent.select(seq_dim, -1).unsqueeze(seq_dim)
    
    tangent = torch.cat([tangent, last_tangent], dim=seq_dim)
    tangent = F.normalize(tangent, dim=channel_dim, eps=eps)

    return tangent

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad