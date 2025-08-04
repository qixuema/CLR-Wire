import torch
import numpy as np
import torch.nn.functional as F
from einops import repeat

START_END = np.array(
    [[0.0, 0.0, 0.0], 
    [0.54020254, -0.77711392, 0.32291667]]
)

START_END_R = np.array([
    [ 0.54020282, -0.77711348,  0.32291649],
    [ 0.77711348,  0.60790503,  0.1629285 ],
    [-0.32291649,  0.1629285 ,  0.93229779]
])

def safe_norm(v: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute the L2 norm of `v` along the last axis and clamp it to be at least `eps`.
    """
    return np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


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

def normalize_curves_to_start_end(edge_points, start_end, eps=1e-6):
    # Step 1: Translate to origin
    translated_points = edge_points - edge_points[:, :1, :]

    # Original and target vectors
    original_vec = translated_points[:, -1, :]
    target_vec = start_end[:, 1, :] - start_end[:, 0, :]

    original_length = safe_norm(original_vec, eps)
    original_norm = original_vec / original_length
    target_length = safe_norm(target_vec, eps)
    target_norm = target_vec / target_length

    dot_product = np.einsum('ij,ij->i', original_norm, target_norm).clip(-1, 1)
    angle = np.arccos(dot_product)

    axis = np.cross(original_norm, target_norm)
    axis_norm = safe_norm(axis, eps)
    axis /= axis_norm

    # Handle special cases
    mask_reverse = np.abs(dot_product + 1) < eps
    mask_same = np.abs(dot_product - 1) < eps

    axis[mask_same] = np.array([0, 0, 1])
    axis[mask_reverse] = np.cross(original_norm[mask_reverse], np.array([1, 0, 0]))
    axis[mask_reverse] /= safe_norm(axis[mask_reverse], eps)
    angle[mask_reverse] = np.pi
    angle[mask_same] = 0

    # Rodrigues' formula
    K = np.zeros((len(edge_points), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]

    R = np.eye(3) + np.sin(angle)[:, None, None] * K + (1 - np.cos(angle))[:, None, None] * np.matmul(K, K)
    rotated_points = np.einsum('bij,bkj->bki', R, translated_points)

    scale = target_length / original_length
    scaled_points = rotated_points * scale[:, :, None]

    final_points = scaled_points + start_end[:, :1, :]

    return final_points


def align_and_scale_to_unit_segment(edge_points):
    """
    assume the edge_points is already normalized to the START_END,
    then transform the edge_points from START_END to [[-1, 0, 0],[1, 0, 0]]
    """
    R = repeat(START_END_R, 'n c -> b n c', b=edge_points.shape[0])
    rotated_points = np.einsum('bij,bkj->bki', R, edge_points)

    scaled_points = rotated_points * 2

    scaled_points[:, :, 1:] -= scaled_points[:, -1:, 1:]
    scaled_points -= np.array([1, 0, 0])
    scaled_points[:, 0] = [-1, 0, 0]
    scaled_points[:, -1] = [1, 0, 0]

    return scaled_points


def normalize_curves(edge_points):
    assert edge_points.ndim == 3
    num_curves = edge_points.shape[0]
    tgt_start_ends = np.tile(START_END, (num_curves, 1, 1))
    edge_points_middle_status = normalize_curves_to_start_end(edge_points, tgt_start_ends)
    norm_edge_points = align_and_scale_to_unit_segment(edge_points_middle_status)
    return norm_edge_points