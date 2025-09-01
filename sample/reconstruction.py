import numpy as np
import os
from typing import List, Optional, Tuple, Dict
from einops import rearrange
import logging
from sample.utils import denorm_curves, save_curves


def refine_segment_coords_by_adj(
    adj: np.ndarray, segment_coords: np.ndarray, 
    check_var: bool = False,
    threshold: float = 1e-4,
) -> np.ndarray:
    """
    Refine segment coords using adjacency information.
    """
    success = True

    node_coord_coords: Dict[int, List[np.ndarray]] = {}

    for i, (first, second) in enumerate(adj):
        first, second = int(first), int(second)
        node0_features = segment_coords[i, :3]  # [3]
        node1_features = segment_coords[i, 3:]  # [3]

        node_coord_coords.setdefault(first, []).append(node0_features)
        node_coord_coords.setdefault(second, []).append(node1_features)

    average_node_coord_coords = {}
    for node, coord_coords_list in node_coord_coords.items():
        coords_array = np.array(coord_coords_list)
        
        if check_var:
            # calculate the variance of coords_array
            var = np.var(coords_array, axis=0)
            
            if np.any(var > threshold):
                # success = False
                return None, False
        
        average_coords = np.mean(coords_array, axis=0)
        # average_coords = np.median(coords_array, axis=0)
        average_node_coord_coords[node] = average_coords

    updated_segment_coords = []
    for i, (first, second) in enumerate(adj):
        first, second = int(first), int(second)
        node1_avg_features = average_node_coord_coords[first]
        node2_avg_features = average_node_coord_coords[second]
        updated_edge = np.concatenate([node1_avg_features, node2_avg_features], axis=0)
        updated_segment_coords.append(updated_edge)

    return np.array(updated_segment_coords), success


def recon_and_save_wireframe_from_logits(
    pred_logits: dict,
    filenames: List[str],
    tgt_dir_path: str,
    recon_curve: bool = False,
    dec_curves: Optional[np.ndarray] = None,
    use_adj_refine_segment = True,
    check_valid: bool = False,
    logger: logging.Logger = None,
    threshold: float = 1e-4,
    **kwargs,
) -> None:
    """
    Reconstruct and save wireframes from decoder output.
    """

    pred_cls_logits = pred_logits['cls'].detach().cpu().numpy()
    num_curves = pred_cls_logits.argmax(axis=-1) + 1

    pred_segments = pred_logits['segments'].detach().cpu().numpy()    

    assert pred_segments.shape[-1] == 6

    pred_diffs_logits = pred_logits['diffs'].detach().cpu().numpy()
    assert pred_diffs_logits.shape[-1] == 38

    pred_col_diff_logits = pred_diffs_logits[..., :6]
    pred_row_diff_logits = pred_diffs_logits[..., 6:]

    col_diff = pred_col_diff_logits.argmax(axis=-1)
    row_diff = pred_row_diff_logits.argmax(axis=-1) + 1

    first_col = np.cumsum(col_diff, axis=-1)
    second_col = first_col + row_diff

    adj = np.stack([first_col, second_col], axis=-1)

    results = {}

    for i, filename in enumerate(filenames):
        num_curves_i = num_curves[i]

        adj_i = adj[i][:num_curves_i]
        
        pred_segments_i = pred_segments[i][:num_curves_i]
        
        if use_adj_refine_segment:
            updated_preds, success = refine_segment_coords_by_adj(
                adj_i, pred_segments_i, check_var=check_valid, threshold=threshold,
            )
        else:
            updated_preds = pred_segments_i
        
        if updated_preds is None:
            continue
        
        segments_i = updated_preds

        if recon_curve and dec_curves is not None:
            dec_curves_i = dec_curves[i][:num_curves_i]
            segments_i = rearrange(segments_i, "nl (c d) -> nl c d", c=2)
            curves = denorm_curves(dec_curves_i, segments_i)
            if curves is not None:
                uid = filename.split('_')[0]
                save_curves(curves, uid, tgt_dir_path, save_png=True)
                
            else:
                logger.warning(f"No curves to save for UID {filename}.")


        logger.info(f"Saved {filename}")
        
        results[filename] = success
    
    return results