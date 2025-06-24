import os
import numpy as np
from typing import Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


START_END = np.array(
    [[0.0, 0.0, 0.0], 
    [0.54020254, -0.77711392, 0.32291667]]
)

def inverse_transform_polyline(transformed_points, start_and_end, handleCollinear=True, epsilon=1e-6):
    tgt_start, tgt_end = start_and_end
    offset = - transformed_points[0]

    lengths = np.linalg.norm(transformed_points[-1] - transformed_points[0])

    # Step 1: inverse the translation
    transformed_points = transformed_points + offset

    # Step 2: calculate the scale factor
    tgt_direction = tgt_end - tgt_start
    scale_factor = np.linalg.norm(tgt_direction)

    # check if the scale factor is zero, avoid division by zero
    if scale_factor == 0:
        raise ValueError("The start and end points are the same, so the scale factor cannot be determined.")

    # Step 3: inverse the scaling
    scaled_back_points = transformed_points * scale_factor / (lengths + epsilon)

    # Step 4: calculate the inverse of the rotation matrix
    target_vector = tgt_direction

    # check if the pn_prime is a zero vector
    pn_prime = scaled_back_points[-1]
    if np.linalg.norm(pn_prime) == 0:
        raise ValueError("The transformed polyline's end point is at the origin, so the direction cannot be determined.")

    # normalize the vector
    pn_prime_norm = pn_prime / (np.linalg.norm(pn_prime) + epsilon)
    target_norm = target_vector / (np.linalg.norm(target_vector) + epsilon)

    # calculate the dot product and angle
    dot_product = np.dot(pn_prime_norm, target_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # check if the dot product is close to -1
    if np.abs(dot_product + 1) < epsilon:
        if not handleCollinear:
            return None
        
        # the vectors are opposite, choose an arbitrary vector orthogonal to pn_prime_norm as the rotation axis
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(pn_prime_norm, arbitrary_vector) or np.allclose(pn_prime_norm, -arbitrary_vector):
            arbitrary_vector = np.array([0, 1, 0])
        axis = np.cross(pn_prime_norm, arbitrary_vector)
        axis = axis / (np.linalg.norm(axis) + epsilon)
        angle = np.pi
    
    elif np.abs(dot_product - 1) < epsilon:
        if not handleCollinear:
            return None
        
        # the vectors are the same, no rotation is needed
        axis = np.array([0, 0, 1])  # the axis is arbitrary, because the angle is 0
        angle = 0.0

    else:
        # calculate the rotation axis
        axis = np.cross(pn_prime_norm, target_norm)
        axis = axis / (np.linalg.norm(axis) + epsilon)

    # Rodrigues' rotation formula for inverse rotation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K) 

    # Step 5: inverse the rotation
    rotated_back_points = np.dot(scaled_back_points, R.T)

    # Step 6: inverse the translation
    restored_points = rotated_back_points + tgt_start

    return restored_points

def denorm_curves(
    norm_curves: np.ndarray, 
    corners: np.ndarray
) -> Optional[np.ndarray]:
    """
    use the given corners to denormalize the curves
    """

    curves = []
    for i, corner in enumerate(corners):
        if np.linalg.norm(corner[0] - corner[1]) == 0:
            logger.warning(f"Corner {i} has zero length.")
            continue
        
        curve_i_temp = inverse_transform_polyline(norm_curves[i], start_and_end=START_END) 
        curve_i = inverse_transform_polyline(curve_i_temp, start_and_end=corner)             
        
        if curve_i is None:
            logger.warning(f"Curve {i} is None.")
            continue
        
        curves.append(curve_i)

    if curves:
        return np.stack(curves, axis=0)
    else:
        return None


def polylines_to_png(
    polylines, 
    filename='multi_polyline.png',
    dpi=72,
    figsize=(5, 5),
    linewidth=2,
    n_ticks=5,
    markersize=3,
):
    """
    render and save the multiple 3D polylines as a PNG file
    
    Args:
    - polylines: List of (N, 3) numpy array or list of points, each element is a polyline
    - filename: path to save the output PNG file
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # plot each polyline, using the default color cycle
    for pts in polylines:
        pts_arr = np.asarray(pts)
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                linewidth=linewidth, markersize=markersize)


    # ==== set the same scale for each axis ====
    for i, axis in enumerate(['x', 'y', 'z']):
        getattr(ax, f'set_{axis}lim')(-1.0, 1.0)


    ax.set_box_aspect((1, 1, 1))
    # ax.set_proj_type('ortho')

    # --------------------------
    # control the number of major ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.zaxis.set_major_locator(MaxNLocator(n_ticks))
    # also adjust the font size
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        for label in axis.get_ticklabels():
            label.set_fontsize(8)
    # --------------------------
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)



def save_curves(
    curves: np.ndarray,
    uid: str,
    tgt_dir_path: str,
    save_png: bool = False,
):
    tgt_npy_dir = tgt_dir_path + '/npy'
    os.makedirs(tgt_npy_dir, exist_ok=True)
    tgt_file_path = tgt_npy_dir + f'/{uid}.npy'
    np.save(tgt_file_path, curves)
    
    if save_png:
        tgt_png_dir = tgt_dir_path + '/png'
        os.makedirs(tgt_png_dir, exist_ok=True)
        tgt_png_file_path = tgt_png_dir + f'/{uid}.png'
        polylines_to_png(curves, filename=tgt_png_file_path)
