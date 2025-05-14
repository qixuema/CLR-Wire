import numpy as np

def normalize(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)

# ========= Transformations =========

def rotation_matrix_z(theta):
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])

def rotation_matrix_x(theta):
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])

def rotation_matrix_y(theta):
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

def check_npy_file(file_path):
    try:
        data = np.load(file_path)
    except Exception as e:
        return False
    
    return True

def check_nan_inf(data):
    contains_nan_inf = np.any(np.isnan(data)) or np.any(np.isinf(data))
    if contains_nan_inf:
        return True
    return False

def is_close(a, b, atol=1e-5):
    return np.isclose(a, b, atol=atol)


def interpolate_1d(
    t,
    data,
):
    """
    Perform 1D linear interpolation on given data.

    Args:
    t (np.ndarray): Interpolation coordinates in [0, 1], shape (n,).
    data (np.ndarray): Source data, shape (num_points, n_channels).

    Returns:
    np.ndarray: Interpolated data, shape (n, n_channels).
    """
    assert t.ndim == 1, "t must be a 1D array with shape (n,)"
    assert data.ndim == 2, "data must be a 2D array with shape (num_points, n_channels)"

    num_reso = data.shape[0]
    t = t * (num_reso - 1)

    left = np.floor(t).astype(np.int32)
    right = np.ceil(t).astype(np.int32)
    alpha = t - left

    left = np.clip(left, a_min=0, a_max=num_reso - 1)
    right = np.clip(right, a_min=0, a_max=num_reso - 1)


    left_values = data[left]
    right_values = data[right]

    alpha = alpha[:, None]

    interpolated = (1 - alpha) * left_values + alpha * right_values
    
    return interpolated