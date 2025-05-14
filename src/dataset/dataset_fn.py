import numpy as np
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor
import random

from src.utils.numpy_tools import (
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
)


def random_viewpoint():
    theta = np.random.uniform(np.radians(20), np.radians(110))
    phi = np.random.uniform(0, 2 * np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vec = np.array([x, y, z])
    return vec / np.linalg.norm(vec)



def spherical_flip(points, center, param):
    translated_points = points - center  # (N, 3)

    norm_points = np.linalg.norm(translated_points, axis=1)  # (N,)

    R = np.max(norm_points) * 1385

    factor = (2.0 * R / norm_points) - 1.0  # (N,)

    flipped_points = factor[:, None] * translated_points  # (N, 3)

    return flipped_points

def hidden_point_removal(cloud, campos, only_return_indices=False):

    cloud = cloud.astype(np.float16)

    center = np.asarray(campos).reshape(1, 3)  # (1, 3)

    flipped_cloud = spherical_flip(cloud, center, param=np.pi)  # (N, 3)

    points_for_hull = np.vstack((flipped_cloud, np.zeros((1, 3), dtype=cloud.dtype)))  # (N+1, 3)

    hull = ConvexHull(points_for_hull, qhull_options='Qx')

    visible_indices = hull.vertices[hull.vertices < len(cloud)]

    if only_return_indices: 
        return visible_indices
    else:
        return cloud[visible_indices]


# ========== scale and jitter ==========

def curve_yz_scale(vertices, interval=(0.9, 1.1), jitter=0.1):
    random_vals = np.random.randn(1, 3) / 3
    range_val = (interval[1] - interval[0])
    scale = (np.clip(random_vals, -1, 1) + 1) * range_val / 2  + interval[0]

    scale[0] = 1 # x no scale
    vertices = vertices * scale
    
    return vertices

def scale_and_jitter_wireframe_set(vertices, interval=(0.85, 1.15), jitter=0.1):
    scale_ratio = np.random.rand()
    vertices *= scale_ratio * (interval[1] - interval[0]) + interval[0]
    
    shift = np.random.rand(3) * 2 - 1
    vertices += shift * jitter
    
    vertices = np.clip(vertices, -0.999, 0.999)
    
    return vertices

@DeprecationWarning
def add_perlin_noise(points):
    std = random.uniform(0.05, 0.15)

    theta_x, theta_y, theta_z = np.random.rand(3) * 360

    rotation_matrix = rotation_matrix_x(theta_x) @ rotation_matrix_y(theta_y) @ rotation_matrix_z(theta_z)

    # 使用合并后的旋转矩阵一次性旋转点云
    rotated_points = points @ rotation_matrix.T

    # 使用并行化加速噪声计算
    def compute_noise(x, y, z):
        return opensimplex.noise3(x, y, z)
    
    with ThreadPoolExecutor() as executor:
        noise = np.array(list(executor.map(compute_noise, rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2])))
    
    noise = noise * std
    
    return points + np.repeat(noise[:, np.newaxis], 3, axis=1)

def add_gaussian_noise(points, noise_level=0.01):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def transform_point_cloud(pc, R=None, S=None, t=None, add_gaussian=False, noise_level=0.01):
    
    transformed_points = pc[:, :3]
    transformed_normals = pc[:,3:6]

    if R is not None:
        transformed_points = transformed_points @ R.T
        transformed_normals = transformed_normals @ R.T

    if S is not None:
        transformed_points = transformed_points @ S.T
        inverse_transpose_S = np.linalg.inv(S).T
        transformed_normals = transformed_normals @ inverse_transpose_S

    if t is not None:
        transformed_points = transformed_points + t

    if add_gaussian:
        transformed_points = add_gaussian_noise(transformed_points, noise_level)
        transformed_normals = add_gaussian_noise(transformed_normals, noise_level)
    
    transformed_points = np.clip(transformed_points, -0.99, 0.99)
    
    transformed_normals = transformed_normals / np.linalg.norm(transformed_normals, axis=1, keepdims=True)

    pc = np.concatenate([transformed_points, transformed_normals], axis=1)
    
    return pc

def scale_and_jitter_pc(pc, interval=(0.8, 1.2), jitter=0.1, is_rotation=True, add_gaussian=True, noise_level=0.01):
    R = None
    if is_rotation:
        random_angle = 0
        
        random_angle_noise = np.random.rand(3) * 20 - 10
        random_angle += random_angle_noise
        
        random_axis = 3
        
        if random_axis == 0:
            R = rotation_matrix_x(random_angle[0])
        if random_axis == 1:
            R = rotation_matrix_y(random_angle[1])
        if random_axis == 2:
            R = rotation_matrix_z(random_angle[2])
        if random_axis == 3:
            Rx = rotation_matrix_x(random_angle[0])
            Ry = rotation_matrix_y(random_angle[1])
            Rz = rotation_matrix_z(random_angle[2])
            R = Rx @ Ry @ Rz

    scale_factors = np.random.rand(3) * (interval[1] - interval[0]) + interval[0]
    S = np.diag(scale_factors)
    
    t = (np.random.rand(3) * 2 - 1) * jitter

    pc = transform_point_cloud(pc, R, S, t, add_gaussian=add_gaussian, noise_level=noise_level)
    
    return pc


# ========== others ==========

def get_rotaion_matrix_3d(idx):
    # idx 0, 1, 2, 3
    angles = [0, 90, 180, 270]
    angle = angles[idx]
    rot_matrix = rotation_matrix_z(angle)
    
    return rot_matrix

def aug_pc_by_idx(pc, idx):
    # idx 0, 1, 2, 3, 4, 5, 6, 7
    
    rotation_idx = idx // 2
    flip_idx = idx % 2
    
    rotation_matrix = get_rotaion_matrix_3d(rotation_idx)    
    
    points = pc[:, :3]
    normals = pc[:, 3:6]
    
    points = np.dot(points, rotation_matrix.T)
    normals = normals @ rotation_matrix.T

    if flip_idx == 1:
        points[..., 0] = -points[..., 0]
        normals[..., 0] = -normals[..., 0]
    
    pc = np.concatenate([points, normals], axis=1)
    
    return pc