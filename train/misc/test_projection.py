import numpy as np
import pyrender

def transform_points(local_points, local_to_world_matrix):
    """
    Transforms 3D points from local to world coordinates using a given transformation matrix.

    Parameters:
    local_points (numpy array of shape (N, 3)): Points in local coordinates
    local_to_world_matrix (numpy array of shape (4, 4)): Transformation matrix from local to world coordinates

    Returns:
    world_points (numpy array of shape (N, 3)): Points in world coordinates
    """

    # Convert local points to homogeneous coordinates
    local_points_hom = np.concatenate([local_points, np.ones((local_points.shape[0], 1))], axis=-1)
    local_to_world_matrix_col_major = local_to_world_matrix.T
    print_matrix(local_to_world_matrix_col_major, "local_to_world_matrix_col_major")
    print_matrix(local_points_hom, "local_points_hom")

    world_points_hom = local_to_world_matrix_col_major @ local_points_hom.T

    # world_points_hom = np.dot(local_to_world_matrix.T, local_points_hom.T)
    # print_matrix(world_points_hom, "world_points_hom")
    print_matrix(local_to_world_matrix, "local_to_world_matrix")

    world_points = world_points_hom[:, :3]
    print_matrix(world_points, "world_points")
    return world_points

def print_matrix(matrix, name):
    print(f"--\n{name}: {matrix.shape} ")
    for row in matrix:
        print(row)

def get_image_space_points(points, view_proj_matrix):
    """
    Args:
        points: numpy array of N points (N, 3) in the world space. Points will be projected into the image space.
        view_proj_matrix: Desired view projection matrix, transforming points from world frame to image space of desired camera
    Returns:
        numpy array of shape (N, 3) of points projected into the image space.
    """

    homo = np.pad(points, ((0, 0), (0, 1)), constant_values=1.0)
    tf_points = np.dot(homo, view_proj_matrix)
    tf_points = tf_points / (tf_points[..., -1:])
    tf_points[..., :2] = 0.5 * (tf_points[..., :2] + 1)
    image_space_points = tf_points[..., :3]

    return image_space_points


import json
json_fp = "../sample_data/000000.json"

with open(json_fp) as f:
    data = json.load(f)

camera_data = data["camera_data"]
camera_view_matrix = np.array(camera_data["camera_view_matrix"])

cx = camera_data["intrinsics"]["cx"]
cy = camera_data["intrinsics"]["cy"]
fx = camera_data["intrinsics"]["fx"]
fy = camera_data["intrinsics"]["fy"]

camera_intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

view_projection_matrix = cam.get_projection_matrix(camera_data["width"], camera_data["height"])

view_projection_matrix = view_projection_matrix.T

points_3d = np.array(data["objects"][0]["local_cuboid"])
local_to_world_matrix = np.array(data["objects"][0]["local_to_world_matrix"])


world_points_3d = transform_points(points_3d, local_to_world_matrix)

print_matrix(view_projection_matrix, "view_projection_matrix")
image_space_points = get_image_space_points(world_points_3d, view_projection_matrix)

resolution = np.array([[camera_data["width"], camera_data["height"], 1.0]])
image_space_points *= resolution

projected_cuboid_points = [
    [pixel_coordinate[0], pixel_coordinate[1]] for pixel_coordinate in image_space_points
]
print("--\nimage_space_points: ")
for row in image_space_points:
    print(row)

