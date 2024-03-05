import numpy as np

# def project_points(points_3d, intrinsics, view_matrix):
#     """
#     Project 3D points into 2D image space.

#     Parameters:
#     points_3d (numpy array of shape (N, 3)): 3D points
#     intrinsics (numpy array of shape (3, 3)): camera intrinsic matrix
#     view_matrix (numpy array of shape (4, 4)): camera view matrix

#     Returns:
#     points_2d (numpy array of shape (N, 2)): projected 2D points
#     """

#     print("points_3d: ", points_3d.shape)
#     print("intrinsics: ", intrinsics.shape)
#     print("view_matrix: ", view_matrix.shape)

#     # convert to homogeneous coordinates
#     points_3d_hom = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=-1)
#     print("points_3d_hom: ", points_3d_hom.shape)
#     # transform to camera coordinates
#     points_cam = np.dot(view_matrix, points_3d_hom.T)

#     # transform to normalized device coordinates
#     points_ndc = np.dot(intrinsics, points_cam[:3, :])

#     # perspective divide
#     points_2d = points_ndc[:2, :] / points_ndc[2, :]

#     return points_2d.T

import pyrender 

# def project_points(world_points, intrinsics, view_matrix):
#     """
#     Projects 3D points from world coordinates to 2D image coordinates.

#     Parameters:
#     world_points (numpy array of shape (N, 3)): Points in world coordinates
#     intrinsics (numpy array of shape (3, 3)): Camera intrinsics matrix
#     view_matrix (numpy array of shape (4, 4)): Camera view matrix

#     Returns:
#     image_points (numpy array of shape (N, 2)): Points in image coordinates
#     """

#     # Convert world points to homogeneous coordinates
#     world_points_hom = np.concatenate([world_points, np.ones((world_points.shape[0], 1))], axis=-1)

#     # Transform world points to camera coordinates
#     camera_points_hom = np.multiply(view_matrix, world_points_hom.T)

#     # Transform to normalized device coordinates
#     points_ndc = np.multiply(intrinsics, camera_points_hom[:3, :])

#     # Perspective divide
#     image_points = points_ndc[:2, :] / points_ndc[2, :]

#     return image_points.T

# def get_view_projection_matrix(camera_view_matrix, camera_intrinsics):

#     intrinsics = np.eye(4)

#     intrinsics[:3, :3] = camera_intrinsics

#     view_projection_matrix = intrinsics @  camera_view_matrix.T 

#     return view_projection_matrix 

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
json_fp = "/home/andrewg/centerpose/data/outf_all/hammers_test/test/003006/centerpose/000003.json"

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

# print_matrix(points_3d, "points_3d")

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


# projected_points = project_points(world_points_3d, camera_intrinsic_matrix, camera_view_matrix)

# print(projected_points)




# camera_data = data["camera_data"]
# camera_view_matrix = np.array(camera_data["camera_view_matrix"])

# cx = camera_data["intrinsics"]["cx"]
# cy = camera_data["intrinsics"]["cy"]
# fx = camera_data["intrinsics"]["fx"]
# fy = camera_data["intrinsics"]["fy"]

# camera_intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# points_3d = np.array(data["objects"][0]["local_cuboid"])

# projected_points = project_points(points_3d, camera_intrinsic_matrix, camera_view_matrix)

# print(projected_points)

# print(transform_and_project(data))


# ########## IGNORE 


# import numpy as np
# import json

# def transform_and_project(json_data):
#     # Get camera intrinsics and create intrinsic matrix
#     intrinsics_data = json_data['camera_data']['intrinsics']
#     intrinsics = np.array([[intrinsics_data['fx'], 0, intrinsics_data['cx']],
#                            [0, intrinsics_data['fy'], intrinsics_data['cy']],
#                            [0, 0, 1]])

#     # Get view matrix
#     view_matrix = np.array(json_data['camera_data']['camera_view_matrix'])

#     # Get local_to_world_matrix
#     local_to_world_matrix = np.array(json_data['objects'][0]['local_to_world_matrix'])

#     # Get local cuboid points
#     local_cuboid = np.array(json_data['objects'][0]['local_cuboid'])

#     # Transform local points to world coordinates
#     local_cuboid_hom = np.concatenate([local_cuboid, np.ones((local_cuboid.shape[0], 1))], axis=-1)
#     world_points = np.dot(local_to_world_matrix, local_cuboid_hom.T)

#     # Project points into 2D image space
#     projected_points = project_points(world_points[:3, :].T, intrinsics, view_matrix)

#     return projected_points

# def project_points(points_3d, intrinsics, view_matrix):
#     # Convert to homogeneous coordinates
#     points_3d_hom = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=-1)

#     # Transform to camera coordinates
#     points_cam = np.dot(view_matrix, points_3d_hom.T)

#     # Transform to normalized device coordinates
#     points_ndc = np.dot(intrinsics, points_cam[:3, :])

#     # Perspective divide
#     points_2d = points_ndc[:2, :] / points_ndc[2, :]
#     # points_2d = points_ndc
#     return points_2d.T


# view_projection_matrix = np.array([[1.66514147, 0.,         0.00469604, 0.        ],
# [ 0.00000000e+00 ,2.21693160e+00,-1.78270601e-03, 0.00000000e+00],
# [ 0. ,         0.  ,       -1.0010005 , -0.10005003],
# [ 0. , 0. ,-1.,  0.]])