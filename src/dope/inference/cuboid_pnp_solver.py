# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import cv2
import numpy as np
from cuboid import CuboidVertexType
from pyrr import Quaternion


class CuboidPNPSolver(object):
    """
    This class is used to find the 6-DoF pose of a cuboid given its projected vertices.

    Runs perspective-n-point (PNP) algorithm.
    """

    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self, object_name="", camera_intrinsic_matrix = None, cuboid3d = None,
            dist_coeffs = np.zeros((4, 1))):
        self.object_name = object_name
        if (not camera_intrinsic_matrix is None):
            self._camera_intrinsic_matrix = camera_intrinsic_matrix
        else:
            self._camera_intrinsic_matrix = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        self._cuboid3d = cuboid3d

        self._dist_coeffs = dist_coeffs

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        '''Sets the camera intrinsic matrix'''
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        '''Sets the camera intrinsic matrix'''
        self._dist_coeffs = dist_coeffs

    def solve_pnp(self, cuboid2d_points, pnp_algorithm = None):
        """
        Detects the rotation and traslation
        of a cuboid object from its vertexes'
        2D location in the image
        """

        location = None
        quaternion = None
        projected_points = cuboid2d_points

        cuboid3d_points = np.array(self._cuboid3d.get_vertices())
        obj_2d_points = []
        obj_3d_points = []

        for i in range(CuboidVertexType.TotalVertexCount):
            check_point_2d = cuboid2d_points[i]
            # Ignore invalid points
            if (check_point_2d is None):
                continue
            obj_2d_points.append(check_point_2d)
            obj_3d_points.append(cuboid3d_points[i])

        obj_2d_points = np.array(obj_2d_points, dtype=float)
        obj_3d_points = np.array(obj_3d_points, dtype=float)

        valid_point_count = len(obj_2d_points)
        print(valid_point_count, "valid points found" )

        # Set PNP algorithm based on OpenCV version and number of valid points
        is_points_valid = False

        if pnp_algorithm is None:
            if CuboidPNPSolver.cv2majorversion == 2:
                is_points_valid = True
                pnp_algorithm = cv2.CV_ITERATIVE
            elif CuboidPNPSolver.cv2majorversion > 2:
                if valid_point_count >= 6:
                    is_points_valid = True
                    pnp_algorithm = cv2.SOLVEPNP_ITERATIVE
                elif valid_point_count >= 4:
                    is_points_valid = True
                    pnp_algorithm = cv2.SOLVEPNP_P3P
                    # This algorithm requires EXACTLY four points, so we truncate our
                    # data
                    obj_3d_points = obj_3d_points[:4]
                    obj_2d_points = obj_2d_points[:4]
                    # Alternative algorithms:
                    # pnp_algorithm = SOLVE_PNP_EPNP
            else:
                assert False, "DOPE will not work with versions of OpenCV earlier than 2.0"

        if is_points_valid:
            try:
                ret, rvec, tvec = cv2.solvePnP(
                    obj_3d_points,
                    obj_2d_points,
                    self._camera_intrinsic_matrix,
                    self._dist_coeffs,
                    flags=pnp_algorithm
                )
            except:
                # solvePnP will assert if there are insufficient points for the
                # algorithm
                print("cv2.solvePnP failed with an error")
                ret = False

            if ret:
                location = list(x[0] for x in tvec)
                quaternion = self.convert_rvec_to_quaternion(rvec)

                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self._camera_intrinsic_matrix, self._dist_coeffs)
                projected_points = np.squeeze(projected_points)

                # If the location.Z is negative or object is behind the camera then flip both location and rotation
                x, y, z = location
                if z < 0:
                    # Get the opposite location
                    location = [-x, -y, -z]

                    # Change the rotation by 180 degree
                    rotate_angle = np.pi
                    rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                    quaternion = rotate_quaternion.cross(quaternion)

        return location, quaternion, projected_points

    def convert_rvec_to_quaternion(self, rvec):
        '''Convert rvec (which is log quaternion) to quaternion'''
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)

        # Alternatively: pyquaternion
        # return Quaternion(axis=raxis, radians=theta)  # uses OpenCV's Quaternion (order is WXYZ)

    def project_points(self, rvec, tvec):
        '''Project points from model onto image using rotation, translation'''
        output_points, tmp = cv2.projectPoints(
            self.__object_vertex_coordinates,
            rvec,
            tvec,
            self.__camera_intrinsic_matrix,
            self.__dist_coeffs)

        output_points = np.squeeze(output_points)
        return output_points
