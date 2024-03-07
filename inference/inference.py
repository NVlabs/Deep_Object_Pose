#!/usr/bin/env python

import argparse
import cv2
import numpy as np
from PIL import Image
import os
import simplejson as json
import sys
import yaml

sys.path.append("../common/")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import loadimages_inference, loadweights, Draw


class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""

    def __init__(
        self,
        config,  # config yaml loaded eg dict
        weight,  # path to weight
        class_name,
    ):
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        # load network model, create PNP solver
        self.model = ModelData(
            name=class_name,
            net_path=weight,
        )
        self.model.load_net_model()
        print("Model Loaded")

        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(
            class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name])
        )
        self.class_name = class_name

        print("Ctrl-C to stop")

    def image_callback(
        self,
        img,
        camera_info,
        img_name,  # this is the name of the img file to save, it needs the .png at the end
        output_folder,  # folder where to put the output
        weight,
    ):
        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(
                camera_info["projection_matrix"]["data"], dtype="float64"
            ).copy()
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype="float64")
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype="float64")
            dist_coeffs.resize((len(camera_info.D), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(
                img, (int(scaling_factor * width), int(scaling_factor * height))
            )

        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        # dictionary for the final output
        dict_out = {"camera_data": {}, "objects": []}

        # Detect object
        results, _ = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect
        )

        # Publish pose and overlay cube on image
        for _, result in enumerate(results):
            if result["location"] is None:
                continue

            loc = result["location"]
            ori = result["quaternion"]

            dict_out["objects"].append(
                {
                    "class": self.class_name,
                    "location": np.array(loc).tolist(),
                    "quaternion_xyzw": np.array(ori).tolist(),
                    "projected_cuboid": np.array(result["projected_points"]).tolist(),
                }
            )

            # Draw the cube
            if None not in result["projected_points"]:
                points2d = []
                for pair in result["projected_points"]:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_color)

        # create directory to save image if it does not exist
        img_name_base = img_name.split("/")[-1]
        output_path = os.path.join(
            output_folder,
            weight.split("/")[-1].replace(".pth", ""),
            *img_name.split("/")[:-1],
        )
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        im.save(os.path.join(output_path, img_name_base))

        json_path = os.path.join(
            output_path, ".".join(img_name_base.split(".")[:-1]) + ".json"
        )
        # save the json files
        with open(json_path, "w") as fp:
            json.dump(dict_out, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outf",
        default="output",
        help="Where to store the output images and inference results.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="folder for data images to load.",
    )
    parser.add_argument(
        "--config",
        default="../config/config_pose.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--camera",
        default="../config/camera_info.yaml",
        help="Path to camera info file",
    )

    parser.add_argument(
        "--weights",
        "--weight",
        "-w",
        required=True,
        help="Path to weights or folder containing weights. If path is to a folder, then script will run inference with all of the weights in the folder. This could take a while if the set of test images is large.",
    )

    parser.add_argument(
        "--exts",
        nargs="+",
        type=str,
        default=["png"],
        help="Extensions for images to use. Can have multiple entries seperated by space. e.g. png jpg",
    )

    parser.add_argument(
        "--object",
        required=True,
        help="Name of class to run detections on.",
    )

    opt = parser.parse_args()

    # load the configs
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(opt.outf, exist_ok=True)

    # Load model weights
    weights = loadweights(opt.weights)

    if len(weights) < 1:
        print(
            "No weights found at specified directory. Please check --weights flag and try again."
        )
        exit()
    else:
        print(f"Found {len(weights)} weights. ")
        # Load inference images
    imgs, imgsname = loadimages_inference(opt.data, extensions=opt.exts)

    if len(imgs) == 0 or len(imgsname) == 0:
        print(
            "No input images found at specified path and extensions. Please check --data and --exts flags and try again."
        )
        exit()

    for w_i, weight in enumerate(weights): 
        dope_node = DopeNode(config, weight, opt.object)

        for i in range(len(imgs)):
            print(
                f"({w_i + 1} of  {len(weights)}) frame {i + 1} of {len(imgs)}: {imgsname[i]}"
            )
            img_name = imgsname[i]

            frame = cv2.imread(imgs[i])

            frame = frame[..., ::-1].copy()

            # call the inference node
            dope_node.image_callback(
                img=frame,
                camera_info=camera_info,
                img_name=img_name,
                output_folder=opt.outf,
                weight=weight,
            )

        print("------")
