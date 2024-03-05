"""
things needed 

predictions for image 
ground thruth for that image 
3d model loaded 

compare the poses.

"""


import argparse
import os
import numpy as np
import math

from scipy import spatial

import simplejson as json
import visii
import csv

from utils_eval import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_prediction",
        default="../inference/output",
        help="Path to prediction data.",
    )
    parser.add_argument(
        "--data",
        default="../sample_data/",
        help="path to ground truth data.",
    )
    parser.add_argument(
        "--models",
        default="../3d_models/YCB_models/",
        help="path to the 3D grocery models. ",
    )
    parser.add_argument("--outf", default="output", help="where to put the results.")
    parser.add_argument(
        "--adds", action="store_true", help="run ADDS, this might take a while"
    )
    parser.add_argument(
        "--cuboid", action="store_true", help="use cuboid to compute the ADD"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[0.02, 0.04, 0.06, 0.08, 0.10],
        help="Thresholds to compute accuracy for.",
    )

    opt = parser.parse_args()

    visii.initialize(headless=True)

    # Create Models
    meshes_gt = loadmodels(opt.models, cuboid=opt.cuboid, suffix="_gt")
    meshes_gu = loadmodels(opt.models, cuboid=opt.cuboid, suffix="_gu")

    # Get Ground Truth and Prediction Data
    data_truth = load_groundtruth(opt.data)
    prediction_folders = load_prediction(opt.data_prediction, data_truth)
    prediction_folders.sort()

    print(f"Number of Ground Truth Data : {len(data_truth)}")
    print(f"Number of Prediction Folders: {len(prediction_folders)}")

    # Make output directory if it does not exist
    os.makedirs(os.path.join(opt.outf), exist_ok=True)

    csv_file = open(os.path.join(opt.outf, "result.csv"), "w+")

    csv_writer = csv.writer(csv_file)

    # Write Header Row
    csv_writer.writerow(["--cuboid", opt.cuboid, "--adds", opt.adds])
    csv_writer.writerow(["Weights", "Object", "Total AUC"] + opt.thresholds)

    for pf_i, pf in enumerate(prediction_folders):
        adds_objects = {}
        adds_all = []

        count_all_gt = 0
        count_by_object = {}

        count_all_preds = 0
        count_by_object_preds = {}

        for f_i, gt_file in enumerate(data_truth):

            print(
                f"({pf_i + 1} of {len(prediction_folders)}) Computing file {f_i + 1} of {len(data_truth)}"
            )

            gt_json = None
            with open(os.path.join(opt.data, gt_file)) as json_file:
                gt_json = json.load(json_file)

            gu_json = None
            with open(os.path.join(opt.data_prediction, pf, gt_file)) as json_file:
                gu_json = json.load(json_file)

            objects_gt = []  # name of obj, pose

            for obj in gt_json["objects"]:

                name_gt = obj["class"]
                # remove "_16k" suffix from name of predicted object
                name_gt = name_gt[:-4] if name_gt.endswith("_16k") else name_gt
                objects_gt.append(
                    [
                        name_gt,
                        {
                            "rotation": visii.quat(
                                obj["quaternion_xyzw"][3],
                                obj["quaternion_xyzw"][0],
                                obj["quaternion_xyzw"][1],
                                obj["quaternion_xyzw"][2],
                            ),
                            "position": visii.vec3(
                                obj["location"][0] / 100,
                                obj["location"][1] / 100,
                                obj["location"][2] / 100,
                            ),
                        },
                    ]
                )

                count_all_gt += 1

                if name_gt in count_by_object:
                    count_by_object[name_gt] += 1
                else:
                    count_by_object[name_gt] = 1

            for obj_pred in gu_json["objects"]:
                name_pred = obj_pred["class"]

                # remove "_16k" suffix from name of predicted object
                name_pred = name_pred[:-4] if name_pred.endswith("_16k") else name_pred

                # need to add rotation for DOPE prediction, if your frames are aligned
                try:
                    pose_mesh = {
                        "rotation": visii.quat(
                            float(obj_pred["quaternion_xyzw"][3]),
                            float(obj_pred["quaternion_xyzw"][0]),
                            float(obj_pred["quaternion_xyzw"][1]),
                            float(obj_pred["quaternion_xyzw"][2]),
                        )
                        * visii.angleAxis(1.57, visii.vec3(1, 0, 0))
                        * visii.angleAxis(1.57, visii.vec3(0, 0, 1)),
                        "position": visii.vec3(
                            float(str(obj_pred["location"][0])) / 100.0,
                            float(str(obj_pred["location"][1])) / 100.0,
                            float(str(obj_pred["location"][2])) / 100.0,
                        ),
                    }
                except:
                    # in case there is an inf or null in the location prediction/gt
                    pose_mesh = {
                        "rotation": visii.quat(
                            float(obj_pred["quaternion_xyzw"][3]),
                            float(obj_pred["quaternion_xyzw"][0]),
                            float(obj_pred["quaternion_xyzw"][1]),
                            float(obj_pred["quaternion_xyzw"][2]),
                        )
                        * visii.angleAxis(1.57, visii.vec3(1, 0, 0))
                        * visii.angleAxis(1.57, visii.vec3(0, 0, 1)),
                        "position": visii.vec3(
                            1000000,
                            1000000,
                            1000000,
                        ),
                    }
                count_all_preds += 1

                if name_pred in count_by_object_preds:
                    count_by_object_preds[name_pred] += 1
                else:
                    count_by_object_preds[name_pred] = 1

                candidates = []
                for i_obj_gt, obj_gt in enumerate(objects_gt):
                    name_gt, pose_mesh_gt = obj_gt

                    if name_gt == name_pred:
                        candidates.append([i_obj_gt, pose_mesh_gt, name_gt])

                best_dist = float("inf")
                best_index = None

                for candi_gt in candidates:

                    # compute the add
                    i_gt, pose_gt, name_gt = candi_gt

                    visii_gt = meshes_gt[name_gt]

                    visii_gt.get_transform().set_position(pose_gt["position"])
                    visii_gt.get_transform().set_rotation(pose_gt["rotation"])

                    # name_pred should match the names of the folders containing 3d models
                    visii_gu = meshes_gu[name_pred]

                    visii_gu.get_transform().set_position(pose_mesh["position"])
                    visii_gu.get_transform().set_rotation(pose_mesh["rotation"])

                    if opt.adds and opt.cuboid:
                        dist = 0
                        for i_p in range(9):
                            corner_gt = visii.transform.get(
                                f"{name_gt + '_gt'}_cuboid_{i_p}"
                            )
                            dist_s = []
                            for i_ps in range(9):
                                corner_gu = visii.transform.get(
                                    f"{name_pred + '_gu'}_cuboid_{i_ps}"
                                )
                                gt_trans = corner_gt.get_local_to_world_matrix()
                                gu_trans = corner_gu.get_local_to_world_matrix()

                                dist_now = math.sqrt(
                                    (gt_trans[3][0] - gu_trans[3][0]) ** 2
                                    + (gt_trans[3][1] - gu_trans[3][1]) ** 2
                                    + (gt_trans[3][2] - gu_trans[3][2]) ** 2
                                )
                                dist_s.append(dist_now)
                            dist += min(dist_s)

                        dist /= 9
                    elif opt.adds and not opt.cuboid:
                        dist = []
                        dist2 = []
                        vertices = visii_gt.get_mesh().get_vertices()
                        points_gt = []
                        points_gu = []

                        for i in range(len(vertices)):
                            v = visii.vec4(
                                vertices[i][0], vertices[i][1], vertices[i][2], 1
                            )
                            p0 = (
                                visii_gt.get_transform().get_local_to_world_matrix() * v
                            )
                            p1 = (
                                visii_gu.get_transform().get_local_to_world_matrix() * v
                            )
                            points_gt.append([p0[0], p0[1], p0[2]])
                            points_gu.append([p1[0], p1[1], p1[2]])

                        dist = np.mean(
                            spatial.distance_matrix(
                                np.array(points_gt), np.array(points_gu), p=2
                            ).min(axis=1)
                        )

                    elif not opt.adds and opt.cuboid:
                        dist = 0

                        for i_p in range(9):
                            corner_gt = visii.transform.get(
                                f"{name_gt + '_gt'}_cuboid_{i_p}"
                            )

                            corner_gu = visii.transform.get(
                                f"{name_pred + '_gu'}_cuboid_{i_p}"
                            )

                            gt_trans = corner_gt.get_local_to_world_matrix()
                            gu_trans = corner_gu.get_local_to_world_matrix()

                            dist += math.sqrt(
                                (gt_trans[3][0] - gu_trans[3][0]) ** 2
                                + (gt_trans[3][1] - gu_trans[3][1]) ** 2
                                + (gt_trans[3][2] - gu_trans[3][2]) ** 2
                            )

                        dist /= 9

                    else:
                        dist = []
                        vertices = visii_gt.get_mesh().get_vertices()
                        for i in range(len(vertices)):
                            v = visii.vec4(
                                vertices[i][0], vertices[i][1], vertices[i][2], 1
                            )
                            p0 = (
                                visii_gt.get_transform().get_local_to_world_matrix() * v
                            )
                            p1 = (
                                visii_gu.get_transform().get_local_to_world_matrix() * v
                            )
                            dist.append(visii.distance(p0, p1))

                        dist = np.mean(dist)

                    if dist < best_dist:
                        best_dist = dist
                        best_index = i_gt

                if best_index is not None:
                    if not name_pred in adds_objects.keys():
                        adds_objects[name_pred] = []
                    adds_all.append(best_dist)
                    adds_objects[name_pred].append(best_dist)

        # Compute Metrics
        for name in adds_objects:

            auc_thresh = calculate_auc(
                thresholds=opt.thresholds,
                add_list=np.array(adds_objects[name]),
                total_objects=count_by_object[name],
            )

            auc_total = calculate_auc_total(
                add_list=np.array(adds_objects[name]),
                total_objects=count_by_object[name],
            )

            csv_writer.writerow([pf, name, auc_total] + auc_thresh)

    csv_file.close()

    visii.deinitialize()
