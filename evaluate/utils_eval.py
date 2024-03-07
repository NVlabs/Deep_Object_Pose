import os
import visii
import numpy as np


def create_obj(
    name="name",
    path_obj="",
    path_tex=None,
    scale=1,
    rot_base=None,  # visii quat
    pos_base=None,  # visii vec3
):

    # This is for YCB like dataset
    if path_obj in create_obj.meshes:
        obj_mesh = create_obj.meshes[path_obj]
    else:
        obj_mesh = visii.mesh.create_from_file(name, path_obj)
        create_obj.meshes[path_obj] = obj_mesh

    obj_entity = visii.entity.create(
        name=name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh=obj_mesh,
        transform=visii.transform.create(name),
        material=visii.material.create(name),
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1
    obj_entity.get_material().set_transmission(0)  # should 0 or 1
    obj_entity.get_material().set_roughness(1)  # default is 1

    if not path_tex is None:

        if path_tex in create_obj.textures:
            obj_texture = create_obj.textures[path_tex]
        else:
            obj_texture = visii.texture.create_from_file(name, path_tex)
            create_obj.textures[path_tex] = obj_texture

        obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    if not rot_base is None:
        obj_entity.get_transform().set_rotation(rot_base)
    if not pos_base is None:
        obj_entity.get_transform().set_position(pos_base)

    return obj_entity


create_obj.meshes = {}
create_obj.textures = {}


def add_cuboid(name, debug=False):
    obj = visii.entity.get(name)

    min_obj = obj.get_mesh().get_min_aabb_corner()
    max_obj = obj.get_mesh().get_max_aabb_corner()
    centroid_obj = obj.get_mesh().get_aabb_center()

    cuboid = [
        visii.vec3(max_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]),
    ]

    # change the ids to be like ndds / DOPE
    cuboid = [
        cuboid[2],
        cuboid[0],
        cuboid[3],
        cuboid[5],
        cuboid[4],
        cuboid[1],
        cuboid[6],
        cuboid[7],
        cuboid[-1],
    ]

    cuboid.append(visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]))

    for i_p, p in enumerate(cuboid):
        child_transform = visii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)
        child_transform.set_scale(visii.vec3(0.3))
        child_transform.set_parent(obj.get_transform())
        if debug:
            visii.entity.create(
                name=f"{name}_cuboid_{i_p}",
                mesh=visii.mesh.create_sphere(f"{name}_cuboid_{i_p}"),
                transform=child_transform,
                material=visii.material.create(f"{name}_cuboid_{i_p}"),
            )

    for i_v, v in enumerate(cuboid):
        cuboid[i_v] = [v[0], v[1], v[2]]

    return cuboid


def loadmodels(root, cuboid, suffix=""):
    models = {}

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [
            os.path.join(path, o)
            for o in os.listdir(path)
            if os.path.isdir(os.path.join(path, o))
        ]

        if len(folders) > 0:
            for path_entry in folders:
                explore(path_entry)
        else:
            path_obj = os.path.join(path, "textured_simple.obj")
            path_tex = os.path.join(path, "texture_map.png")

            if os.path.exists(path_obj) and os.path.exists(path_tex):
                path = path.rstrip("/")
                model_name = path.split("/")[-1]

                print(f"Loading Model: {model_name}")

                models[model_name] = create_obj(
                    name=model_name + suffix,
                    path_obj=path_obj,
                    path_tex=path_tex,
                    scale=0.01,
                )

                if cuboid:
                    add_cuboid(model_name + suffix)

                if "gu" in suffix:
                    models[model_name].get_material().set_metallic(1)
                    models[model_name].get_material().set_roughness(0.05)

    explore(root)

    return models


def load_groundtruth(root):
    gts = []

    def explore(path):

        if not os.path.isdir(path):
            return
        folders = [
            os.path.join(path, o)
            for o in os.listdir(path)
            if os.path.isdir(os.path.join(path, o))
        ]

        for path_entry in folders:
            explore(path_entry)

        gts.extend(
            [
                os.path.join(path, gt).replace(root, "").lstrip("/")
                for gt in os.listdir(path)
                if gt.endswith(".json") and not "settings" in gt
            ]
        )

    explore(root)

    return gts


def load_prediction(root, groundtruths):
    """
    Supports multiple prediction folders for one set of testing data.
    Each prediction folder must contain the same folder structure as the testing data directory.
    """
    subdirs = os.listdir(root)
    subdirs.append("")

    prediction_folders = []

    for dir in subdirs:
        valid_folder = True
        for gt in groundtruths:
            file_path = os.path.join(os.path.abspath(root), dir, gt)

            if not os.path.exists(file_path):
                valid_folder = False
                break

        if valid_folder:
            prediction_folders.append(dir)

    return prediction_folders


def calculate_auc(thresholds, add_list, total_objects):
    res = []
    for thresh in thresholds:
        under_thresh = len(np.where(add_list <= thresh)[0]) / total_objects

        res.append(under_thresh)

    return res


def calculate_auc_total(
    add_list, total_objects, delta_threshold=0.00001, max_threshold=0.1
):
    add_threshold_values = np.arange(0.0, max_threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add_list <= value)[0]) / total_objects
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / max_threshold

    return auc
