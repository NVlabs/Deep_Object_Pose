"""
NVIDIA from jtremblay@gmail.com
"""
import albumentations as A
import colorsys
import glob
import io
import json
import math
from math import acos
from math import sqrt
from math import pi
import numpy as np
import os
from os.path import exists, basename, join
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms

from cuboid import CuboidVertexType, CuboidLineIndexes

def default_loader(path):
    return Image.open(path).convert("RGB")


def length(v):
    return sqrt(v[0] ** 2 + v[1] ** 2)


def dot_product(v, w):
    return v[0] * w[0] + v[1] * w[1]


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    cosx = dot_product(v, w) / (length(v) * length(w))
    rad = acos(cosx)  # in radians
    return rad * 180 / pi  # returns degrees


def py_ang(A, B=(1, 0)):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if (
        det < 0
    ):  # this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360 - inner



def append_dot(extensions):
    res = []

    for ext in extensions:
        if not ext.startswith("."):
            res.append(f".{ext}")
        else:
            res.append(ext)

    return res


def loadimages(root, extensions=["png"]):
    imgs = []
    extensions = append_dot(extensions)

    def add_json_files(
        path,
    ):
        for ext in extensions:
            for file in os.listdir(path):
                imgpath = os.path.join(path, file)
                if (
                    imgpath.endswith(ext)
                    and exists(imgpath)
                    and exists(imgpath.replace(ext, ".json"))
                ):
                    imgs.append(
                        (
                            imgpath,
                            imgpath.replace(path, "").replace("/", ""),
                            imgpath.replace(ext, ".json"),
                        )
                    )

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

        add_json_files(path)

    explore(root)

    return imgs


def loadweights(root):
    if root.endswith(".pth") and os.path.isfile(root):
        return [root]
    else:
        weights = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f)) and f.endswith(".pth")
        ]

        weights.sort()
        return weights


def loadimages_inference(root, extensions):
    imgs, imgsname = [], []
    extensions = append_dot(extensions)

    def add_imgs(
        path,
    ):
        for ext in extensions:
            for file in os.listdir(path):
                imgpath = os.path.join(path, file)
                if imgpath.endswith(ext) and exists(imgpath):
                    imgs.append(imgpath)
                    imgsname.append(imgpath.replace(root, ""))

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

        add_imgs(path)

    explore(root)

    return imgs, imgsname


class CleanVisiiDopeLoader(data.Dataset):
    def __init__(
        self,
        path_dataset,
        objects=None,
        sigma=1,
        output_size=400,
        extensions=["png"],
        debug=False,
        use_s3=False,
        buckets=[],
        endpoint_url=None,
    ):
        ###################
        self.path_dataset = path_dataset
        self.objects_interest = list(map(str.lower, objects))
        self.sigma = sigma
        self.output_size = output_size
        self.extensions = append_dot(extensions)
        self.debug = debug
        ###################

        self.imgs = []
        self.s3_buckets = {}
        self.use_s3 = use_s3

        if self.use_s3:
            import boto3
            self.session = boto3.Session()
            self.s3 = self.session.resource(
                service_name="s3", endpoint_url=endpoint_url
            )

            for bucket_name in buckets:
                try:
                    self.s3_buckets[bucket_name] = self.s3.Bucket(bucket_name)
                except Exception as e:
                    print(
                        f"Error trying to load bucket {bucket_name} for training data:",
                        e,
                    )

            for bucket in self.s3_buckets:
                bucket_objects = [
                    str(obj.key) for obj in self.s3_buckets[bucket].objects.all()
                ]

                jsons = set([json for json in bucket_objects if json.endswith(".json")])
                imgs = [
                    img
                    for img in bucket_objects
                    if img.endswith(tuple(self.extensions))
                ]

                for ext in self.extensions:
                    for img in imgs:
                        # Only add images that have a ground truth file
                        if img.endswith(ext) and img.replace(ext, ".json") in jsons:
                            # (img key, bucket name, json key)
                            self.imgs.append((img, bucket, img.replace(ext, ".json")))

        else:
            for path_look in path_dataset:
                self.imgs += loadimages(path_look, extensions=self.extensions)

        # np.random.shuffle(self.imgs)
        print("Number of Training Images:", len(self.imgs))

        if debug:
            print("Debuging will be save in debug/")
            if os.path.isdir("debug"):
                print(f'folder {"debug"}/ exists')
            else:
                os.mkdir("debug")
                print(f'created folder {"debug"}/')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        # load the data
        if self.use_s3:
            img_key, bucket, json_key = self.imgs[index]
            mem_img = io.BytesIO()

            object_img = self.s3_buckets[bucket].Object(img_key)
            object_img.download_fileobj(mem_img)

            img = np.array(Image.open(mem_img).convert("RGB"))

            object_json = self.s3_buckets[bucket].Object(json_key)
            data_json = json.load(object_json.get()["Body"])

            img_name = img_key[:-3]

        else:
            path_img, img_name, path_json = self.imgs[index]

            # load the image
            img = np.array(Image.open(path_img).convert("RGB"))

            # load the json file
            with open(path_json) as f:
                data_json = json.load(f)

        all_projected_cuboid_keypoints = []

        # load the projected cuboid keypoints
        for obj in data_json["objects"]:
            if (
                self.objects_interest is not None
                    and not obj["class"].lower() in self.objects_interest
            ):
                continue
            # load the projected_cuboid_keypoints
            if obj["visibility"] > 0:
                projected_cuboid_keypoints = obj["projected_cuboid"]
                # FAT dataset only has 8 corners for 'projected_cuboid'
                if len(projected_cuboid_keypoints) == 8:
                    projected_cuboid_keypoints.append(obj["projected_cuboid_centroid"])
            else:
                projected_cuboid_keypoints = [
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                ]
            all_projected_cuboid_keypoints.append(projected_cuboid_keypoints)

        if len(all_projected_cuboid_keypoints) == 0:
            all_projected_cuboid_keypoints = [
                [
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                    [-100, -100],
                ]
            ]

        # flatten the keypoints
        flatten_projected_cuboid = []
        for obj in all_projected_cuboid_keypoints:
            for p in obj:
                flatten_projected_cuboid.append(p)

        #######
        if self.debug:
            img_to_save = Image.fromarray(img)
            draw = ImageDraw.Draw(img_to_save)

            for ip, p in enumerate(flatten_projected_cuboid):
                draw.ellipse(
                    (int(p[0]) - 2, int(p[1]) - 2, int(p[0]) + 2, int(p[1]) + 2),
                    fill="green",
                )

            img_to_save.save(f"debug/{img_name.replace('.png','_original.png')}")
        #######

        # data augmentation
        transform = A.Compose(
            [
                A.RandomCrop(width=400, height=400),
                A.Rotate(limit=180),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.15, p=1
                ),
                A.GaussNoise(p=1),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        transformed = transform(image=img, keypoints=flatten_projected_cuboid)
        img_transformed = transformed["image"]
        flatten_projected_cuboid_transformed = transformed["keypoints"]

        #######

        # transform to the final output
        if not self.output_size == 400:
            transform = A.Compose(
                [
                    A.Resize(width=self.output_size, height=self.output_size),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            transformed = transform(
                image=img_transformed, keypoints=flatten_projected_cuboid_transformed
            )
            img_transformed_output_size = transformed["image"]
            flatten_projected_cuboid_transformed_output_size = transformed["keypoints"]

        else:
            img_transformed_output_size = img_transformed
            flatten_projected_cuboid_transformed_output_size = (
                flatten_projected_cuboid_transformed
            )

        #######
        if self.debug:
            img_transformed_saving = Image.fromarray(img_transformed)

            draw = ImageDraw.Draw(img_transformed_saving)

            for ip, p in enumerate(flatten_projected_cuboid_transformed):
                draw.ellipse(
                    (int(p[0]) - 2, int(p[1]) - 2, int(p[0]) + 2, int(p[1]) + 2),
                    fill="green",
                )

            img_transformed_saving.save(
                f"debug/{img_name.replace('.png','_transformed.png')}"
            )
        #######

        # update the keypoints list
        # obj x keypoint_id x (x,y)
        i_all = 0
        for i_obj, obj in enumerate(all_projected_cuboid_keypoints):
            for i_p, point in enumerate(obj):
                all_projected_cuboid_keypoints[i_obj][
                    i_p
                ] = flatten_projected_cuboid_transformed_output_size[i_all]
                i_all += 1

        # generate the belief maps
        beliefs = CreateBeliefMap(
            size=int(self.output_size),
            pointsBelief=all_projected_cuboid_keypoints,
            sigma=self.sigma,
            nbpoints=9,
            save=False,
        )
        beliefs = torch.from_numpy(np.array(beliefs))
        # generate affinity fields with centroid.
        affinities = GenerateMapAffinity(
            size=int(self.output_size),
            nb_vertex=8,
            pointsInterest=all_projected_cuboid_keypoints,
            objects_centroid=np.array(all_projected_cuboid_keypoints)[:, -1].tolist(),
            scale=1,
        )

        # prepare for the image tensors
        normalize_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img_tensor = normalize_tensor(Image.fromarray(img_transformed))
        img_original = to_tensor(img_transformed)

        ########
        if self.debug:
            imgs = VisualizeBeliefMap(beliefs)
            img, grid = save_image(
                imgs,
                f"debug/{img_name.replace('.png','_beliefs.png')}",
                mean=0,
                std=1,
                nrow=3,
                save=True,
            )
            imgs = VisualizeAffinityMap(affinities)
            save_image(
                imgs,
                f"debug/{img_name.replace('.png','_affinities.png')}",
                mean=0,
                std=1,
                nrow=3,
                save=True,
            )
        ########
        img_tensor[torch.isnan(img_tensor)] = 0
        affinities[torch.isnan(affinities)] = 0
        beliefs[torch.isnan(beliefs)] = 0

        img_tensor[torch.isinf(img_tensor)] = 0
        affinities[torch.isinf(affinities)] = 0
        beliefs[torch.isinf(beliefs)] = 0

        return {
            "img": img_tensor,
            "affinities": torch.clamp(affinities, -1, 1),
            "beliefs": torch.clamp(beliefs, 0, 1),
            "file_name": img_name,
            "img_original": img_original,
        }


def VisualizeAffinityMap(
    tensor,
    # tensor of (len(keypoints)*2)xwxh
    threshold_norm_vector=0.4,
    # how long does the vector has to be to be drawn
    points=None,
    # list of points to draw in white on top of the image
    factor=1.0,
    # by how much the image was reduced, scale factor
    translation=(0, 0)
    # by how much the points were moved
    # return len(keypoints)x3xwxh # stack of images
):
    images = torch.zeros(tensor.shape[0] // 2, 3, tensor.shape[1], tensor.shape[2])
    for i_image in range(0, tensor.shape[0], 2):  # could be read as i_keypoint

        indices = (
            torch.abs(tensor[i_image, :, :]) + torch.abs(tensor[i_image + 1, :, :])
            > threshold_norm_vector
        ).nonzero()

        for indice in indices:

            i, j = indice

            angle_vector = np.array([tensor[i_image, i, j], tensor[i_image + 1, i, j]])
            if length(angle_vector) > threshold_norm_vector:
                angle = py_ang(angle_vector)
                c = colorsys.hsv_to_rgb(angle / 360, 1, 1)
            else:
                c = [0, 0, 0]
            for i_c in range(3):
                images[i_image // 2, i_c, i, j] = c[i_c]
        if not points is None:
            point = points[i_image // 2]

            print(
                int(point[1] * factor + translation[1]),
                int(point[0] * factor + translation[0]),
            )
            images[
                i_image // 2,
                :,
                int(point[1] * factor + translation[1])
                - 1 : int(point[1] * factor + translation[1])
                + 1,
                int(point[0] * factor + translation[0])
                - 1 : int(point[0] * factor + translation[0])
                + 1,
            ] = 1

    return images


def VisualizeBeliefMap(
    tensor,
    # tensor of len(keypoints)xwxh
    points=None,
    # list of points to draw on top of the image
    factor=1.0,
    # by how much the image was reduced, scale factor
    translation=(0, 0)
    # by how much the points were moved
    # return len(keypoints)x3xwxh # stack of images in torch tensor
):
    images = torch.zeros(tensor.shape[0], 3, tensor.shape[1], tensor.shape[2])
    for i_image in range(0, tensor.shape[0]):  # could be read as i_keypoint

        belief = tensor[i_image].clone()
        belief -= float(torch.min(belief).item())
        belief /= float(torch.max(belief).item())

        belief = torch.clamp(belief, 0, 1)
        belief = torch.cat(
            [belief.unsqueeze(0), belief.unsqueeze(0), belief.unsqueeze(0)]
        ).unsqueeze(0)

        images[i_image] = belief

    return images


def GenerateMapAffinity(
    size, nb_vertex, pointsInterest, objects_centroid, scale, save=False
):
    # Apply the downscale right now, so the vectors are correct.

    img_affinity = Image.new("RGB", (int(size / scale), int(size / scale)), "black")
    # create the empty tensors
    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(2, int(size / scale), int(size / scale)))

    for i_pointsImage in range(len(pointsInterest)):
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]

            affinity_pair, img_affinity = getAfinityCenter(
                int(size / scale),
                int(size / scale),
                tuple((np.array(pointsImage[i_points]) / scale).tolist()),
                tuple((np.array(center) / scale).tolist()),
                img_affinity=img_affinity,
                radius=1,
            )

            affinities[i_points] = (affinities[i_points] + affinity_pair) / 2

            # Normalizing
            v = affinities[i_points].numpy()

            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero] /= norms[nonzero]
            yvec[nonzero] /= norms[nonzero]

            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec], [yvec]]))
    affinities = torch.cat(affinities, 0)

    return affinities


def getAfinityCenter(
    width, height, point, center, radius=7, tensor=None, img_affinity=None
):
    """
    Create the affinity map
    """
    if tensor is None:
        tensor = torch.zeros(2, height, width).float()

    # create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width, height), "black")
    totensor = transforms.Compose([transforms.ToTensor()])
    draw = ImageDraw.Draw(imgAffinity)
    r1 = radius
    p = point
    draw.ellipse((p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1), (255, 255, 255))

    del draw

    # compute the array to add the afinity
    array = (np.array(imgAffinity) / 255)[:, :, 0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array * angle_vector[0]], [array * angle_vector[1]]])

    if not img_affinity is None:
        # find the angle vector
        if length(angle_vector) > 0:
            angle = py_ang(angle_vector)
        else:
            angle = 0
        c = np.array(colorsys.hsv_to_rgb(angle / 360, 1, 1)) * 255
        draw = ImageDraw.Draw(img_affinity)
        draw.ellipse(
            (p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1),
            fill=(int(c[0]), int(c[1]), int(c[2])),
        )
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity


def CreateBeliefMap(size, pointsBelief, nbpoints, sigma=16, save=False):
    # Create the belief maps in the points
    beliefsImg = []
    for numb_point in range(nbpoints):
        array = np.zeros([size, size])
        out = np.zeros([size, size])

        for point in pointsBelief:
            p = [point[numb_point][1], point[numb_point][0]]
            w = int(sigma * 2)
            if p[0] - w >= 0 and p[0] + w < size and p[1] - w >= 0 and p[1] + w < size:
                for i in range(int(p[0]) - w, int(p[0]) + w + 1):
                    for j in range(int(p[1]) - w, int(p[1]) + w + 1):

                        # if there is already a point there.
                        array[i, j] = max(
                            np.exp(
                                -(
                                    ((i - p[0]) ** 2 + (j - p[1]) ** 2)
                                    / (2 * (sigma**2))
                                )
                            ),
                            array[i, j],
                        )

        beliefsImg.append(array.copy())

        if save:
            stack = np.stack([array, array, array], axis=0).transpose(2, 1, 0)
            imgBelief = Image.fromarray((stack * 255).astype("uint8"))
            imgBelief.save("debug/{}.png".format(numb_point))
    return beliefsImg


def crop(img, i, j, h, w):
    """Crop the given PIL.Image.
    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


class AddRandomContrast(object):
    """
    Apply some random image filters from PIL
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):

        contrast = ImageEnhance.Contrast(im)

        im = contrast.enhance(np.random.normal(1, self.sigma))

        return im


class AddRandomBrightness(object):
    """
    Apply some random image filters from PIL
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):

        contrast = ImageEnhance.Brightness(im)
        im = contrast.enhance(np.random.normal(1, self.sigma))
        return im


class AddNoise(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        t = torch.FloatTensor(tensor.size()).normal_(0, self.std)

        t = tensor.add(t)
        t = torch.clamp(t, -1, 1)  # this is expansive
        return t


irange = range


def make_grid(
    tensor,
    nrow=8,
    padding=2,
    normalize=False,
    range=None,
    scale_each=False,
    pad_value=0,
):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(
            "tensor or list of tensors expected, got {}".format(type(tensor))
        )

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(
                range, tuple
            ), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(
        pad_value
    )
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2, mean=None, std=None, save=True):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """

    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10, pad_value=1)
    if not mean is None:
        # ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        ndarr = (
            grid.mul(std)
            .add(mean)
            .mul(255)
            .byte()
            .transpose(0, 2)
            .transpose(0, 1)
            .numpy()
        )
    else:
        ndarr = (
            grid.mul(0.5)
            .add(0.5)
            .mul(255)
            .byte()
            .transpose(0, 2)
            .transpose(0, 1)
            .numpy()
        )
    im = Image.fromarray(ndarr)
    if save is True:
        im.save(filename)
    return im, grid


class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)
        self.width = im.size[0]

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_rectangle(self, point1, point2, line_color=(0, 255, 0), line_width=2):
        self.draw.rectangle([point1, point2], outline=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius,
            ]
            self.draw.ellipse(xy, fill=point_color, outline=point_color)

    def draw_text(self, point, text, text_color):
        """Draws text on image"""
        if point is not None:
            self.draw.text(point, text, fill=text_color)

    def draw_cube(self, points, color=(0, 255, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """
        # draw cuboid
        for l in CuboidLineIndexes:
            self.draw_line(points[l[0]], points[l[1]], color, line_width=2)
        # draw 'X' on top
        X_Indexes = [[CuboidVertexType.FrontTopRight, CuboidVertexType.RearTopLeft],
                     [CuboidVertexType.FrontTopLeft, CuboidVertexType.RearTopRight]]
        for x in X_Indexes:
            self.draw_line(points[x[0]], points[x[1]], color, line_width=2)
        # Draw center
        self.draw_dot(points[CuboidVertexType.Center], point_color=color, point_radius=6)

        for i in range(9):
            self.draw_text(points[i], str(i), (255, 0, 0))


def get_image_grid(tensor, nrow=3, padding=2, mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """

    # tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=1)
    if not mean is None:
        # ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    else:
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr)
    return im
