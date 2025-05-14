#!/usr/bin/env -S blenderproc run

import blenderproc as bp  # must be first!
from blenderproc.python.utility.Utility import Utility
import bpy

import argparse
import cv2
import glob
import json
from math import acos, atan, cos, pi, sin, sqrt
import numpy as np
import os
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import random
import sys


def random_object_position(near=5.0, far=40.0):
    # Specialized function to randomly place the objects in a visible
    # location
    x = 20 - 40*random.random()
    y = near + (far-near)*random.random()
    z = 20 - 40*random.random()
    return np.array([x, y, z])


def random_depth_in_frustrum(tw, th, bw, bh, depth):
    '''
    Generate a random depth within a frustrum. In order to get a uniform volume
    distribution, we want the probability density function to be proportional to
    the cross-sectional area of the frustum at the generated depth.

    tw, th - the width and height, respectively at the "top" (narrowest part)
             of the frustrum
    bw, bh - the width and height, respectively at the "bottom" (widest part)
             of the frustrum
    depth  - depth of frustrum (distance between 'top' and 'bottom')
    '''
    A = (tw - bw) * (th - bw)/(depth * depth)
    B = (bw * (tw - bw) + bw * (th - bw))/depth
    C = bw * bw
    area = depth * (C + depth * (0.5 * B + depth * A / 3.0))
    r = random.random() * area
    det = B * B - 4 * A * C
    part1 = B * (B * B - 6 * A * C) - 12 * A * A * r
    part2 = sqrt(part1 * part1 - det * det * det)
    part3 = pow(part1 + part2, 1./3.)
    return (-(B + det / part3 + part3) / (2 * A))


def point_in_frustrum(camera, near=10, far=20):
    fov_w, fov_h = camera.get_fov()

    tw = sin(fov_w)*near # top (nearest to camera) width of frustrum
    th = sin(fov_h)*near # top (nearest to camera) height of frustrum
    bw = sin(fov_w)*far  # bottom width
    bh = sin(fov_h)*far  # bottom height

    # calculate random inverse depth: 0 at the 'far' plane and 1 at the 'near' plane
    inv_depth = random_depth_in_frustrum(tw, th, bw, bh, far-near)
    depth = far-inv_depth

    nd = depth/(far-near) # normalized depth
    w = nd*(bw-tw)
    h = nd*(bh-th)

    # construct points so that we are looking down -Z, +Y is up, +X to right
    x = (0.5 - random.random())*w
    y = (0.5 - random.random())*h
    z = depth

    # orient them along the camera's view direction
    xform = camera.get_camera_pose()
    return (np.array([x,y,z,1]) @ xform)[0:3]


def Rx(A):
    return np.array([[1,      0,      0],
                     [0, cos(A), -sin(A)],
                     [0, sin(A),  cos(A)]])

def Ry(A):
    return np.array([[ cos(A), 0, sin(A)],
                     [      0, 1,      0],
                     [-sin(A), 0, cos(A)]])

def Rz(A):
    return np.array([[cos(A), -sin(A), 0],
                     [sin(A),  cos(A), 0],
                     [0,            0, 1]])

def ur():
    return 2.0*random.random() - 1.0

def random_rotation_matrix(max_angle=180):
    mr = pi*(max_angle/180.0)
    # Orient the board so a white square (sq #0) in UL corner
    RY = Ry(-0.5*pi)
    # add some random rotations
    return RY @ Rx(mr*ur()) @ Ry(mr*ur()) @ Rz(mr*ur())


def rotated_rectangle_extents(w, h, angle):
    """
    Given a rectangle of size W x H that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same, it
    # suffices to only look at the first quadrant and the absolute values of
    # sin,cos:
    sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr


def crop_around_center(image, width, height):
    """
    Crop 'image' (a PIL image) to 'width' and height' around the images center
    point
    """
    size = image.size
    center = (int(size[0] * 0.5), int(size[1] * 0.5))

    if(width > size[0]):
        width = size[0]

    if(height > size[1]):
        height = size[1]

    x1 = int(center[0] - width * 0.5)
    x2 = int(center[0] + width * 0.5)
    y1 = int(center[1] - height * 0.5)
    y2 = int(center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2)) # (left, upper, right, lower)


def crop_to_rotation(img, angle):
    # 'img' is a PIL Image of uint8 RGB values
    # 'angle' is in degrees
    angle_rad = angle*pi/180.0
    width, height = img.size

    img = img.rotate(angle)
    # Crop out black border resulting from rotation
    wr, hr = rotated_rectangle_extents(width, height, angle_rad)
    return crop_around_center(img, wr, hr)


def scale_to_original_shape(img, o_width, o_height):
    c_width, c_height = img.size
    o_ar = o_width/o_height
    c_ar = c_width/c_height
    if o_ar > c_ar:
        cropped = crop_around_center(img, c_width, c_width/o_ar)
    else:
        cropped = crop_around_center(img, c_height*o_ar, c_height)

    return cropped.resize((o_width, o_height))


def get_cuboid_image_space(mesh, camera):
    # object aligned bounding box coordinates in world coordinates
    bbox = mesh.get_bound_box()
    '''
    bbox is a list of the world-space coordinates of the corners of a
    blender object's oriented bounding box
     https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box
    The points from Blender are ordered like so:


            2 +-----------------+ 6
             /|                /|
            /                 / |
         1 +-----------------+ 5
           |     z    y      |  |
           |      | /        |  |
           |      |/         |  |
           |  |   *--- x     |  |
            3 +------------  |  + 7
           | /               | /
           |                 |/
         0 +-----------------+ 4

      But these points must be arranged to match the NVISII ordering
      (Deep_Object_Pose/data_generation/nvisii_data_gen/utils.py:927)

           3 +-----------------+ 0
            /                 /|
           /                 / |
        2 +-----------------+ 1|
          |     z    x      |  |
          |       | /       |  |
          |       |/        |  |
          |  y <--*         |  |
          | 7 +----         |  + 4
          |  /              | /
          | /               |/
        6 +-----------------+ 5

    '''

    centroid = np.array([0.,0.,0.])
    for ii in range(8):
        centroid += bbox[ii]
    centroid = centroid / 8

    cam_pose = np.linalg.inv(camera.get_camera_pose()) # 4x4 world to camera transformation matrx.
    # rvec & tvec describe the world to camera coordinate system
    tvec = -cam_pose[0:3,3]
    rvec = -cv2.Rodrigues(cam_pose[0:3,0:3])[0]
    K = camera.get_intrinsics_as_K_matrix()

    # However these points are in a different order than the original DOPE data format,
    # so we must reorder them (including coordinate frame changes)
    dope_order = [6, 2, 1, 5, 7, 3, 0, 4]

    cuboid = [None for ii in range(9)]
    for ii in range(8):
        cuboid[dope_order[ii]] = cv2.projectPoints(bbox[ii], rvec, tvec, K, np.array([]))[0][0][0]
    cuboid[8] = cv2.projectPoints(centroid, rvec, tvec, K, np.array([]))[0][0][0]

    return np.array(cuboid, dtype=float).tolist()


def write_json(outf, args, camera, objects, objects_data, seg_map):
    cam_xform = camera.get_camera_pose()
    eye = -cam_xform[0:3,3]
    at = -cam_xform[0:3,2]
    up = cam_xform[0:3,0]

    K = camera.get_intrinsics_as_K_matrix()

    data = {
        "camera_data" : {
            "width" : args.width,
            'height' : args.height,
            'camera_look_at':
            {
                'at': [
                    at[0],
                    at[1],
                    at[2],
                ],
                'eye': [
                    eye[0],
                    eye[1],
                    eye[2],
            ],
                'up': [
                    up[0],
                    up[1],
                    up[2],
                ]
            },
            'intrinsics':{
                'fx':K[0][0],
                'fy':K[1][1],
                'cx':K[0][2],
                'cy':K[1][2]
            }
        },
        "objects" : []
    }

    ## Object data
    ##
    for ii, oo in enumerate(objects):
        idx = ii+1 # objects ID indices start at '1'

        num_pixels = int(np.sum((seg_map == idx)))

        if num_pixels < args.min_pixels:
            continue
        projected_keypoints = get_cuboid_image_space(oo, camera)

        data['objects'].append({
            'class': objects_data[ii]['class'],
            'name': objects_data[ii]['name'],
            'visibility': num_pixels,
            'projected_cuboid': projected_keypoints,
            ## 'location' and 'quaternion_xyzw' are both optional data fields,
            ## not used for training
            'location': objects_data[ii]['location'],
            'quaternion_xyzw': objects_data[ii]['quaternion_xyzw']
        })

    with open(outf, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return data


def draw_cuboid_markers(objects, camera, im):
    colors = ['yellow', 'magenta', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'white']
    R = 2 # radius
    # draw dots on image to label the cuiboid vertices
    draw = ImageDraw.Draw(im)
    for oo in objects:
        projected_keypoints = get_cuboid_image_space(oo, camera)
        for idx, pp in enumerate(projected_keypoints):
            x = int(pp[0])
            y = int(pp[1])
            draw.ellipse((x-R, y-R, x+R, y+R), fill=colors[idx])

    return im


def randomize_background(path, width, height):
    img = Image.open(path)

    # Randomly rotate
    angle = 45.0 - random.random()*90.0
    img = crop_to_rotation(img, angle)
    img = scale_to_original_shape(img, width, height)

    # Randomly flip in horizontal and vertical directions
    if random.random() > 0.5:
        # flip horizontal
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        # flip vertical
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img


def set_world_background_hdr(filename, strength=1.0, rotation_euler=None):
    """
    Sets the background with a Poly Haven HDRI file

    strength: The brightness of the background.
    rot_euler: Optional euler angles to rotate the background.
    """
    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(filename, check_existing=True)

    # get the background node of the world shader and link the new texture node
    background_node = Utility.get_the_one_node_with_type(nodes, "Background")
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # Set the brightness
    background_node.inputs["Strength"].default_value = strength

    # add a mapping node and a texture coordinate node
    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    #link the texture coordinate node to mapping node and vice verse
    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    mapping_node.inputs["Rotation"].default_value = rotation_euler


def main(args):
    ## Segmentation values
    SEG_DISTRACT = 0

    ## All units used are in centimeters

    # Make output directories
    out_directory = os.path.join(args.outf, str(args.run_id))
    os.makedirs(out_directory, exist_ok=True)

    # Construct list of background images
    image_types = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.hdr', '*.HDR')
    backdrop_images = []
    if args.backgrounds_folder is not None:
        for ext in image_types:
            backdrop_images.extend(glob.glob(os.path.join(args.backgrounds_folder,
                                                          os.path.join('**', ext)),
                                             recursive=True))
        if len(backdrop_images) == 0:
            print(f"No images found in backgrounds directory '{args.backgrounds_folder}'")
        else:
            print(f"{len(backdrop_images)} images found in backgrounds directory "
                  f"'{args.backgrounds_folder}'")

    # Construct list of object models
    object_models = []
    tmp_p = None
    if args.path_single_obj:
        object_models.append(args.path_single_obj)
        tmp_p = args.path_single_obj
    else:
        object_models = glob.glob(args.objs_folder + "**/textured.obj", recursive=True)
        tmp_p = args.objs_folder
    if len(object_models) == 0:
        print(f"Failed to find any loadable models at {tmp_p}")
        exit(1)

    # Construct list of distractors
    distractor_objs = glob.glob(args.distractors_folder + "**/model.obj", recursive=True)
    print(f"{len(distractor_objs)} distractor objects found.")

    # Set up blenderproc
    bp.init()

    # Set the camera to be in front of the object
    cam_pose = bp.math.build_transformation_mat([0, -25, 0], [np.pi / 2, 0, 0])
    bp.camera.add_camera_pose(cam_pose)
    bp.camera.set_resolution(args.width, args.height)
    if args.focal_length:
        K = np.array([[args.focal_length, 0, args.width/2],
                      [0, args.focal_length, args.height/2],
                      [0,0,1]])
        bp.camera.set_intrinsics_from_K_matrix(K, args.width, args.height, clip_start=1.0,
                                               clip_end=1000.0)
    else:
        bp.camera.set_intrinsics_from_blender_params(lens=0.785398, # FOV in radians
                                                     lens_unit='FOV',
                                                     clip_start=1.0, clip_end=1000.0)

    # Create lights
    #bp.renderer.set_world_background([1,1,1], 1.0)
    #static_light = bp.types.Light()
    #static_light.set_type('SUN')
    #static_light.set_energy(1) # watts per sq. meter

    #light = bp.types.Light()
    #light.set_type('POINT')
    #light.set_energy(100) # watts per sq. meter

    light = bp.lighting.add_intersecting_spot_lights_to_camera_poses(5.0, 50.0)


    # Renderer setup
    bp.renderer.set_output_format('PNG')
    bp.renderer.set_render_devices(desired_gpu_ids=[0])


    # Create objects
    objects = []
    objects_data = []
    for idx in range(args.nb_objects):
        model_path =  object_models[random.randint(0, len(object_models) - 1)]
        obj = bp.loader.load_obj(model_path)[0]
        obj.set_cp("category_id", 1+idx)
        objects.append(obj)
        obj_class = args.object_class
        if obj_class is None:
            # e.g. 'models/Ketchup/google_16k/textured.obj'
            obj_class = os.path.split(os.path.split(os.path.split(model_path)[0])[0])[1]
        obj_name = obj_class + "_" + str(idx).zfill(3)
        objects_data.append({'class': obj_class,
                            'name': obj_name,
                            'id':1+idx
                            })

    # Create distractor(s)
    distractors = []
    if len(distractor_objs) > 0:
        for idx_obj in range(int(args.nb_distractors)):
            distractor_fn = distractor_objs[random.randint(0,len(distractor_objs)-1)]
            distractor = bp.loader.load_obj(distractor_fn)[0]
            distractor.set_cp("category_id", SEG_DISTRACT)
            distractors.append(distractor)
            print(f"loaded {distractor_fn}")

    for frame in range(args.nb_frames):
        # Randomize light
        #light.set_location([10-random.random()*20, 10-random.random()*20,
        #                    150+random.random()*100])

        # Place object(s)
        for idx, oo in enumerate(objects):
            # Set a random pose
            xform = np.eye(4)
            xform[0:3,3] = random_object_position(near=20, far=100)
            xform[0:3,0:3] = random_rotation_matrix()
            oo.set_local2world_mat(xform)

            # 'location' and 'quaternion_xyzw' describe the position and orientation of the
            # object in the camera coordinate system
            xform_in_cam = np.linalg.inv(bp.camera.get_camera_pose()) @ xform
            objects_data[idx]['location'] = xform_in_cam[0:3,3].tolist()
            tmp_wxyz = Quaternion(matrix=xform_in_cam[0:3,0:3]).elements  # [scalar, x, y, z]
            q_xyzw = [tmp_wxyz[1], tmp_wxyz[2], tmp_wxyz[3], tmp_wxyz[0]] # [x, y, z, scalar]
            objects_data[idx]['quaternion_xyzw'] = q_xyzw

            # Scale 3D model to cm
            oo.set_scale([args.scale, args.scale, args.scale])

        # Place distractors
        for dd in distractors:
            xform = np.eye(4)
            xform[0:3,3] = point_in_frustrum(bp.camera, near=5.0, far=100.)
            xform[0:3,0:3] = random_rotation_matrix()
            dd.set_local2world_mat(xform)
            dd.set_scale([args.distractor_scale, args.distractor_scale, args.distractor_scale])

        # Render the scene
        background_path = None
        if args.backgrounds_folder:
            background_path = backdrop_images[random.randint(0, len(backdrop_images) - 1)]
            if os.path.splitext(background_path)[1].lower() == ".hdr":
                strength = random.random()+0.5
                rotation = [random.random()*0.2-0.1, random.random()*0.2-0.1,
                            random.random()*0.2-0.1]
                set_world_background_hdr(background_path, strength, rotation)
            else:
                bp.renderer.set_output_format(enable_transparency=True)

        # redirect blenderproc output to log file
        logfile = '/tmp/blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)

        segs = bp.renderer.render_segmap()
        data = bp.renderer.render()

        # disable output redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

        im = Image.fromarray(data['colors'][0])

        if args.backgrounds_folder:
            if os.path.splitext(background_path)[1].lower() != ".hdr":
                # We have an ordinary image. We randomize its rotation and crop
                # and paste it in as a background
                background = randomize_background(background_path, args.width, args.height)
                background = background.convert('RGB') # some images may be B&W
                # Pasting the current image on the selected background
                background.paste(im, mask=im.convert('RGBA'))
                im = background

        if args.debug:
            im = draw_cuboid_markers(objects, bp.camera, im)

        filename = os.path.join(out_directory, str(frame).zfill(6) + ".png")
        im.save(filename)

        ## Export JSON file
        filename = os.path.join(out_directory, str(frame).zfill(6) + ".json")
        write_json(filename, args, bp.camera, objects, objects_data, segs['class_segmaps'][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Parameters passed from run script; they are ignored here
    parser.add_argument(
        '--nb_runs',
        default=1,
        type=int,
        help='Number of times the datagen script is run. Each time it is run, a new set of '
        'distractors is selected.'
    )
    ## Parameters for this script
    parser.add_argument(
        '--run_id',
        default=0,
        type=int,
        help='Output files will be put in a subdirectory of this name. This parameter should '
        'not be set by the user'
    )
    parser.add_argument(
        '--width',
        default=500,
        type=int,
        help = 'image output width'
    )
    parser.add_argument(
        '--height',
        default=500,
        type=int,
        help = 'image output height'
    )
    parser.add_argument(
        '--focal-length',
        default=None,
        type=float,
        help = "focal length of the camera"
    )
    parser.add_argument(
        '--distractors_folder',
        default='google_scanned_models/',
        help = "folder containing distraction objects"
    )
    parser.add_argument(
        '--objs_folder',
        default='models/',
        help = "folder containing training objects, if using multiple"
    )
    parser.add_argument(
        '--path_single_obj',
        default=None,
        help='If you have a single obj file, path to the obj directly.'
    )
    parser.add_argument(
        '--object_class',
        default=None,
        help="The class name of the object(s). If none is provided, the name of the directory "
        "containing the model(s) will be used."
    )
    parser.add_argument(
        '--scale',
        default=1,
        type=float,
        help='Scaling to apply to the target object(s) to put in units of centimeters; e.g if '
             'the object scale is meters -> scale=0.01; if it is in cm -> scale=1.0'
    )
    parser.add_argument(
        '--backgrounds_folder',
        default=None,
        help = "folder containing background images. Images can .jpeg, .png, or .hdr."
    )
    parser.add_argument(
        '--nb_objects',
        default=1,
        type = int,
        help = "how many objects"
    )
    parser.add_argument(
        '--nb_distractors',
        default=1,
        help = "how many distractor objects"
    )
    parser.add_argument(
        '--distractor_scale',
        default=50,
        type=float,
        help='Scaling to apply to distractor objects in order to put in units of centimeters; '
             'e.g if the object scale is meters -> scale=100; if it is in cm -> scale=1'
    )
    parser.add_argument(
        '--nb_frames',
        type = int,
        default=2000,
        help = "how many total frames to generate"
    )
    parser.add_argument(
        '--min_pixels',
        type = int,
        default=1,
        help = "How many visible pixels an object must have to be included in the JSON data"
    )
    parser.add_argument(
        '--outf',
        default='output_example/',
        help = "output filename inside output/"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help="Render the cuboid corners as small spheres. Only for debugging purposes;"
        "do not use for training!"
    )

    opt = parser.parse_args()
    main(opt)
