#!/usr/bin/env python3

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
import random
import sys



def point_in_frustrum(camera, width, height, near=0.0, far=25.0):
    fov = camera.get_fov()
    aspect = width/height

    ## To simplify this function, we assume a camera position of [0,-25,0] with a
    ## rotation of [np.pi / 2, 0, 0]
    cam_pos = [0, -25, 0]

    ## Generate a random point in the unit cube
    pos = np.random.random((3,))
    # shift the Y and Z values to be centered around 0
    pos[0] = 2.0*pos[1] - 1.0
    pos[2] = 2.0*pos[2] - 1.0

    # rescale depth (Y) to fit near/far
    pos[1] = pos[1]*(far-near) + near
    # rescale X and Z to fit frustrum
    pos[0] = pos[0] * sin(fov[0]/2)*(25+pos[1])
    pos[2] = pos[2] * sin(fov[1]/2)*aspect*(25+pos[1])
    return pos


def random_rotation_matrix():
    # Generate a random quaternion (XYZW)
    a,b,c = random.random(), random.random(), random.random()
    x,y,z,w = sqrt(1-a)*sin(2*pi*b), sqrt(1-a)*cos(2*pi*b), sqrt(a)*sin(2*pi*c), sqrt(a)*cos(2*pi*c)
    # Convert to a 3x3 rotation matrix
    return np.array([[1-2*y*y-2*z*z, 2*x*y+2*z*w, 2*x*z-2*y*w],
                     [2*x*y-2*z*w, 1-2*x*x-2*z*z, 2*y*z+2*x*w],
                     [2*x*z+2*y*w, 2*y*z-2*x*w, 1-2*x*x-2*y*y]])


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


def get_cuboid_image_space(mesh, camera):
    # object aligned bounding box coordinates in world coordinates
    bbox = mesh.get_bound_box()
    '''
    bbox is a list of the world-space coordinates of the corners of the
    object's oriented bounding box
    https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box

                   TOP
           3 +-----------------+ 7
            /                 /|
           /                 / |
        2 +-----------------+ 6
          |     z    y      |  |
          |      | /        |  |
          |      |/         |  |
          |  |   +--- x     |  |
           0 +-             |  + 4
          | /               | /
          |                 |/
        1 +-----------------+ 5
                FRONT

    Point '8' (the ninth entry) is the centroid
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
    # so we must reorder them
    dope_order = [5, 1, 2, 6, 4, 0, 3, 7]
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
                'cx':K[2][0],
                'cy':K[2][1]
            }
        },
        "objects" : []
    }

    ## Object data
    ##
    for ii, oo in enumerate(objects):
        idx = ii+1 # objects ID indices start at '1'

        num_pixels = int(np.sum((seg_map == idx)))
        projected_keypoints = get_cuboid_image_space(oo, camera)

        data['objects'].append({
            'class': objects_data[ii]['class'],
            'name': objects_data[ii]['name'],
            'visibility': num_pixels,
            'projected_cuboid': projected_keypoints
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
    img = Image.open(path).resize([width, height])
    (cY, cX) = (height // 2, width // 2)

    # Randomly rotate
    angle = random.random()*90.0
    img = img.rotate(angle)
    # Crop out black border resulting from rotation
    wr, hr = rotated_rectangle_extents(width, height, angle*pi/180.0)
    img = img.crop((cY-int(hr/2.0), cX-int(wr/2.0), cY+int(hr/2.0),
                    cX+int(wr/2.0))).resize([width, height])

    # Randomly recrop: between 50%-100%
    rr = random.random()*0.5 + 0.5
    hrr = int(height*rr)
    wrr = int(width*rr)
    cr_h = int((height-hrr)*random.random())
    cr_w = int((width-wrr)*random.random())
    img = img.crop((cr_h,cr_w, cr_h+hrr, cr_w+wrr)).resize([width, height])

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
    if opt.backgrounds_folder is not None:
        for ext in image_types:
            backdrop_images.extend(glob.glob(os.path.join(opt.backgrounds_folder,
                                                          os.path.join('**', ext)),
                                             recursive=True))
        if len(backdrop_images) == 0:
            print(f"No images found in backgrounds directory '{opt.backgrounds_folder}'")
        else:
            print(f"{len(backdrop_images)} images found in backgrounds directory "
                  f"'{opt.backgrounds_folder}'")

    # Construct list of object models
    object_models = []
    if args.path_single_obj:
        object_models.append(args.path_single_obj)
    else:
        object_models = glob.glob(opt.objs_folder + "**/textured.obj", recursive=True)

    # Construct list of distractors
    distractor_objs = glob.glob(args.distractors_folder + "**/model.obj", recursive=True)
    print(f"{len(distractor_objs)} distractor objects found.")

    # Set up blenderproc
    bp.init()

    # Create lights
    bp.renderer.set_world_background([1,1,1], 1.0)
    static_light = bp.types.Light()
    static_light.set_type('SUN')
    static_light.set_energy(100) # watts per sq. meter
    light = bp.types.Light()
    light.set_type('SUN')

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
        light.set_location([10-random.random()*20, 10-random.random()*20,
                            150+random.random()*100])
        # Place object(s)
        for oo in objects:
            # Set a random pose
            xform = np.eye(4)
            xform[0:3,3] = point_in_frustrum(bp.camera, args.width, args.height,
                                             near=5.0, far=300.)
            xform[0:3,0:3] = random_rotation_matrix()
            oo.set_local2world_mat(xform)

            # Scale 3D model to cm
            oo.set_scale([args.scale, args.scale, args.scale])

        # Place distractors
        for dd in distractors:
            xform = np.eye(4)
            xform[0:3,3] = point_in_frustrum(bp.camera, args.width, args.height,
                                             near=5.0, far=300.)
            xform[0:3,0:3] = random_rotation_matrix()
            dd.set_local2world_mat(xform)

            # Scale 3D model to cm ('google_scanned_models' is in scale of meters)
            dd.set_scale([100*args.scale, 100*args.scale, 100*args.scale])

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

        segs = bp.renderer.render_segmap()
        data = bp.renderer.render()
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
            im = draw_cuboid_markers(objects, bp.camera, background)

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
    # TODO: change for an array
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
        '--nb_frames',
        type = int,
        default=2000,
        help = "how many total frames to generate"
    )
    parser.add_argument(
        '--outf',
        default='output_example/',
        help = "output filename inside output/"
    )
    parser.add_argument(
        '--focal-length',
        default=None,
        type=float,
        help = "focal length of the camera"
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
