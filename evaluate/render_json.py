
import nvisii as visii 
import argparse
import numpy as np

import simplejson as json
import pyrender 
import cv2 
import numpy as np
import os 
import glob
import pyrr 

parser = argparse.ArgumentParser()

parser.add_argument(
    '--spp', 
    default=100,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)
parser.add_argument(
    '--opencv',
    action='store_true',
    default=False,
    help = "Use the opencv coordinate frame, e.g., dope inference,\
            otherwise it uses the opengl coordinate frame."
)

parser.add_argument(
    '--bop',
    action='store_true',
    default=False,
    help = "Use the bop format for the camera poses and .ply format \
            for the 3d object."
)
parser.add_argument(
    '--bop_scene',
    default=0,
    type=int,
    help = "Which scene to load, only applied to bop format."
)

parser.add_argument(
    '--contour',
    action='store_true',
    default=False,
    help = "Only draw the contour instead of the 3d model overlay"
)

parser.add_argument(
    '--overlay',
    action='store_true',
    default=False,
    help = "add the overlay"
)

parser.add_argument(
    '--gray',
    action='store_true',
    default=False,
    help = "draw the 3d model in gray"
)

parser.add_argument(
    '--path_json',
    required=True,
    help = "path to the json files you want loaded,\
            it assumes that there is a png accompanying."
)

parser.add_argument(
    '--objs_folder',
    default='content/',
    help = "object to load folder, should follow YCB structure"
)

parser.add_argument(
    '--scale',
    default=1,
    type=float,
    help='Specify the scale of the target object(s). If the obj mesh is in '
         'meters -> scale=1; if it is in cm -> scale=0.01.'
)

parser.add_argument(
    '--out',
    default='overlay.png',
    help = "output filename"
)

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #

def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = None,
    scale = 1, 
    rot_base = None, #visii quat
    pos_base = (-10,-10,-10), # visii vec3
    ):

    
    # This is for YCB like dataset
    if path_obj in create_obj.meshes:
        obj_mesh = create_obj.meshes[path_obj]
    else:
        obj_mesh = visii.mesh.create_from_obj(name, path_obj)
        create_obj.meshes[path_obj] = obj_mesh

    
    obj_entity = visii.entity.create(
        name = name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(1) # default is 1  

    if not path_tex is None:

        if path_tex in create_obj.textures:
            obj_texture = create_obj.textures[path_tex]
        else:
            obj_texture = visii.texture.create_from_image(name,path_tex)
            create_obj.textures[path_tex] = obj_texture


        obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    if not rot_base is None:
        obj_entity.get_transform().set_rotation(rot_base)
    if not pos_base is None:
        obj_entity.get_transform().set_position(pos_base)
    print(f' created: {obj_entity.get_name()}')
    return obj_entity

create_obj.meshes = {}
create_obj.textures = {}

# # # # # # # # # # # # # # # # # # # # # # # # #


visii.initialize_headless()
visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera", 
    )
)

camera.get_transform().look_at(
    visii.vec3(0,0,-1), # look at (world coordinate)
    visii.vec3(0,1,0), # up vector
    visii.vec3(0,0,0), # camera_origin    
)

visii.set_camera_entity(camera)

visii.set_dome_light_intensity(1)

try:
    visii.set_dome_light_color(visii.vec3(1, 1, 1), 0)
except TypeError:
    # Support for alpha transparent backgrounds was added in nvisii ef1880aa,
    # but as of 2022-11-03, the latest released version (1.1) does not include
    # that change yet.
    print("WARNING! Your version of NVISII does not support alpha transparent backgrounds yet; --contour will not work properly.")
    visii.set_dome_light_color(visii.vec3(1, 1, 1))

# # # # # # # # # # # # # # # # # # # # # # # # #

# LOAD THE SCENE 

objects_added = []
dope_trans = []
gt = None

with open(opt.path_json) as f:
    data_json = json.load(f)

if opt.bop: 
    # opt.opencv = True

    with open(opt.path_json.replace("gt","camera")) as f:
        camera_data_json = json.load(f)
        # a little bit of a hack, use the first camera, would need 
    # print(camera_data_json[str(opt.bop_scene)]["cam_K"])
    # raise()
    data_json['camera_data'] = {}
    data_json['camera_data']['intrinsics'] = {}
    data_json['camera_data']['intrinsics']['fx'] = camera_data_json[str(opt.bop_scene)]["cam_K"][0]
    data_json['camera_data']['intrinsics']['fy'] = camera_data_json[str(opt.bop_scene)]["cam_K"][4]
    data_json['camera_data']['intrinsics']['cx'] = camera_data_json[str(opt.bop_scene)]["cam_K"][2]
    data_json['camera_data']['intrinsics']['cy'] = camera_data_json[str(opt.bop_scene)]["cam_K"][5]

    # load an image for resolution 
    path_imgs = "/".join(opt.path_json.split("/")[:-1]+ ['rgb/'])
    path_imgs = sorted(glob.glob(path_imgs + "*.png"))
    opt.im_path = path_imgs[0]
    img = cv2.imread(path_imgs[0])
    data_json['camera_data']['height'] = img.shape[0]
    data_json['camera_data']['width'] = img.shape[1]

# set the camera
if "camera_data" in data_json.keys() and "intrinsics" in data_json['camera_data'].keys():
    intrinsics = data_json['camera_data']['intrinsics']
    im_height = data_json['camera_data']['height']
    im_width = data_json['camera_data']['width']

    cam = pyrender.IntrinsicsCamera(intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy'])

    proj_matrix = cam.get_projection_matrix(im_width, im_height)

    proj_matrix = visii.mat4(
            proj_matrix.flatten()[0],
            proj_matrix.flatten()[1],
            proj_matrix.flatten()[2],
            proj_matrix.flatten()[3],
            proj_matrix.flatten()[4],
            proj_matrix.flatten()[5],
            proj_matrix.flatten()[6],
            proj_matrix.flatten()[7],
            proj_matrix.flatten()[8],
            proj_matrix.flatten()[9],
            proj_matrix.flatten()[10],
            proj_matrix.flatten()[11],
            proj_matrix.flatten()[12],
            proj_matrix.flatten()[13],
            proj_matrix.flatten()[14],
            proj_matrix.flatten()[15],
    )
    proj_matrix = visii.transpose(proj_matrix)

    camera.get_camera().set_projection(proj_matrix)
else:
    im_height = 512
    im_width = 512
    intrinsics = {  "cx": 964.957,
                    "cy": 522.586,
                    "fx": 1390.53,
                    "fy": 1386.99,
                }

    cam = pyrender.IntrinsicsCamera(intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy'])

    proj_matrix = cam.get_projection_matrix(im_width, im_height)
    # print(proj_matrix)
    proj_matrix = visii.mat4(
            proj_matrix.flatten()[0],
            proj_matrix.flatten()[1],
            proj_matrix.flatten()[2],
            proj_matrix.flatten()[3],
            proj_matrix.flatten()[4],
            proj_matrix.flatten()[5],
            proj_matrix.flatten()[6],
            proj_matrix.flatten()[7],
            proj_matrix.flatten()[8],
            proj_matrix.flatten()[9],
            proj_matrix.flatten()[10],
            proj_matrix.flatten()[11],
            proj_matrix.flatten()[12],
            proj_matrix.flatten()[13],
            proj_matrix.flatten()[14],
            proj_matrix.flatten()[15],
    )
    proj_matrix = visii.transpose(proj_matrix)
    # print(proj_matrix)
    camera.get_camera().set_projection(proj_matrix)

if opt.bop: 
    # get the objects to load. 
    scene_objs = data_json[str(opt.bop_scene)]
    data_json['objects'] = []
    for obj in scene_objs:
        # print(obj)
        to_add = {}
        to_add['class'] = str(obj['obj_id'])
        to_add['location'] = obj['cam_t_m2c']

        # figuring out the rotation now
        rot = obj['cam_R_m2c']
        m = pyrr.Matrix33(
            [
                [rot[0],rot[1],rot[2]],
                [rot[3],rot[4],rot[5]],
                [rot[6],rot[7],rot[8]],
                # [rot[0],rot[3],rot[6]],
                # [rot[1],rot[4],rot[7]],
                # [rot[2],rot[5],rot[8]],                
            ])
        quat = m.quaternion
        to_add['quaternion_xyzw'] = [quat.x,quat.y,quat.z,quat.w]
        data_json['objects'].append(to_add)


for i_obj, obj in enumerate(data_json['objects']):
    name = obj['class']

    # hack because notation is old
    if name == '003':
        name = '003_cracker_box_16k'

    print('loading',name)
    if opt.bop:
        entity_visii = create_obj(
            name = obj['class'] + "_" + str(i_obj),
            path_obj = f"{opt.objs_folder}/obj_{str(name).zfill(6)}.ply",
            path_tex = f"{opt.objs_folder}/obj_{str(name).zfill(6)}.png",
            scale = .001, 
            rot_base = None
        )      

    else:    
        entity_visii = create_obj(
            name = obj['class'] + "_" + str(i_obj),
            path_obj = opt.objs_folder + "/"+name + "/google_16k/textured.obj",
            path_tex = opt.objs_folder + "/"+name + "/google_16k/texture_map_flat.png",
            scale = opt.scale,
            rot_base = None
        )        
    
    pos = obj['location']
    rot = obj['quaternion_xyzw']

    entity_visii.get_transform().set_rotation(
        visii.quat(
            rot[3],
            rot[0],
            rot[1],
            rot[2],
        )
    )
    if opt.opencv:
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0]/100,
                pos[1]/100,
                pos[2]/100,
            )
        )
    elif opt.bop: 
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0]/1000,
                pos[1]/1000,
                pos[2]/1000,
            )
        )
    else:
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0],
                pos[1],
                pos[2],
            )
        )
    if opt.opencv or opt.bop:
        entity_visii.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))
    # print(entity_visii.get_transform().get_position())
    # entity_visii.get_transform().set_position(visii.vec3(-0.04,-.2,-0.6))    
    # entity_visii.get_transform().set_rotation(visii.quat(0,0,1,0))    
    # print(entity_visii.get_transform().get_position())
    
    # camera.get_transform().look_at(
    #     entity_visii.get_transform().get_world_position(), # look at (world coordinate)
    #     visii.vec3(0,1,0), # up vector
    #     visii.vec3(0,0,0), # camera_origin    
    # )

    # break

# # # # # # # # # # # # # # # # # # # # # # # # #


visii.render_to_file(
    width=im_width, 
    height=im_height, 
    samples_per_pixel=400,
    file_path=opt.out
)
# raise()

# # # # # # # # # # # # # # # # # # # # # # # # #

# # create overlay
if opt.bop: 
    im_path = opt.im_path
if os.path.exists(opt.path_json.replace("json",'png')):
    im_path = opt.path_json.replace("json",'png')
    
elif os.path.exists(opt.path_json.replace("json",'jpg')):
    im_path = opt.path_json.replace("json",'jpg')

im = cv2.imread(im_path)

if im is not None: 
    im_pred = cv2.imread(opt.out,cv2.IMREAD_UNCHANGED)

    alpha = im_pred[:,:,-1]/255.0 * 0.75
    alpha = np.stack((alpha,)*3, axis=-1)
    im_pred = im_pred.astype(float)
    
    im = im.astype(float)
    



    # if opt.gray:
    #     im_pred_gray = cv2.imread(opt.out)
    #     im_pred_gray = cv2.cvtColor(im_pred_gray, cv2.COLOR_BGR2GRAY)
    #     im_pred_gray = im_pred_gray.astype(float)
    #     im_pred_gray = np.stack((im_pred_gray,)*3, axis=-1)
    #     foreground = cv2.multiply(alpha,im_pred_gray)
    # else:

    foreground = cv2.multiply(alpha,im_pred[:,:,:3])
    

    if opt.gray:
        im_gray = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)
        im_gray = im_gray.astype(float)
        im_gray = np.stack((im_gray,)*3, axis=-1)
        background = cv2.multiply(1-alpha,im_gray)
    elif opt.overlay:
        background = cv2.multiply(1-alpha,im[:,:,:3])
    else:
        background = im[:,:,:3]
        foreground = background

    outrgb = cv2.add(foreground,background)
    
    if opt.contour:     
        gray = cv2.cvtColor((alpha*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(outrgb, [c], -1, (36, 255, 12), thickness=3,lineType=cv2.LINE_AA)
    print(f'outputing {opt.out}')
    cv2.imwrite(opt.out,outrgb)

# let's clean up the GPU
visii.deinitialize()
