
import nvisii as visii 
import argparse
import numpy as np

import simplejson as json
import pyrender 
import cv2 
import numpy as np
import os 

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
    '--out',
    default='tmp.png',
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

visii.set_dome_light_intensity(2)
visii.set_dome_light_color(visii.vec3(1,1,1),0)

# # # # # # # # # # # # # # # # # # # # # # # # #
objects_added = []
dope_trans = []
gt = None

with open(opt.path_json) as f:
    data_json = json.load(f)

# set the camera
if "camera_data" in data_json.keys() and "intrinsics" in data_json['camera_data'].keys():
    intrinsics = data_json['camera_data']['intrinsics']
    im_height = data_json['camera_data']['height']
    im_width = data_json['camera_data']['width']
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
else:
    im_height = 512
    im_width = 512 


for i_obj, obj in enumerate(data_json['objects']):
    name = obj['class']

    # hack because notation is old
    if name == '003':
        name = '003_cracker_box_16k'

    print('loading',name)

    entity_visii = create_obj(
        name = obj['class'] + "_" + str(i_obj),
        path_obj = opt.objs_folder + "/"+name + "/google_16k/textured.obj",
        path_tex = opt.objs_folder + "/"+name + "/google_16k/texture_map_flat.png",
        scale = 0.01, 
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
    else:
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0],
                pos[1],
                pos[2],
            )
        )
    if opt.opencv:
        entity_visii.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))

# # # # # # # # # # # # # # # # # # # # # # # # #


visii.render_to_file(
    width=im_width, 
    height=im_height, 
    samples_per_pixel=400,
    file_path=opt.out
)


# # # # # # # # # # # # # # # # # # # # # # # # #

# # create overlay
if os.path.exists(opt.path_json.replace("json",'png')):
    im = cv2.imread(opt.path_json.replace("json",'png'))
elif os.path.exists(opt.path_json.replace("json",'jpg')):
    im = cv2.imread(opt.path_json.replace("json",'jpg'))

if im is not None: 
    im_pred = cv2.imread(opt.out,cv2.IMREAD_UNCHANGED)

    alpha = im_pred[:,:,-1]/255.0 * 0.85
    alpha = np.stack((alpha,)*3, axis=-1)
    im_pred = im_pred.astype(float)
    im = im.astype(float)


    foreground = cv2.multiply(alpha,im_pred[:,:,:3])
    background = cv2.multiply(1-alpha,im[:,:,:3])
    outrgb = cv2.add(foreground,background)
    cv2.imwrite(opt.out,outrgb)
# let's clean up the GPU
visii.deinitialize()