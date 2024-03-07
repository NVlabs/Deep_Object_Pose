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
import glob
import math 

# from pymesh import obj 
# from pymesh import ply 
# import pywavefront
# import pymesh 
from scipy import spatial

import simplejson as json 
import copy 
from pyquaternion import Quaternion
import pickle 
import nvisii as visii 
import subprocess 



parser = argparse.ArgumentParser()

parser.add_argument('--data_prediction', 
    default = "data/table_dope_results/", 
    help='path to prediction data')
parser.add_argument('--data', 
    default="data/table_ground_truth/", 
    help='path to data ground truth')
parser.add_argument('--models', 
    # /home/jtremblay/code/nvdu/nvdu/data/ycb/aligned_cm/AlphabetSoup/google_16k
    # default="/home/jtremblay/code/nvdu/nvdu/data/ycb/original/", 
    default="content/", 
    help='path to the 3D grocery models')
parser.add_argument("--outf",
    default="results/",
    help="where to put the data"
    )
parser.add_argument('--adds',
    action='store_true',
    help="run ADDS, this might take a while"
    )
parser.add_argument("--cuboid",
    action='store_true',
    help="use cuboid to compute the ADD"
    )
parser.add_argument("--show",
    action='store_true',
    help="show the graph at the end. "
    )

opt = parser.parse_args()



if opt.outf is None:
    opt.outf = opt.data_prediction

if not os.path.isdir(opt.outf):
    print(f'creating the folder: {opt.outf}')
    os.mkdir(opt.outf)

if os.path.isdir(opt.outf + "/tmp"):
    print(f'folder {opt.outf + "/tmp"}/ exists')
else:
    os.mkdir(opt.outf + "/tmp")
    print(f'created folder {opt.outf + "/tmp"}/')



def get_all_entries(path_to_explore, what='*.json'):

    imgs = []

    def add_images(path): 
        # print(path)
        # print(glob.glob(path+"/*json"))
        # print(glob.glob(path+"/"+what))
        for j in sorted(glob.glob(path+"/"+what)):
            # print(j)
            imgs.append(j)
            # imgsname.append(j.replace(path,"").replace("/",""))


    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        # if len(folders)>0:
        for path_entry in folders:                
            explore(path_entry)

           
        add_images(path)

    explore(path_to_explore)
    return imgs






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
    cuboid = [  cuboid[2],cuboid[0],cuboid[3],
                cuboid[5],cuboid[4],cuboid[1],
                cuboid[6],cuboid[7],cuboid[-1]]

    cuboid.append(visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]))
        
    for i_p, p in enumerate(cuboid):
        child_transform = visii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)
        child_transform.set_scale(visii.vec3(0.1))
        child_transform.set_parent(obj.get_transform())
        if debug: 
            visii.entity.create(
                name = f"{name}_cuboid_{i_p}",
                mesh = visii.mesh.create_sphere(f"{name}_cuboid_{i_p}"),
                transform = child_transform, 
                material = visii.material.create(f"{name}_cuboid_{i_p}")
            )
    
    for i_v, v in enumerate(cuboid):
        cuboid[i_v]=[v[0], v[1], v[2]]


     
    return cuboid


def get_models(path,suffix=""):
    models = {}
    for folder in glob.glob(path+"/*/"):

        model_name = folder.replace(path,"").replace('/',"")
        print('loading',model_name + suffix)
        models[model_name] = create_obj(
            name = model_name + suffix,
            path_obj = folder + "/google_16k/textured.obj",
            path_tex = folder + "/google_16k/texture_map_flat.png",
            scale = 0.01
        )
        if opt.cuboid: 
            add_cuboid(model_name + suffix)
    if 'gu' in suffix: 
        models[model_name].get_material().set_metallic(1)
        models[model_name].get_material().set_roughness(0.05)

    return models



# START OF THE PROGRAM HERE
visii.initialize_headless()


# data_thruth = get_all_entries(opt.data,'scene_all_realsense.json')
# if len(data_thruth) == 0: 
#     data_thruth = get_all_entries(opt.data,'scene_realsense.json')
data_thruth = get_all_entries(opt.data,"*.json")
data_prediction = get_all_entries(opt.data_prediction,"*.json")


print('number of ground thruths found',len(data_thruth))
print("number of predictions found",len(data_prediction))

meshes_gt = get_models(opt.models,'_gt')
meshes_gu = get_models(opt.models,'_gu')

adds_objects = {}

adds_all = []
all_gts = []
count_all_annotations = 0
count_by_object = {}

count_all_guesses = 0
count_by_object_guesses = {}


for gt_file in data_thruth:
    scene_gt = gt_file.replace(opt.data,"").replace('.json','')
    pred_scene = None


    for d in data_prediction:
        scene_d = d.replace(opt.data_prediction,'').replace('json','').replace('.','')

        # if scene in d:
        # print(scene_d,scene_gt)
        if scene_d.split('/')[-1] == scene_gt.split('/')[-1]:
            pred_scene = d
            break

    if pred_scene is None:
        continue
    # print(gt_file)
    gt_json = None
    with open(gt_file) as json_file:
        gt_json = json.load(json_file)

    gu_json = None
    with open(pred_scene) as json_file:
        gu_json = json.load(json_file)


    objects_gt = [] #name obj, pose

    for obj in gt_json['objects']:

        name_gt = obj['class']

        # little hack from bug in the data
        if name_gt == '003':
            name_gt = "003_cracker_box_16k"
        objects_gt.append(
            [
                name_gt,
                {
                    "rotation":visii.quat(
                        obj['quaternion_xyzw'][3],
                        obj['quaternion_xyzw'][0],
                        obj['quaternion_xyzw'][1],
                        obj['quaternion_xyzw'][2],
                    ),
                    "position":visii.vec3(
                        obj['location'][0],
                        obj['location'][1],
                        obj['location'][2],
                    )
                }
            ]
        )
        
        count_all_annotations += 1
        
        if name_gt in count_by_object: 
            count_by_object[name_gt] +=1 
        else:
            count_by_object[name_gt] = 1

    for obj_guess in gu_json['objects']:

        name_guess = obj_guess['class']
        # name_look_up = obj_guess['class'].split("_")[0]
        name_look_up = obj_guess['class']

        # need to add rotation for DOPE prediction, if your frames are aligned 
        try:
            pose_mesh = {
                "rotation":visii.quat(
                    float(obj_guess['quaternion_xyzw'][3]),
                    float(obj_guess['quaternion_xyzw'][0]),
                    float(obj_guess['quaternion_xyzw'][1]),
                    float(obj_guess['quaternion_xyzw'][2]),
                ) 
                # * visii.angleAxis(1.57, visii.vec3(1,0,0)) * visii.angleAxis(1.57, visii.vec3(0,0,1))
                # * visii.angleAxis(1.57*2, visii.vec3(0,0,1)) 
                # * visii.angleAxis(1.57, visii.vec3(0,1,0))
                ,
                "position":visii.vec3(
                    float(str(obj_guess['location'][0]))/100.0,
                    float(str(obj_guess['location'][1]))/100.0,
                    float(str(obj_guess['location'][2]))/100.0,
                )
            }
        except:
            # in case there is an inf or null in the location prediction/gt
            pose_mesh = {
                "rotation":visii.quat(
                    float(obj_guess['quaternion_xyzw'][3]),
                    float(obj_guess['quaternion_xyzw'][0]),
                    float(obj_guess['quaternion_xyzw'][1]),
                    float(obj_guess['quaternion_xyzw'][2]),
                ) * visii.angleAxis(1.57, visii.vec3(1,0,0)) * visii.angleAxis(1.57, visii.vec3(0,0,1))
                ,
                "position":visii.vec3(
                    1000000,
                    1000000,
                    1000000,
                )
            }
        count_all_guesses += 1
        
        if name_guess in count_by_object_guesses: 
            count_by_object_guesses[name_guess] +=1 
        else:
            count_by_object_guesses[name_guess] = 1


        # print (name, pose_mesh)
        candidates = []
        for i_obj_gt, obj_gt in enumerate(objects_gt):
            name_gt, pose_mesh_gt = obj_gt

            # print(name_look_up,name_gt)

            if name_look_up == name_gt:
                candidates.append([i_obj_gt, pose_mesh_gt, name_gt])

        best_dist = 10000000000 
        best_index = -1 

        for candi_gt in candidates:
            # compute the add
            i_gt, pose_gt, name_gt = candi_gt
            # if i_gt in used_index:
            #     continue
            # print(meshes_gt.keys())
            visii_gt = meshes_gt[name_gt]
            
            visii_gt.get_transform().set_position(pose_gt['position'])
            visii_gt.get_transform().set_rotation(pose_gt['rotation'])

            # visii_gt.get_transform().set_position(visii.vec3(-10,-10,-10))
            # visii_gt.get_transform().set_rotation(pose_gt['rotation'])


            visii_gu = meshes_gu[name_look_up]

            visii_gu.get_transform().set_position(pose_mesh['position'])
            visii_gu.get_transform().set_rotation(pose_mesh['rotation'])

            # dope is in the opencv frame, need to be put in the opengl frame
            visii_gu.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))

            
            if opt.adds:
                if opt.cuboid:                                            
                    dist = 0
                    for i_p in range(9):
                        corner_gt = visii.transform.get(f"{name_gt + '_gt'}_cuboid_{i_p}")
                        dist_s = []
                        for i_ps in range(9):
                            corner_gu = visii.transform.get(f"{name_look_up+ '_gu'}_cuboid_{i_ps}")
                            gt_trans = corner_gt.get_local_to_world_matrix()
                            gu_trans = corner_gu.get_local_to_world_matrix()

                            # print(corner_pos,cuboid_gt[i_p])
                            dist_now =\
                                math.sqrt(
                                    (gt_trans[3][0]-gu_trans[3][0])**2+\
                                    (gt_trans[3][1]-gu_trans[3][1])**2+\
                                    (gt_trans[3][2]-gu_trans[3][2])**2
                                )
                            dist_s.append(dist_now)
                        dist += min(dist_s) 

                    dist /= 9
                    print(dist)
                else:
                    dist = []
                    dist2 = []
                    vertices = visii_gt.get_mesh().get_vertices()
                    points_gt = []
                    points_gu = []

                    for i in range(len(vertices)):
                        v = visii.vec4(vertices[i][0],vertices[i][1],vertices[i][2],1)
                        p0 = visii_gt.get_transform().get_local_to_world_matrix() * v
                        p1 = visii_gu.get_transform().get_local_to_world_matrix() * v
                        points_gt.append([p0[0],p0[1],p0[2]])
                        points_gu.append([p1[0],p1[1],p1[2]])

                    dist = np.mean(spatial.distance_matrix(
                                        np.array(points_gt), 
                                        np.array(points_gu),p=2).min(axis=1))


            else:
                if opt.cuboid:                                            
                    dist = 0
                    for i_p in range(9):
                        corner_gt = visii.transform.get(f"{name_gt + '_gt'}_cuboid_{i_p}")
                        corner_gu = visii.transform.get(f"{name_look_up+ '_gu'}_cuboid_{i_p}")
                        gt_trans = corner_gt.get_local_to_world_matrix()
                        gu_trans = corner_gu.get_local_to_world_matrix()

                        # print(corner_pos,cuboid_gt[i_p])
                        dist +=\
                            math.sqrt(
                                (gt_trans[3][0]-gu_trans[3][0])**2+\
                                (gt_trans[3][1]-gu_trans[3][1])**2+\
                                (gt_trans[3][2]-gu_trans[3][2])**2
                            )
                        
                    dist /= 9
                else:
                    dist = []
                    vertices = visii_gt.get_mesh().get_vertices()
                    for i in range(len(vertices)):
                        v = visii.vec4(vertices[i][0],vertices[i][1],vertices[i][2],1)
                        p0 = visii_gt.get_transform().get_local_to_world_matrix() * v
                        p1 = visii_gu.get_transform().get_local_to_world_matrix() * v
                        dist.append(visii.distance(p0, p1))


                    dist = np.mean(dist)

            if dist < best_dist:
                best_dist = dist
                best_index = i_gt

        if best_index != -1:
            if not name_guess in adds_objects.keys():
                 adds_objects[name_guess] = []
            adds_all.append(best_dist)
            adds_objects[name_guess].append(best_dist)

# save the data
if len(opt.outf.split("/"))>1:
    path = None
    for folder in opt.outf.split("/"):
        if path is None:
            path = folder
        else:
            path = path + "/" + folder 
        try:
            os.mkdir(path)
        except:
            pass        
else:
    try:
        os.mkdir(opt.outf)
    except:
        pass
print(adds_objects.keys())
count_by_object["all"] = count_all_annotations
pickle.dump(count_by_object,open(f'{opt.outf}/count_all_annotations.p','wb'))
pickle.dump(adds_all,open(f'{opt.outf}/adds_all.p','wb'))

count_by_object_guesses["all"] = count_all_guesses
pickle.dump(count_by_object,open(f'{opt.outf}/count_all_guesses.p','wb'))


labels = []
data = []
for key in adds_objects.keys():
    pickle.dump(adds_objects[key],open(f'{opt.outf}/adds_{key}.p','wb'))
    labels.append(key)
    data.append(f'{opt.outf}/adds_{key}.p')


array_to_call = ["python", "make_graphs.py","--outf", opt.outf,'--labels']

for label in labels:
    array_to_call.append(label)

array_to_call.append('--data')
for d_p in data:
    array_to_call.append(d_p)

array_to_call.append('--colours')
for i in range(len(data)):
    array_to_call.append(str(i))
if opt.show:
    array_to_call.append('--show')

print(array_to_call)
subprocess.call(array_to_call)

# subprocess.call(
#     [
#         "python", "make_graphs.py", 
#         "--data", f'{opt.outf}/adds_{key}.p', 
#         "--labels", key, 
#         "--outf", opt.outf,
#         '--colours', "0",
#     ]
# )


visii.deinitialize()

