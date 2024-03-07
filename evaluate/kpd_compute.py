"""
This script computes the average distance metric at the keypoint level 
from GT to GU.
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
parser.add_argument("--outf",
    default="results_kpd/",
    help="where to put the data"
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





###### START #######

data_thruth = get_all_entries(opt.data,"*.json")
data_prediction = get_all_entries(opt.data_prediction,"*.json")


print('number of ground thruths found',len(data_thruth))
print("number of predictions found",len(data_prediction))

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


    objects_gt = [] #name obj, keypoints

    for obj in gt_json['objects']:
        if 'class' not in obj:
            name_gt = obj['name']
        else:
            name_gt = obj['class']
        # little hack from bug in the data
        if name_gt == '003':
            name_gt = "003_cracker_box_16k"

        objects_gt.append(
            [
                name_gt,
                obj["projected_cuboid"]
            ]
        )
        
        count_all_annotations += 1
        
        if name_gt in count_by_object: 
            count_by_object[name_gt] +=1 
        else:
            count_by_object[name_gt] = 1

    for obj_guess in gu_json['objects']:

        if 'class' not in obj:
            name_guess = obj_guess['name']
            name_look_up = obj_guess['name']
        else:
            name_guess = obj_guess['class']
            name_look_up = obj_guess['class']


        keypoints_gu = obj_guess["projected_cuboid"]

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
            i_gt, keypoint_gt, name_gt = candi_gt
            dist = []

            for i in range(len(keypoints_gu)):
                dist_key = 100000
                for j in range(len(keypoints_gu)):
                    d = np.sqrt((keypoint_gt[i][0]-keypoints_gu[j][0])**2+(keypoint_gt[i][1]-keypoints_gu[j][1])**2)
                    # print(keypoint_gt[i],keypoints_gu[i],i,d)
                    if d < dist_key:
                        dist_key = d
                dist.append(dist_key)


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


array_to_call = ["python", 
                "make_graphs.py",
                '--pixels',
                '--threshold',"50.0",
                "--outf", 
                opt.outf,
                '--labels',
                ]

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

