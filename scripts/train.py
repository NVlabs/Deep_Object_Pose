#!/usr/bin/env python3

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from __future__ import print_function

######################################################
"""
REQUIREMENTS:
simplejson==3.16.0
numpy==1.14.1
opencv_python==3.4.3.18
horovod==0.13.5
photutils==0.5
scipy==1.1.0
torch==0.4.0
pyquaternion==0.9.2
tqdm==4.25.0
pyrr==0.9.2
Pillow==5.2.0
torchvision==0.2.1
PyYAML==3.13
"""

######################################################
"""
HOW TO TRAIN DOPE

This is the DOPE training code.  
It is provided as a convenience for researchers, but it is otherwise unsupported.

Please refer to `python3 train.py --help` for specific details about the 
training code. 

If you download the FAT dataset 
(https://research.nvidia.com/publication/2018-06_Falling-Things)
you can train a YCB object DOPE detector as follows: 

```
python3 train.py --data path/to/FAT --object soup --outf soup 
--gpuids 0 1 2 3 4 5 6 7 
```

This will create a folder called `train_soup` where the weights will be saved 
after each epoch. It will use the 8 gpus using pytorch data parallel. 
"""


import argparse
import configparser
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
import datetime
import json
import glob
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi    

from os.path import exists

import cv2
import colorsys

from dope.utils import make_grid

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"


##################################################
# NEURAL NETWORK MODEL
##################################################

class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage
        
        if pretrained is False:
            print("Training network without imagenet weights.")
        else:
            print("Training network pretrained on imagenet.")

        vgg_full = models.vgg19(pretrained=pretrained).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)


    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2],\
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2],\
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],\
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],\
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],\
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],\
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
                        
    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model



##################################################
# UTILS CODE FOR LOADING THE DATA
##################################################

def default_loader(path):
    return Image.open(path).convert('RGB')          

def loadjson(path, objectofinterest):
    """
    Loads the data from a json file.
    If there are no objects of interest, then load all the objects.
    """
    with open(path) as data_file:
        data = json.load(data_file)
    pointsBelief = []
    centroids = []

    translations = []
    rotations = []
    points = []

    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]
        if not objectofinterest is None and \
                not objectofinterest in info['class'].lower():
            continue

        # 3d bbox with belief maps
        points3d = []

        pointdata = info['projected_cuboid']
        for p in pointdata:
            points3d.append((p[0], p[1]))

        if len(points3d) == 8:
            # NDDS format: 8 points in 'projected_cuboid', 1 point in 'projected_cuboid_centroid'
            pcenter = info['projected_cuboid_centroid']
            points3d.append((pcenter[0], pcenter[1]))
        elif len(points3d) == 9:
            # nvisii format: 9 points in 'projected_cuboid', no 'projected_cuboid_centroid' key
            pcenter = points3d[-1]
        else:
            raise RuntimeError(f'projected_cuboid has to have 8 or 9 points while reading "{path}"')

        pointsBelief.append(points3d)
        points.append(points3d + [(pcenter[0], pcenter[1])])  # NOTE: Adding the centroid again is probably a bug.
        centroids.append((pcenter[0], pcenter[1]))

        # load translations
        location = info['location']
        translations.append([location[0], location[1], location[2]])

        # quaternion
        rot = info["quaternion_xyzw"]
        rotations.append(rot)

    return {
        "pointsBelief": pointsBelief,
        "rotations": rotations,
        "translations": translations,
        "centroids": centroids,
        "points": points,
        "keypoints_2d": [],
    }

def loadimages(root):
    """
    Find all the images in the path and folders, return them in imgs. 
    """
    imgs = []

    def add_json_files(path,):
        for imgpath in glob.glob(path+"/*.png"):
            if exists(imgpath) and exists(imgpath.replace('png',"json")):
                imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                    imgpath.replace('png',"json")))
        for imgpath in glob.glob(path+"/*.jpg"):
            if exists(imgpath) and exists(imgpath.replace('jpg',"json")):
                imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                    imgpath.replace('jpg',"json")))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        if len(folders)>0:
            for path_entry in folders:                
                explore(path_entry)
        add_json_files(path)

    explore(root)

    return imgs

class MultipleVertexJson(data.Dataset):
    """
    Dataloader for the data generated by NDDS (https://github.com/NVIDIA/Dataset_Synthesizer). 
    This is the same data as the data used in FAT.
    """
    def __init__(self, root,transform=None, nb_vertex = 8,
            keep_orientation = True, 
            normal = None, test=False, 
            target_transform = None,
            loader = default_loader, 
            objectofinterest = "",
            img_size = 400,
            save = False,  
            noise = 2,
            data_size = None,
            sigma = 16,
            random_translation = (25.0,25.0),
            random_rotation = 15.0,
            ):
        ###################
        self.objectofinterest = objectofinterest
        self.img_size = img_size
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.imgs = []
        self.test = test
        self.normal = normal
        self.keep_orientation = keep_orientation
        self.save = save 
        self.noise = noise
        self.data_size = data_size
        self.sigma = sigma
        self.random_translation = random_translation
        self.random_rotation = random_rotation

        def load_data(path):
            '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
            imgs = loadimages(path)

            # Check all the folders in path 
            for name in os.listdir(str(path)):
                imgs += loadimages(path +"/"+name)
            return imgs


        self.imgs = load_data(root)

        # Shuffle the data, this is useful when we want to use a subset. 
        np.random.shuffle(self.imgs)

    def __len__(self):
        # When limiting the number of data
        if not self.data_size is None:
            return int(self.data_size)

        return len(self.imgs)   

    def __getitem__(self, index):
        """
        Depending on how the data loader is configured,
        this will return the debug info with the cuboid drawn on it, 
        this happens when self.save is set to true. 
        Otherwise, during training this function returns the 
        belief maps and affinity fields and image as tensors.  
        """
        path, name, txt = self.imgs[index]
        img = self.loader(path)

        img_size = img.size
        img_size = (400,400)

        loader = loadjson
        
        data = loader(txt, self.objectofinterest)

        pointsBelief        =   data['pointsBelief'] 
        objects_centroid    =   data['centroids']
        points_all          =   data['points']
        points_keypoints    =   data['keypoints_2d']
        translations        =   torch.from_numpy(np.array(
                                data['translations'])).float()
        rotations           =   torch.from_numpy(np.array(
                                data['rotations'])).float() 

        if len(points_all) == 0:
            points_all = torch.zeros(1, 10, 2).double()
        
        # self.save == true assumes there is only 
        # one object instance in the scene. 
        if translations.size()[0] > 1:
            translations = translations[0].unsqueeze(0)
            rotations = rotations[0].unsqueeze(0)

        # If there are no objects, still need to return similar shape array
        if len(translations) == 0:
            translations = torch.zeros(1,3).float()
            rotations = torch.zeros(1,4).float()

        # Camera intrinsics
        path_cam = path.replace(name,'_camera_settings.json')
        with open(path_cam) as data_file:    
            data = json.load(data_file)
        # Assumes one camera
        cam = data['camera_settings'][0]['intrinsic_settings']

        matrix_camera = np.zeros((3,3))
        matrix_camera[0,0] = cam['fx']
        matrix_camera[1,1] = cam['fy']
        matrix_camera[0,2] = cam['cx']
        matrix_camera[1,2] = cam['cy']
        matrix_camera[2,2] = 1

        # Load the cuboid sizes
        path_set = path.replace(name,'_object_settings.json')
        with open(path_set) as data_file:    
            data = json.load(data_file)

        cuboid = torch.zeros(1)

        if self.objectofinterest is None:
            cuboid = np.array(data['exported_objects'][0]['cuboid_dimensions'])
        else:
            for info in data["exported_objects"]:
                if self.objectofinterest in info['class']:
                    cuboid = np.array(info['cuboid_dimensions'])

        img_original = img.copy()        

        
        def Reproject(points,tm, rm):
            """
            Reprojection of points when rotating the image
            """
            proj_cuboid = np.array(points)

            rmat = np.identity(3)
            rmat[0:2] = rm
            tmat = np.identity(3)
            tmat[0:2] = tm

            new_cuboid = np.matmul(
                rmat, np.vstack((proj_cuboid.T, np.ones(len(points)))))
            new_cuboid = np.matmul(tmat, new_cuboid)
            new_cuboid = new_cuboid[0:2].T

            return new_cuboid

        # Random image manipulation, rotation and translation with zero padding
    	# These create a bug, thank you to 
	    # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
	    # dx = round(np.random.normal(0, 2) * float(self.random_translation[0]))
        # dy = round(np.random.normal(0, 2) * float(self.random_translation[1]))
        # angle = round(np.random.normal(0, 1) * float(self.random_rotation))

        dx = round(float(torch.normal(torch.tensor(0.0), torch.tensor(2.0)) * float(self.random_translation[0])))
        dy = round(float(torch.normal(torch.tensor(0.0), torch.tensor(2.0)) * float(self.random_translation[1])))
        angle = round(float(torch.normal(torch.tensor(0.0), torch.tensor(1.0)) * float(self.random_rotation)))	
	
        tm = np.float32([[1, 0, dx], [0, 1, dy]])
        rm = cv2.getRotationMatrix2D(
            (img.size[0]/2, img.size[1]/2), angle, 1)

        for i_objects in range(len(pointsBelief)):
            points = pointsBelief[i_objects]
            new_cuboid = Reproject(points, tm, rm)
            pointsBelief[i_objects] = new_cuboid.tolist()
            objects_centroid[i_objects] = tuple(new_cuboid.tolist()[-1])
            pointsBelief[i_objects] = list(map(tuple, pointsBelief[i_objects]))

        for i_objects in range(len(points_keypoints)):
            points = points_keypoints[i_objects]
            new_cuboid = Reproject(points, tm, rm)
            points_keypoints[i_objects] = new_cuboid.tolist()
            points_keypoints[i_objects] = list(map(tuple, points_keypoints[i_objects]))
                
        image_r = cv2.warpAffine(np.array(img), rm, img.size)
        result = cv2.warpAffine(image_r, tm, img.size)
        img = Image.fromarray(result)

        # Note:  All point coordinates are in the image space, e.g., pixel value.
        # This is used when we do saving --- helpful for debugging
        if self.save or self.test:   
            # Use the save to debug the data
            if self.test:
                draw = ImageDraw.Draw(img_original)
            else:
                draw = ImageDraw.Draw(img)
            
            # PIL drawing functions, here for sharing draw
            def DrawKeypoints(points):
                for key in points:
                    DrawDot(key,(12, 115, 170),7) 
                                       
            def DrawLine(point1, point2, lineColor, lineWidth):
                if not point1 is None and not point2 is None:
                    draw.line([point1,point2],fill=lineColor,width=lineWidth)

            def DrawDot(point, pointColor, pointRadius):
                if not point is None:
                    xy = [point[0]-pointRadius, point[1]-pointRadius, point[0]+pointRadius, point[1]+pointRadius]
                    draw.ellipse(xy, fill=pointColor, outline=pointColor)

            def DrawCube(points, which_color = 0, color = None):
                '''Draw cube with a thick solid line across the front top edge.'''
                lineWidthForDrawing = 2
                lineColor1 = (255, 215, 0)  # yellow-ish
                lineColor2 = (12, 115, 170)  # blue-ish
                lineColor3 = (45, 195, 35)  # green-ish
                if which_color == 3:
                    lineColor = lineColor3
                else:
                    lineColor = lineColor1

                if not color is None:
                    lineColor = color        

                # draw front
                DrawLine(points[0], points[1], lineColor, 8) #lineWidthForDrawing)
                DrawLine(points[1], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[0], lineColor, lineWidthForDrawing)
                
                # draw back
                DrawLine(points[4], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[7], lineColor, lineWidthForDrawing)
                DrawLine(points[4], points[7], lineColor, lineWidthForDrawing)
                
                # draw sides
                DrawLine(points[0], points[4], lineColor, lineWidthForDrawing)
                DrawLine(points[7], points[3], lineColor, lineWidthForDrawing)
                DrawLine(points[5], points[1], lineColor, lineWidthForDrawing)
                DrawLine(points[2], points[6], lineColor, lineWidthForDrawing)

                # draw dots
                DrawDot(points[0], pointColor=(255,255,255), pointRadius = 3)
                DrawDot(points[1], pointColor=(0,0,0), pointRadius = 3)

            # Draw all the found objects. 
            for points_belief_objects in pointsBelief:
                DrawCube(points_belief_objects)
            for keypoint in points_keypoints:
                DrawKeypoints(keypoint)

            img = self.transform(img)
            
            return {
                "img":img,
                "translations":translations,
                "rot_quaternions":rotations,
                'pointsBelief':np.array(points_all[0]),
                'matrix_camera':matrix_camera,
                'img_original': np.array(img_original),
                'cuboid': cuboid,
                'file_name':name,
            }

        # Create the belief map
        beliefsImg = CreateBeliefMap(
            img, 
            pointsBelief=pointsBelief,
            nbpoints = 9,
            sigma = self.sigma)

        # Create the image maps for belief
        transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])

        for j in range(len(beliefsImg)):
            beliefsImg[j] = self.target_transform(beliefsImg[j])
            # beliefsImg[j].save('{}.png'.format(j))
            beliefsImg[j] = totensor(beliefsImg[j])

        beliefs = torch.zeros((len(beliefsImg),beliefsImg[0].size(1),beliefsImg[0].size(2)))
        for j in range(len(beliefsImg)):
            beliefs[j] = beliefsImg[j][0]
        

        # Create affinity maps
        scale = 8
        if min (img.size) / 8.0  != min (img_size)/8.0:
            # print (scale)
            scale = min (img.size)/(min (img_size)/8.0)

        affinities = GenerateMapAffinity(img,8,pointsBelief,objects_centroid,scale)
        img = self.transform(img)

        # Transform the images for training input
        w_crop = np.random.randint(0, img.size[0] - img_size[0]+1)
        h_crop = np.random.randint(0, img.size[1] - img_size[1]+1)
        transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])

        if not self.normal is None:
            normalize = transforms.Compose([transforms.Normalize
                ((self.normal[0],self.normal[0],self.normal[0]),
                 (self.normal[1],self.normal[1],self.normal[1])),
                AddNoise(self.noise)])
        else:
            normalize = transforms.Compose([AddNoise(0.0001)])
        
        img = crop(img,h_crop,w_crop,img_size[1],img_size[0])
        img = totensor(img)

        img = normalize(img)

        w_crop = int(w_crop/8)
        h_crop = int(h_crop/8)

        affinities = affinities[:,h_crop:h_crop+int(img_size[1]/8),w_crop:w_crop+int(img_size[0]/8)]
        beliefs = beliefs[:,h_crop:h_crop+int(img_size[1]/8),w_crop:w_crop+int(img_size[0]/8)]

        if affinities.size()[1] == 49 and not self.test:
            affinities = torch.cat([affinities,torch.zeros(16,1,50)],dim=1)

        if affinities.size()[2] == 49 and not self.test:
            affinities = torch.cat([affinities,torch.zeros(16,50,1)],dim=2)

        return {
            'img':img,    
            "affinities":affinities,            
            'beliefs':beliefs,
        }

"""
Some simple vector math functions to find the angle
between two points, used by affinity fields. 
"""
def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def GenerateMapAffinity(img,nb_vertex,pointsInterest,objects_centroid,scale):
    """
    Function to create the affinity maps, 
    e.g., vector maps pointing toward the object center. 

    Args:
        img: PIL image
        nb_vertex: (int) number of points 
        pointsInterest: list of points 
        objects_centroid: (x,y) centroids for the obects
        scale: (float) by how much you need to scale down the image 
    return: 
        return a list of tensors for each point except centroid point      
    """

    # Apply the downscale right now, so the vectors are correct. 
    img_affinity = Image.new(img.mode, (int(img.size[0]/scale),int(img.size[1]/scale)), "black")
    # Create the empty tensors
    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(2,int(img.size[1]/scale),int(img.size[0]/scale)))
    
    for i_pointsImage in range(len(pointsInterest)):    
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]
            affinity_pair, img_affinity = getAfinityCenter(int(img.size[0]/scale),
                int(img.size[1]/scale),
                tuple((np.array(pointsImage[i_points])/scale).tolist()),
                tuple((np.array(center)/scale).tolist()), 
                img_affinity = img_affinity, radius=1)

            affinities[i_points] = (affinities[i_points] + affinity_pair)/2


            # Normalizing
            v = affinities[i_points].numpy()                    
            
            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero]/=norms[nonzero]
            yvec[nonzero]/=norms[nonzero]

            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec],[yvec]]))
    affinities = torch.cat(affinities,0)

    return affinities

def getAfinityCenter(width, height, point, center, radius=7, img_affinity=None):
    """
    Function to create the affinity maps, 
    e.g., vector maps pointing toward the object center. 

    Args:
        width: image wight
        height: image height
        point: (x,y) 
        center: (x,y)
        radius: pixel radius
        img_affinity: tensor to add to 
    return: 
        return a tensor
    """
    tensor = torch.zeros(2,height,width).float()

    # Create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width,height), "black")
    totensor = transforms.Compose([transforms.ToTensor()])
    
    draw = ImageDraw.Draw(imgAffinity)    
    r1 = radius
    p = point
    draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),(255,255,255))

    del draw

    # Compute the array to add the afinity
    array = (np.array(imgAffinity)/255)[:,:,0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array*angle_vector[0]],[array*angle_vector[1]]])

    # print (tensor)
    if not img_affinity is None:
        # Find the angle vector
        # print (angle_vector)
        if length(angle_vector) >0:
            angle=py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle/360,1,1)) * 255
        draw = ImageDraw.Draw(img_affinity)    
        draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),fill=(int(c[0]),int(c[1]),int(c[2])))
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity

       
def CreateBeliefMap(img,pointsBelief,nbpoints,sigma=16):
    """
    Args: 
        img: image
        pointsBelief: list of points in the form of 
                      [nb object, nb points, 2 (x,y)] 
        nbpoints: (int) number of points, DOPE uses 8 points here
        sigma: (int) size of the belief map point
    return: 
        return an array of PIL black and white images representing the 
        belief maps         
    """
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):    
        array = np.zeros(img.size)
        out = np.zeros(img.size)

        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma*2)
            if p[0]-w>=0 and p[0]+w<img.size[0] and p[1]-w>=0 and p[1]+w<img.size[1]:
                for i in range(int(p[0])-w, int(p[0])+w):
                    for j in range(int(p[1])-w, int(p[1])+w):
                        array[i,j] = np.exp(-(((i - p[0])**2 + (j - p[1])**2)/(2*(sigma**2))))

        stack = np.stack([array,array,array],axis=0).transpose(2,1,0)
        imgBelief = Image.new(img.mode, img.size, "black")
        beliefsImg.append(Image.fromarray((stack*255).astype('uint8')))
    return beliefsImg


def crop(img, i, j, h, w):
    """
    Crop the given PIL.Image.
    
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
    Apply some random contrast from PIL
    """
    
    def __init__(self,sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        contrast = ImageEnhance.Contrast(im)
        im = contrast.enhance( np.random.normal(1,self.sigma) )        
        return im


class AddRandomBrightness(object):
    """
    Apply some random brightness from PIL
    """

    def __init__(self,sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        bright = ImageEnhance.Brightness(im)
        im = bright.enhance( np.random.normal(1,self.sigma) )
        return im        

class AddNoise(object):
    """
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    
    def __init__(self,std=0.1):
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        # t = torch.FloatTensor(tensor.size()).uniform_(self.min,self.max)
        t = torch.FloatTensor(tensor.size()).normal_(0,self.std)

        t = tensor.add(t)
        t = torch.clamp(t,-1,1) #this is expansive
        return t


def save_image(tensor, filename, nrow=4, padding=2,mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10,pad_value=1)
    if not mean is None:
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    else:      
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def DrawLine(point1, point2, lineColor, lineWidth,draw):
    if not point1 is None and not point2 is None:
        draw.line([point1,point2],fill=lineColor,width=lineWidth)

def DrawDot(point, pointColor, pointRadius, draw):
    if not point is None:
        xy = [point[0]-pointRadius, point[1]-pointRadius, point[0]+pointRadius, point[1]+pointRadius]
        draw.ellipse(xy, fill=pointColor, outline=pointColor)

def DrawCube(points, which_color = 0, color = None, draw = None):
    '''Draw cube with a thick solid line across the front top edge.'''
    lineWidthForDrawing = 2
    lineWidthThick = 8
    lineColor1 = (255, 215, 0)  # yellow-ish
    lineColor2 = (12, 115, 170)  # blue-ish
    lineColor3 = (45, 195, 35)  # green-ish
    if which_color == 3:
        lineColor = lineColor3
    else:
        lineColor = lineColor1

    if not color is None:
        lineColor = color        

    # draw front
    DrawLine(points[0], points[1], lineColor, lineWidthThick, draw)
    DrawLine(points[1], points[2], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[3], points[2], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[3], points[0], lineColor, lineWidthForDrawing, draw)
    
    # draw back
    DrawLine(points[4], points[5], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[6], points[5], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[6], points[7], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[4], points[7], lineColor, lineWidthForDrawing, draw)
    
    # draw sides
    DrawLine(points[0], points[4], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[7], points[3], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[5], points[1], lineColor, lineWidthForDrawing, draw)
    DrawLine(points[2], points[6], lineColor, lineWidthForDrawing, draw)

    # draw dots
    DrawDot(points[0], pointColor=lineColor, pointRadius = 4,draw = draw)
    DrawDot(points[1], pointColor=lineColor, pointRadius = 4,draw = draw)



##################################################
# TRAINING CODE MAIN STARTING HERE
##################################################

print ("start:" , datetime.datetime.now().time())

conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )
conf_parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

parser.add_argument('--data',  
    default = "", 
    help='path to training data')

parser.add_argument('--datatest', 
    default="", 
    help='path to data testing set')

parser.add_argument('--object', 
    default=None, 
    help='In the dataset which objet of interest')

parser.add_argument('--workers', 
    type=int, 
    default=8,
    help='number of data loading workers')

parser.add_argument('--batchsize', 
    type=int, 
    default=32, 
    help='input batch size')

parser.add_argument('--imagesize', 
    type=int, 
    default=400, 
    help='the height / width of the input image to network')

parser.add_argument('--lr', 
    type=float, 
    default=0.0001, 
    help='learning rate, default=0.001')

parser.add_argument('--noise', 
    type=float, 
    default=2.0, 
    help='gaussian noise added to the image')

parser.add_argument('--net', 
    default='', 
    help="path to net (to continue training)")

parser.add_argument('--namefile', 
    default='epoch', 
    help="name to put on the file of the save weights")

parser.add_argument('--manualseed', 
    type=int, 
    help='manual seed')

parser.add_argument('--epochs', 
    type=int, 
    default=60,
    help="number of epochs to train")

parser.add_argument('--loginterval', 
    type=int, 
    default=100)

parser.add_argument('--gpuids',
    nargs='+', 
    type=int, 
    default=[0], 
    help='GPUs to use')

parser.add_argument('--outf', 
    default='tmp', 
    help='folder to output images and model checkpoints, it will \
    add a train_ in front of the name')

parser.add_argument('--sigma', 
    default=4, 
    help='keypoint creation size for sigma')

parser.add_argument('--save', 
    action="store_true", 
    help='save a visual batch and quit, this is for\
    debugging purposes')

parser.add_argument("--pretrained",
    default=True,
    help='do you want to use vgg imagenet pretrained weights')

parser.add_argument('--nbupdates', 
    default=None, 
    help='nb max update to network, overwrites the epoch number\
    otherwise uses the number of epochs')

parser.add_argument('--datasize', 
    default=None, 
    help='randomly sample that number of entries in the dataset folder') 

# Read the config but do not overwrite the args written 
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }

if args.config:
    config = ConfigParser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

if opt.pretrained in ['false', 'False']:
	opt.pretrained = False

if not "/" in opt.outf:
    opt.outf = "train_{}".format(opt.outf)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

# save the hyper parameters passed
with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt)+"\n")

with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt))
    file.write("seed: "+ str(opt.manualseed)+'\n')
    with open (opt.outf+'/test_metric.csv','w') as file:
        file.write("epoch, passed,total \n")

# set the manual seed. 
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

# save 
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59,0.25]
    transform = transforms.Compose([
                               AddRandomContrast(contrast),
                               AddRandomBrightness(brightness),
                               transforms.Scale(opt.imagesize),
                               ])
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose([
                           transforms.Resize(opt.imagesize),
                           transforms.ToTensor()])

print ("load data")
#load the dataset using the loader in utils_pose
trainingdata = None
if not opt.data == "":
    train_dataset = MultipleVertexJson(
        root = opt.data,
        objectofinterest=opt.object,
        keep_orientation = True,
        noise = opt.noise,
        sigma = opt.sigma,
        data_size = opt.datasize,
        save = opt.save,
        transform = transform,
        normal = normal_imgs,
        target_transform = transforms.Compose([
                               transforms.Scale(opt.imagesize//8),
            ]),
        )
    trainingdata = torch.utils.data.DataLoader(train_dataset,
        batch_size = opt.batchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True
        )

if opt.save:
    for i in range(2):
        images = iter(trainingdata).next()
        if normal_imgs is None:
            normal_imgs = [0,1]
        save_image(images['img'],'{}/train_{}.png'.format( opt.outf,str(i).zfill(5)),mean=normal_imgs[0],std=normal_imgs[1])

        print (i)        

    print ('things are saved in {}'.format(opt.outf))
    quit()

testingdata = None
if not opt.datatest == "": 
    testingdata = torch.utils.data.DataLoader(
        MultipleVertexJson(
            root = opt.datatest,
            objectofinterest=opt.object,
            keep_orientation = True,
            noise = opt.noise,
            sigma = opt.sigma,
            data_size = opt.datasize,
            save = opt.save,
            transform = transform,
            normal = normal_imgs,
            target_transform = transforms.Compose([
                                   transforms.Scale(opt.imagesize//8),
                ]),
            ),
        batch_size = opt.batchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True)

if not trainingdata is None:
    print('training data: {} batches'.format(len(trainingdata)))
if not testingdata is None:
    print ("testing data: {} batches".format(len(testingdata)))
print('load models')

net = DopeNetwork(pretrained=opt.pretrained).cuda()
net = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda()

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters,lr=opt.lr)

with open (opt.outf+'/loss_train.csv','w') as file: 
    file.write('epoch,batchid,loss\n')

with open (opt.outf+'/loss_test.csv','w') as file: 
    file.write('epoch,batchid,loss\n')

nb_update_network = 0

def _runnetwork(epoch, loader, train=True):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

    for batch_idx, targets in enumerate(loader):

        data = Variable(targets['img'].cuda())
        
        output_belief, output_affinities = net(data)
                       
        if train:
            optimizer.zero_grad()
        target_belief = Variable(targets['beliefs'].cuda())        
        target_affinity = Variable(targets['affinities'].cuda())

        loss = None
        
        # Belief maps loss
        for l in output_belief: #output, each belief map layers. 
            if loss is None:
                loss = ((l - target_belief) * (l-target_belief)).mean()
            else:
                loss_tmp = ((l - target_belief) * (l-target_belief)).mean()
                loss += loss_tmp
        
        # Affinities loss
        for l in output_affinities: #output, each belief map layers. 
            loss_tmp = ((l - target_affinity) * (l-target_affinity)).mean()
            loss += loss_tmp 

        if train:
            loss.backward()
            optimizer.step()
            nb_update_network+=1

        if train:
            namefile = '/loss_train.csv'
        else:
            namefile = '/loss_test.csv'

        with open (opt.outf+namefile,'a') as file:
            s = '{}, {},{:.15f}\n'.format(
                epoch,batch_idx,loss.data.item()) 
            # print (s)
            file.write(s)

        if train:
            if batch_idx % opt.loginterval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data.item()))
        else:
            if batch_idx % opt.loginterval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data.item()))

        # break
        if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
            torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
            break


for epoch in range(1, opt.epochs + 1):

    if not trainingdata is None:
        _runnetwork(epoch,trainingdata)

    if not opt.datatest == "":
        _runnetwork(epoch,testingdata,train = False)
        if opt.data == "":
            break # lets get out of this if we are only testing
    try:
        torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(opt.outf, opt.namefile ,epoch))
    except:
        pass

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break

print ("end:" , datetime.datetime.now().time())
