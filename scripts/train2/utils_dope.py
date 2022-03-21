"""
NVIDIA from jtremblay@gmail.com
"""
import numpy as np
import scipy.ndimage
import torch

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.data as data
import glob
import os 
import copy
import pickle

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi    

from os.path import exists, basename
import json
from os.path import join

import cv2
import albumentations as A

def default_loader(path):
    return Image.open(path).convert('RGB')          

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
import colorsys,math

def getAffinity(width,height,p1,p2,tickness=12,tensor=None,img_affinity=None):
    """
    take as input image width, and height or a tensor 
    and returns the affinity field from p1 to p2.
    when tensor added then return into the tensor.
    tensor here is pytorch tensor not a numpy array     
    """

    if tensor is None:
        tensor = torch.zeros(2,width,height).float()

    # create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width,height), "black")
    draw = ImageDraw.Draw(imgAffinity)    
    draw.line([p1,p2],fill=(255/4,255/4,255/4),width=tickness)
    draw.line([p1,p2],fill=(255/2,255/2,255/2),width=2*tickness/3)
    draw.line([p1,p2],fill=(255,255,255),width=tickness/3)
    del draw

    # compute the array to add the afinity
    array = (np.array(imgAffinity)/255)[:,:,0]
    angle_vector = np.array(p2) - np.array(p1)
    angle_vector = normalize(angle_vector)
    
    affinity = np.concatenate([[array*angle_vector[0]],[array*angle_vector[1]]])
    # print (tensor)
    if not img_affinity is None:
        # find the angle vector
        # print (angle_vector)
        if length(angle_vector) >0:
            angle=py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle/360,1,1)) * 255
        draw = ImageDraw.Draw(img_affinity)    
        draw.line([p1,p2],fill=(int(c[0]/4),int(c[1]/4),int(c[2]/4)),width=tickness)
        draw.line([p1,p2],fill=(int(c[0]/2),int(c[1]/2),int(c[2]/2)),width=2*tickness/3)
        draw.line([p1,p2],fill=(int(c[0]),int(c[1]),int(c[2])),width=tickness/3)
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity

def loadimages(root,datastyle = "json",extensions= ['png']):
    imgs = []
    loadimages.extensions = extensions

    def add_json_files(path,):

        # print (path)
        # print(len(glob.glob(path+"/*.json")))
        # for file_json in glob.glob(path+"/*.json"):
        #     if "setting" in file_json or "right" in file_json:
        #         continue
        #     data_jsons.append(file_json)
        for ext in loadimages.extensions:
            # for imgpath in glob.glob(path+"/*.rgb.{}".format(ext.replace('.',''))):
            #     if exists(imgpath) and exists(imgpath.replace(ext,"json").replace('.rgb','')):
            #         imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
            #             imgpath.replace(ext,"json").replace(".rgb",'')))
            for imgpath in glob.glob(path+"/*.{}".format(ext.replace('.',''))):
                if exists(imgpath) and exists(imgpath.replace(ext,"json")):
                    imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                        imgpath.replace(ext,"json")))


    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        if len(folders)>0:
            for path_entry in folders:
                
                
                explore(path_entry)
            # raise()
        else:
            add_json_files(path)

    explore(root)

    return imgs

class CleanVisiiDopeLoader(data.Dataset):
    def __init__(
        self,
        path_dataset,
        objects = None,
        sigma = 1,
        output_size = 400, 
        extensions= ["png",'jpg'],
        debug = False
    ):
    ###################    
        self.path_dataset = path_dataset
        self.objects_interest = objects 
        self.sigma = sigma 
        self.output_size = output_size 
        self.extensions = extensions
        self.debug = debug 
        ###################    

        def load_data(path,extensions):
            imgs = loadimages(path,extensions = extensions)

            # Check all the folders in path 
            for name in os.listdir(str(path)):
                imgs += loadimages(path +"/"+name,extensions = extensions)
            return imgs
        self.imgs = []
        for path_look in path_dataset:
            self.imgs += load_data(path_look,extensions = self.extensions)
        
        # np.random.shuffle(self.imgs)        

        if debug:
            print("Debuging will be save in debug/")
            if os.path.isdir("debug"):
                print(f'folder {"debug"}/ exists')
            else:
                os.mkdir("debug")
                print(f'created folder {"debug"}/')


    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        
        # load the data
        path_img, img_name, path_json = self.imgs[index]
    
        # load the image
        # img = cv2.imread(path_img,cv2.COLOR_BGR2RGB)
        img = np.array(Image.open(path_img).convert('RGB')) 

        # load the json file
        all_projected_cuboid_keypoints = []
        with open(path_json) as f:
            data_json = json.load(f)
        
        # load the projected cuboid keypoints
        for obj in data_json['objects']:    
            if not self.objects_interest is None and \
                not obj['class'] in self.objects_interest\
                :
                continue
            # load the projected_cuboid_keypoints
            if obj['visibility'] > 0:
                projected_cuboid_keypoints = obj['projected_cuboid']
            else:
                projected_cuboid_keypoints = [[-100,-100],[-100,-100],[-100,-100],\
                    [-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100]]
            all_projected_cuboid_keypoints.append(projected_cuboid_keypoints)
        
        if len(all_projected_cuboid_keypoints) == 0:
            all_projected_cuboid_keypoints = [[[-100,-100],[-100,-100],[-100,-100],\
                    [-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100]]]

        # flatten the keypoints
        flatten_projected_cuboid = []
        for obj in all_projected_cuboid_keypoints: 
            for p in obj:
                flatten_projected_cuboid.append(p)

        #######
        if self.debug:
            img_to_save = Image.fromarray(img)
            draw = ImageDraw.Draw(img_to_save)

            for ip,p in enumerate(flatten_projected_cuboid):
                draw.ellipse((int(p[0])-2,int(p[1])-2,int(p[0])+2,int(p[1])+2),fill='green')
                # draw.text((p[0]*2+4, p[1]*2+4),str(ip),'green',font=font)

            img_to_save.save(f"debug/{img_name.replace('.png','_original.png')}")
        #######


        # data augmentation
        transform = A.Compose(
            [
                A.RandomCrop(width=400, height=400),
                A.Rotate(limit=180),
                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.15,p=1),
                A.GaussNoise(p=1),

            ], 
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
        )
        transformed = transform(image=img, keypoints=flatten_projected_cuboid)
        img_transformed = transformed['image']
        flatten_projected_cuboid_transformed = transformed['keypoints']
        # img_transformed[:,:,3] = 255

        #######

        # transform to the final output 
        if not self.output_size == 400: 
            transform = A.Compose(
                [
                    A.Resize(width=self.output_size, height=self.output_size),

                ], 
                keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
            )
            transformed = transform(image=img_transformed, keypoints=flatten_projected_cuboid_transformed)
            img_transformed_output_size = transformed['image']
            flatten_projected_cuboid_transformed_output_size = transformed['keypoints']

        else:
            img_transformed_output_size = img_transformed
            flatten_projected_cuboid_transformed_output_size = flatten_projected_cuboid_transformed


        #######
        if self.debug:
            img_transformed_saving = Image.fromarray(img_transformed)

            draw = ImageDraw.Draw(img_transformed_saving)

            for ip,p in enumerate(flatten_projected_cuboid_transformed):
                draw.ellipse((int(p[0])-2,int(p[1])-2,int(p[0])+2,int(p[1])+2),fill='green')
                # draw.text((p[0]*2+4, p[1]*2+4),str(ip),'green',font=font)

            img_transformed_saving.save(f"debug/{img_name.replace('.png','_transformed.png')}")
        #######

        # update the keypoints list
        # obj x keypoint_id x (x,y)
        i_all = 0
        for i_obj, obj in enumerate(all_projected_cuboid_keypoints):
            for i_p, point in enumerate(obj):
                all_projected_cuboid_keypoints[i_obj][i_p] = flatten_projected_cuboid_transformed_output_size[i_all]
                i_all +=1


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
        # def GenerateMapAffinity(img,nb_vertex,pointsInterest,objects_centroid,scale):
        affinities = GenerateMapAffinity(
            size=int(self.output_size),
            nb_vertex = 8, 
            pointsInterest = all_projected_cuboid_keypoints,
            objects_centroid= np.array(all_projected_cuboid_keypoints)[:,-1].tolist(),
            scale = 1,
            # save = True, 
            )        

        # prepare for the image tensors 
        normalize_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
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
            img,grid = save_image(
                imgs, 
                f"debug/{img_name.replace('.png','_beliefs.png')}", 
                mean=0, std=1,nrow=3, save=True
            )
            imgs = VisualizeAffinityMap(affinities)
            save_image(
                imgs, 
                f"debug/{img_name.replace('.png','_affinities.png')}", 
                mean=0, std=1, nrow=3, save=True
            )
        ########
        img_tensor[torch.isnan(img_tensor)] = 0
        affinities[torch.isnan(affinities)] = 0
        beliefs[torch.isnan(beliefs)] = 0

        img_tensor[torch.isinf(img_tensor)] = 0
        affinities[torch.isinf(affinities)] = 0
        beliefs[torch.isinf(beliefs)] = 0


        return {
            'img':img_tensor,  
            "affinities":torch.clamp(affinities,-1,1),            
            'beliefs':torch.clamp(beliefs,0,1),
            'file_name':img_name,
            'img_original':img_original,
        }



def VisualizeAffinityMap(
    tensor, 
    # tensor of (len(keypoints)*2)xwxh
    threshold_norm_vector = 0.4,
    # how long does the vector has to be to be drawn
    points = None, 
    # list of points to draw in white on top of the image
    factor = 1.0, 
    # by how much the image was reduced, scale factor
    translation = (0,0)
    # by how much the points were moved 

    # return len(keypoints)x3xwxh # stack of images
    ):
    images = torch.zeros(tensor.shape[0]//2,3,tensor.shape[1],tensor.shape[2])
    for i_image in range(0,tensor.shape[0],2): #could be read as i_keypoint
        # for i in range(tensor.shape[1]):
            # for j in range(tensor.shape[2]):
        indices = (torch.abs(tensor[i_image,:,:]) + torch.abs(tensor[i_image+1,:,:]) > threshold_norm_vector).nonzero()
        # print(indices)
        for indice in indices:
            # print (indice)
            i,j = indice
            # print(tensor[i,j,i_image,0].item())
            angle_vector = np.array(
                                [tensor[i_image,i,j],
                                tensor[i_image+1,i,j]]
                                )
            if length(angle_vector) > threshold_norm_vector:
                angle=py_ang(angle_vector)
                c = colorsys.hsv_to_rgb(angle/360,1,1)
            else:
                c = [0,0,0]
            for i_c in range(3):
                images[i_image//2,i_c,i,j] = c[i_c]
        if not points is None: 
            point = points[i_image//2]
            # print (images.shape)
            print (                
                int(point[1]*factor+translation[1]),
                int(point[0]*factor+translation[0]),
            )
            images[i_image//2,:,
                int(point[1]*factor+translation[1])-1:int(point[1]*factor+translation[1])+1,
                int(point[0]*factor+translation[0])-1:int(point[0]*factor+translation[0])+1,
                # int(point[0])-1:int(point[0])+1
                ] = 1

    return images

def VisualizeBeliefMap(
    tensor, 
    # tensor of len(keypoints)xwxh
    points = None, 
    # list of points to draw on top of the image
    factor = 1.0, 
    # by how much the image was reduced, scale factor
    translation = (0,0)
    # by how much the points were moved 

    # return len(keypoints)x3xwxh # stack of images in torch tensor
    ):
    images = torch.zeros(tensor.shape[0],3,tensor.shape[1],tensor.shape[2])
    for i_image in range(0,tensor.shape[0]): #could be read as i_keypoint

        belief = tensor[i_image].clone()
        belief -= float(torch.min(belief).item())
        belief /= float(torch.max(belief).item())  

        belief = torch.clamp(belief,0,1)  
        belief = torch.cat([belief.unsqueeze(0),belief.unsqueeze(0),belief.unsqueeze(0)]).unsqueeze(0)

        images[i_image] = belief

    return images


def GenerateAffinityPoints2Tensor(
        points, 
        # list of points in image space nx2
        tensor_mask, 
        # mask of the object, wxh
        factor = 1.0, 
        # by how much the image was reduced, scale factor
        translation = (0,0)
        # by how much were the keypoint moved
        #returns a tensor (len(points)*2)xwxh
    ):
    
    tensor = torch.zeros(len(points)*2,tensor_mask.shape[0],tensor_mask.shape[1])

    # find where there are ones. 
    indices = (tensor_mask > 0).nonzero()
    for i_stack in range(0,len(points)*2,2):
        i_point = i_stack//2
        for indice in indices:
            i,j = indice
            # j,i = indice
            angle_vector = np.array([points[i_point][0] * factor + translation[0],
                 points[i_point][1] * factor + translation[1]]) - np.array((float(i),float(j)))
            angle_vector = normalize(angle_vector)

            tensor[i_stack,i,j] = angle_vector[0]
            tensor[i_stack+1,i,j] = angle_vector[1]
    return tensor


def GenerateMapAffinity(size,nb_vertex,pointsInterest,objects_centroid,scale,save=False):
    # Apply the downscale right now, so the vectors are correct. 

    img_affinity = Image.new("RGB", (int(size/scale),int(size/scale)), "black")
    # create the empty tensors
    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(2,int(size/scale),int(size/scale)))
    
    for i_pointsImage in range(len(pointsInterest)):    
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]
            # print (pointsImage[i_points])
            affinity_pair, img_affinity = getAfinityCenter(
                int(size/scale),
                int(size/scale),
                tuple((np.array(pointsImage[i_points])/scale).tolist()),
                tuple((np.array(center)/scale).tolist()), 
                img_affinity = img_affinity, 
                radius=1)

            # affinities[i_points] = (affinities[i_points] + affinity_pair)
            affinities[i_points] = (affinities[i_points] + affinity_pair)/2


            #Normalizing
            v = affinities[i_points].numpy()                    
            
            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero]/=norms[nonzero]
            yvec[nonzero]/=norms[nonzero]

            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec],[yvec]]))
    affinities = torch.cat(affinities,0)

    # img_affinity.save('aff.png')
    return affinities

def getAfinityCenter(width,height,point,center,radius=7,tensor=None,img_affinity=None):
    """
    Create the affinity map
    """
    if tensor is None:
        tensor = torch.zeros(2,height,width).float()

    # create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width,height), "black")
    totensor = transforms.Compose([transforms.ToTensor()])
    # raise()
    draw = ImageDraw.Draw(imgAffinity)    
    r1 = radius
    p = point
    draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),(255,255,255))

    del draw

    # compute the array to add the afinity
    array = (np.array(imgAffinity)/255)[:,:,0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array*angle_vector[0]],[array*angle_vector[1]]])

    # print (tensor)
    if not img_affinity is None:
        # find the angle vector
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

       
def CreateBeliefMap(size,pointsBelief,nbpoints,sigma=16,save=False):
    #Create the belief maps in the points        
    beliefsImg = []
    # sigma = sigma
    # print(img.shape)
    for numb_point in range(nbpoints):    
        array = np.zeros([size,size])
        out = np.zeros([size,size])

        for point in pointsBelief:
            p = [point[numb_point][1],point[numb_point][0]]
            w = int(sigma*2)
            if p[0]-w>=0 and p[0]+w<size and p[1]-w>=0 and p[1]+w<size:
                for i in range(int(p[0])-w, int(p[0])+w+1):
                    for j in range(int(p[1])-w, int(p[1])+w+1):
                        # print (i,p[0],j,p[1])
                        # raise()
                        # if there is already a point there. 
                        array[i,j] = max(np.exp(-(((i - p[0])**2 + (j - p[1])**2)/(2*(sigma**2)))),array[i,j])
                        # out[i,j] = (-((i - p[0])*(i - p[0]) - (j - p[1])*(j - p[1]))/(2*sigma*sigma))
                        # print (array[i,j])
                        # raise()
                # print (array[int(p[0])-w:int(p[0])+w,int(p[1])-w:int(p[1])+w])
                # print (out[int(p[0])-w:int(p[0])+w,int(p[1])-w:int(p[1])+w])
        # print ('----')
        # print (np.min(array),np.max(array))
        # stack = stack.astype("uint8")

        # print (np.array(imgBelief).dtype)
        # print (stack.dtype)
        beliefsImg.append(array.copy())
        # raise()
        # break
        if save:
            stack = np.stack([array,array,array],axis=0).transpose(2,1,0)
            # imgBelief = Image.new('RGB', [size,size], "black")
            imgBelief = Image.fromarray((stack*255).astype('uint8'))
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
    # def __init__(self,min=0.1,max=1.9):
    #     self.min = min
    #     self.max = max
    def __init__(self,sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):

        contrast = ImageEnhance.Contrast(im)
        # im = contrast.enhance( np.random.rand() * (self.max-self.min) + self.min )
        im = contrast.enhance( np.random.normal(1,self.sigma) )
        # im = contrast.enhance( 1 )
        
        return im


class AddRandomBrightness(object):
    """
    Apply some random image filters from PIL
    """
    # def __init__(self,min=0.1,max=1.9):
    #     self.min = min
    #     self.max = max
    def __init__(self,sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):

        contrast = ImageEnhance.Brightness(im)
        # im = contrast.enhance( np.random.rand() * (self.max-self.min) + self.min )
        im = contrast.enhance( np.random.normal(1,self.sigma) )
        # im = contrast.enhance( 1 )
        return im        

class AddNoise(object):
    """Given mean: (R, G, B) and std: (R, G, B),
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


irange = range

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
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
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

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
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

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
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2,mean=None, std=None, save=True):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10,pad_value=1)
    if not mean is None:
        # ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    else:      
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    im = Image.fromarray(ndarr)
    if save is True:
        im.save(filename)
    return im, grid




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
    DrawLine(points[0], points[1], lineColor, 8, draw) #lineWidthForDrawing)
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

def OverlayBeliefOnImage(img, beliefs, name="tmp", path="", factor=0.7, grid=3, 
        norm_belief = True, save = False, scale_factor=1):
    """
    take as input 
    img: a tensor image in pytorch normalized at 0.5
            3xwxh
    belief: tensor of the same size as the image to overlay over img 
            nb_beliefxwxh
    name: str to name the image, e.g., output.png
    path: where to save, e.g., /where/to/save/
    factor: float [0,1] how much to keep the original, 1 = fully, 0 black
    grid: how big the grid, e.g., 3 wide. 
    norm_belief: bool to normalize the values [0,1]
    save: write to disk 
    scale_factor: how much to scale the belief map
    returns the image 
    """

    belief_imgs = []
    if not img is None:
        in_img = img
        in_img *= factor     
    tensor = nn.functional.upsample(beliefs.unsqueeze(0),scale_factor=scale_factor ).squeeze()      
    for j in range(tensor.size()[0]):
        belief = tensor[j].clone()
        if norm_belief:
            belief -= float(torch.min(belief).data.cpu().numpy())
            belief /= float(torch.max(belief).data.cpu().numpy())
        belief = torch.clamp(belief,0,1).cpu()
        
        if img is None:
            belief = torch.cat([
                        belief.unsqueeze(0),belief.unsqueeze(0),belief.unsqueeze(0)
                        ]).unsqueeze(0)
        else:
            belief = torch.cat([
                        belief.unsqueeze(0)* in_img[0,:,:] + in_img[0,:,:],
                        belief.unsqueeze(0)* in_img[1,:,:] + in_img[1,:,:],
                        belief.unsqueeze(0)* in_img[2,:,:] + in_img[2,:,:]
                        ]).unsqueeze(0)
            
        # print (in_img[0,:,:].shape)
        # print ((belief.unsqueeze(0) * in_img[1,:,:] * factor + in_img[1,:,:]).shape)
        # print (in_img[2,:,:].shape)
        # belief = torch.cat([
        #             in_img[0,:,:].unsqueeze(0),
        #             belief.unsqueeze(0) * in_img[1,:,:] + in_img[1,:,:],
        #             in_img[2,:,:].unsqueeze(0)
        #             ]).unsqueeze(0)
        belief = torch.clamp(belief,0,1) 

        belief_imgs.append(belief.data.squeeze().numpy())

    # Create the image grid
    belief_imgs = torch.tensor(np.array(belief_imgs))

    img,grid = save_image(belief_imgs, "{}{}".format(path, name), 
        mean=0, std=1, nrow=grid, save=save)

    return img, grid

def GetPoseMatrix(location,rotation):
    """
        Return the rotation Matrix from a vector translation 
        and a quaternion rotation vector

    """
    from pyquaternion import Quaternion 
    
    pose_matrix = np.zeros([4,4])
    q = Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])

    pose_matrix[0:3,0:3] = q.rotation_matrix
    pose_matrix[0:3,3] = np.array(location)
    pose_matrix[3,3] = 1

    return pose_matrix

def ADDErrorCuboid(pose_gu, pose_gt, cuboid):
    """
        Compute the ADD error for a given cuboid. 
        pose_gu is the predicted pose as a matrix
        pose_gt is the ground thruth pose as a matrix
        cuboid is a Cuboid3D object (see inference/cuboid.py)
    """
    from scipy import spatial

    #obj = self.__obj_model
    vertices = np.array(cuboid._vertices)
    # print (vertices.shape)
    vertices = np.insert(vertices,3,1,axis=1)
    vertices = np.rot90(vertices,3)
    # print(vertices)

    obj = vertices
    pred_obj = np.matmul(pose_gu, obj)
    
    # obj = self.__obj_model
    # pred_obj = np.matmul(pose_gu, obj)
    # print (pred_obj)

    actual_obj = np.matmul(pose_gt, obj)
    #actual_obj = np.matmul(self.LtNT, actual_obj_l)
    #print("PREDICTED OBJECT\n", pred_obj)
    #print("ACTUAL OBJECT\n", actual_obj)   
    dist = spatial.distance.cdist(pred_obj.T, actual_obj.T, 'euclidean')
    true_dist = [dist[i][i] for i in range(len(dist))]
    #for i in range(len(true_dist)):
    #    if true_dist[i] >6000:
    #        print(i, true_dist[i])
    # print (true_dist)
    # raise()
    return np.mean(true_dist)

if __name__ == '__main__':
    # mask = torch.zeros(200,200)
    # mask[50:100,50:100] = 1
    # m = GenerateAffinityPoints2Tensor([(50,50),(50,100),(100,50),(100,100)],mask)
    # # print(m.shape)
    # imgs = VisualizeAffinityMap(m,points = [(50,50),(50,100),(100,50),(100,100)])

    # # save the images using the grid
    # img,grid = save_image(imgs, "some_img.png", 
    #     mean=0, std=1, nrow=2, save=True)
    # raise()

    # path = '../data/fat_sugar_simple_20/'
    # path = '../data/fat_sugar_simple_20/'
    # path = "../../data/simple_cube/"

    # train_dataset = MultipleVertexJson(
    #     root = path,
    #     objectsofinterest = ['cube_red'],
    #     # random_translation = [0,0],
    #     # random_rotation = 0,
    #     # objectsofinterest = ['004_sugar_box_16k']
    #     # scale_output = 2,
    #     debug = True,
    #     )

    path = "../../data/single_object_dr_chocolat_pudding/"
    # path = '/home/trump/raid/data/home_dataset/single_ndds1/AlphabetSoup/single'
    path = '/home/trump/raid/data/visii/dr/dr_Ketchup_000/'
    path = '/home/trump/raid/data/visii/dr/Ketchup'
    path = ['../data/visii_fat/', '../data/TomatoSauce/']
    path = ['../../data/visii/meat_can_shiny/']
    # path = "../../data/home_dataset/pudding_1/"
    # path = '/media/jtremblay/data/home_dataset/ndds1/alphabet_soup_small_single/'
    # path = "../../data/home_dataset/single_object_dr_chocolat_pudding/"
    # path = "../../data/home_dataset/pudding_1/"
    # path = '/media/jtremblay/data/home_dataset/ndds1/alphabet_soup_small/'


    train_dataset = CleanVisiiDopeLoader(
        path,
        output_size = 50, 
        sigma = 0.75,
        debug = True,
        # objects_interest = "TomatoSauce"
    )

    trainingdata = torch.utils.data.DataLoader(train_dataset,
        batch_size = 12, 
        shuffle = True,
        num_workers = 1, 
        pin_memory = False
    )

    # print(len(trainingdata))
    targets = iter(trainingdata).next()

    # raise()


    # train_dataset = MultipleVertexJson(
    #     root = path,
    #     objectsofinterest = None,
    #     # random_translation = [0,0],
    #     # random_rotation = 0,
    #     # objectsofinterest = ['004_sugar_box_16k']
    #     # output_size = 50,
    #     # sigma = 1, # for 50x50
    #     output_size = 50,
    #     sigma = 4, # for 50x50
    #     debug = False,
    #     )

    # trainingdata = torch.utils.data.DataLoader(train_dataset,
    #     batch_size = 10, 
    #     shuffle = False,
    #     num_workers = 1, 
    #     pin_memory = False
    # )

    # print(len(trainingdata))
    # targets = iter(trainingdata).next()

    # # print(targets['pointsBelief'].shape)
    # # print(targets['pointsBelief'].min())
    # # print(targets['pointsBelief'].max())
    # print(targets.keys())

    
    # print(targets['file_name'])

    # imgs = VisualizeBeliefMap(targets['beliefs'][0])

    # img,grid = save_image(imgs, "some_img.png", 
    #     mean=0, std=1, nrow=3, save=True)

    # imgs = VisualizeAffinityMap(targets['affinities'][0])

    # img,grid = save_image(imgs, "some_aff.png", 
    #     mean=0, std=1, nrow=3, save=True)

    # imgs = VisualizeAffinityMap(targets['affinities'][0])

    # img,grid = save_image(
    #     torch.cat([
    #         targets['img_original'],
    #         targets['img_original'],
    #         targets['img_original'],
    #         targets['img_original'],
    #         targets['img_original'],
    #         targets['img_original'],
    #         targets['img_original'],
    #         targets['img_original'],
    #         ])
    #     , "imgs.png", 
    #     mean=0, std=1, nrow=3, save=True)    


    # ####### test the inference on the data output 
    # import sys 
    # sys.path.append("inference")
    
    # from cuboid import Cuboid3d
    # from cuboid_pnp_solver import CuboidPNPSolver
    # from detector import ModelData, ObjectDetector

    # config = lambda: None
    # config.thresh_angle = 0.5
    # config.thresh_map = 0.0001
    # config.sigma = 3
    # config.thresh_points = 0.01
    # scale_factor = 8
    # objects, all_peaks = ObjectDetector.find_objects(
    #                         targets['beliefs'][0], 
    #                         targets['affinities'][0], 
    #                         config,
    #                         scale_factor=scale_factor)
    # for obj in objects:
    #     if obj is None: 
    #         continue
    #     # Run PNP
    #     points = obj[1] + [(obj[0][0]*scale_factor, obj[0][1]*scale_factor)]
    #     cuboid2d = np.copy(points)
    #     try:
    #         print(cuboid2d/scale_factor)
    #     except:
    #         pass
    #     avg_dist = 0 
    #     denominator = 0 
    #     # for i_p,p in enumerate(cuboid2d):
    #     #     if p is None:
    #     #         continue
    #     #     # print(p/scale_factor,targets['pointsBelief'][0][i_p].numpy(),np.linalg.norm(p/scale_factor-targets['pointsBelief'][0][i_p].numpy()))
    #     #     avg_dist += np.linalg.norm(p/scale_factor-targets['pointsBelief'][0][i_p].numpy())
    #     #     denominator += 1
    #     # print(avg_dist/denominator)
    #     # print(targets['pointsBelief'][0])
