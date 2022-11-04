# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

'''
Contains the following classes:
   - ModelData - High level information encapsulation
   - ObjectDetector - Greedy algorithm to build cuboids from belief maps 
'''

import time
import json
import os, shutil
import sys
import traceback
from os import path
import threading
from threading import Thread

import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models

from scipy import ndimage
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import gaussian_filter
from scipy import optimize

import sys 
sys.path.append("../")
from models import * 

# Import the definition of the neural network model and cuboids
from cuboid_pnp_solver import *

#global transform for image input
transform = transforms.Compose([
    # transforms.Scale(IMAGE_SIZE),
    # transforms.CenterCrop((imagesize,imagesize)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(    
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                    )
    ])


#================================ Models ================================


class DopeNetwork(nn.Module):
    def __init__(
            self,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        vgg_full = models.vgg19(pretrained=False).features
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



class ModelData(object):
    '''This class contains methods for loading the neural network'''

    def __init__(self, name="", net_path="", gpu_id=0,architecture='dope'):
        self.name = name
        self.net_path = net_path  # Path to trained network model
        self.net = None  # Trained network
        self.gpu_id = gpu_id
        self.architecture = architecture

    def get_net(self):
        '''Returns network'''
        if not self.net:
            self.load_net_model()
        return self.net

    def load_net_model(self):
        '''Loads network model from disk'''
        if not self.net and path.exists(self.net_path):
            self.net = self.load_net_model_path(self.net_path)
        if not path.exists(self.net_path):
            print("ERROR:  Unable to find model weights: '{}'".format(
                self.net_path))
            exit(0)

    def load_net_model_path(self, path):
        '''Loads network model from disk with given path'''
        model_loading_start_time = time.time()
        print("Loading DOPE model '{}'...".format(path))
        if self.architecture == 'dope':
            net = DopeNetwork()
        else:
            net = ResnetSimple()
        net = torch.nn.DataParallel(net, [0]).cuda()
        net.load_state_dict(torch.load(path))
        net.eval()
        print('    Model loaded in {} seconds.'.format(
            time.time() - model_loading_start_time))
        return net

    def __str__(self):
        '''Converts to string'''
        return "{}: {}".format(self.name, self.net_path)


#================================ ObjectDetector ================================
class ObjectDetector(object):
    '''This class contains methods for object detection'''

    @staticmethod
    def gaussian(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(
                    -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    @staticmethod
    def moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    @staticmethod
    def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = ObjectDetector.moments(data)
        errorfunction = lambda p: np.ravel(ObjectDetector.gaussian(*p)(*np.indices(data.shape)) -
                                    data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
    @staticmethod
    def make_grid(tensor, nrow=8, padding=2,
                  normalize=False, range_=None, scale_each=False, pad_value=0):
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
        import math

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
            if range_ is not None:
                assert isinstance(range_, tuple), \
                    "range has to be a tuple (min, max) if specified. min and max are numbers"

            def norm_ip(img, min, max):
                img.clamp_(min=min, max=max)
                img.add_(-min).div_(max - min + 1e-5)

            def norm_range(t, range_):
                if range_ is not None:
                    norm_ip(t, range_[0], range_[1])
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
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                grid.narrow(1, y * height + padding, height - padding)\
                    .narrow(2, x * width + padding, width - padding)\
                    .copy_(tensor[k])
                k = k + 1
        return grid

    @staticmethod
    def get_image_grid(tensor, filename, nrow=3, padding=2,mean=None, std=None):
        """
        Saves a given Tensor into an image file.
        If given a mini-batch tensor, will save the tensor as a grid of images.
        """
        from PIL import Image
        
        # tensor = tensor.cpu()
        grid = ObjectDetector.make_grid(tensor, nrow=nrow, padding=10,pad_value=1)
        if not mean is None:
            # ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
            ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        else:      
            ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        im = Image.fromarray(ndarr)
        # im.save(filename)
        return im

    @staticmethod
    def detect_object_in_image(net_model, pnp_solver, in_img, config,
                               make_belief_debug_img=False, norm_belief=True, overlay_image=True):
        """
        Detect objects in a image using a specific trained network model
        Returns the poses of the objects and the belief maps
        """
        if in_img is None:
            return []

        # Run network inference
        image_tensor = transform(in_img)
        image_torch = Variable(image_tensor).cuda().unsqueeze(0)
        out, seg = net_model(image_torch)
        vertex2 = out[-1][0]
        aff = seg[-1][0]

        # Find objects from network output
        detected_objects = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, config)

        if not make_belief_debug_img:
            return detected_objects, None
        else:
            # Run the belief maps debug display on the belief maps
            tensor = vertex2
            belief_imgs = []
            if overlay_image:
                upsampling = nn.UpsamplingNearest2d(size=in_img.shape[:2])
                in_img = (torch.tensor(in_img).float() / 255.0)
                in_img *= 0.5

            for j in range(tensor.size()[0]):
                belief = tensor[j].clone()
                if norm_belief:
                    belief -= float(torch.min(belief).item())
                    belief /= float(torch.max(belief).item())

                belief = torch.clamp(belief, 0, 1).cpu()
                if overlay_image:
                    belief = upsampling(belief.unsqueeze(0).unsqueeze(0)).squeeze().squeeze().data
                    belief = torch.cat([
                        belief.unsqueeze(0) + in_img[:, :, 0],
                        belief.unsqueeze(0) + in_img[:, :, 1],
                        belief.unsqueeze(0) + in_img[:, :, 2]
                    ]).unsqueeze(0)
                    belief = torch.clamp(belief, 0, 1)
                else:
                    belief = torch.cat([
                        belief.unsqueeze(0),
                        belief.unsqueeze(0),
                        belief.unsqueeze(0)
                    ]).unsqueeze(0)
                belief_imgs.append(belief.data.squeeze().numpy())

            # Create the image grid
            belief_imgs = torch.tensor(np.array(belief_imgs))

            im_belief = ObjectDetector.get_image_grid(belief_imgs, None,
                mean=0, std=1)

            return detected_objects, im_belief


            
    @staticmethod
    def find_object_poses(vertex2, aff, pnp_solver, config, run_sampling=False, num_sample=100,scale_factor=8):
        '''Detect objects given network output'''

        # run_sampling = True
        
        # Detect objects from belief maps and affinities
        objects, all_peaks = ObjectDetector.find_objects(vertex2, aff, config,
            run_sampling=run_sampling, num_sample=num_sample,scale_factor=scale_factor)
        detected_objects = []
        obj_name = pnp_solver.object_name
        
        # print(all_peaks)

        #print("find_object_poses:  found {} objects ================".format(len(objects)))
        for obj in objects:
            # Run PNP
            points = obj[1] + [(obj[0][0]*scale_factor, obj[0][1]*scale_factor)]
            cuboid2d = np.copy(points)
            location, quaternion, projected_points = pnp_solver.solve_pnp(points)

            # run multiple sample
            if run_sampling:
                lx,ly,lz = [],[],[]
                qx,qy,qz,qw = [],[],[],[]

                for i_sample in range(num_sample):
                    sample = []
                    for i_point in range(len(obj[-1])):
                        if not obj[-1][i_point][i_sample] is None:
                            sample.append( (obj[-1][i_point][i_sample][0]*scale_factor,
                                obj[-1][i_point][i_sample][1]*scale_factor))
                        else:
                            sample.append( None)
                    # final_cuboids.append(sample)
                    pnp_sample = pnp_solver.solve_pnp(sample)

                    try:
                        lx.append(pnp_sample[0][0])
                        ly.append(pnp_sample[0][1])
                        lz.append(pnp_sample[0][2])

                        qx.append(pnp_sample[1][0])
                        qy.append(pnp_sample[1][1])
                        qz.append(pnp_sample[1][2])
                        qw.append(pnp_sample[1][3])
                    except:
                        pass
                    # TODO
                    # RUN quaternion as well for the std and avg. 
                    
                try:
                    print ("----")
                    print ("location:")
                    print (location[0],location[1],location[2])
                    print (np.mean(lx),np.mean(ly),np.mean(lz))
                    print (np.std(lx),np.std(ly),np.std(lz))
                    print ("quaternion:")
                    print (quaternion[0],quaternion[1],quaternion[2],quaternion[3])
                    print (np.mean(qx),np.mean(qy),np.mean(qz),np.mean(qw))
                    print (np.std(qx),np.std(qy),np.std(qz),np.std(qw))
                
                
                except:
                    pass
            if not location is None:
                detected_objects.append({
                    'name': obj_name,
                    'location': location,
                    'quaternion': quaternion,
                    'cuboid2d': cuboid2d,
                    'projected_points': projected_points,
                    'confidence': obj[-1],
                    'raw_points': points               
                })

            #print("find_object_poses:  points = ", type(points), points)
            #print("find_object_poses:  locn = ", location, "quat =", quaternion)
            #print("find_object_poses:  projected_points = ", type(projected_points), projected_points)

        return detected_objects

    @staticmethod
    def find_objects(vertex2, aff, config, numvertex=8, run_sampling=False, num_sample=100,scale_factor=8):
        '''Detects objects given network belief maps and affinities, using heuristic method'''

        all_peaks = []
        all_samples = []

        peak_counter = 0
        for j in range(vertex2.size()[0]):
            belief = vertex2[j].clone()
            map_ori = belief.cpu().data.numpy()
            
            map = gaussian_filter(belief.cpu().data.numpy(), sigma=config.sigma)
            p = 1
            map_left = np.zeros(map.shape)
            map_left[p:,:] = map[:-p,:]
            map_right = np.zeros(map.shape)
            map_right[:-p,:] = map[p:,:]
            map_up = np.zeros(map.shape)
            map_up[:,p:] = map[:,:-p]
            map_down = np.zeros(map.shape)
            map_down[:,:-p] = map[:,p:]

            peaks_binary = np.logical_and.reduce(
                                (
                                    map >= map_left, 
                                    map >= map_right, 
                                    map >= map_up, 
                                    map >= map_down, 
                                    map > config.thresh_map)
                                )
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) 
            
            # Computing the weigthed average for localizing the peaks
            peaks = list(peaks)
            win = 11
            ran = win//2
            peaks_avg = []
            point_sample_list = []
            for p_value in range(len(peaks)):
                p = peaks[p_value]
                weights = np.zeros((win,win))
                i_values = np.zeros((win,win))
                j_values = np.zeros((win,win))
                for i in range(-ran,ran+1):
                    for j in range(-ran,ran+1):
                        if p[1]+i < 0 \
                                or p[1]+i >= map_ori.shape[0] \
                                or p[0]+j < 0 \
                                or p[0]+j >= map_ori.shape[1]:
                            continue 

                        i_values[j+ran, i+ran] = p[1] + i
                        j_values[j+ran, i+ran] = p[0] + j

                        weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])
                # if the weights are all zeros
                # then add the none continuous points
                OFFSET_DUE_TO_UPSAMPLING = 0.4395

                # Sample the points using the gaussian
                if run_sampling:
                    data = weights
                    params = ObjectDetector.fitgaussian(data)
                    fit = ObjectDetector.gaussian(*params)
                    _, mu_x,mu_y,std_x,std_y = params
                    points_sample = np.random.multivariate_normal(
                        np.array([p[1] + mu_x + OFFSET_DUE_TO_UPSAMPLING,
                            p[0] - mu_y + OFFSET_DUE_TO_UPSAMPLING]), 
                        # np.array([[std_x*std_x,0],[0,std_y*std_y]]), size=num_sample)
                        np.array([[std_x,0],[0,std_y]]), size=num_sample)
                    point_sample_list.append(points_sample)


                try:
                    peaks_avg.append(
                        (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                         np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
                except:
                    peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))

            
            # Note: Python3 doesn't support len for zip object
            peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

            peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

            id = range(peak_counter, peak_counter + peaks_len)

            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            all_samples.append(point_sample_list)
            peak_counter += peaks_len

        objects = []

        if aff is None:
            # Assume there is only one object 
            points = [None for i in range(numvertex)]
            for i_peak, peaks in enumerate(all_peaks):
                # print (peaks)
                for peak in peaks:
                    if peak[2] > config.threshold:
                        points[i_peak] = (peak[0],peak[1])

            return points 


        # Check object centroid and build the objects if the centroid is found
        for nb_object in range(len(all_peaks[-1])):
            if all_peaks[-1][nb_object][2] > config.thresh_points:
                objects.append([
                    [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
                    [None for i in range(numvertex)],
                    [None for i in range(numvertex)],
                    all_peaks[-1][nb_object][2],
                    [[None for j in range(num_sample)] for i in range(numvertex+1)]
                ])

                # Check if the object was added before
                if run_sampling and nb_object < len(objects):
                    # add the samples to the object centroids 
                    objects[nb_object][4][-1] = all_samples[-1][nb_object]


        # Working with an output that only has belief maps
        if aff is None:
            if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
                for i_points in range(8):
                    if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > config.threshold:
                        objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
        else:
            # For all points found
            for i_lists in range(len(all_peaks[:-1])):
                lists = all_peaks[i_lists]

                # Candidate refers to point that needs to be match with a centroid object 
                for i_candidate, candidate in enumerate(lists):
                    if candidate[2] < config.thresh_points:
                        continue

                    i_best = -1
                    best_dist = 10000 
                    best_angle = 100

                    # Find the points that links to that centroid. 
                    for i_obj in range(len(objects)):
                        center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                        # integer is used to look into the affinity map, 
                        # but the float version is used to run 
                        point_int = [int(candidate[0]), int(candidate[1])]
                        point = [candidate[0], candidate[1]]

                        # look at the distance to the vector field.
                        v_aff = np.array([
                                        aff[i_lists*2, 
                                        point_int[1],
                                        point_int[0]].data.item(),
                                        aff[i_lists*2+1, 
                                            point_int[1], 
                                            point_int[0]].data.item()]) * 10

                        # normalize the vector
                        xvec = v_aff[0]
                        yvec = v_aff[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec/=norms
                        yvec/=norms
                            
                        v_aff = np.concatenate([[xvec],[yvec]])

                        v_center = np.array(center) - np.array(point)
                        xvec = v_center[0]
                        yvec = v_center[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)
                            
                        xvec /= norms
                        yvec /= norms

                        v_center = np.concatenate([[xvec],[yvec]])
                        
                        # vector affinity
                        dist_angle = np.linalg.norm(v_center - v_aff)

                        # distance between vertexes
                        dist_point = np.linalg.norm(np.array(point) - np.array(center))
                        
                        if dist_angle < config.thresh_angle \
                                and best_dist > 1000 \
                                or dist_angle < config.thresh_angle \
                                and best_dist > dist_point:
                            i_best = i_obj
                            best_angle = dist_angle
                            best_dist = dist_point

                    if i_best == -1:
                        continue
                    
                    if objects[i_best][1][i_lists] is None \
                            or best_angle < config.thresh_angle \
                            and best_dist < objects[i_best][2][i_lists][1]:
                        # set the points 
                        objects[i_best][1][i_lists] = ((candidate[0])*scale_factor, (candidate[1])*scale_factor)
                        # set information about the points: angle and distance
                        objects[i_best][2][i_lists] = (best_angle, best_dist)
                        # add the sample points
                        if run_sampling:
                            # print ("---")
                            # print(len(all_samples[i_lists]))
                            # print (len(all_peaks[i_lists]))
                            # print(i_obj)
                            objects[i_best][4][i_lists] = all_samples[i_lists][i_candidate]
        return objects, all_peaks
