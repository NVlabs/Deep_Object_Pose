"""
python train.py --data data/path/to/images
Use the NDDS format for images and json. 

"""


from __future__ import print_function

import argparse
import os
import random
import shutil

try: 
    import configparser as configparser
except ImportError: 
    import ConfigParser as configparser

# import cv2
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
import datetime
import json

from tensorboardX import SummaryWriter


from PIL import Image
from PIL import ImageDraw

from models import *
from utils_dope import *


import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.gradcheck = False
torch.backends.cudnn.benchmark = True

# import ctypes

# _libcudart = ctypes.CDLL('libcudart.so')
# # Set device limit on the current device
# # cudaLimitMaxL2FetchGranularity = 0x05
# pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
# _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
# _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
# assert pValue.contents.value == 128




# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
parser.add_argument('--data', nargs='+', help='path to training data')
parser.add_argument('--datatest', nargs='+', default="", help='path to data testing set')
parser.add_argument('--testonly', action='store_true', help='only run inference') 
parser.add_argument('--testbatchsize', default=1,type=int, help='size of the batchsize for testing')
parser.add_argument('--freeze', action='store_true', help='use affinity maps for training')
parser.add_argument('--test', action='store_true', help='use affinity maps for training')
parser.add_argument('--debug', action='store_true', help='gt vs. prediction show difference')
parser.add_argument('--savetest', action='store_true', help='Save the output of the testing -- this might slow things')
parser.add_argument('--horovod', action='store_true', help='run the network with horovod')
parser.add_argument('--objects',nargs='+', type=str, default=None, 
    help='In the dataset which objets of interest')
parser.add_argument('--optimizer', default='adam', 
    help='which optimizer to use, default=adam')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
parser.add_argument('--imagesize', type=int, default=448, help='the height / width of the input image to network')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
parser.add_argument('--noise', type=float, default=2.0, help='gaussian noise added to the image')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--net_dope', default=None, help="path to pretrained dope_network")
parser.add_argument('--network', default='dream_full', help="[dream_full,dope,mobile]")
parser.add_argument('--namefile', default='epoch', help="name to put on the file of the save weightss")
parser.add_argument('--manualseed', type=int, help='manual seed')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--loginterval', type=int, default=100)
# parser.add_argument('--gpuids',nargs='+', type=int, default=[0,1,2,3], help='GPUs to use')
parser.add_argument('--gpuids',nargs='+', type=int, default=[0], help='GPUs to use')
parser.add_argument('--extensions',nargs='+', type=str, default=["png"], 
    help='Extensions to use, you can have multiple entries seperated by space, e.g., png jpeg. ')
parser.add_argument('--outf', default='tmp', help='folder to output images and model checkpoints')
parser.add_argument('--sigma', default=4, help='keypoint creation sigma')
parser.add_argument('--keypoints', default=None, 
    help='list of keypoints to load from the json files.')
parser.add_argument('--no_affinity', default=False,
    help='remove the affinity part of DOPE')
parser.add_argument('--datastyle', default='json', help='What data style are you using: nvdl,lov,pascal')
parser.add_argument('--save', action="store_true", help='save a batch and quit')
parser.add_argument('--verbose', action="store_true", help='speak out your mind program')
parser.add_argument('--dontsave', action="store_true", default=False,help='dont_save_weights')

parser.add_argument("--pretrained",default=True,help='do you want to have pretrained weights')

# Feature extractor
parser.add_argument('--features', default="vgg", help='vgg or resnet')

# datasize is used 
parser.add_argument('--datasize', default=None, help='the size in absolute to use, e.g. 2000') 
parser.add_argument('--nbupdates', default=None, help='nb max update to network')

# These are used when you want to mix two datasets with different percentage. 
parser.add_argument('--data1', default=None, help='path to dataset1')
parser.add_argument('--data2', default=None, help='path to dataset2')
parser.add_argument('--size1', default=None, help='size of dataset1 in percentage (0,1)')
parser.add_argument('--size2', default=None, help='size of dataset2 in percentage (0,1)')
parser.add_argument("--local_rank", type=int)

# Read the config but do not overwrite the args written 
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }

if args.config:
    config = configparser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

if opt.keypoints:
    opt.keypoints = eval(opt.keypoints)

if isinstance(opt.extensions, str):
    opt.extensions = eval(opt.extensions)

if isinstance(opt.objects, str):
    opt.objects = eval(opt.objects)

try:
    for i,o in enumerate(opt.objects):
        opt.objects[i] = opt.objects[i].lower()
except:
    pass 

train_sampler = None
if opt.horovod:
    import horovod.torch as hvd
    import torch.utils.data.distributed
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

if opt.testonly: 
    opt.epochs = 1

if not "/" in opt.outf:
    opt.outf = "train_{}".format(opt.outf)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Delete the folder for testing
try:
    shutil.rmtree("{}/test/".format(opt.outf))
    print ('deleted the test folder')
except:
    pass

# Delete the folder for testing
try:
    shutil.rmtree("{}/test/".format(opt.outf))
    print ('deleted the test folder')
except:
    pass


# Create the folders for debugging
if opt.debug:
    try:
        shutil.rmtree("{}/debug/".format(opt.outf))
        # print ('deleted the debug folder')
    except:
        pass
    try:
        os.makedirs("{}/debug/".format(opt.outf))
        # print ("created debug dir")
    except OSError:
        pass


with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt)+"\n")

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt))
    file.write("seed: "+ str(opt.manualseed)+'\n')
with open (opt.outf+'/test_metric.csv','w') as file:
    file.write("epoch, passed, total \n")


# Data check from args
# if opt.testbatchsize > 1:
#     print ('ERROR: test batchsize has to be one')
#     opt.testbatchsize = 1 
if opt.local_rank == 0:
    writer = SummaryWriter(opt.outf+"/runs/")

random.seed(opt.manualseed)


torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(backend='NCCL',
                                     init_method='env://')


torch.manual_seed(opt.manualseed)

torch.cuda.manual_seed_all(opt.manualseed)

if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59,0.25]
    transform = transforms.Compose([
                               AddRandomContrast(0.2),
                               AddRandomBrightness(0.2),
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


if opt.network == 'resnetsimple':
    net = ResnetSimple()
    output_size = 208
    
elif opt.network == 'dope':
    net = DopeNetwork()
    output_size = 50
    opt.sigma = 0.5

elif opt.network == 'full':
    net = ()
    output_size = 400
    opt.sigma = 2
    net = DreamHourglassMultiStage(
        9,
        n_stages = 2,
        internalize_spatial_softmax = False,
        deconv_decoder = False,
        full_output = True)

elif opt.network == 'mobile':
    net = ()
    output_size = 50
    opt.sigma = 0.5
    net = DopeMobileNet() 
    
elif opt.network == 'boundary':

    # if not opt.net_dope is None:
    #     net_dope = DopeNetwork()
    #     net_dope = torch.nn.DataParallel(net_dope).cuda()
    #     tmp = torch.load(opt.net_dope)
    #     net_dope.load_state_dict(tmp)

    net = BoundaryAwareNet(opt.net_dope)
    output_size = 50
    opt.sigma = 1

else:
    print(f'network {opt.network} does not exists')
    quit()


print (f"load data: {opt.data}")
print (f"load data: {opt.datatest}")
#load the dataset using the loader in utils_pose
trainingdata = None

if not opt.data == "":
    train_dataset = CleanVisiiDopeLoader(
        opt.data,
        sigma = opt.sigma,
        output_size = output_size,
        objects = opt.objects
        )
    trainingdata = torch.utils.data.DataLoader(train_dataset,
        batch_size = opt.batchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True
        )

testingdata = None

if not opt.datatest == "":
    test_dataset = CleanVisiiDopeLoader(
        opt.datatest,
        sigma = opt.sigma,
        output_size = output_size,
        objects = opt.objects
        )
    testingdata = torch.utils.data.DataLoader(test_dataset,
        batch_size = opt.testbatchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True
        )

if not trainingdata is None:
    print('training data: {} batches'.format(len(trainingdata)))
if not testingdata is None:
    print('testing data: {} batches'.format(len(testingdata)))
print('load models')

# Might want to modfify to include the cropped features from 
# yolov3

# net = DopeHourglassBlock(input_size_channels=3,output_size_channels = 9)
# net = DopeHourglass(num_stages = 2)



# net = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda()
# net = torch.nn.DataParallel(net).cuda()
net = torch.nn.parallel.DistributedDataParallel(net.cuda(),
    device_ids=[opt.local_rank],
    output_device=opt.local_rank)
# print(net)

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters,lr=opt.lr)


with open (opt.outf+'/loss_train.txt','w') as file: 
    file.write('epoch,batchid,loss\n')

with open (opt.outf+'/loss_test.txt','w') as file: 
    file.write('epoch,batchid,loss,l1,l1std\n')

print("ready to train!")

nb_update_network = 0
best_results = {"epoch":None,'passed':None,'add_mean':None,"add_std":None}

scaler = torch.cuda.amp.GradScaler() 

def _runnetwork(epoch,train_loader,train=True,syn=False):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

        config_detect = lambda: None
        config_detect.mask_edges = 1
        config_detect.mask_faces = 1
        config_detect.vertex = 1
        config_detect.treshold = 0.5
        config_detect.softmax = 1000
        config_detect.thresh_angle = 0.5
        config_detect.thresh_map = 0.01
        config_detect.sigma = 4
        config_detect.thresh_points = 0.1
        config_detect.threshold = 0.1
        
        # todo opt.model path to the 3d model

        #Create the folders for output
        # try:
        #     namefile = '/test_{}.csv'.format(epoch)
        #     with open (opt.outf+namefile,'w') as file:
        #         file.write("img, trans_gu, trans_gt, agg_error \n")
        #     os.makedirs(opt.outf+"/test/{}/".format(str(epoch).zfill(3)))
        all_data_to_save = []   

    loss_avg_to_log = {}
    loss_avg_to_log['loss'] = []
    loss_avg_to_log['loss_affinities'] = []
    loss_avg_to_log['loss_belief'] = []
    loss_avg_to_log['loss_class'] = []
    for batch_idx, targets in enumerate(train_loader):
        optimizer.zero_grad()
        logged = 0

        data = Variable(targets['img'].cuda())
        target_belief = Variable(targets['beliefs'].cuda())        
        target_affinities = Variable(targets['affinities'].cuda())
        
        # target_segmentation = Variable(targets['segmentation'].cuda())
        
        # target_affinity_map = Variable(targets['affinity_map'][:,2,:,:]).cuda()
        # print(target_belief.min(),target_belief.max())
        # print(target_affinities.min(),target_affinities.max())
        # print(target_classification.min(),target_classification.max())


        # print (f'data: {data.shape}')        
        # print (f'target_belief: {target_belief.shape}')        
        # print (f'target_affinities: {target_affinities.shape}')        
        # print (f'target_segmentation: {target_segmentation.shape}')        
        output_belief, output_aff = net(data)
        
        loss = None
        
        # print(f'len: {len(output_net)}')
        # print(f'1: {output_net[0][0].shape}')
        # print(f'2: {output_net[0][1].shape}')
        # print(f'3: {output_net[0][2].shape}')
        # len: 2
        # 1: torch.Size([4, 9, 100, 100])
        # 2: torch.Size([4, 16, 100, 100])
        # 3: torch.Size([4, 21, 100, 100])

        # raise()

        loss_belief = torch.tensor(0).float().cuda() 
        loss_affinities = torch.tensor(0).float().cuda()
        loss_class = torch.tensor(0).float().cuda()
        # loss_segmentation = torch.tensor(0).float().cuda()

        for stage in range(len(output_aff)): #output, each belief map layers. 
            # print(stage[0].shape)
            # print(target_affinity_map.shape)
            # raise()
            # loss_tmp = (( - target_affinity_map) * (stage[0]-target_affinity_map)).mean()



            loss_affinities += ((output_aff[stage] - target_affinities)*(output_aff[stage] - target_affinities)).mean()
            
            # print(output_belief[stage].shape)
            # print(target_belief.shape)

            loss_belief += ((output_belief[stage] - target_belief)*(output_belief[stage] - target_belief)).mean()

            # loss_tmp = ((stage[1] - target_affinities) * (stage[1]-target_affinities)).mean()
            # loss_affinities += loss_tmp 

            # loss_tmp = ((stage[2] - target_segmentation) * (stage[2]-target_segmentation)).mean()
            # loss_segmentation += loss_tmp

        # loss = loss_belief + loss_affinities * 0.9 + loss_segmentation * 0.00001

        # compute classification loss 
        # loss_class = ((target_classification.flatten(1) - output_classification) * (target_classification.flatten(1) - output_classification)).mean()
        # print(loss_class.item(),loss_belief.item(),loss_affinities.item() )
        loss = loss_affinities + loss_belief

        #save one output of the network and one gt
        # if False : 
        if batch_idx == 0 : 

            if train:
                post = "train"
            else:
                post = 'test'

            if opt.local_rank == 0:

                for i_output in range(1):

                    # input images
                    writer.add_image(f"{post}_input_{i_output}",
                        targets['img_original'][i_output],
                        epoch,
                        dataformats="CWH",
                        )

                    # belief maps gt
                    imgs = VisualizeBeliefMap(target_belief[i_output])
                    img,grid = save_image(imgs, "some_img.png", 
                        mean=0, std=1, nrow=3, save=False)
                    writer.add_image(f"{post}_belief_ground_truth_{i_output}",
                                grid,
                                epoch, 
                                dataformats="CWH")

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    img,grid = save_image(imgs, "some_img.png", 
                        mean=0, std=1, nrow=3, save=False)
                    writer.add_image(f"{post}_belief_guess_{i_output}",
                                grid,
                                epoch, 
                                dataformats="CWH")

        if not train:
            # TODO look into using batchsize > 1 when the input data is 
            # constant
            # comes as Batch size x nb object x 3 or 4

            # save the images then continue
            data_to_save = {}
            data_to_save['loss_test'] = float(loss.item())


            # @ray.remote
            from multiprocessing import Process


        if train:
            # optimizer.zero_grad()
            # for param in net.parameters():
                # param.grad = None

            loss.backward()
            # scaler.scale(loss).backward() 
                
            optimizer.step()
            # scaler.step(optimizer)

            # scaler.update()
            nb_update_network+=1
            
            
            # namefile = '/loss_train.txt'
            # with open (opt.outf+namefile,'a') as file:
            #     s = '{}, {},{:.15f}\n'.format(
            #         epoch,batch_idx,loss.item()) 
            #     file.write(s)
        
        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_class"].append(loss_class.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())
            
        if batch_idx % opt.loginterval == 0:
            if not opt.horovod or hvd.rank() == 0:
                if train:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

                else:
                    print('Test  Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        # break
        # if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        #     torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
        #     break

    # log the loss values
    if opt.local_rank == 0:

        if train:
            writer.add_scalar('loss/train_loss',np.mean(loss_avg_to_log["loss"]),epoch)
            writer.add_scalar('loss/train_cls',np.mean(loss_avg_to_log["loss_class"]),epoch)
            writer.add_scalar('loss/train_aff',np.mean(loss_avg_to_log["loss_affinities"]),epoch)
            writer.add_scalar('loss/train_bel',np.mean(loss_avg_to_log["loss_belief"]),epoch)
        else:
            # import pandas as pd
            # add the loss

            writer.add_scalar('loss/test_loss',np.mean(loss_avg_to_log["loss"]),epoch)
            writer.add_scalar('loss/test_cls',np.mean(loss_avg_to_log["loss_class"]),epoch)
            writer.add_scalar('loss/test_aff',np.mean(loss_avg_to_log["loss_affinities"]),epoch)
            writer.add_scalar('loss/test_bel',np.mean(loss_avg_to_log["loss_belief"]),epoch)

for epoch in range(1, opt.epochs + 1):

    if not trainingdata is None and not opt.testonly:
        _runnetwork(epoch,trainingdata)
        if opt.optimizer == 'sgd':
            scheduler.step()

    if not opt.datatest == "":
        _runnetwork(epoch,testingdata,train = False)
        if opt.data == "":
            break # lets get out of this if we are only testing
    try:
        if opt.local_rank == 0:
            if not opt.dontsave is True:
                torch.save(net.state_dict(), f'{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(2)}.pth')
            else:
                torch.save(net.state_dict(), f'{opt.outf}/net_{opt.namefile}.pth')
    except:
        pass

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break
# print(best_results)
if opt.local_rank == 0:
    torch.save(net.state_dict(), f'{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(2)}.pth')
print ("end:" , datetime.datetime.now().time())

