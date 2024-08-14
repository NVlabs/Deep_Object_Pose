#! /usr/bin/env python3

# To run on a single node:
#  torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS ./train_ddp.py --epochs 20

import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import argparse
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import random
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '../common')
from models import *
from utils import *



def init_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
      args.world_size = int(os.environ['WORLD_SIZE'])
      args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.gpu = 0
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)

    os.environ['MASTER_ADDR']= '127.0.0.1'
    os.environ['MASTER_PORT']= '29500'
    dist.init_process_group(backend='nccl', world_size=args.world_size)
    dist.barrier()

def _runnetwork(gpu_id, local_rank, epoch, nb_update_network, net, train_loader, optimizer, writer):
    net.train()

    loss_avg_to_log = {}
    loss_avg_to_log["loss"] = []
    loss_avg_to_log["loss_affinities"] = []
    loss_avg_to_log["loss_belief"] = []
    loss_avg_to_log["loss_class"] = []
    for batch_idx, targets in enumerate(train_loader):
        optimizer.zero_grad()

        data = Variable(targets["img"].to(gpu_id))
        target_belief = Variable(targets["beliefs"].to(gpu_id))
        target_affinities = Variable(targets["affinities"].to(gpu_id))

        output_belief, output_aff = net(data)

        loss = None

        loss_belief = torch.tensor(0).float().to(gpu_id)
        loss_affinities = torch.tensor(0).float().to(gpu_id)
        loss_class = torch.tensor(0).float().to(gpu_id)

        for stage in range(len(output_aff)):  # output, each belief map layers.
            loss_affinities += (
                (output_aff[stage] - target_affinities)
                * (output_aff[stage] - target_affinities)
            ).mean()

            loss_belief += (
                (output_belief[stage] - target_belief)
                * (output_belief[stage] - target_belief)
            ).mean()

        loss = loss_affinities + loss_belief

        if batch_idx == 0:
            post = "train"

            if local_rank == 0:

                for i_output in range(1):

                    # input images
                    writer.add_image(
                        f"{post}_input_{i_output}",
                        targets["img_original"][i_output],
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps gt
                    imgs = VisualizeBeliefMap(target_belief[i_output])
                    img, grid = save_image(
                        imgs, "some_img.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_ground_truth_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    img, grid = save_image(
                        imgs, "some_img.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_guess_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

        loss.backward()

        optimizer.step()

        nb_update_network += 1

        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_class"].append(loss_class.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())

        if batch_idx % args.loginterval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.15f} \tLocal Rank: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    local_rank,
                )
            )

    # log the loss values
    if local_rank == 0:

        writer.add_scalar("loss/train_loss", np.mean(loss_avg_to_log["loss"]), epoch)
        writer.add_scalar(
            "loss/train_cls", np.mean(loss_avg_to_log["loss_class"]), epoch
        )
        writer.add_scalar(
            "loss/train_aff", np.mean(loss_avg_to_log["loss_affinities"]), epoch
        )
        writer.add_scalar(
            "loss/train_bel", np.mean(loss_avg_to_log["loss_belief"]), epoch
        )


def main(args):
    init_distributed(args)

    random.seed(args.manualseed)
    torch.manual_seed(args.manualseed)
    torch.cuda.manual_seed_all(args.manualseed)

    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank == 0:
        writer = SummaryWriter(args.outf + "/runs/")
    else:
        writer = None

    # Data Augmentation
    if not args.save:
        contrast = 0.2
        brightness = 0.2
        noise = 0.1
        normal_imgs = [0.59, 0.25]
        transform = transforms.Compose(
            [
                AddRandomContrast(0.2),
                AddRandomBrightness(0.2),
                transforms.Resize(args.imagesize),
            ]
        )
    else:
        contrast = 0.00001
        brightness = 0.00001
        noise = 0.00001
        normal_imgs = None
        transform = transforms.Compose(
            [transforms.Resize(args.imagesize), transforms.ToTensor()]
        )

    output_size = 50

    train_dataset = CleanVisiiDopeLoader(
        args.data,
        sigma=args.sigma,
        output_size=output_size,
        objects=args.object,
        use_s3=args.use_s3,
        buckets=args.train_buckets,
        endpoint_url=args.endpoint,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=1,#args.workers,
        pin_memory=True,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)


    # Load Model
    net = DopeNetwork().to(args.gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])


    start_epoch = 1
    if args.pretrained:
        if args.net_path is not None:
            start_epoch = int(os.path.splitext(os.path.basename(args.net_path).split('_')[2])[0]) + 1
            net.load_state_dict(torch.load(args.net_path))
        else:
            print("Error: Did not specify path to pretrained weights.")
            quit()

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    nb_update_network = 0

    for epoch in range(start_epoch, args.epochs+1):  # loop over the dataset multiple times
        train_sampler.set_epoch(epoch)
        _runnetwork(args.gpu, local_rank, epoch, nb_update_network, net, train_loader, optimizer,
                    writer)

        try:
            if local_rank == 0:
                torch.save(
                    net.state_dict(),
                    f"{args.outf}/net_{args.namefile}_{str(epoch).zfill(2)}.pth",
                )
        except Exception as e:
            print(f"Encountered Exception: {e}")

        if not args.nbupdates is None and nb_update_network > int(args.nbupdates):
            break

    if local_rank == 0:
        PATH = './final_net.pth'
        torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    # Read the config but do not overwrite the args written
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
    add_help=False,
    )
    conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {"option": "default"}

    # Parse non-torchrun commandline options
    parser = argparse.ArgumentParser()

    # Specify Training Data
    parser.add_argument("--data", nargs="+",
                        help="Path to training data")
    parser.add_argument("--use_s3", action="store_true",
                        help="Use s3 buckets for training data")
    parser.add_argument("--train_buckets", nargs="+", default=[],
                        help="s3 buckets containing training data. Can list multiple buckets "
                        "separated by a space.")
    parser.add_argument("--endpoint", "--endpoint_url", type=str, default=None)

    # Specify Training Object
    parser.add_argument("--object", nargs="+", required=True, default=[],
                        help='Object to train network for. Must match "class" field in groundtruth'
                        ' .json file. For best performance, only put one object of interest.')

    parser.add_argument("--world_size", help="Total number of nodes")
    parser.add_argument("--batchsize", "--batch_size", type=int, default=32,
                        help="input batch size")
    parser.add_argument("--imagesize", type=int, default=512,
                        help="the height / width of the input image to network")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate, default=0.0001")
    parser.add_argument("--net_path", default=None,
                        help="path to net (to continue training)")
    parser.add_argument("--namefile", default="epoch",
                        help="name to put on the file of the save weightss")
    parser.add_argument("--manualseed", type=int,
                        help="manual seed")
    parser.add_argument("--epochs", "--epoch", "-e", type=int, default=60,
                        help="Number of epochs to train for")

    parser.add_argument("--loginterval", type=int, default=100)
    parser.add_argument("--exts", nargs="+", type=str, default=["png"],
                        help="Extensions for images to use. Can have multiple entries seperated "
                        "by space. e.g. png jpg")
    parser.add_argument("--outf", default="output/weights",
                        help="folder to output images and model checkpoints")

    parser.add_argument("--sigma", default=0.5,
                        help="keypoint creation sigma")

    parser.add_argument("--save", action="store_true",
                        help="save a batch and quit")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained weights. Must also specify --net_path.")

    parser.add_argument("--nbupdates", default=None,
                        help="nb max update to network")


    if args.config:
        config = configparser.SafeConfigParser()
        config.read([args.config])
        defaults.update(dict(config.items("defaults")))

    parser.set_defaults(**defaults)
    parser.add_argument("--option")
    args = parser.parse_args(remaining_argv)

    # Validate Arguments
    if args.use_s3 and (args.train_buckets is None or args.endpoint is None):
        raise ValueError(
            "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
        )

    if not args.use_s3 and args.data is None:
        raise ValueError("--data field must be specified.")

    if args.manualseed is None:
        args.manualseed = random.randint(1, 10000)

    # Make output directory
    os.makedirs(args.outf, exist_ok=True)

    # write options to the 'header.txt' file so we can recreate this training
    with open(args.outf + "/header.txt", "w") as file:
        file.write(str(args) + "\n")
        file.write("seed: " + str(args.manualseed) + "\n")


    main(args)
