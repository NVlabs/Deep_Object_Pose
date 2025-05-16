#!/usr/bin/python3

"""
Example usage:

 python -m torch.distributed.launch --nproc_per_node=1 train.py --data ../sample_data/ --object cracker
"""


import argparse
import datetime
import os
from queue import Queue
import random
import warnings
warnings.filterwarnings("ignore")

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '../common')
from models import *
from utils import *



def _runnetwork(net, optimizer, local_rank, epoch, train_loader, writer=None):
    loss_avg_to_log = {}
    loss_avg_to_log["loss"] = []
    loss_avg_to_log["loss_affinities"] = []
    loss_avg_to_log["loss_belief"] = []
    loss_avg_to_log["loss_class"] = []
    for batch_idx, targets in enumerate(train_loader):
        optimizer.zero_grad()

        data = Variable(targets["img"].cuda())
        target_belief = Variable(targets["beliefs"].cuda())
        target_affinities = Variable(targets["affinities"].cuda())

        output_belief, output_aff = net(data)

        loss = None

        loss_belief = torch.tensor(0).float().cuda()
        loss_affinities = torch.tensor(0).float().cuda()
        loss_class = torch.tensor(0).float().cuda()

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

            if writer is not None and local_rank == 0:
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
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps_gt.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_ground_truth_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_guess_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )


        loss.backward()

        optimizer.step()

        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_class"].append(loss_class.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())

        if batch_idx % opt.loginterval == 0:
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
    if writer is not None and local_rank == 0:
        writer.add_scalar(
            "loss/train_loss", np.mean(loss_avg_to_log["loss"]), epoch
        )
        writer.add_scalar(
            "loss/train_cls", np.mean(loss_avg_to_log["loss_class"]), epoch
        )
        writer.add_scalar(
            "loss/train_aff", np.mean(loss_avg_to_log["loss_affinities"]), epoch
        )
        writer.add_scalar(
            "loss/train_bel", np.mean(loss_avg_to_log["loss_belief"]), epoch
        )


def main(opt):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.gradcheck = False
    torch.backends.cudnn.benchmark = True

    local_rank = opt.local_rank

    # Validate Arguments
    if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
        raise ValueError(
            "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
        )

    if not opt.use_s3 and opt.data is None:
        raise ValueError("--data field must be specified.")

    os.makedirs(opt.outf, exist_ok=True)

    random_seed = random.randint(1, 10000)
    if opt.manualseed is not None:
        random_seed = opt.manualseed

    # Save run parameters in a file
    with open(opt.outf + "/header.txt", "w") as file:
        file.write(str(opt) + "\n")
        file.write("seed: " + str(random_seed) + "\n")

    writer = None
    if local_rank == 0:
        writer = SummaryWriter(opt.outf + "/runs/")

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="NCCL", init_method="env://")


    # Data Augmentation
    transform = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.ToTensor()
    ])

    # Load Model
    net = DopeNetwork()
    output_size = 50
    opt.sigma = 0.5

    # Convert object names to lower-case for comparison later
    for idx in range(len(opt.object)):
        opt.object[idx] = opt.object[idx].lower()

    training_dataset = CleanVisiiDopeLoader(
        opt.data,
        sigma=opt.sigma,
        output_size=output_size,
        objects=opt.object,
        use_s3=opt.use_s3,
        buckets=opt.train_buckets,
        endpoint_url=opt.endpoint,
    )
    training_data = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
    )

    if not training_data is None:
        print("training data: {} batches".format(len(training_data)))

        print("Loading Model...")
        net = torch.nn.parallel.DistributedDataParallel(
            net.cuda(),
            device_ids=[local_rank],
            output_device=local_rank
        )

    # Load any previous checkpoint (i.e. current job is a follow-up job)
    if opt.net_path is not None:
        net.load_state_dict(torch.load(opt.net_path))

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=opt.lr)

    print("ready to train!")
    start_time = datetime.datetime.now()
    print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

    ckpt_q = None
    if opt.nb_checkpoints > 0:
        ckpt_q = Queue(maxsize=opt.nb_checkpoints)

    start_epoch = 0
    if opt.net_path is not None:
        # We started with a saved checkpoint, we start numbering checkpoints
        # after the loaded one
        try:
            start_epoch = int(os.path.splitext(os.path.basename(opt.net_path).split('_')[-1])[0]) + 1
        except:
            start_epoch = 1
        print(f"Starting at epoch {start_epoch}")

    net.train()
    for epoch in range(start_epoch, opt.epochs + 1):
        _runnetwork(net, optimizer, local_rank, epoch, training_data, writer)

        try:
            if local_rank == 0 and epoch > 0 and epoch % opt.save_every == 0:
                out_fn = f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
                torch.save(net.state_dict(), out_fn)

                # Clean up old checkpoints if we're limiting the number saved
                if ckpt_q is not None:
                    if ckpt_q.full():
                        to_del = ckpt_q.get()
                        os.remove(to_del)
                    ckpt_q.put(out_fn)

        except Exception as e:
            print(f"Encountered Exception: {e}")

    if local_rank == 0:
        torch.save(
            net.state_dict(),
            f"{opt.outf}/final_net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
        )

    print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    print("Total time taken: ", str(datetime.datetime.now() - start_time).split(".")[0])
    return


if __name__ == "__main__":
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False,
    )
    conf_parser.add_argument(
        "-c", "--config",
        help="Specify config file",
        metavar="FILE"
    )
    # Read the config but do not overwrite the args written
    args, remaining_argv = conf_parser.parse_known_args()


    parser = argparse.ArgumentParser()
    # Specify Training Data
    parser.add_argument(
        "--data",
        nargs="+",
        help="Path to training data"
    )
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="Use s3 buckets for training data"
    )
    parser.add_argument(
        "--train_buckets",
        nargs="+",
        default=[],
        help="s3 buckets containing training data. Can list multiple buckets separated by a space.",
    )
    parser.add_argument(
        "--endpoint",
        "--endpoint_url",
        type=str,
        default=None
    )

    # Specify Training Object
    parser.add_argument(
        "--object",
        nargs="+",
        required=True,
        default=[],
        help='Object to train network for. Must match "class" field in groundtruth .json file.'
        ' For best performance, only put one object of interest.',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers"
    )
    parser.add_argument(
        "--batchsize", "--batch_size",
        type=int,
        default=32,
        help="input batch size"
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        default=448,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate, default=0.0001"
    )
    parser.add_argument(
        "--net_path",
        default=None, help="path to net (to continue training)"
    )
    parser.add_argument(
        "--namefile",
        default="epoch",
        help="name to put on the file of the save weights"
    )
    parser.add_argument(
        "--manualseed",
        type=int,
        help="manual random number seed"
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        "-e",
        type=int,
        default=60,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=100
    )
    parser.add_argument(
        "--outf",
        default="output/weights",
        help="folder to output images and model checkpoints",
    )
    parser.add_argument(
        "--nb_checkpoints",
        type=int,
        default=0,
        help="Number of checkpoints (.pth files) to save. Older ones will be "
        "deleted as new ones are saved. A value of 0 means an unlimited "
        "number will be saved"
    )
    parser.add_argument(
        '--save_every',
        type=int, default=1,
        help='How often (in epochs) to save a snapshot'
    )
    parser.add_argument(
        "--sigma",
        default=4,
        help="keypoint creation sigma")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0
    )

    parser.add_argument("--save", action="store_true", help="save a batch and quit")
    opt = parser.parse_args(remaining_argv)

    main(opt)
