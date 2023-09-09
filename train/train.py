"""
python train.py --data data/path/to/images

"""


from __future__ import print_function

import argparse
import os
import random

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

# import cv2
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import datetime

from tensorboardX import SummaryWriter


from models import *
from utils import *


import warnings

warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.gradcheck = False
torch.backends.cudnn.benchmark = True

start_time = datetime.datetime.now()
print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

conf_parser = argparse.ArgumentParser(
    description=__doc__,  # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False,
)
conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

# Specify Training Data
parser.add_argument("--data", nargs="+", help="Path to training data")
parser.add_argument(
    "--use_s3", action="store_true", help="Use s3 buckets for training data"
)
parser.add_argument(
    "--train_buckets",
    nargs="+",
    default=[],
    help="s3 buckets containing training data. Can list multiple buckets separated by a space.",
)
parser.add_argument("--endpoint", "--endpoint_url", type=str, default=None)

# Specify Training Object
parser.add_argument(
    "--object",
    nargs="+",
    required=True,
    default=[],
    help='Object to train network for. Must match "class" field in groundtruth .json file. For best performance, only put one object of interest.',
)
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=8
)
parser.add_argument(
    "--batchsize", "--batch_size", type=int, default=32, help="input batch size"
)
parser.add_argument(
    "--imagesize",
    type=int,
    default=512,
    help="the height / width of the input image to network",
)
parser.add_argument(
    "--lr", type=float, default=0.0001, help="learning rate, default=0.0001"
)
parser.add_argument(
    "--net_path", default=None, help="path to net (to continue training)"
)
parser.add_argument(
    "--namefile", default="epoch", help="name to put on the file of the save weightss"
)
parser.add_argument("--manualseed", type=int, help="manual seed")
parser.add_argument(
    "--epochs",
    "--epoch",
    "-e",
    type=int,
    default=60,
    help="Number of epochs to train for",
)
parser.add_argument("--loginterval", type=int, default=100)
parser.add_argument("--gpuids", nargs="+", type=int, default=[0], help="GPUs to use")
parser.add_argument(
    "--exts",
    nargs="+",
    type=str,
    default=["png"],
    help="Extensions for images to use. Can have multiple entries seperated by space. e.g. png jpg",
)
parser.add_argument(
    "--outf",
    default="output/weights",
    help="folder to output images and model checkpoints",
)
parser.add_argument("--sigma", default=4, help="keypoint creation sigma")
parser.add_argument("--local_rank", type=int)

parser.add_argument("--save", action="store_true", help="save a batch and quit")
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="Use pretrained weights. Must also specify --net_path.",
)
parser.add_argument("--nbupdates", default=None, help="nb max update to network")

# Read the config but do not overwrite the args written
args, remaining_argv = conf_parser.parse_known_args()
defaults = {"option": "default"}

if args.config:
    config = configparser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))


parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

local_rank = opt.local_rank

# Validate Arguments
if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
    raise ValueError(
        "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
    )

if not opt.use_s3 and opt.data is None:
    raise ValueError("--data field must be specified.")

os.makedirs(opt.outf, exist_ok=True)


with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt) + "\n")

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt))
    file.write("seed: " + str(opt.manualseed) + "\n")


if local_rank == 0:
    writer = SummaryWriter(opt.outf + "/runs/")

random.seed(opt.manualseed)


torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="NCCL", init_method="env://")

torch.manual_seed(opt.manualseed)

torch.cuda.manual_seed_all(opt.manualseed)

# Data Augmentation
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59, 0.25]
    transform = transforms.Compose(
        [
            AddRandomContrast(0.2),
            AddRandomBrightness(0.2),
            transforms.Resize(opt.imagesize),
        ]
    )
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose(
        [transforms.Resize(opt.imagesize), transforms.ToTensor()]
    )

# Load Model
net = DopeNetwork()
output_size = 50
opt.sigma = 0.5


train_dataset = CleanVisiiDopeLoader(
    opt.data,
    sigma=opt.sigma,
    output_size=output_size,
    objects=opt.object,
    use_s3=opt.use_s3,
    buckets=opt.train_buckets,
    endpoint_url=opt.endpoint,
)
trainingdata = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True,
)


if not trainingdata is None:
    print("training data: {} batches".format(len(trainingdata)))

print("Loading Model...")

net = torch.nn.parallel.DistributedDataParallel(
    net.cuda(), device_ids=[local_rank], output_device=local_rank
)

if opt.pretrained:
    if opt.net_path is not None:
        net.load_state_dict(torch.load(opt.net))
    else:
        print("Error: Did not specify path to pretrained weights.")
        quit()

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters, lr=opt.lr)


print("ready to train!")

nb_update_network = 0
best_results = {"epoch": None, "passed": None, "add_mean": None, "add_std": None}

scaler = torch.cuda.amp.GradScaler()


def _runnetwork(epoch, train_loader, syn=False):
    global nb_update_network
    # net
    net.train()

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


for epoch in range(1, opt.epochs + 1):

    _runnetwork(epoch, trainingdata)

    try:
        if local_rank == 0:

            torch.save(
                net.state_dict(),
                f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(2)}.pth",
            )
    except Exception as e:
        print(f"Encountered Exception: {e}")

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break

if local_rank == 0:
    torch.save(
        net.state_dict(), f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(2)}.pth"
    )
else:
    torch.save(
        net.state_dict(),
        f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(2)}_rank_{local_rank}.pth",
    )

print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
print(
    "Total time taken: ",
    str(datetime.datetime.now() - start_time).split(".")[0],
)
