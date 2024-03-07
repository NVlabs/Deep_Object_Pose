#!/usr/bin/env python3

import argparse
import multiprocessing
import os
from queue import Queue
import subprocess
import sys


parser = argparse.ArgumentParser()
## Parameters for this script
parser.add_argument(
    '--nb_runs',
    default=1,
    type=int,
    help='Number of times the datagen script is run. Each time it is run, a new set of '
    'distractors is selected.'
)
parser.add_argument(
    '--nb_workers',
    default=0,
    type=int,
    help='Number of parallel blenderproc workers to run.  The default of 0 will create '
    'one worker for every CPU core'
)

## Parameters to pass on to the blenderproc code
parser.add_argument(
    '--width',
    default=500,
    type=int,
    help = 'image output width'
)
parser.add_argument(
    '--height',
    default=500,
    type=int,
    help = 'image output height'
)
# TODO: change for an array
parser.add_argument(
    '--distractors_folder',
    default='google_scanned_models/',
    help = "folder containing distraction objects"
)
parser.add_argument(
    '--objs_folder',
    default='models/',
    help = "folder containing training objects, if using multiple"
)
parser.add_argument(
    '--path_single_obj',
    default=None,
    help='If you have a single obj file, path to the obj directly.'
)
parser.add_argument(
    '--object_class',
    default=None,
    help="The class name of the object(s). If none is provided, the name of the directory "
    "containing the model(s) will be used."
)
parser.add_argument(
    '--scale',
    default=1,
    type=float,
    help='Scaling to apply to the target object(s) to put in units of centimeters; e.g if '
         'the object scale is meters -> scale=0.01; if it is in cm -> scale=1.0'
)
parser.add_argument(
    '--backgrounds_folder',
    default=None,
    help = "folder containing background images"
)
parser.add_argument(
    '--nb_objects',
    default=1,
    type = int,
    help = "how many objects"
)
parser.add_argument(
    '--nb_distractors',
    default=1,
    help = "how many distractor objects"
)
parser.add_argument(
    '--nb_frames',
    type = int,
    default=2000,
    help = "how many total frames to generate"
)
parser.add_argument(
    '--outf',
    default='output_example/',
    help = "output filename inside output/"
)
parser.add_argument(
    '--focal-length',
    default=None,
    type=float,
    help = "focal length of the camera"
)
parser.add_argument(
    '--debug',
    action='store_true',
    default=False,
    help="Render the cuboid corners as small spheres. Only for debugging purposes;"
    "do not use for training!"
)

opt = parser.parse_args()

num_workers = min(opt.nb_workers, multiprocessing.cpu_count())
if num_workers == 0:
    num_workers = multiprocessing.cpu_count()

amount_of_runs = opt.nb_runs

# set the folder in which the generation script is located
rerun_folder = os.path.abspath(os.path.dirname(__file__))

# pass the arguments given to this command to the subprocess
used_arguments = sys.argv[1:]
output_location = opt.outf

print("nb workers", num_workers)
Q = Queue(maxsize = num_workers)
for run_id in range(amount_of_runs):
    if Q.full():
        proc = Q.get()
        proc.wait()

    # execute one BlenderProc run
    cmd = ["blenderproc", "run", os.path.join(rerun_folder, "generate_training_data.py")]
    cmd.extend(used_arguments)
    cmd.extend(['--run_id', str(run_id)])
    p = subprocess.Popen(" ".join(cmd), shell=True)
    Q.put(p)
