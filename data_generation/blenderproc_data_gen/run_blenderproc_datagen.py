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


opt, unknown = parser.parse_known_args()

num_workers = min(opt.nb_workers, multiprocessing.cpu_count())
if num_workers == 0:
    num_workers = multiprocessing.cpu_count()

amount_of_runs = opt.nb_runs

# set the folder in which the generation script is located
rerun_folder = os.path.abspath(os.path.dirname(__file__))

Q = Queue(maxsize = num_workers)
for run_id in range(amount_of_runs):
    if Q.full():
        proc = Q.get()
        proc.wait()

    # execute one BlenderProc run
    cmd = ["blenderproc", "run", os.path.join(rerun_folder, "generate_training_data.py")]
    cmd.extend(unknown)
    cmd.extend(['--run_id', str(run_id)])
    p = subprocess.Popen(" ".join(cmd), shell=True)
    Q.put(p)
