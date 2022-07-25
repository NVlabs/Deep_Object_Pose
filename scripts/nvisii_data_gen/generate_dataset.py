#!/usr/bin/env python3
import random 
import subprocess


# 20 000 images

for i in range(0, 100):
	to_call = [
		"python",'single_video_pybullet.py',
		'--spp','10',
		'--nb_frames', '200',
		'--nb_objects',str(int(random.uniform(50,75))),
		'--scale', '0.01',
		'--outf',f"dataset/{str(i).zfill(3)}",
	]
	subprocess.call(to_call)