#!/usr/bin/env python3
import random 
import subprocess


# 20 000 images

for i in range(97, 100):
	to_call = [
		"python",'single_video_pybullet.py',
		'--spp','10',
		'--nb_frames', '200',
		'--nb_distractors','10',
		'--nb_objects',str(int(random.uniform(2,5))),
		'--scale', '0.015',
		'--skyboxes_folder','/home/jtremblay/code/visii_dr/dome_hdri_haven/',
		"--skip_frame",'100',
		'--objs_folder_distrators','/media/jtremblay/data_large/google_scanned/google_scanned_models/',
		'--path_single_obj','/home/jtremblay/code/visii_dr/content/models/grocery_ycb/003_cracker_box/google_16k/textured.obj',
		'--outf',f"andrew_ycb_dope/{str(i).zfill(3)}",
	]
	subprocess.call(to_call)
	# raise()