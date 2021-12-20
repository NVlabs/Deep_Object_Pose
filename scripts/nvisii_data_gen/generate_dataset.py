import random 
import subprocess



num_loop = 5#40  # num_loop * nb_frames images


for i in range(0,num_loop):
	to_call = [
		# "python",'single_video.py',
		"python",'single_video_pybullet.py',
		'--spp','10',
		'--nb_frames', '200',
		'--nb_objects',str(int(random.uniform(50,75))),
		'--static_camera',
		'--outf',f"dataset/{str(i).zfill(3)}",
	]
	subprocess.call(to_call)
	#subprocess.call(['mv',f'dataset/{str(i).zfill(3)}/video.mp4',f"dataset/{str(i).zfill(3)}.mp4"])
	# break
	# subprocess.Popen(["rsync",'-r',f'output/dataset/{str(i).zfill(3)}',
	# 	"/mnt/adlr/dataset_2/",";",
	# 	'rm','-rf',f'output/dataset/{str(i).zfill(3)}'])
	# subprocess.call([])
	# break