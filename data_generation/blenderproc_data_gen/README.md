Blenderproc is made to create a single scene and render many frames. Adding/removing objects (such as varying the number of distractors) will cause memory bloat and poor performance.  Instead, we use a batching script (`run_blenderproc_datagen.py`) to run blenderproc several times.


## Usage example:

Run the blenderproc script 50 times, each time generating 1000 frames:
```

./run_blenderproc_datagen.py --nb_runs 5 --objs_folder ../models/ --nb_objects 5 --distractors_folder ../google_scanned_models/ --nb_distractors 10 --backgrounds_folder ~/data/ImageNet2012/val --nb_frames 10 --outf ~/data/DOPE/mine/Ketchup_Nov17/


```

All parameters can be shown by running `python ./run_blenderproc_datagen.py --help`


