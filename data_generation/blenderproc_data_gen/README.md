# Synthetic Data Generation with Blenderproc


[Blenderproc](https://github.com/DLR-RM/BlenderProc) is intended to create a single scene and render multiple frames of it. Adding and removing objects (such as varying the number of distractors) will cause memory bloat and poor performance.  To avoid this issue, we use a batching script (`run_blenderproc_datagen.py`) to run a standalone blenderproc script several times.


## Usage example:

Run the blenderproc script 5 times, each time generating 1000 frames. Each frame will have five copies of the object and ten randomly chosen distractor objects:
```

./run_blenderproc_datagen.py --nb_runs 5 --nb_frames 1000 --objs_folder ../models/ --nb_objects 5 --distractors_folder ../google_scanned_models/ --nb_distractors 10 --backgrounds_folder ~/data/ImageNet2012/val --nb_frames 10 --outf ~/data/DOPE/Ketchup/
```

All parameters can be shown by running `python ./run_blenderproc_datagen.py --help`

Note that, as a blenderproc script, `generate_training_data.py` cannot be invoked with Python. It must be run via the `blenderproc` launch script.  