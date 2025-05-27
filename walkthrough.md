# DOPE Pipeline Walkthrough

Here we provide a detailed example of using the data generation, training, and inference tools provided in this repo. This example is
given not only to demonstrate the various tools, but also to show what
kind of results you can expect.

We have uploaded our final PTH file as well as some sample data 
to [Google Drive](https://drive.google.com/drive/folders/1zq4yJUj8lTn56bWdOMnkCr1Wmj0dq-GL).


## Preparation
We assume you have installed the project dependencies and are in an environment where you have access to GPUs.


Follow [the instructions](data_generation/readme.md) for downloading environment maps and distractors.

Download or create a textured model of your object of interest.  For this walkthrough, we will use the [Ketchup](https://drive.google.com/drive/folders/1ICXdhNhahDiUrjh_r5aBMPYbb2aWMCJF?usp=drive_link) model from the [HOPE 3D Model Set](https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft/).


For the sake of the example commands, we will assume the following folder
structure:
`~/data/dome_hdri_haven/` contains the HDR environment maps; 
`~/data/google_scanned_models/` contains the distractor objects;
`~/data/models/` contains our "hero" models in subdirectories; e.g. `~/data/models/Ketchup`.

## Data Generation
We will use the BlenderProc data generation utilities.  In the `data_generation/blenderproc_data_gen` directory, run the following command:

```
./run_blenderproc_datagen.py --nb_runs 10 --nb_frames 50000 --path_single_obj ~/data/models/Ketchup/google_16k/textured.obj --nb_objects 1 --distractors_folder ~/data/google_scanned_models/ --nb_distractors 10 --backgrounds_folder ~/data/dome_hdri_haven/ --outf ~/data/KetchupData
```

This will create ten subdirectories under the `~/data/KetchupData` directory, each containing 5000 images (`nb_images` divided by `nb_runs`).  For Blender efficiency reasons, the distractors are only changed from run to run. That is, we will have 10 different selections of distractors in our 50,000 images.  If you want 
a greater selection of distractors, increase the `nb_runs` parameter.

## Training

Assuming your machine has *N* GPUs, run the following command:

```
python -m torch.distributed.launch --nproc_per_node=N ./train.py --data  ~/data/KetchupData --object Ketchup --epochs 2000 --save_every 100
```

This command will train DOPE for 2000 epochs, saving a checkpoint every 100 epochs.

## Inference
When training is finished, you will have several saved checkpoints including the final one: `final_net_epoch_2000.pth`.  We will use this checkpoint for inference.


Generate a small number of new images in the same distribution as your training images. We will use these for inference testing and evaluation.
```
./run_blenderproc_datagen.py --nb_runs 2 --nb_frames 20 --path_single_obj ~/data/models/Ketchup/google_16k/textured.obj --nb_objects 1 --distractors_folder ~/data/google_scanned_models/ --nb_distractors 10 --backgrounds_folder ~/data/dome_hdri_haven/ --outf ~/data/KetchupTest
```
For convenience, we have uploaded 20 test images and JSON files to the  [Google Drive](https://drive.google.com/drive/folders/1zq4yJUj8lTn56bWdOMnkCr1Wmj0dq-GL)
 location mentioned above.


Inside the `inference` directory, run the following command:
```
python ./inference.py --camera ../config/blenderproc_camera_info_example.yaml --object Ketchup --parallel --weights final_net_epoch_2000.pth --data ~/data/KetchupTest/
```

The inference output will be in the `output` directory. Using our provided `final_net_epoch_2000.pth` and our provided test images, DOPE finds the object of interest in 13 out of 20 images.

