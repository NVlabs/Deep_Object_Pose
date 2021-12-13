# Description

This uses NViSII to generate synthetic data for training DOPE. You will need a few things in order to generate the data. 
You need NVIDIA drivers 450 or above, and we highly recommend a GPU with RTX as ray tracing can be costly on a non RTX gpu. 

# Installation
```
pip install -r requirements.txt
```

## HDRI maps
You need to download HDRI maps to use to illuminate the scene. You can find some freely on [polyhaven](https://polyhaven.com/hdris). 
For testing purposes, you can download a single one here: 
```
wget https://www.dropbox.com/s/na3vo8rca7feoiq/teatro_massimo_2k.hdr
mv teatro_massimo_2k.hdr dome_hdri_haven/
```


## Distractors

If you want to run the script as is, it is expecting some object to be used as distractor, it is currently using the [google scanned dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects). You can download it automatically with the following: 

```
python download_google_scanned_objects.py
```

If you do not want to use the distractors, use the following argument when running the script, `--nb_distractors 0`.

# Running the script




# Adding your own 3d models 



