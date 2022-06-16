# ADD metrics computation

This folder contains scripts to generate ADD metrics and/or graphs similar to the original DOPE paper. The script provided is to work closely with the *HOPE dataset*. 
If you are interested in making it more flexible please send a PR :D. 

## Requirements
```
pip install nvisii gdown
git clone git@github.com:swtyree/hope-dataset.git 
cd hope-dataset
python setup.py
```

This process downloads the hope-dataset and the 3d models associated with it. 


# TODO 
- Make a `requirement.txt` file. 
- Make the folder structure not as tight
- Possibly subsamble vertices so computation is faster