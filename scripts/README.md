# Steps to start training

Use [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) to create your synthetic training dataset.
## Prerequisites for NDDS:
- 3d model of your object. (_You can use blender for example_)
- Export your model in FBX format (_recommended_) to UE4.

## Steps:
* Create a domain randomization with scene with your object being exported.
* Generate around 20k images (*should be enough*).
* Use *train.py* script. It is native to NDDS exported data. Train for
about 30 epochs.
```python
python train.py --data path/to/FAT --object soup --outf soup --gpuids 0 1 2 3 4 5 6 7 
```
* Deploy the trained weights to DOPE ROS adding weights and object dimensions.

## Useful videos (and inspiration):
* [DOPE+NDDS for Cautery Tracking -- Simple Training Data](https://www.youtube.com/watch?v=g1adPMSmrXY)
* [Running DOPE with Zed in Extreme Lighting](https://www.youtube.com/watch?v=rf-Hnc4QBsk)
* [DOPE in extremely bright light](https://www.youtube.com/watch?v=hMfBv_JHpnM)

## FAQ
* How to train for transparent objects?(*Unclosed issue*)
    - https://github.com/NVlabs/Deep_Object_Pose/issues/65.

* How to train for symmetrical objects?
    - Refer to Sec. 3.3 of the paper "BB8: A Scalable, Accurate, Robust to Partial Occlusion Method for Predicting the 3D Poses of Challenging Objects without Using Depth."
    - https://github.com/NVlabs/Deep_Object_Pose/issues/37.

* How to randomly simulate real scenes and random poses in UE4?
    - https://github.com/NVlabs/Deep_Object_Pose/issues/45#issuecomment-502249962
    - https://github.com/NVlabs/Deep_Object_Pose/issues/45#issuecomment-495479767

* How to avoid object overlaps with multiple objects in NDDS?
    - https://github.com/NVlabs/Deep_Object_Pose/issues/45#issuecomment-505363501.

* How to move the camera in UE4?
    - https://github.com/NVlabs/Deep_Object_Pose/issues/8#issuecomment-529226666
