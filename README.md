# Human-Object Interaction Imitation

### 📖[*Paper*](https://arxiv.org/abs/2312.04393)|🖼️[*Project Page*](https://wyhuai.github.io/physhoi-page/)

This repository contains the **code** and **dataset** release for the paper: "PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction"

Our whole-body humanoid follows the **SMPL-X** kinematic tree and has a total of **51x3 DoF** actuators, with fully **dextrous hands**.

🏀 Now simulated humanoids can learn diverse basketball skills **without designing task-specific rewards!**

![image](https://github.com/wyhuai/PhysHOI_dev/assets/95485229/6013e448-05ed-4a12-9164-aa5b34896598)


## TODOs

- [ ] Add more data to the BallPlay dataset.

- [ ] Provide a physically rectified version of basic BallPlay (using PhysHOI).

- [x] Release the basic BallPlay dataset.

- [x] Release training and evaluation code. 


## Requirements 🖥️

It is suggested to perform inference with a graphical interface, which may need a local computer with a screen.

You may need an NVIDIA GPU. The inference needs at least 6G memory. The training needs at least 12G memory (with 1024 envs).

## Installation 💽

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```


## PhysHOI 🎯

### Pre-Trained Models 📁
Download the trained models from this [link](https://drive.google.com/file/d/1jPnzd6PVVpiWNA1-MTVuUgIR_GOJMcLu/view?usp=sharing), unzip the files, and put them into `physhoi/data/models/`. The directory structure should be like `physhoi/data/models/backdribble/nn/PhysHOI.pth`, `physhoi/data/models/pass/nn/PhysHOI.pth`, etc.

### Inference ⛹️‍♂️

#### Basic Evaluation ⛹️‍♂️
For toss, fingerspin, pass, walkpick, and backspin, use the following command. Please change the `[task]` correspondingly.
```
python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 16 --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/[task].pt --checkpoint physhoi/data/models/[task]/nn/PhysHOI.pth
```
For rebound, we need to give the ball an initial velocity, otherwise it will fall vertically downward:
```
python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 16 --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/rebound.pt --checkpoint physhoi/data/models/rebound/nn/PhysHOI.pth --init_vel
```
For changeleg, we provide a trained model that use 60hz control frequency and 60fps data frame rate:
```
python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 16 --cfg_env physhoi/data/cfg/physhoi_60hz.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/changeleg.pt --checkpoint physhoi/data/models/changeleg_60fps/nn/PhysHOI.pth
```
For backdribble, we provide a trained model that use 30hz control frequency and 25fps data frame rate:
```
python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 16 --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/backdribble.pt --checkpoint physhoi/data/models/backdribble/nn/PhysHOI.pth --frames_scale 1.
```

#### Other Options 💡
To view the HOI dataset, add `--play_dataset`.

To throw projectiles at the humanoid, add `--projtype Mouse`, and keep on clicking the screen with your mouse:

To change the size of the ball, add `--ball_size 1.5`, and you can change the value as you like:

To test with different data frame rates, change the value of `--frames_scale` as you like, e.g., 1.5.

To save the images, add `--save_images` to the command, and the images will be saved in `physhoi/data/images`.

To transform the images into a video, run the following command, and the video can be found in `physhoi/data/videos`.
```
python physhoi/utils/make_video.py --image_path physhoi/data/images/backdribble --fps 30
```

&nbsp;

### Training 🏋️

All tasks share the same training code and most of the Hyper-parameters. To train the model, run the following command, and you may change the `--motion_file` to different HOI data: 
```
python physhoi/run.py --task PhysHOI_BallPlay --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/toss.pt --headless
```
During the training, the latest checkpoint PhysHOI.pth will be regularly saved to output/, along with a Tensorboard log.

It takes different epochs to reach convergence depending on the difficulty and data quality. For example, it takes 10000 epochs for toss and backdribble to converge, which takes about 9 hours on an NVIDIA 4090 Ti GPU.

#### Tips for Hyper-Parameters 💡
- For fingerspin, `cg2` is suggested to be `0.01`, considering the default contact graph is not detailed enough for finger-level operations.
- For walkpick, `stateInit` is suggested to be Random, due to the data inaccuracy.
- Too large `cg2` and `cg2` may yield unnatural movements; Too small `cg2` and `cg2` may lead to fail grabs or false interaction. 

&nbsp;

### The BallPlay dataset 🏀

The basic BallPlay HOI dataset, including 8 human-basketball interaction skills, is placed in `physhoi/data/motions/BallPlay`. The frame rate is 25 FPS. The contact label denotes the contact between the ball and hands. The details of the data structure can be found in function `_load_motion` in `physhoi/env/tasks/physhoi.py`. The humanoid robot and basketball model are placed in `physhoi/data/assets/smplx/smplx_capsule.xml` and `physhoi/data/assets/mjcf/ball.urdf`, respectively. 

&nbsp;

## References
If you find this repository useful for your research, please cite the following work.
```
@article{wang2023physhoi,
  author    = {Wang, Yinhuai and Lin, Jing and Zeng, Ailing and Luo, Zhengyi and Zhang, Jian and Zhang, Lei},
  title     = {PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction},
  journal   = {arXiv preprint arXiv:2312.04393},
  year      = {2023},
}
```
The code implementation is based on ASE:
- https://github.com/nv-tlabs/ASE

The SMPL-X humanoid robot is generated using UHC:
- https://github.com/ZhengyiLuo/UniversalHumanoidControl
