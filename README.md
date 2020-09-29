## Requirements
- python 3.6+
- pytorch 1.5+

## Installation
1. install dependent library project: [pytorch utilities](https://github.com/UM-ARM-Lab/arm_pytorch_utilities), 
[pytorch mppi](https://github.com/LemonPi/pytorch_mppi)
2. `pip3 install -e .`

## Installation with ROS
```
conda create --name <env_name> --channel conda-forge
   ros-core \
   ros-actionlib \
   ros-dynamic-reconfigure
   python=3.7.5 
```
Use the environment created with ROS if you need real robot experiments.
This could work well with a system-level ROS install. Either through the conda ROS
or the system ROS, `catkin_make` the required messages from the `tampc_or_msgs` package
and also install `tampc_or`. Also be sure to add the `devel` libraries to 
Project Structure (for PyCharm) so the IDE knows where the paths are.
Lastly, when running, add an environment variable `ROS_MASTER_URI` to point
to the right ROS master.

## Usage
1. (optional and requires ROS) generate urdf files `python3 build_models.py`
    
2. (only for training models) open tensorboard server (see below)

3. run scripts

### Scripts
Scripts are located in `scripts`. Those with `_main` suffix are 
the main simulation scripts that have methods for collecting data,
training model, and testing trained models on novel environments.
    
### Tensorboard logging
`pip3 install tensorboardX`

(Also install `tensorboard`) To start the tensorboard server, 

`tensorboard --logdir scripts/runs`

### Remove empty runs
`python clean_empty_runs.py`

