## Installation
1. install dependent library project: [pytorch utilities](https://github.com/UM-ARM-Lab/arm_pytorch_utilities), 
[pytorch mppi](https://github.com/LemonPi/pytorch_mppi)
2. `pip3 install -e .`

## Usage
1. generate urdf files `python3 build_models.py`
    
2. open tensorboard server (see below)

3. run scripts

## Scripts
Scripts are located in `scripts`. Those with `_main` suffix are 
the main simulation scripts that have methods for collecting data,
training model, and testing trained models on novel environments.
    
## Tensorboard logging
`pip3 install tensorboardX`

(Also install `tensorboard`) To start the tensorboard server, 

`tensorboard --logdir scripts/runs`

## Remove empty runs
`python clean_empty_runs.py`