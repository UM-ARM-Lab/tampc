## Installation
`pip3 install -e .`

## Usage
1. generate urdf files `python3 build_models.py`
    
2. open tensorboard server (see below)

3. run scripts

## Scripts
Scripts are located in `simulation`.
    
## Tensorboard logging
`pip3 install tensorboardX`

(Also install `tensorboard`) To start the tensorboard server, 

`tensorboard --logdir simulation/runs`