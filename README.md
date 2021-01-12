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


# Reproducing Paper Results
Use the `update_post_ral` branch of this repository (although master should work also).
The codebase is entirely in python, and it is recommended to create a new [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) environment and set up the dependencies with the following commands. For the 3rd line, replace 9.2 with your CUDA version from `nvcc --version`.
```
conda create --name tampc python=3.7.5
conda activate tampc
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
pip install pybullet
pip install gpytorch==1.1.1
pip install tensorboardX
pip install gym
```

At the time of writing, our `conda list` looks like (`tensorflow` is optional and only needed if you want the web client view of tensorboard)
```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_tflow_select             2.3.0                       mkl  
absl-py                   0.8.1                    py37_0  
arm-pytorch-utilities     0.2.0                     dev_0    <develop>
astor                     0.8.0                    py37_0  
baselines                 0.1.6                     dev_0    <develop>
blas                      1.0                         mkl  
c-ares                    1.15.0            h7b6447c_1001  
ca-certificates           2020.6.24                     0  
certifi                   2020.6.20                py37_0  
cffi                      1.13.2           py37h2e261b9_0  
click                     7.1.2                    pypi_0    pypi
cloudpickle               1.2.2                    pypi_0    pypi
cudatoolkit               9.2                           0  
dill                      0.3.1.1                  pypi_0    pypi
fasteners                 0.15                     pypi_0    pypi
freetype                  2.9.1                h8a8886c_1  
future                    0.18.2                   pypi_0    pypi
gast                      0.2.2                    py37_0  
glfw                      1.10.0                   pypi_0    pypi
google-pasta              0.1.8                      py_0  
gpytorch                  1.1.1                    pypi_0    pypi
grpcio                    1.16.1           py37hf8bcb03_1  
gym                       0.15.4                   pypi_0    pypi
h5py                      2.9.0            py37h7918eee_0  
hdf5                      1.10.4               hb1b8bf9_0  
imageio                   2.6.1                    pypi_0    pypi
imageio-ffmpeg            0.3.0                    pypi_0    pypi
importlib-metadata        1.6.0                    pypi_0    pypi
intel-openmp              2019.4                      243  
jpeg                      9b                   h024ee3a_2  
keras-applications        1.0.8                      py_0  
keras-preprocessing       1.1.0                      py_1  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libpng                    1.6.37               hbc83047_0  
libprotobuf               3.11.2               hd408876_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.1.0                h2733197_0  
markdown                  3.1.1                    py37_0  
meta-contact              0.0.0                     dev_0    <develop>
mkl                       2019.4                      243  
mkl-service               2.3.0            py37he904b0f_0  
mkl_fft                   1.0.15           py37ha843d7b_0  
mkl_random                1.1.0            py37hd6b4f25_0  
mock                      3.0.5                    py37_0  
monotonic                 1.5                      pypi_0    pypi
mujoco-py                 2.0.2.9                  pypi_0    pypi
ncurses                   6.1                  he6710b0_1  
ninja                     1.9.0            py37hfd86e86_0  
numpy                     1.17.4           py37hc1035e2_0  
numpy-base                1.17.4           py37hde5b4d6_0  
olefile                   0.46                     py37_0  
opencv-python             4.1.2.30                 pypi_0    pypi
openexr                   1.3.2                    pypi_0    pypi
openssl                   1.1.1g               h7b6447c_0  
opt_einsum                3.1.0                      py_0  
packaging                 20.4                     pypi_0    pypi
patchelf                  0.10                 he6710b0_0  
pillow                    7.0.0            py37hb39fc2d_0  
pip                       20.1.1                   pypi_0    pypi
pluggy                    0.13.1                   pypi_0    pypi
protobuf                  3.11.2           py37he6710b0_0  
pybullet                  2.7.9                    pypi_0    pypi
pycocotools               2.0                      pypi_0    pypi
pycparser                 2.19                     py37_0  
pyglet                    1.3.2                    pypi_0    pypi
pytest                    5.4.2                    pypi_0    pypi
python                    3.7.5                h0371630_0  
pytorch                   1.5.1           py3.7_cuda9.2.148_cudnn7.6.3_0    pytorch
pytorch-mppi              0.1.0                     dev_0    <develop>
readline                  7.0                  h7b6447c_5  
scipy                     1.3.2            py37h7c811a0_0  
setuptools                44.0.0                   py37_0  
six                       1.13.0                   py37_0  
sqlite                    3.30.1               h7b6447c_0  
tensorboard               2.0.0              pyhb38c66f_1  
tensorflow                2.0.0           mkl_py37h66b46cc_0  
tensorflow-base           2.0.0           mkl_py37h9204916_0  
tensorflow-estimator      2.0.0              pyh2649769_0  
termcolor                 1.1.0                    py37_1  
tk                        8.6.8                hbc83047_0  
torchvision               0.6.1                 py37_cu92    pytorch
werkzeug                  0.16.0                     py_0  
wheel                     0.33.6                   py37_0  
wrapt                     1.11.2           py37h7b6447c_0  
xz                        5.2.4                h14c3975_4  
zipp                      3.1.0                    pypi_0    pypi
zlib                      1.2.11               h7b6447c_3  
zstd                      1.3.7                h0b5b093_0  
```

## Local packages
Clone and install (`cd <dir>` then `pip install -e .`) these repositories first: [arm_pytorch_utilities](https://github.com/UM-ARM-Lab/arm_pytorch_utilities) and [pytorch_mppi](https://github.com/LemonPi/pytorch_mppi). Then install this repository with `pip install -e .`

First change working directory to `tampc/scripts`.
Most steps can be applied in the same way to planar pushing (`push_main.py`) and peg-in-hole (`peg_main.py`) environments.  
Differences will be pointed out. When running the scripts, logs of the messages can be found under `tampc/logs`.

Note that different versions of pytorch and gpytorch might yield slightly different results. The models we used are included in `tampc/checkpoints`.  
If you use these you can skip the representation learning and dynamics fine-tuning steps which can take a long time.  
For the block tasks, pass in `--rep_name saved` to use the saved learned representation and for peg tasks pass in `--rep_name saved_peg`.  
A copy of the saved dynamics model is present with the name (copy) after it in case you override it with fine tuning.  
The estimated time for some steps is listed at the end of each line for you to consider if you wish to skip some steps.

Whether you are running with CUDA or not will also affect exact reproducibility. We recommend reviewers run this on a CUDA enabled machine (although everything still works out of the box without CUDA, just that the results will be slightly different).

1. (optional) collect nominal data  
	```
	python push_main.py collect --seed 4
	```  
	you can visualize the collection process by passing in `--gui` flag. Should take about 1 hour. The data is included as part of the repository, and this step is mainly for you to validate the collection process.
2. (optional) learn dynamics representations and compare against feedforward baseline (reproduce figure 4)
	```
	python push_main.py learn_representation --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --batch 2048 --rep_name eval
	python push_main.py learn_representation --seed 0 1 2 3 4 5 6 7 8 9 --representation feedforward_baseline --batch 500 --rep_name eval
    python plot_run_result.py 
	```  
	learning across the 10 seeds is only necessary for reproducing figure 4. Should take 1-2 hours per seed. If you only care to run the tasks, you can use the saved model with `--rep_name saved` for the block tasks and `--rep_name saved_peg` for the peg tasks. Alternatively, just learn the seed 1 for pushing and seed 0 for peg (default seeds). Should take about 1 hour per seed.
3. (optional) fine tune dynamics  
	```
	python push_main.py fine_tune_dynamics --representation learned_rex --rep_name saved
	```
	to select a different learned representation, for example pass in `--rep_name s1` to use the seed 1 model (`saved` for the one used in the paper).  
	If you decide to not use the default name, then you will have to also pass the same `--rep_name` argument to other commands.  
	Note that 1 dynamics model is saved per representation type (`learned_rex`, `rex_ablation`, ...), so if you fine tune the dynamics on one representation model but forget to pass the `--rep_name` argument to the other commands, the resulting model will output garbage! If you want to use a different trained representation, you"ll have to re-fine tune the dynamics. One hint is if the logged network error is higher than the least squares error (we check and log this at the start of running the controller). Note that you can also pass in `--seed 1` here to learn the dynamics with a different seed (only the first seed in the list is used). Should take about 3 minutes.
4. also learn dynamics in original space for comparison later `python push_main.py fine_tune_dynamics --representation none`
5. run task with options
	```
	python push_main.py run --task "Block-H" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved
	python push_main.py run --task "Block-D" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved
	```
	```
	python peg_main.py run --task "Peg-U" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved_peg --tampc_param dynamics_minimum_window=15 --mpc_param horizon=15 --run_prefix h15_larger_min_window
	python peg_main.py run --task "Peg-I" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved_peg --tampc_param trap_cost_annealing_rate=0.95 --mpc_param horizon=20 --run_prefix h20_less_anneal
	python peg_main.py run --task "Peg-T" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved_peg
	python peg_main.py run --task "Peg-T(T)" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved_peg
	```
	there is a `--visualize_rollout` option to show what the planned MPC trajectory would take the state. Should take about 10 minutes per 500 frame run.
6. run tasks with adaptive baseline with the `--adaptive_baseline` option; for example
	```
	python push_main.py run --task "Block-H" --seed 0 1 2 3 4 5 6 7 8 9 --adaptive_baseline
	```
7. run tasks with artificial potential field (APF) baselines; for example
	```
	python peg_main.py run --task "Peg-U" --seed 0 1 2 3 4 5 6 7 8 9 --apfvo_baseline --representation learned_rex --rep_name saved_peg
	python peg_main.py run --task "Peg-U" --seed 0 1 2 3 4 5 6 7 8 9 --apfsp_baseline --representation learned_rex --rep_name saved_peg
	```
8. run tasks with random recovery policy ablation option; for example
	```
	python push_main.py run --task "Block-H" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --random_ablation --rep_name saved
	python push_main.py run --task "Block-D" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --random_ablation --rep_name saved
	```
9. run tasks with non-adaptive baseline with the `--nonadaptive_baseline` option; for example
	```
	python push_main.py run --task "Block-H" --seed 0 1 2 3 4 5 6 7 8 9 --nonadaptive_baseline
	```
10. run Peg-T(T) with dynamics in the original space
	```
	python peg_main.py run --task "Peg-T(T)" --seed 0 1 2 3 4 5 6 7 8 9 --representation none 
	```
11. run tasks with the no error estimation ablation with the `--never_estimate_error` option; for example
	```
	python peg_main.py run --task "Peg-T" --seed 0 1 2 3 4 5 6 7 8 9 --representation learned_rex --rep_name saved_peg --never_estimate_error
	```
12. evaluate the performance of the runs to prepare for visualization (saved to cache); for example (see the `_main.py` scripts' dictionary of runs to visualize for full list of names)
	```
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__NO_E__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__RANDOM__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__APFVO__NONE__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__APFSP__NONE__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__GP_KERNEL_INDEP_OUT__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	```
	```
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__NO_E__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__RANDOM__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__APFVO__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__APFSP__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	python push_main.py evaluate --eval_run_prefix auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	```
	```
	python peg_main.py evaluate --eval_run_prefix auto_recover__h15_larger_min_window__NONE__MAB__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__NO_E__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__RANDOM__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFVO__NONE__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFSP__NONE__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__GP_KERNEL_INDEP_OUT__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	```
	```
	python peg_main.py evaluate --eval_run_prefix auto_recover__h20_less_anneal__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__NO_E__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__RANDOM__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFVO__NONE__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFSP__NONE__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__GP_KERNEL_INDEP_OUT__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	```
	```
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__NO_E__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__RANDOM__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFVO__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFSP__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	```
	```
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__NO_E__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__RANDOM__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFVO__NONE__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__APFSP__NONE__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__NONE__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	python peg_main.py evaluate --eval_run_prefix auto_recover__GP_KERNEL_INDEP_OUT__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST
	```
	also run this for any other series you decide to perform (including the SAC baseline) with different names by passing in `--run_prefix myname`. 
13. plot the performance results (columns of figure 7); will also report the success rate for Table 1.
	```
	python push_main.py visualize
	python peg_main.py visualize1
	python peg_main.py visualize2
	```

## Controller parameters
To change matrix-valued controller parameters, modify the dictionaries after line 115 of `push_main.py` and 185 of `peg_main.py`. You can modify scalar values more simply by passing the arguments `--tampc_param p1name=p1value p2name=p2value` and simiarly for MPC parameters `--mpc_param ...`. See the lines mentioned above for the parameter names.

## SAC baseline
This is implemented in the `soft-actor-critic` repository; see the README inside it for training and running instructions (training the nominal policy may take up to 10 hours, but saved models are provided).
To plot the SAC results together with the other results, copy the result files such as `sac__5*` to either `tampc/data/pushing/` or `tampc/data/peg/`.

