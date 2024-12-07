# "Optimal control for a class of linear transport dominated systems via the shifted proper orthogonal decomposition".

## Prerequisites

Before running the scripts, please ensure that you have the necessary packages installed. You need to install the Conda package manager and use the provided environment files to create a Conda virtual environment.

## Setting up the Environment

After installing Conda:

For Mac users:

```bash
conda env create -f env_with_accel_Mac.yml
conda activate env_with_accel
```

For other OS:

```bash
conda env create -f env_with_accel_others.yml
conda activate env_with_accel
```


## Running the Scripts

This repository includes three script files that need to be executed to reproduce the results. The script files are:

1. `run_FOM.sh`  
   Usage: `./run_FOM.sh arg`  
   - The value of `arg` can be `1`, `2`, or `3`, corresponding to the three examples shown in the paper.  
   - This script runs the tests for the Full-Order Model (FOM) for all example problems.

2. `run_PODG.sh`  
   Usage: `./run_PODG.sh arg1 arg2`  
   - `arg1`: Same as for FOM  
   - `arg2`: Can be either `modes` for mode-based study or `tol` for tolerance-based study.  
   - This script runs the tests for the POD-Galerkin model.

3. `run_sPODG.sh`  
   Usage: `./run_sPODG.sh arg1 arg2`  
   - The arguments are the same as those in the POD-Galerkin case.  
   - This script runs the tests for the sPOD-Galerkin model.

## Plotting Results

Once all the runs are complete, the results can be visualized using the following plotting scripts:

1. `state_target_plot.py`  
   Usage: `python3 state_target_plot.py arg`  
   - This script plots the state and target snapshot profiles for the specified `arg`, where `arg` is `1`, `2`, or `3`, corresponding to the example problems.

2. `sPOD_vs_POD_SV_plot.py`  
   Usage: `python3 sPOD_vs_POD_SV_plot.py`  
   - This script plots the singular value decay of the POD and sPOD methods for a traveling wave system, which is a toy example for illustrating the advantage of sPOD over POD.

3. `J_vs_modes_plot.py`  
   Usage: `python3 J_vs_modes_plot.py arg`  
   - This script plots the cost functional value against the number of modes needed for the POD-Galerkin and sPOD-Galerkin methods. The argument `arg` specifies the example number.

4. `J_vs_runtime_plot.py`  
   Usage: `python3 J_vs_runtime_plot.py arg`  
   - This script plots the cost functional value against the computational time for the given `arg`, where `arg` specifies the example problem.

5. `control_adjoint_state_plot.py`  
   Usage: `python3 control_adjoint_state_plot.py arg`  
   - This script plots the combined snapshots for the optimal controls, adjoints, and optimal states for the specified `arg`, where `arg` specifies the example problem.


Researchers are encouraged to try out the examples and extend it for their own research problems.
