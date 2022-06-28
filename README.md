# Cerjan-Miller Saddle Point Search Algorithm
Python implementation of Cerjan-Miller method for saddle point search using GAUSSIAN as calculation engine.

## Usage:
Create a `CM_TSS` object and pass the path to the `settings.json` file. The `settings.json` file is formated as (default values in parantheses):
```
{
    "N": {number of atoms} (required),
    "Dim": {dimesions of the system} (3),
    "working-dir": {path to directory with all the files} (required),
    "submit-f-dir": {path to Gaussian submission file} (required),
    "basis-f-name": {filename of the basis file} ("" skipped),
    "init-f-name": {filename of the initial structure} (required),
    "gaussian-f-name": {name for gaussian input file} ("in"), 
    "charge": {charge of the system} (0),
    "spin": {spin of the system} (1),
    "N-procs": {number of processors to use} (8),
    "max-iter": {max number of iterations} (10),
    "energy-header-calc": {Gaussian header for energy calculations} ("#P wB97XD/6-31G** nosymm force"),
    "hess-header-calc": {Gaussian header for Hessian calculations } ("#P b3lyp/6-31G** nosymm freq"),
    "conv-radius":{atom position convergence radius} (1e-6),
    "conv-grad":{gradient convergence radius} (1e-6),
    "R-trust": {Trust radius} (0.2)
}
```

A GAUSSIAN submission script is necessary. This is a system depandant file.

`basis-f-name` is the name of the file containing basis specifications if necessary and will be added to the bottom of the GAUSSIAN input file.

Input coordinates file should have the format:
```
{atomic number} {-1 for frozen 0 otherwise} {x} {y} {z}
```

There's an example calculation for HCN available under the `HCN` directory. The `CM_TSS.py` should either be added to PATH or copied to the same directory as the test script.

