# GREG
MATLAB R2023b codes of GREG in the paper entitled "Fast Data-driven Greedy Sensor Selection for Ridge Regression".

## Code
### F_GREG.m
Function of GREG.
* Args
  * X: All-measurement data matrix
  * Y: Target data matrix
  * p: Number of selected sensors
  * lambda: Reguralization parameter
* Returns
  * S: Selected sensors
  * time: computation times for sensor selection

### P_Selction4LS.m
Main program for Section V-A.

### P_Selction4PSP.m
Main program for Section V-B.
