# Inferring time from temperature

Run with L-BFGS
```
./heat_tmax.py
```

Run with Newton using direct linear solver with damping
```
./heat_tmax.py --optimizer newton --multigrid 0 --linsolver direct --every_factor 0.002 --linsolver_damp 1e-9
```
The multigrid decomposition has to be disabled for Newton, since the extra unknowns would make the problem underdetermined.
Newton's method needs fewer iterations (`--every_factor` sets the factor for the number of steps).
