# Wave equation

## L-BFGS without multigrid

```
./wave.py --optimizer lbfgsb --multigrid 0 --kimp 1 --every_factor 10
```

## L-BFGS with multigrid

```
./wave.py --optimizer lbfgsb --multigrid 1 --kimp 100 --every_factor 2
```

Output directory `out_wave`

* [`train.log`](https://cselab.github.io/odil/examples/out_wave/train.log)
* [`u_00002.png`](https://cselab.github.io/odil/examples/out_wave/u_00002.png)  
  <img src="https://cselab.github.io/odil/examples/out_wave/u_00002.png" height=200>

## Newton

```
./wave.py --optimizer newton --multigrid 0 --linsolver direct --every_factor 0.01
```

The multigrid decomposition has to be disabled for Newton, since the extra unknowns would make the problem underdetermined.
Newton's method needs newer iterations (`--every_factor` sets the factor for the number of steps).
