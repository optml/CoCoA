# CoCoA - Distributed Optimization with Arbitrary Local Solvers

This code contains efficient C++ implementations of the communication efficient CoCoA+ distributed optimization framework, as described in [the paper](http://arxiv.org/abs/1512.04039) below.
MPI is used for communication accross machines.
The current code supports a variety of loss functions (such as Hinge loss, quadratic/least squares loss and logistic loss), with strongly convex regularizers. Different implementations of many local solvers are provided for comparison.
Accuracy certificates in the form of the duality gap are readily computed and can be plotted.

Both additive or averaging udpates of the local solvers are available, as supported by the CoCoA+ algorithm framework.

For the experiments in the paper, the code was run on the public Amazon EC2 cloud.

### Getting Started
use `./configure` and `make`. (This assumes you have `automake` installed on your system).
Assuming executable program is named "Cocoa":

```
mpirun -np 4 Cocoa -A data/a1a.4/a1a -l 0.0001 -C 50 -I 100 -f 3 -a 1 -M 0

The meaning of the parameters:
-A: path of dataset
-l: value of the regularization parameter \lambda
-C: number of outer iterations of CoCoA+ algorithm that you wish to run
-I: number of inner iterations of local solvers during each epoch
-f: type of loss functions, can be set
        0 (Logistic Loss),
        1 (Hinge Loss),
        2 (Squared Hinge Loss),
        3 (Quadratic Loss)
-a: set to 0 if using the default cocoa+ subproblems, set to 1 for using more aggressive subproblems 
-M: type of local solver, default 0 is using SDCA solver for the local subproblems
```

## References
The algorithmic framework is described in more detail in the following paper:

_Chenxin Ma, Jakub Konečný, Martin Jaggi, Virginia Smith, Michael I. Jordan, Peter Richtárik, Martin Takáč [Distributed Optimization with Arbitrary Local Solvers](http://arxiv.org/abs/1512.04039). Optimization Methods and Software, Vol23-Iss4, pages 813-848, 2017.
