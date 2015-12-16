# CoCoA - Distributed Optimization with Arbitrary Local Solvers

This code contains efficient C++ implementations of the communication efficient CoCoA+ distributed optimization framework, as described in [the paper](http://arxiv.org/abs/1512.04039) below.
The current supports a variety of loss functions (such as Hinge loss, quadratic/least squares loss and logistic loss), with strongly convex regularizers. Different implementations of many local solvers are provided for comparison.
Accuracy certificates in the form of the duality gap are readily computed and can be plotted.

Both additive or averaging udpates of the local solvers are available, as supported by the CoCoA+ algorithm framework.

For the experiments in the paper, the code was run on the public Amazon EC2 cloud.

## References
The algorithmic framework is described in more detail in the following paper:

_Chenxin Ma, Jakub Konečný, Martin Jaggi, Virginia Smith, Michael I. Jordan, Peter Richtárik, Martin Takáč [Distributed Optimization with Arbitrary Local Solvers](http://arxiv.org/abs/1512.04039). arXiv 2015._
